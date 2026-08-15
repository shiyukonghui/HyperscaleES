//! 融合泊松编码内核（cuda-oxide 版，阶段 C-4）。
//!
//! 构建源：`cuda-oxide-0.2.1/.../examples/snn_poisson/src/main.rs`（外部源码树，
//! 不入库）；本文件为内核源码存档（与构建源保持一致）。
//!
//! 目标：替换 `hyperscalees_envs::snn_mnist::poisson_encode` 的
//! 「(t, batch, in) Uniform 随机 + lower 比较 + float 转换」多内核路径
//! （[5b] poisson_single ≈ 2.4ms/chunk），改为单次启动：每线程一个元素，沿 T
//! 用 xorshift32 现场生成 Uniform(0,1) 并与像素强度比较（[4P] ≈ 1.6ms/chunk）。
//!
//! 语义：统计等价（每元素每时间步独立 Bernoulli，发放率 ≈ 像素值）；与 burn 版
//! 不要求逐位一致（参考实现即如此）。
//!
//! ## 踩坑记录（详见集成文档 §10）
//!
//! 1. **LLVM 优化 bug**：taus+lcg 四状态机的 `int_random = s0^s1^s2^s3` 在
//!    opt -O2/-O3 后被错写成用旧的部分值（taus_step_1 的 `(z<<4)` 替代完整
//!    结果、s0' 丢失）——发放率与 p 无关（平 0.387）。`#[noinline]` 无效、
//!    编译期展开无效、-O1 同样触发。换成单变量自依赖的 **xorshift32** 后正确。
//! 2. **burn 256B 行对齐 pitch**：784 f32 = 3136B → 行 stride 832 ≠ 784。
//!    内核必须按 `row·s + col` 寻址（同 einsum 的 g_s 机制）；扁平假设下
//!    发放率与 p 无关（读写全部错位）。

use cuda_device::{kernel, launch_bounds, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNEL —— 编译到 PTX
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn thread_seed(thread_id: u32) -> u32 {
        1000000007u32.wrapping_mul(thread_id)
    }

    /// xorshift32 步进（统计均匀，u32 -> (0,1) 取高 23 位）。
    /// 注意：taus+lcg 四状态机在此编译器上会被 LLVM 优化错误改写（见文档 §10）；
    /// xorshift32 是单变量自依赖链，未触发该 bug。泊松编码只要求统计等价。
    #[inline(always)]
    fn next_unit(x: &mut u32) -> f32 {
        let mut v = *x;
        v ^= v << 13;
        v ^= v >> 17;
        v ^= v << 5;
        *x = v;
        let shifted = v >> 9;
        (shifted as f32 + 1.0) / 8388609.0 // 2^23 + 1
    }

    /// 融合泊松编码：`out[t][idx] = (u_t < probs[idx]) ? 1 : 0`。
    /// 每线程处理一个元素，沿 T 每步推进一次状态机生成一个均匀数。
    ///
    /// 布局（burn 256B 行对齐 pitch）：`probs`/`out` 行 stride 可能 ≠ in_dim
    /// （如 784 f32 = 3136B → pitch 832），按 (row·s + col) 寻址，不做扁平假设。
    #[kernel]
    #[launch_bounds(256)]
    pub fn poisson_encode_fused(
        probs: *const f32, // (batch, in_dim) 行主序，行 stride = s
        out: *mut f32,     // (t, batch, in_dim) 行主序，行 stride = s
        total: u32,        // batch * in_dim
        in_dim: u32,       // 每行元素数
        s: u32,            // 行 stride（元素单位，burn 256B 对齐 pitch）
        t: u32,            // 时间步 T（宿主保证 ≤ 8）
        seed_0: u32,
        seed_1: u32,
        seed_2: u32,
        seed_3: u32,
    ) {
        let idx =
            (thread::threadIdx_x() + thread::blockDim_x() * thread::blockIdx_x()) as usize;
        if idx >= total as usize {
            return;
        }
        let row = idx / in_dim as usize;
        let col = idx % in_dim as usize;
        let base = row * s as usize + col;
        let out_step = (total as usize / in_dim as usize) * s as usize;
        let p = unsafe { *probs.add(base) };
        // 单状态 xorshift32：种子混合（+0x9E37_79B9 保证非零——xorshift 全零态退化）。
        let mut x = thread_seed(idx as u32)
            .wrapping_add(seed_0 ^ seed_1 ^ seed_2 ^ seed_3)
            .wrapping_add(0x9E37_79B9);
        macro_rules! step {
            ($tt:literal) => {{
                if $tt < t {
                    let u = next_unit(&mut x);
                    let spike = if u < p { 1.0f32 } else { 0.0f32 };
                    unsafe {
                        *out.add($tt as usize * out_step + base) = spike;
                    }
                }
            }};
        }
        step!(0);
        step!(1);
        step!(2);
        step!(3);
        step!(4);
        step!(5);
        step!(6);
        step!(7);
    }
}
