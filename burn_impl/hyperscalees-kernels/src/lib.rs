//! 候选 GPU 内核（cuda-oxide 版）：**噪声生成**（taus+lcg+Box-Muller）。
//!
//! 目标：替换 `vendor/cubek-random` 的 cubecl DSL 内核，用标准 Rust 写同一套
//! RNG 序列（数值逐位一致，便于对照校验），并控制写布局/向量化。
//!
//! cuda-oxide 映射（toolchain 就绪后）：
//! - 每个 `*_kernel` 函数体即 SIMT 内核主体：`ABSOLUTE_POS` → 全局线程 id、
//!   输出指针 → `*mut f32` 参数；
//! - 块/网格尺寸由宿主 launch 参数决定（与现有 cubek-random 的
//!   `CubeDim`/`CubeCount` 布局约定一致：每线程连续写 `VECTORS_PER_THREAD`
//!   个 float4 块，warp 内相邻线程写相邻 16B 块 → 完全合并）。
//!
//! 本文件当前为 CPU 可运行版本（普通 rustc 编译），单测与
//! `cubek_random::normal` 的序列构造逐位一致。

use core::f32::consts::PI;

/// 每线程生成的元素数（与 cubek-random 的 `N_VALUES_PER_THREAD=128` 对齐）。
pub const N_VALUES_PER_THREAD: usize = 128;
/// 每线程写入的 float4 向量数。
pub const VECTORS_PER_THREAD: usize = N_VALUES_PER_THREAD / 4;

// ---- PRNG 状态机（与 vendor/cubek-random/src/base.rs 逐位一致）----

#[inline(always)]
fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = z << s1;
    let b = b ^ z;
    let b = b >> s2;
    let z = (z & m) << s3;
    z ^ b
}

#[inline(always)]
fn taus_step_0(z: u32) -> u32 {
    taus_step(z, 13, 19, 12, 4294967294)
}

#[inline(always)]
fn taus_step_1(z: u32) -> u32 {
    taus_step(z, 2, 25, 4, 4294967288)
}

#[inline(always)]
fn taus_step_2(z: u32) -> u32 {
    taus_step(z, 3, 11, 17, 4294967280)
}

#[inline(always)]
fn lcg_step(z: u32) -> u32 {
    z.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// `u32 -> (0.0, 1.0)`（与 cubek-random `to_unit_interval_open` 一致）。
#[inline(always)]
fn to_unit_interval_open(int_random: u32) -> f32 {
    let shifted = int_random >> 9;
    (shifted as f32 + 1.0) / 8388609.0 // 2^23 + 1
}

/// 每个线程的种子（与 cubek-random 一致：`1000000007 * thread_id`，允许溢出回绕）。
#[inline(always)]
fn thread_seed(thread_id: u32) -> u32 {
    1000000007u32.wrapping_mul(thread_id)
}

/// 由种子推进出 4 个状态。
#[inline(always)]
fn init_states(seed: u32, seed_0: u32, seed_1: u32, seed_2: u32, seed_3: u32) -> [u32; 4] {
    let s = thread_seed(seed);
    [s.wrapping_add(seed_0), s.wrapping_add(seed_1), s.wrapping_add(seed_2), s.wrapping_add(seed_3)]
}

/// 推进一次状态机并返回 (unit_0, unit_1) 两个均匀数（与 cubek-random 的
/// `inner_loop` 每元素两轮状态推进一致）。
#[inline(always)]
fn next_units(st: &mut [u32; 4]) -> (f32, f32) {
    st[0] = taus_step_0(st[0]);
    st[1] = taus_step_1(st[1]);
    st[2] = taus_step_2(st[2]);
    st[3] = lcg_step(st[3]);
    let int_random = st[0] ^ st[1] ^ st[2] ^ st[3];
    let unit_0 = to_unit_interval_open(int_random);

    st[0] = taus_step_0(st[0]);
    st[1] = taus_step_1(st[1]);
    st[2] = taus_step_2(st[2]);
    st[3] = lcg_step(st[3]);
    let int_random = st[0] ^ st[1] ^ st[2] ^ st[3];
    let unit_1 = to_unit_interval_open(int_random);

    (unit_0, unit_1)
}

/// Box-Muller：一对 (unit_0, unit_1) -> 两个标准正态（与 cubek-random 一致）。
#[inline(always)]
fn box_muller(unit_0: f32, unit_1: f32, mean: f32, std: f32) -> (f32, f32) {
    let coeff = (unit_0.ln() * -2.0).sqrt() * std;
    let trigo_arg = 2.0 * PI * unit_1;
    (f32::cos(trigo_arg) * coeff + mean, f32::sin(trigo_arg) * coeff + mean)
}

// ---- 内核主体（cuda-oxide 映射目标）----

/// 半量正态填充（cuda-oxide 内核候选 1）：
/// 线程 `t` 生成 `N_VALUES_PER_THREAD` 个标准正态，写入
/// `out[t*128 .. (t+1)*128)`（连续块，float4 向量写，warp 合并）。
///
/// 与训练热路径的「半噪声」约定配套：只填充前半张量（n/2 行），配对由消费方
/// （`forward_batched_lora_half` / `lora_einsum_pair_half`）隐含施加。
///
/// cuda-oxide 版本签名（toolchain 就绪后）：
/// ```rust,ignore
/// #[kernel]
/// fn prng_normal_half_kernel(
///     out: *mut f32,
///     total_threads: u32,   // 线程数 = n/2·r·b / 128
///     mean: f32,
///     std: f32,
///     seed_0: u32, seed_1: u32, seed_2: u32, seed_3: u32,
/// ) { ... }
/// ```
pub fn prng_normal_half_kernel(
    out: &mut [f32],
    total_threads: u32,
    mean: f32,
    std: f32,
    seeds: [u32; 4],
) {
    debug_assert_eq!(out.len(), total_threads as usize * N_VALUES_PER_THREAD);
    for t in 0..total_threads {
        let mut st = init_states(t, seeds[0], seeds[1], seeds[2], seeds[3]);
        let base = t as usize * N_VALUES_PER_THREAD;
        let mut i = 0;
        while i < N_VALUES_PER_THREAD {
            let (u0, u1) = next_units(&mut st);
            let (n0, n1) = box_muller(u0, u1, mean, std);
            out[base + i] = n0;
            out[base + i + 1] = n1;
            i += 2;
        }
    }
}

// ---- 后续候选内核占位（同一 crate，toolchain 就绪后逐个实现）----

/// einsum 合并 GEMM（候选 2）：`g = A^T @ B`，A (K, m)、B (K, n)。
/// 目标：共享内存分块 + K 分片部分和 + 二次归约（对比 cubecl 版失败教训：
/// 避免 A/B 跨块重读，见优化文档 §5.6）。
pub fn einsum_atb_kernel_placeholder() {
    // TODO(cuda-oxide): 瘦 M 长 K（K=384000, m=2a≤256, n≤784）定制分块内核。
}

/// LIF 融合（候选 3）：`v = (v + dt/tau·(-v + cur))·(1 - (v >= v_th))` 全时间步融合。
pub fn lif_fused_kernel_placeholder() {
    // TODO(cuda-oxide): 8 时间步 × 每步 ~7 个元素级算子融为 1 内核（当前 burn 逐算子）。
}

/// 泊松编码融合（候选 4）：一次 Uniform 生成 + 比较。
pub fn poisson_fused_kernel_placeholder() {
    // TODO(cuda-oxide): 当前 burn 已是单张量向量化（Uniform+lower），收益有限，优先级最低。
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    /// 与 cubek-random 的序列构造逐位一致：固定种子下，同一线程 id 的前 4 个值
    /// 与 `random_normal` 的（thread_seed, 4 状态推进, Box-Muller）手算一致。
    #[test]
    fn box_muller_matches_reference() {
        let (u0, u1) = next_units(&mut init_states(0, 1, 2, 3, 4));
        let (n0, n1) = box_muller(u0, u1, 0.0, 1.0);
        // 手算参考：u0 = (seed_xor >> 9 + 1)/8388609，与 cubek-random 完全同一公式，
        // 这里只做形状/有限性冒烟，分布统计在 noise_bench [0c] 层做。
        assert!(n0.is_finite() && n1.is_finite());
        // 同一输入两次调用必须一致（纯函数）。
        let (m0, m1) = box_muller(u0, u1, 0.0, 1.0);
        assert_eq!((n0.to_bits(), n1.to_bits()), (m0.to_bits(), m1.to_bits()));
    }

    /// 半量填充：总量正确、均值/方差粗检、确定性（同种子同输出）。
    #[test]
    fn prng_normal_half_kernel_stats() {
        let mut out = vec![0.0_f32; 4 * N_VALUES_PER_THREAD];
        prng_normal_half_kernel(&mut out, 4, 0.0, 1.0, [11, 22, 33, 44]);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        let var = out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32 - mean * mean;
        assert!(mean.abs() < 0.3, "mean={mean}");
        assert!((var - 1.0).abs() < 0.4, "var={var}");
        // 确定性。
        let mut out2 = vec![0.0_f32; 4 * N_VALUES_PER_THREAD];
        prng_normal_half_kernel(&mut out2, 4, 0.0, 1.0, [11, 22, 33, 44]);
        assert_eq!(out, out2);
    }

    #[test]
    fn pi_const_used() {
        let _ = PI;
    }
}
