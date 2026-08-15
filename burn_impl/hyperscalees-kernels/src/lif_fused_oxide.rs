//! 融合 LIF 扫描内核（cuda-oxide 版，阶段 C-3）。
//!
//! 构建源：`cuda-oxide-0.2.1/.../examples/snn_lif/src/main.rs`（外部源码树，不入库）；
//! 本文件为内核源码存档（与构建源保持一致）。
//!
//! 目标：替换 `hyperscalees_models::snn::run_lif` 在训练热路径上的一次次
//! 逐时间步元素级内核（[8] lif_loop ≈ 1.7ms/chunk：T 步 × 每步 ~5 个 kernel
//! launch + 中间张量分配），改为单次启动、每线程一个 (n, h) 元素沿 T 扫描。
//!
//! 数学（与 `lif_step` 逐位一致）：
//!   charged = v + (cur - v) · (1/tau_m)
//!   spike   = (charged >= v_th) ? 1 : 0
//!   v       = charged · (1 - spike)        （hard reset）
//!
//! 布局：`cur` (T, total) 行主序连续、`v0` (total,)、`out` (T, total)（0/1 f32）。
//! 数据契约与 einsum 内核相同：裸指针 + 显式形状，宿主侧经 cudarc 驱动 API
//! 在 cubecl 主流上启动（同流有序、零同步）。

use cuda_device::{kernel, launch_bounds, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNEL —— 编译到 PTX
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// 融合 LIF 扫描：每线程处理一个 (n, h) 元素，沿 T 顺序扫描。
    #[kernel]
    #[launch_bounds(256)]
    pub fn lif_fused(
        cur: *const f32, // (T, total) 行主序连续
        v0: *const f32,  // (total,)
        out: *mut f32,   // (T, total)，写 0/1
        total: u32,      // n * h
        t: u32,          // 时间步 T
        tau_m: f32,
        v_th: f32,
    ) {
        let idx = (thread::threadIdx_x() + thread::blockDim_x() * thread::blockIdx_x()) as usize;
        if idx >= total as usize {
            return;
        }
        let leak = 1.0f32 / tau_m;
        let mut v = unsafe { *v0.add(idx) };
        let mut tt = 0usize;
        while tt < t as usize {
            let c = unsafe { *cur.add(tt * total as usize + idx) };
            // FMA 友好：v + (c - v)·leak
            let charged = v + (c - v) * leak;
            let spike = if charged >= v_th { 1.0f32 } else { 0.0f32 };
            v = charged * (1.0f32 - spike);
            unsafe {
                *out.add(tt * total as usize + idx) = spike;
            }
            tt += 1;
        }
    }
}
