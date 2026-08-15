//! SNN-ES 训练「配对合并 einsum」定制内核（cuda-oxide 版）。
//!
//! 目标：替换 `lora_einsum_pair_cublas`（cuBLAS TF32 gemm_atb + cat/mul 预处理）在
//! 瘦 M 长 K 形状（M=2a ≤ 256, N=b, K=half·r = 384000）上的路径。
//!
//! 数学（与 `lora_einsum_pair_half` 完全一致，反对称配对 + 双负抵消）：
//!   g_raw[m, n]  = Σ_k A_fused[k, m] · B[k, n]，  m ∈ [0, a)
//!   g_ones[m, n] = 2 · Σ_k A[k, m] · B[k, n]，    m ∈ [0, a)
//! 其中 A_fused[k, m] = A[k, m] · f[i]（i = k / r，f[i] = scores[i] + scores[i+half]）。
//! 即把训练侧的「slice + f_pair 加权 + cat 拼接（(half,r,2a)）」全部**融合进内核**：
//! 共享内存平铺的 A 列 [0..a) 乘 f[i]、[a..2a) 原值，消除 a_w 乘法与 cat 拷贝。
//!
//! 并行化：split-K（每块一个 K 切片）+ 输出直接 f32 原子累加（`red.global.add.f32`）。
//! 每块覆盖完整 M=2a（≤ 256）与 112 列 N 切片；A 每列最多读 N/112 次（L2 命中），
//! B 只读一次 → 全局流量 ≈ A + B 各一遍。
//!
//! 输入布局：`A` (half, r, a) 行主序（可能带行对齐 pitch，stride 显式传入）、
//! `B` (half, r, b) 行主序、`scores` (n,)；输出 `g_raw`/`g_ones` (a, b) 行主序
//! （调用方先置零，内核原子累加）。

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{SharedArray, kernel, launch_bounds, thread};
use cuda_host::cuda_module;
use std::ffi::c_void;

// =============================================================================
// KERNEL —— 编译到 PTX
// =============================================================================

const BK: usize = 32; // K 平铺（共享 47KB，1 block/SM）
const BN: usize = 112; // N 平铺（7×16）
const M_MAX: usize = 256; // 2a 上限（宿主断言）
const APAD: usize = M_MAX + 1; // 奇数行填充，规避共享内存 bank 冲突
const THREADS: usize = 512; // 主内核：32(m 组) × 16(n 组)，8×7 寄存器块

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn f_pair(scores: *const f32, i: usize, half: usize) -> f32 {
        unsafe { *scores.add(i) + *scores.add(i + half) }
    }

    /// 静态展开的 FMA：acc[dm·7+dn] += av · bv[dn]（dm/dn 均为字面量 →
    /// 数组访问常量折叠 → LLVM SROA 提升到寄存器，避免 local memory 溢出）。
    macro_rules! fma_dm {
        ($acc:expr, $dm:literal, $av:expr, $b0:expr, $b1:expr, $b2:expr, $b3:expr, $b4:expr, $b5:expr, $b6:expr) => {
            $acc[($dm) * 7 + 0] += $av * $b0;
            $acc[($dm) * 7 + 1] += $av * $b1;
            $acc[($dm) * 7 + 2] += $av * $b2;
            $acc[($dm) * 7 + 3] += $av * $b3;
            $acc[($dm) * 7 + 4] += $av * $b4;
            $acc[($dm) * 7 + 5] += $av * $b5;
            $acc[($dm) * 7 + 6] += $av * $b6;
        };
    }

    /// 静态展开的原子输出：acc[dm·7+dn] → g_raw/g_ones（dm/dn 字面量）。
    macro_rules! atomic_dn {
        ($acc:expr, $dm:literal, $dn:literal, $m2:expr, $n:expr, $g_s:expr, $b_dim:expr, $a_dim:expr, $g_raw:expr, $g_ones:expr) => {
            if $n < $b_dim {
                let v = $acc[($dm) * 7 + ($dn)];
                unsafe {
                    if $m2 < $a_dim {
                        let p = $g_raw.add($m2 * $g_s + $n) as *const DeviceAtomicF32;
                        (*p).fetch_add(v, AtomicOrdering::Relaxed);
                    } else {
                        let p = $g_ones.add(($m2 - $a_dim) * $g_s + $n) as *const DeviceAtomicF32;
                        (*p).fetch_add(2.0 * v, AtomicOrdering::Relaxed);
                    }
                }
            }
        };
    }

    /// 一个 dm 组的 7 个原子输出（dm 字面量，m2 = tx + 16·dm 显式传入）。
    macro_rules! atomic_dm_group {
        ($acc:expr, $dm:literal, $m2:expr, $n_base:expr, $m2max:expr, $g_s:expr, $b_dim:expr, $a_dim:expr, $g_raw:expr, $g_ones:expr) => {
            if $m2 < $m2max {
                atomic_dn!($acc, $dm, 0, $m2, $n_base + 0, $g_s, $b_dim, $a_dim, $g_raw, $g_ones);
                atomic_dn!($acc, $dm, 1, $m2, $n_base + 1, $g_s, $b_dim, $a_dim, $g_raw, $g_ones);
                atomic_dn!($acc, $dm, 2, $m2, $n_base + 2, $g_s, $b_dim, $a_dim, $g_raw, $g_ones);
                atomic_dn!($acc, $dm, 3, $m2, $n_base + 3, $g_s, $b_dim, $a_dim, $g_raw, $g_ones);
                atomic_dn!($acc, $dm, 4, $m2, $n_base + 4, $g_s, $b_dim, $a_dim, $g_raw, $g_ones);
                atomic_dn!($acc, $dm, 5, $m2, $n_base + 5, $g_s, $b_dim, $a_dim, $g_raw, $g_ones);
                atomic_dn!($acc, $dm, 6, $m2, $n_base + 6, $g_s, $b_dim, $a_dim, $g_raw, $g_ones);
            }
        };
    }

    /// 融合配对 einsum：C = A_fused^T @ B（A 配对加权/拼接在共享加载时完成）。
    #[kernel]
    #[launch_bounds(512)]
    pub fn einsum_pair_fused(
        a_flat: *const f32, // (half, r, a) 行主序，A 半量噪声
        a_s0: u32,          // dim0 stride（元素单位）
        a_s1: u32,          // dim1 stride（= a 的 pitch）
        a: u32,             // a 维（M = 2a）
        b_flat: *const f32, // (half, r, b) 行主序
        b_s0: u32,          // dim0 stride
        b_s1: u32,          // dim1 stride（= b 的 pitch）
        b: u32,             // b 维（N）
        scores: *const f32, // (n,) 原始分数（含配对后半）
        half: u32,          // half = n/2
        r: u32,             // rank
        g_raw: *mut f32,    // (a, b) 输出（调用方已置零）
        g_ones: *mut f32,   // (a, b) 输出（调用方已置零，×2 在原子累加时折叠）
        g_s: u32,           // 输出张量的行 stride（元素单位，burn 有 256B 对齐 pitch）
        k_total: u32,       // K = half * r
        k_slices: u32,      // K 切片数（grid.y）
    ) {
        static mut A_SH: SharedArray<f32, { BK * APAD }> = SharedArray::UNINIT;
        static mut B_SH: SharedArray<f32, { BK * BN }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let tx = tid % 32; // m 组（8 列/组）
        let ty = tid / 32; // n 组（16 组 × 7 列 = 112）
        let n0 = (thread::blockIdx_x() as usize) * BN; // 本块 N 起始列
        let ks = ((k_total as usize) + (k_slices as usize) - 1) / (k_slices as usize);
        let k_start = (thread::blockIdx_y() as usize) * ks;
        let k_start_ks = k_start + ks;
        let k_end = if k_start_ks < k_total as usize { k_start_ks } else { k_total as usize };

        let a_dim = a as usize;
        let m2max = 2 * a_dim; // ≤ M_MAX（宿主断言）
        let b_dim = b as usize;
        let g_sz = g_s as usize;
        let hh = half as usize;
        let rr = r as usize;
        let a_off0 = a_s0 as usize;
        let a_off1 = a_s1 as usize;
        let b_off0 = b_s0 as usize;
        let b_off1 = b_s1 as usize;

        // 寄存器累加器（扁平，8×7=56）：acc[dm·7 + dn] = C[8·tx + 8·dm, n0 + 7·ty + dn]
        let mut acc = [0.0f32; 56];

        let mut kc = k_start;
        while kc < k_end {
            let knext = kc + BK;
            let kend2 = if knext < k_end { knext } else { k_end };
            let klen = kend2 - kc;

            // 加载 A 平铺（融合配对）：A_SH[k][m2] =
            //   m2 < a : a_flat[row][m2] · f(i)     （raw 半）
            //   m2 ≥ a : a_flat[row][m2 - a]        （ones 半）
            let mut idx = tid;
            while idx < klen * m2max {
                let k = idx / m2max;
                let m2 = idx % m2max;
                let row = kc + k;
                let i = row / rr;
                let off = i * a_off0 + (row - i * rr) * a_off1;
                unsafe {
                    if m2 < a_dim {
                        let f = f_pair(scores, i, hh);
                        A_SH[k * APAD + m2] = *a_flat.add(off + m2) * f;
                    } else {
                        A_SH[k * APAD + m2] = *a_flat.add(off + m2 - a_dim);
                    }
                }
                idx += THREADS;
            }
            // 加载 B 平铺
            let mut idx = tid;
            while idx < klen * BN {
                let k = idx / BN;
                let n = idx % BN;
                let nn = n0 + n;
                if nn < b_dim {
                    let row = kc + k;
                    let i = row / rr;
                    let off = i * b_off0 + (row - i * rr) * b_off1;
                    unsafe {
                        B_SH[k * BN + n] = *b_flat.add(off + nn);
                    }
                }
                idx += THREADS;
            }
            thread::sync_threads();

            // 计算：每线程 8×7 输出（全部静态展开 → acc 提升到寄存器）
            for k in 0..klen {
                let ka = k * APAD + 8 * tx;
                let kb = k * BN + 7 * ty;
                let bv0 = unsafe { B_SH[kb + 0] };
                let bv1 = unsafe { B_SH[kb + 1] };
                let bv2 = unsafe { B_SH[kb + 2] };
                let bv3 = unsafe { B_SH[kb + 3] };
                let bv4 = unsafe { B_SH[kb + 4] };
                let bv5 = unsafe { B_SH[kb + 5] };
                let bv6 = unsafe { B_SH[kb + 6] };
                fma_dm!(acc, 0, unsafe { A_SH[ka + 0] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
                fma_dm!(acc, 1, unsafe { A_SH[ka + 1] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
                fma_dm!(acc, 2, unsafe { A_SH[ka + 2] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
                fma_dm!(acc, 3, unsafe { A_SH[ka + 3] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
                fma_dm!(acc, 4, unsafe { A_SH[ka + 4] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
                fma_dm!(acc, 5, unsafe { A_SH[ka + 5] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
                fma_dm!(acc, 6, unsafe { A_SH[ka + 6] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
                fma_dm!(acc, 7, unsafe { A_SH[ka + 7] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            }
            thread::sync_threads();
            kc += klen;
        }

        // 原子累加输出（ones 折叠 ×2；每个 dm 组独立展开、m2 显式 = 8·tx + dm，
        // 避免循环内常量索引被提升导致错写 acc[0]）。
        let n_base = n0 + 7 * ty;
        atomic_dm_group!(acc, 0, 8 * tx + 0, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
        atomic_dm_group!(acc, 1, 8 * tx + 1, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
        atomic_dm_group!(acc, 2, 8 * tx + 2, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
        atomic_dm_group!(acc, 3, 8 * tx + 3, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
        atomic_dm_group!(acc, 4, 8 * tx + 4, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
        atomic_dm_group!(acc, 5, 8 * tx + 5, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
        atomic_dm_group!(acc, 6, 8 * tx + 6, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
        atomic_dm_group!(acc, 7, 8 * tx + 7, n_base, m2max, g_sz, b_dim, a_dim, g_raw, g_ones);
    }

    /// 诊断内核：与 einsum_pair_fused 相同的 A/B 平铺加载（只处理前 2 行），
    /// 把 A_SH / B_SH 前 2 行原样写到 global 输出，供宿主对比定位加载错误。
    #[kernel]
    #[launch_bounds(256)]
    pub fn einsum_pair_dump_tiles(
        a_flat: *const f32,
        a_s0: u32,
        a_s1: u32,
        a: u32,
        b_flat: *const f32,
        b_s0: u32,
        b_s1: u32,
        b: u32,
        scores: *const f32,
        half: u32,
        r: u32,
        a_dump: *mut f32, // 2·APAD floats
        b_dump: *mut f32, // 2·BN floats
        k_total: u32,
        k_slices: u32,
    ) {
        static mut A_SH: SharedArray<f32, { BK * APAD }> = SharedArray::UNINIT;
        static mut B_SH: SharedArray<f32, { BK * BN }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let a_dim = a as usize;
        let m2max = 2 * a_dim;
        let b_dim = b as usize;
        let hh = half as usize;
        let rr = r as usize;
        let a_off0 = a_s0 as usize;
        let a_off1 = a_s1 as usize;
        let b_off0 = b_s0 as usize;
        let b_off1 = b_s1 as usize;

        let klen = 2usize; // 只加载前 2 行
        let mut idx = tid;
        while idx < klen * m2max {
            let k = idx / m2max;
            let m2 = idx % m2max;
            let i = k / rr;
            let off = i * a_off0 + (k - i * rr) * a_off1;
            unsafe {
                if m2 < a_dim {
                    let f = f_pair(scores, i, hh);
                    A_SH[k * APAD + m2] = *a_flat.add(off + m2) * f;
                } else {
                    A_SH[k * APAD + m2] = *a_flat.add(off + m2 - a_dim);
                }
            }
            idx += THREADS;
        }
        let mut idx = tid;
        while idx < klen * BN {
            let k = idx / BN;
            let n = idx % BN;
            if n < b_dim {
                let i = k / rr;
                let off = i * b_off0 + (k - i * rr) * b_off1;
                unsafe {
                    B_SH[k * BN + n] = *b_flat.add(off + n);
                }
            }
            idx += THREADS;
        }
        thread::sync_threads();
        let mut idx = tid;
        while idx < klen * APAD {
            unsafe {
                *a_dump.add(idx) = A_SH[idx];
            }
            idx += THREADS;
        }
        let mut idx = tid;
        while idx < klen * BN {
            unsafe {
                *b_dump.add(idx) = B_SH[idx];
            }
            idx += THREADS;
        }
    }

    /// 诊断内核 v2：完整加载首个 chunk（klen=32）+ 一次 FMA 迭代，
    /// 输出 tx<2 的线程的 acc 槽（(tx·16+ty)·112 + dm·7+dn）。
    #[kernel]
    #[launch_bounds(256)]
    pub fn einsum_pair_dump_acc(
        a_flat: *const f32,
        a_s0: u32,
        a_s1: u32,
        a: u32,
        b_flat: *const f32,
        b_s0: u32,
        b_s1: u32,
        b: u32,
        scores: *const f32,
        half: u32,
        r: u32,
        acc_dump: *mut f32, // 2·16·112 floats
        k_total: u32,
        k_slices: u32,
    ) {
        static mut A_SH: SharedArray<f32, { 32 * APAD }> = SharedArray::UNINIT;
        static mut B_SH: SharedArray<f32, { 32 * BN }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let tx = tid % 16;
        let ty = tid / 16;
        let a_dim = a as usize;
        let m2max = 2 * a_dim;
        let b_dim = b as usize;
        let hh = half as usize;
        let rr = r as usize;
        let a_off0 = a_s0 as usize;
        let a_off1 = a_s1 as usize;
        let b_off0 = b_s0 as usize;
        let b_off1 = b_s1 as usize;

        let klen = 32usize; // 一个完整 chunk
        let mut idx = tid;
        while idx < klen * m2max {
            let k = idx / m2max;
            let m2 = idx % m2max;
            let i = k / rr;
            let off = i * a_off0 + (k - i * rr) * a_off1;
            unsafe {
                if m2 < a_dim {
                    let f = f_pair(scores, i, hh);
                    A_SH[k * APAD + m2] = *a_flat.add(off + m2) * f;
                } else {
                    A_SH[k * APAD + m2] = *a_flat.add(off + m2 - a_dim);
                }
            }
            idx += THREADS;
        }
        let mut idx = tid;
        while idx < klen * BN {
            let k = idx / BN;
            let n = idx % BN;
            if n < b_dim {
                let i = k / rr;
                let off = i * b_off0 + (k - i * rr) * b_off1;
                unsafe {
                    B_SH[k * BN + n] = *b_flat.add(off + n);
                }
            }
            idx += THREADS;
        }
        thread::sync_threads();

        // FMA（与主内核相同）
        let mut acc = [0.0f32; 112];
        for k in 0..klen {
            let ka = k * APAD + tx;
            let kb = k * BN + 7 * ty;
            let bv0 = unsafe { B_SH[kb + 0] };
            let bv1 = unsafe { B_SH[kb + 1] };
            let bv2 = unsafe { B_SH[kb + 2] };
            let bv3 = unsafe { B_SH[kb + 3] };
            let bv4 = unsafe { B_SH[kb + 4] };
            let bv5 = unsafe { B_SH[kb + 5] };
            let bv6 = unsafe { B_SH[kb + 6] };
            fma_dm!(acc, 0, unsafe { A_SH[ka + 0] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 1, unsafe { A_SH[ka + 16] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 2, unsafe { A_SH[ka + 32] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 3, unsafe { A_SH[ka + 48] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 4, unsafe { A_SH[ka + 64] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 5, unsafe { A_SH[ka + 80] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 6, unsafe { A_SH[ka + 96] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 7, unsafe { A_SH[ka + 112] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 8, unsafe { A_SH[ka + 128] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 9, unsafe { A_SH[ka + 144] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 10, unsafe { A_SH[ka + 160] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 11, unsafe { A_SH[ka + 176] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 12, unsafe { A_SH[ka + 192] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 13, unsafe { A_SH[ka + 208] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 14, unsafe { A_SH[ka + 224] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
            fma_dm!(acc, 15, unsafe { A_SH[ka + 240] }, bv0, bv1, bv2, bv3, bv4, bv5, bv6);
        }
        // 输出 tx<2 线程的 acc
        if tx < 2 {
            let base = (tx * 16 + ty) * 112;
            for i in 0..112 {
                unsafe {
                    *acc_dump.add(base + i) = acc[i];
                }
            }
        }
    }
}

// =============================================================================
// HOST —— CPU 参考验证 + 真实形状计时
// =============================================================================

/// CPU 参考（f64 累加，f32 输出）：g_raw / g_ones（×2 已折叠）。
fn cpu_ref(
    a_t: &[f32],
    a_pitch: usize,
    a: usize,
    b_t: &[f32],
    b_pitch: usize,
    b: usize,
    scores: &[f32],
    half: usize,
    r: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut raw = vec![0.0f64; a * b];
    let mut ones = vec![0.0f64; a * b];
    for i in 0..half {
        let f = scores[i] + scores[i + half];
        for j in 0..r {
            let kk = i * r + j;
            let ao = kk * a_pitch;
            let bo = kk * b_pitch;
            for m in 0..a {
                let av = a_t[ao + m] as f64;
                for n in 0..b {
                    let bv = b_t[bo + n] as f64;
                    raw[m * b + n] += av * f as f64 * bv;
                    ones[m * b + n] += av * bv;
                }
            }
        }
    }
    (
        raw.iter().map(|x| *x as f32).collect(),
        ones.iter().map(|x| (2.0 * x) as f32).collect(),
    )
}

/// 确定性伪随机（LCG）。
fn fill_rng(data: &mut [f32], seed: &mut u64, scale: f32) {
    for v in data.iter_mut() {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((*seed >> 33) as u32 as f64) / (1u64 << 31) as f64 - 1.0;
        *v = (u as f32) * scale;
    }
}

fn run_test(
    stream: &cuda_core::stream::CudaStream,
    module: &kernels::LoadedModule,
    half: usize,
    r: usize,
    a: usize,
    b: usize,
    seed: &mut u64,
) -> bool {
    // 默认 pitch = 16B 对齐（示例原行为）
    let a_pitch = (a * 4 + 15) / 16 * 4;
    let b_pitch = (b * 4 + 15) / 16 * 4;
    run_test_pitched(stream, module, half, r, a, b, a_pitch, b_pitch, seed)
}

fn run_test_pitched(
    stream: &cuda_core::stream::CudaStream,
    module: &kernels::LoadedModule,
    half: usize,
    r: usize,
    a: usize,
    b: usize,
    a_pitch: usize,
    b_pitch: usize,
    seed: &mut u64,
) -> bool {
    println!("--- shape (half={half}, r={r}, a={a}, b={b}, a_pitch={a_pitch}, b_pitch={b_pitch}) ---");
    let k = half * r;
    assert!(2 * a <= M_MAX, "2a={} 超出 M_MAX={M_MAX}", 2 * a);

    let mut a_t = vec![0.0f32; k * a_pitch];
    let mut b_t = vec![0.0f32; k * b_pitch];
    fill_rng(&mut a_t, seed, 1.0);
    fill_rng(&mut b_t, seed, 1.0);
    let mut scores = vec![0.0f32; 2 * half];
    fill_rng(&mut scores, seed, 1.0);

    let da = DeviceBuffer::<f32>::from_host(&stream, &a_t).unwrap();
    let db = DeviceBuffer::<f32>::from_host(&stream, &b_t).unwrap();
    let ds = DeviceBuffer::<f32>::from_host(&stream, &scores).unwrap();
    let d_raw = DeviceBuffer::<f32>::zeroed(&stream, a * b).unwrap();
    let d_ones = DeviceBuffer::<f32>::zeroed(&stream, a * b).unwrap();

    let n_tiles = (b + BN - 1) / BN;
    let k_slices = core::cmp::max(1, (k + 3999) / 4000);
    let cfg = LaunchConfig {
        grid_dim: (n_tiles as u32, k_slices as u32, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module
        .einsum_pair_fused(
            stream,
            cfg,
            da.cu_deviceptr() as *const f32,
            (r * a_pitch) as u32,
            a_pitch as u32,
            a as u32,
            db.cu_deviceptr() as *const f32,
            (r * b_pitch) as u32,
            b_pitch as u32,
            b as u32,
            ds.cu_deviceptr() as *const f32,
            half as u32,
            r as u32,
            d_raw.cu_deviceptr() as *mut f32,
            d_ones.cu_deviceptr() as *mut f32,
            b as u32,
            k as u32,
            k_slices as u32,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let g_raw = d_raw.to_host_vec(&stream).unwrap();
    let g_ones = d_ones.to_host_vec(&stream).unwrap();
    let (ref_raw, ref_ones) = cpu_ref(&a_t, a_pitch, a, &b_t, b_pitch, b, &scores, half, r);

    // dump 诊断：A_SH/B_SH 前 2 行 vs CPU 期望
    {
        let mut d_adump = DeviceBuffer::<f32>::zeroed(&stream, 2 * APAD).unwrap();
        let mut d_bdump = DeviceBuffer::<f32>::zeroed(&stream, 2 * BN).unwrap();
        module
            .einsum_pair_dump_tiles(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                da.cu_deviceptr() as *const f32,
                (r * a_pitch) as u32,
                a_pitch as u32,
                a as u32,
                db.cu_deviceptr() as *const f32,
                (r * b_pitch) as u32,
                b_pitch as u32,
                b as u32,
                ds.cu_deviceptr() as *const f32,
                half as u32,
                r as u32,
                d_adump.cu_deviceptr() as *mut f32,
                d_bdump.cu_deviceptr() as *mut f32,
                k as u32,
                1u32,
            )
            .unwrap();
        stream.synchronize().unwrap();
        let ad = d_adump.to_host_vec(&stream).unwrap();
        let bd = d_bdump.to_host_vec(&stream).unwrap();
        let mut bad_a = 0usize;
        for k in 0..2usize {
            let i = k / r;
            let off = i * (r * a_pitch) + (k - i * r) * a_pitch;
            for m2 in 0..2 * a {
                let exp = if m2 < a {
                    a_t[off + m2] * (scores[i] + scores[i + half])
                } else {
                    a_t[off + m2 - a]
                };
                let got = ad[k * APAD + m2];
                if (got - exp).abs() > 1e-4 * (1.0 + exp.abs()) {
                    if bad_a < 4 {
                        println!("  [dump] A_SH[{k}][{m2}] got={got:.4} exp={exp:.4}  ✗");
                    }
                    bad_a += 1;
                }
            }
        }
        let mut bad_b = 0usize;
        for k in 0..2usize {
            let i = k / r;
            let off = i * (r * b_pitch) + (k - i * r) * b_pitch;
            for n in 0..b.min(BN) {
                let exp = b_t[off + n];
                let got = bd[k * BN + n];
                if (got - exp).abs() > 1e-4 * (1.0 + exp.abs()) {
                    if bad_b < 4 {
                        println!("  [dump] B_SH[{k}][{n}] got={got:.4} exp={exp:.4}  ✗");
                    }
                    bad_b += 1;
                }
            }
        }
        println!(
            "  [dump] A_SH bad={bad_a}/{}  B_SH bad={bad_b}/{}",
            2 * 2 * a,
            2 * b
        );

        // A8 手动 kernelParams 对照（模拟 burn 侧启动方式）
        {
            use cuda_core::launch_kernel_on_stream;
            let cm = module.as_cuda_module().clone();
            let func = cm.load_function("einsum_pair_fused").unwrap();
            let mut d_r2 = DeviceBuffer::<f32>::zeroed(&stream, a * b).unwrap();
            let mut d_o2 = DeviceBuffer::<f32>::zeroed(&stream, a * b).unwrap();
            #[repr(C, align(8))]
            struct A8<T>(T);
            let mut p1 = A8(da.cu_deviceptr() as *mut c_void);
            let mut p2 = A8((r * a_pitch) as u32);
            let mut p3 = A8(a_pitch as u32);
            let mut p4 = A8(a as u32);
            let mut p5 = A8(db.cu_deviceptr() as *mut c_void);
            let mut p6 = A8((r * b_pitch) as u32);
            let mut p7 = A8(b_pitch as u32);
            let mut p8 = A8(b as u32);
            let mut p9 = A8(ds.cu_deviceptr() as *mut c_void);
            let mut p10 = A8(half as u32);
            let mut p11 = A8(r as u32);
            let mut p12 = A8(d_r2.cu_deviceptr() as *mut c_void);
            let mut p13 = A8(d_o2.cu_deviceptr() as *mut c_void);
            let mut p14 = A8(b as u32);
            let mut p15 = A8(k as u32);
            let mut p16 = A8(1u32);
            let mut args: [*mut c_void; 16] = [
                &mut p1.0 as *mut *mut c_void as *mut c_void,
                &mut p2.0 as *mut u32 as *mut c_void,
                &mut p3.0 as *mut u32 as *mut c_void,
                &mut p4.0 as *mut u32 as *mut c_void,
                &mut p5.0 as *mut *mut c_void as *mut c_void,
                &mut p6.0 as *mut u32 as *mut c_void,
                &mut p7.0 as *mut u32 as *mut c_void,
                &mut p8.0 as *mut u32 as *mut c_void,
                &mut p9.0 as *mut *mut c_void as *mut c_void,
                &mut p10.0 as *mut u32 as *mut c_void,
                &mut p11.0 as *mut u32 as *mut c_void,
                &mut p12.0 as *mut *mut c_void as *mut c_void,
                &mut p13.0 as *mut *mut c_void as *mut c_void,
                &mut p14.0 as *mut u32 as *mut c_void,
                &mut p15.0 as *mut u32 as *mut c_void,
                &mut p16.0 as *mut u32 as *mut c_void,
            ];
            let n_tiles = (b + BN - 1) / BN;
            let k_slices = core::cmp::max(1, (k + 3999) / 4000);
            unsafe {
                launch_kernel_on_stream(
                    &func,
                    (n_tiles as u32, k_slices as u32, 1),
                    (THREADS as u32, 1, 1),
                    0,
                    &stream,
                    &mut args,
                )
                .unwrap();
            }
            stream.synchronize().unwrap();
            let g2 = d_r2.to_host_vec(&stream).unwrap();
            let d2 = g2
                .iter()
                .zip(ref_raw.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("  [A8] maxdiff vs ref = {d2:.3e}");
        }
    }

    let d_raw = g_raw
        .iter()
        .zip(ref_raw.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let d_ones = g_ones
        .iter()
        .zip(ref_ones.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // 打印前几个失配位置（m, n, got, ref）以定位错误模式
    let mut shown = 0;
    for m in 0..a {
        for n in 0..b {
            let i = m * b + n;
            if (g_raw[i] - ref_raw[i]).abs() > 1e-3 * (1.0 + ref_raw[i].abs()) && shown < 6 {
                println!(
                    "  [raw] m={m} n={n} got={:.3} ref={:.3}",
                    g_raw[i], ref_raw[i]
                );
                shown += 1;
            }
        }
    }
    if shown == 0 {
        for m in 0..a {
            for n in 0..b {
                let i = m * b + n;
                if (g_ones[i] - ref_ones[i]).abs() > 1e-3 * (1.0 + ref_ones[i].abs()) && shown < 6 {
                    println!(
                        "  [ones] m={m} n={n} got={:.3} ref={:.3}",
                        g_ones[i], ref_ones[i]
                    );
                    shown += 1;
                }
            }
        }
    }
    let scale = ref_raw.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    println!("  max|Δ raw| = {d_raw:.3e}  max|Δ ones| = {d_ones:.3e}  (ref 量级 {scale:.1})");
    d_raw.max(d_ones) < 1e-3 * (1.0 + scale)
}

fn main() {
    println!("=== SNN einsum_pair_fused (cuda-oxide) ===\n");
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();
    let mut seed = 0x1234_5678_9abc_def0u64;

    // 转储嵌入的 payload（诊断 burn 侧 PTX 差异用）
    {
        use cuda_core::embedded::{ArtifactPayloadKind, embedded_modules_from_current_exe};
        for module in embedded_modules_from_current_exe().unwrap() {
            println!("bundle: {}", module.name());
            for kind in [
                ArtifactPayloadKind::Cubin,
                ArtifactPayloadKind::Ptx,
                ArtifactPayloadKind::NvvmIr,
                ArtifactPayloadKind::Ltoir,
            ] {
                if let Some(p) = module.payload(kind) {
                    println!("  payload {:?}: {} bytes", kind, p.len());
                }
            }
            if let Some(p) = module.payload(ArtifactPayloadKind::Ptx) {
                std::fs::write("embedded_dump.ptx", p).unwrap();
                println!("  wrote embedded_dump.ptx ({} bytes)", p.len());
            }
        }
    }

    let mut ok = true;
    // 1) 极小形状（近似 [0b] 校验）
    ok &= run_test(&stream, &module, 8, 3, 2, 3, &mut seed);
    // 1a) chunk 数二分：K=32（1 chunk）/ K=64（2 chunks）/ K=128（4 chunks）
    ok &= run_test(&stream, &module, 2, 16, 8, 8, &mut seed);
    ok &= run_test(&stream, &module, 2, 32, 8, 8, &mut seed);
    ok &= run_test(&stream, &module, 4, 32, 8, 8, &mut seed);
    // 1a2) chunk 数二分续：K=512（16）/ K=1024（32）/ K=2048（64）
    ok &= run_test(&stream, &module, 8, 64, 8, 8, &mut seed);
    ok &= run_test(&stream, &module, 16, 64, 8, 8, &mut seed);
    ok &= run_test(&stream, &module, 32, 64, 8, 8, &mut seed);
    // 1a3) 交叉：大 K × 小 m2max（m2max=16, K=3200, b=128）
    ok &= run_test(&stream, &module, 50, 64, 8, 128, &mut seed);
    // 1a4) 交叉：大 m2max × 小 b（m2max=128, K=3200, b=8）
    ok &= run_test(&stream, &module, 50, 64, 64, 8, &mut seed);
    // 1a5) m2max 阈值二分（K=128, 4 chunks）：a=12/16/24/32
    ok &= run_test(&stream, &module, 2, 64, 12, 8, &mut seed);
    ok &= run_test(&stream, &module, 2, 64, 16, 8, &mut seed);
    ok &= run_test(&stream, &module, 2, 64, 24, 8, &mut seed);
    ok &= run_test(&stream, &module, 2, 64, 32, 8, &mut seed);
    // 1b) 诊断：单 K 切片、多 N 块（K=3200 → k_slices=1）
    ok &= run_test(&stream, &module, 50, 64, 64, 128, &mut seed);
    // 1c) 诊断：多 K 切片、单 N 块（K=6400 → k_slices=2）
    ok &= run_test(&stream, &module, 100, 64, 64, 48, &mut seed);
    // 2) 中层形状（含 pitch、N 单块）
    ok &= run_test(&stream, &module, 100, 64, 64, 128, &mut seed);
    // 3) fc3 形状（a=10 → M=20，小 M 路径）
    ok &= run_test(&stream, &module, 64, 64, 10, 128, &mut seed);
    // 4) fc1 形状（N=784 → 7 个 N 块，多 K 切片）
    ok &= run_test(&stream, &module, 200, 64, 128, 784, &mut seed);
    // 4b) noise_bench [0m] 同形状（half=1000, r=16, a=32, b=48）
    ok &= run_test(&stream, &module, 1000, 16, 32, 48, &mut seed);
    // 4c) pitch 变体：b=48 但 b_pitch=64（burn 的 256B 行对齐策略）
    ok &= run_test_pitched(&stream, &module, 1000, 16, 32, 48, 32, 64, &mut seed);
    // 4d) burn [0m] 最小化形状：K=128（4 chunks），r=64, a=32, b=48（pitch 64）
    ok &= run_test_pitched(&stream, &module, 2, 64, 32, 48, 32, 64, &mut seed);
    // 4e) 同形状无 pitch（对照）
    ok &= run_test(&stream, &module, 2, 64, 32, 48, &mut seed);

    if ok {
        println!("\n✓ 正确性全部通过");
    } else {
        println!("\n✗ 存在失败项");
        std::process::exit(1);
    }

    // 5) 真实形状计时：(6000, 64, 128, 784)，K=384000
    println!("\n--- 真实形状计时 (half=6000, r=64, a=128, b=784) ---");
    let half = 6000usize;
    let r = 64usize;
    let a = 128usize;
    let b = 784usize;
    let k = half * r;
    let a_pitch = a;
    let b_pitch = b;
    let n_tiles = (b + BN - 1) / BN;
    let k_slices = core::cmp::max(1, (k + 3999) / 4000);
    let cfg = LaunchConfig {
        grid_dim: (n_tiles as u32, k_slices as u32, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let da = DeviceBuffer::<f32>::from_host(&stream, &a_t_real(half, r, a)).unwrap();
    let db = DeviceBuffer::<f32>::from_host(&stream, &b_t_real(half, r, b)).unwrap();
    let ds = DeviceBuffer::<f32>::from_host(&stream, &scores_real(2 * half)).unwrap();
    let d_raw = DeviceBuffer::<f32>::zeroed(&stream, a * b).unwrap();
    let d_ones = DeviceBuffer::<f32>::zeroed(&stream, a * b).unwrap();

    // 预热
    module
        .einsum_pair_fused(
            &stream,
            cfg,
            da.cu_deviceptr() as *const f32,
            (r * a_pitch) as u32,
            a_pitch as u32,
            a as u32,
            db.cu_deviceptr() as *const f32,
            (r * b_pitch) as u32,
            b_pitch as u32,
            b as u32,
            ds.cu_deviceptr() as *const f32,
            half as u32,
            r as u32,
            d_raw.cu_deviceptr() as *mut f32,
            d_ones.cu_deviceptr() as *mut f32,
            b as u32,
            k as u32,
            k_slices as u32,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        module
            .einsum_pair_fused(
                &stream,
                cfg,
                da.cu_deviceptr() as *const f32,
                (r * a_pitch) as u32,
                a_pitch as u32,
                a as u32,
                db.cu_deviceptr() as *const f32,
                (r * b_pitch) as u32,
                b_pitch as u32,
                b as u32,
                ds.cu_deviceptr() as *const f32,
                half as u32,
                r as u32,
                d_raw.cu_deviceptr() as *mut f32,
                d_ones.cu_deviceptr() as *mut f32,
                b as u32,
                k as u32,
                k_slices as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
    let flops = 2.0 * (2 * a) as f64 * b as f64 * k as f64;
    println!(
        "  单次内核 = {ms:.3} ms  ({:.1} TFLOP/s, 流量 {:.2} GB)",
        flops / ms / 1e9,
        (2 * a) as f64 * 4.0 * k as f64 / 1e9 + b as f64 * 4.0 * k as f64 / 1e9
    );
}

fn a_t_real(half: usize, r: usize, a: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; half * r * a];
    let mut s = 42u64;
    fill_rng(&mut v, &mut s, 0.025);
    v
}
fn b_t_real(half: usize, r: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; half * r * b];
    let mut s = 43u64;
    fill_rng(&mut v, &mut s, 1.0);
    v
}
fn scores_real(n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    let mut s = 44u64;
    fill_rng(&mut v, &mut s, 1.0);
    v
}
