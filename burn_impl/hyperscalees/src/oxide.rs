//! cuda-oxide 内核的宿主侧加载/启动封装（仅 `gpu` feature）。
//!
//! 集成路径（见 `docs/cuda_oxide_integration_plan.md`）：
//! 用 cuda-oxide 编译器把 `hyperscalees-kernels` 的 Rust 内核编译为 PTX 文本，
//! 宿主经 CUDA driver API 加载（`cuModuleLoadData`）并启动（`cuLaunchKernel`）。
//!
//! 与 [`crate::cublas`] 同一套机制：module/function 挂在 cubecl 的 context 上
//! （复用 `cublas::state` 的 ctx），启动绑到 cubecl 的原始 stream
//! （`raw_stream`）→ 与 burn 算子天然同流有序、零同步。
//!
//! 内核参数直接传 burn 张量的原始设备指针（`cublas::raw_ptr` 同款 resolve
//! 机制）与显式标量（kernelParams 指向参数值，见 [`launch`]）。

use std::ffi::c_void;

use burn::backend::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;
use cubecl::device::Device;
use cubecl::device_handle::DeviceHandle;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;

use crate::cublas::state as cublas_state;

/// 服务器类型（vendored cubecl-cuda 的 `CudaServer`）。
type Server = <CudaRuntime as Runtime>::Server;

/// 已加载的 cuda-oxide 内核（module + function 句柄）。
pub struct OxideKernel {
    module: cudarc::driver::sys::CUmodule,
    function: cudarc::driver::sys::CUfunction,
}

/// 供探针/集成代码访问 function 句柄。
pub fn kernel_function(kernel: &OxideKernel) -> cudarc::driver::sys::CUfunction {
    kernel.function
}

// 句柄只在本模块内按序使用，不跨线程共享可变访问（与 CublasState 相同约定）。
unsafe impl Send for OxideKernel {}
unsafe impl Sync for OxideKernel {}

impl Drop for OxideKernel {
    fn drop(&mut self) {
        // 释放 module（function 句柄随 module 失效，无需单独释放）。
        unsafe {
            cudarc::driver::sys::cuModuleUnload(self.module);
        }
    }
}

/// 从 PTX 文本加载一个内核函数。
///
/// `ptx` 为 cuda-oxide 编译产出的 PTX 文本字节（**必须以 NUL 结尾**，
/// `cuModuleLoadData` 要求）；`kernel_name` 为 PTX 内的函数名。
/// 加载后内核与 cubecl 共享同一 context（复用 cuBLAS 的 context）。
pub fn load_kernel(
    device: &CudaDevice,
    ptx: &[u8],
    kernel_name: &str,
) -> Result<OxideKernel, String> {
    let st = cublas_state(device);
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx)
            .map_err(|e| format!("设置 CUDA 上下文失败: {e}"))?;

        let mut module = std::mem::MaybeUninit::<cudarc::driver::sys::CUmodule>::uninit();
        let status = cudarc::driver::sys::cuModuleLoadData(
            module.as_mut_ptr(),
            ptx.as_ptr() as *const c_void,
        );
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuModuleLoadData 失败: {status:?}"));
        }
        let module = module.assume_init();

        let mut function = std::mem::MaybeUninit::<cudarc::driver::sys::CUfunction>::uninit();
        let name =
            std::ffi::CString::new(kernel_name).map_err(|_| "内核名含 NUL".to_string())?;
        let status = cudarc::driver::sys::cuModuleGetFunction(
            function.as_mut_ptr(),
            module,
            name.as_ptr(),
        );
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            let _ = cudarc::driver::sys::cuModuleUnload(module);
            return Err(format!("cuModuleGetFunction({kernel_name}) 失败: {status:?}"));
        }
        Ok(OxideKernel {
            module,
            function: function.assume_init(),
        })
    }
}

/// 启动内核（绑到 cubecl 主 stream，零同步；与 cuBLAS 调用同序）。
///
/// `args`：内核参数的指针数组——每个元素是指向**参数值**的指针
/// （`&mut arg as *mut _ as *mut c_void`），与 `cuLaunchKernel` 的
/// `kernelParams` 约定一致。调用方保证参数布局与 PTX 内核签名匹配。
/// 注意：参数值必须由**具名变量**承载并存活到 launch 返回（临时值会悬垂）。
///
/// # Safety
/// 参数指针必须指向与内核签名匹配的值且在内核执行期间有效。
pub unsafe fn launch(
    kernel: &OxideKernel,
    device: &CudaDevice,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
    args: &mut [*mut c_void],
) -> Result<(), String> {
    let st = cublas_state(device);

    // 直接 launch 到 cubecl 主流（同流有序、零同步；与 cuBLAS 集成同机制）。
    // 注意：cuLaunchKernel 参数顺序是 (..., hStream, kernelParams, extra)——kernelParams
    // 传 args 数组、extra 传 null（曾把两者写反导致 CUDA_ERROR_INVALID_VALUE）。
    let dh = DeviceHandle::<Server>::new(device.to_id());
    let stream = dh
        .submit_blocking(|s| s.raw_stream(StreamId::current()) as usize)
        .expect("取 CUDA stream 失败") as *mut cudarc::driver::sys::CUstream_st;
    cudarc::driver::result::ctx::set_current(st.ctx)
        .map_err(|e| format!("设置 CUDA 上下文失败: {e}"))?;

    let status = cudarc::driver::sys::cuLaunchKernel(
        kernel.function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_mem_bytes,
        stream,
        args.as_mut_ptr(),    // kernelParams
        std::ptr::null_mut(), // extra
    );
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuLaunchKernel 失败: {status:?}"));
    }
    Ok(())
}

/// 在指定流上启动内核（内部路径；供探针/特殊场景复用）。
pub unsafe fn launch_on_stream(
    kernel: &OxideKernel,
    device: &CudaDevice,
    stream: *mut cudarc::driver::sys::CUstream_st,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
    args: &mut [*mut c_void],
) -> Result<(), String> {
    let st = cublas_state(device);
    cudarc::driver::result::ctx::set_current(st.ctx)
        .map_err(|e| format!("设置 CUDA 上下文失败: {e}"))?;
    let status = cudarc::driver::sys::cuLaunchKernel(
        kernel.function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_mem_bytes,
        stream,
        args.as_mut_ptr(),    // kernelParams
        std::ptr::null_mut(), // extra
    );
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuLaunchKernel 失败: {status:?}"));
    }
    Ok(())
}

// ===========================================================================
// 具体内核：半量正态噪声生成（cuda-oxide 编译，PTX 内嵌）
// ===========================================================================

/// PRNG 内核 PTX（cuda-oxide 编译：`snn_prng` 示例 → llvm-link libdevice →
/// opt 裁剪 → llc，见 `docs/cuda_oxide_integration_plan.md`）。
/// 注意：cuModuleLoadData 要求 PTX 文本以 NUL 结尾，故用 include_str + "\0"。
const PRNG_PTX: &[u8] =
    concat!(include_str!("../../hyperscalees-kernels/ptx/prng_normal_half.ptx"), "\0").as_bytes();
/// 每线程元素数（与内核常量 ELEMS_PER_THREAD 一致）。
const PRNG_ELEMS_PER_THREAD: usize = 128;

/// 已加载的 PRNG 内核（进程级缓存，加载一次）。
fn prng_kernel(device: &CudaDevice) -> &'static OxideKernel {
    static KERNEL: std::sync::OnceLock<OxideKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        load_kernel(device, PRNG_PTX, "prng_normal_half").expect("加载 prng_normal_half PTX 失败")
    })
}

/// 内核参数种子（每次调用自增 + 时间混合，保证每次生成不同序列）。
fn next_seeds() -> [u32; 4] {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);
    let c = CTR.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0) as u64;
    let mix = |x: u64| -> u32 {
        let mut h = x;
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        h as u32
    };
    [mix(c), mix(c >> 32), mix(t), mix(t >> 32)]
}

/// 半量正态填充：`out` **1D 连续张量**（n/2·r·b 元素），填充 `N(mean, std²)`。
///
/// 与训练热路径的「半噪声」约定配套（配对由消费方隐含施加）。内核为
/// cuda-oxide 编译的 PTX，经 cudarc 在 cubecl 主流上启动（同流有序，零同步）。
///
/// 注意：必须传**连续 1D** 张量（内核按扁平连续写；3D 张量带 256B 行 pitch，
/// 扁平写会错位/覆盖不足——见集成文档 §10 bug 3/4）。调用方如需 3D，用
/// `Tensor::reshape`（零拷贝视图，连续 strides）。
pub fn prng_normal_half(
    out: &burn::tensor::Tensor<crate::B, 1>,
    mean: f32,
    std: f32,
    device: &CudaDevice,
) -> Result<(), String> {
    let n_elems = out.shape().dims::<1>()[0];
    debug_assert_eq!(n_elems % PRNG_ELEMS_PER_THREAD, 0, "元素数需为 128 的整数倍");
    let total_threads = (n_elems / PRNG_ELEMS_PER_THREAD) as u32;

    let cube = crate::cublas::as_cube(out);
    let ptr = crate::cublas::raw_ptr(&cube, device) as *mut f32;

    let seeds = next_seeds();
    // cuLaunchKernel 要求 kernelParams 指向的**参数值按 8 字节对齐**（u32/f32
    // 普通栈变量只保证 4 字节对齐，会随机触发 CUDA_ERROR_INVALID_VALUE）；
    // 用 repr(C, align(8)) 包装每个参数值，再取内部字段的指针。
    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_ptr = A8(ptr as *mut c_void);
    let mut arg_threads = A8(total_threads);
    let mut arg_mean = A8(mean);
    let mut arg_std = A8(std);
    let mut arg_s0 = A8(seeds[0]);
    let mut arg_s1 = A8(seeds[1]);
    let mut arg_s2 = A8(seeds[2]);
    let mut arg_s3 = A8(seeds[3]);
    let mut args: [*mut c_void; 8] = [
        &mut arg_ptr.0 as *mut *mut c_void as *mut c_void,
        &mut arg_threads.0 as *mut u32 as *mut c_void,
        &mut arg_mean.0 as *mut f32 as *mut c_void,
        &mut arg_std.0 as *mut f32 as *mut c_void,
        &mut arg_s0.0 as *mut u32 as *mut c_void,
        &mut arg_s1.0 as *mut u32 as *mut c_void,
        &mut arg_s2.0 as *mut u32 as *mut c_void,
        &mut arg_s3.0 as *mut u32 as *mut c_void,
    ];

    const BLOCK: u32 = 256;
    let grid = total_threads.div_ceil(BLOCK);
    let kernel = prng_kernel(device);
    unsafe { launch(kernel, device, grid, 1, 1, BLOCK, 1, 1, 0, &mut args) }
}

// ===========================================================================
// 具体内核：配对合并 einsum（融合预处理，cuda-oxide 编译，PTX 内嵌）
// ===========================================================================

/// einsum 内核 PTX（cuda-oxide 编译：`snn_einsum` 示例 → llvm-link libdevice →
/// opt 裁剪 → llc（-fp-contract=fast 融合 FMA）→ PTX，见集成计划文档）。
/// 含 3 个 entry：einsum_pair_fused（主）+ 2 个 dump 诊断内核。
const EINSUM_PTX: &[u8] = concat!(
    include_str!("../../hyperscalees-kernels/ptx/einsum_pair_fused.ptx"),
    "\0"
)
.as_bytes();

/// 已加载的 einsum 内核（进程级缓存，加载一次）。
fn einsum_kernel(device: &CudaDevice) -> &'static OxideKernel {
    static KERNEL: std::sync::OnceLock<OxideKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        let k = load_kernel(device, EINSUM_PTX, "einsum_pair_fused")
            .expect("加载 einsum_pair_fused PTX 失败");
        if std::env::var("DEBUG_OXIDE").map(|v| v == "1").unwrap_or(false) {
            // 查询 ptxas 实际分配：寄存器数与 local memory（溢出）字节数。
            let mut regs: i32 = 0;
            let mut local: i32 = 0;
            unsafe {
                cudarc::driver::sys::cuFuncGetAttribute(
                    &mut regs,
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
                    k.function,
                );
                cudarc::driver::sys::cuFuncGetAttribute(
                    &mut local,
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    k.function,
                );
            }
            eprintln!("[oxide] einsum_pair_fused: numRegs={regs} localSizeBytes={local}");
        }
        k
    })
}

/// 融合配对 einsum（cuda-oxide 内核）：与 `cublas::lora_einsum_pair_cublas`
/// 数学一致（`g_raw = Σ_i (f_i+f_{half+i})·A'_i⊗B'_i`，`g_ones = 2·Σ_i A'_i⊗B'_i`），
/// 但把「slice + f_pair 加权 + cat 拼接」全部融合进共享内存加载，A/B 各只读一遍；
/// 输出经 f32 原子累加（split-K 合并）。
///
/// 要求：`a_t`/`b_t` 行主序连续（dim2 stride 1，dim0/dim1 stride 显式传入，
/// 支持 burn 的 256B 行对齐 pitch）；`2a ≤ 256`；输出 `(a, b)` 先置零。
pub fn einsum_pair_fused(
    a_t: &burn::tensor::Tensor<crate::B, 3>,    // (half, r, a) 半量 A' 噪声
    b_t: &burn::tensor::Tensor<crate::B, 3>,    // (half, r, b) 半量 B' 噪声
    scores: &burn::tensor::Tensor<crate::B, 1>, // (n,) 原始分数
    device: &CudaDevice,
) -> Result<(burn::tensor::Tensor<crate::B, 2>, burn::tensor::Tensor<crate::B, 2>), String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [half, r, a] = a_t.dims();
    let [_, _, b] = b_t.dims();
    assert!(half * r > 0 && 2 * a <= 256, "einsum 内核要求 0 < 2a ≤ 256，实际 2a={}", 2 * a);
    assert_eq!(b_t.dims(), [half, r, b], "b_t 形状不匹配");

    let ca = as_cube(a_t);
    let cb = as_cube(b_t);
    let cs = as_cube(scores);
    let s_a = ca.meta.strides();
    let s_b = cb.meta.strides();
    assert_eq!(
        (s_a[2], s_b[2]),
        (1, 1),
        "einsum 内核要求行主序连续输入（innermost stride 1），实际 {:?}/{:?}",
        s_a,
        s_b
    );

    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let ps = raw_ptr(&cs, device);
    let k_total = half * r;

    // 输出（调用方语义 = 原子累加，必须零初始化）。
    let g_raw: burn::tensor::Tensor<crate::B, 2> = burn::tensor::Tensor::zeros([a, b], device);
    let g_ones: burn::tensor::Tensor<crate::B, 2> = burn::tensor::Tensor::zeros([a, b], device);
    let cg_raw = as_cube(&g_raw);
    let cg_ones = as_cube(&g_ones);
    let pgr = raw_ptr(&cg_raw, device);
    let pgo = raw_ptr(&cg_ones, device);

    // split-K：K=384000 → kslice=2000（192 个切片，grid.y=192 实测最优：
    // 96→7.66ms / 192→7.23ms / 384→7.30ms，同窗口 cuBLAS 8.6-8.9ms）。
    let k_slices = std::env::var("OXIDE_KS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::cmp::max(1, k_total.div_ceil(2000))) as u32;
    let n_tiles = (b as u32).div_ceil(112);

    if std::env::var("DEBUG_OXIDE").map(|v| v == "1").unwrap_or(false) {
        eprintln!(
            "[oxide] einsum: a({:?}) strides={:?} b({:?}) strides={:?} scores({:?}) \
             half={} r={} a={} b={} K={} k_slices={} n_tiles={}",
            a_t.dims(),
            s_a,
            b_t.dims(),
            s_b,
            scores.dims(),
            half,
            r,
            a,
            b,
            k_total,
            k_slices,
            n_tiles
        );
    }

    let kernel = einsum_kernel(device);
    // cuLaunchKernel 参数值需 8 字节对齐（见 prng_normal_half 注释）。
    #[repr(C, align(8))]
    struct A8<T>(T);
    // 输出张量的行 stride（burn 256B 对齐 pitch，内核原子写地址用）。
    let g_s_raw = cg_raw.meta.strides()[0] as u32;
    let g_s_ones = cg_ones.meta.strides()[0] as u32;
    debug_assert_eq!(g_s_raw, g_s_ones, "输出张量 stride 应一致");
    let mut arg_a = A8(pa as *mut c_void);
    let mut arg_a0 = A8(s_a[0] as u32);
    let mut arg_a1 = A8(s_a[1] as u32);
    let mut arg_ad = A8(a as u32);
    let mut arg_b = A8(pb as *mut c_void);
    let mut arg_b0 = A8(s_b[0] as u32);
    let mut arg_b1 = A8(s_b[1] as u32);
    let mut arg_bd = A8(b as u32);
    let mut arg_s = A8(ps as *mut c_void);
    let mut arg_h = A8(half as u32);
    let mut arg_r = A8(r as u32);
    let mut arg_gr = A8(pgr as *mut c_void);
    let mut arg_go = A8(pgo as *mut c_void);
    let mut arg_gs = A8(g_s_raw);
    let mut arg_k = A8(k_total as u32);
    let mut arg_ks = A8(k_slices);
    let mut args: [*mut c_void; 16] = [
        &mut arg_a.0 as *mut *mut c_void as *mut c_void,
        &mut arg_a0.0 as *mut u32 as *mut c_void,
        &mut arg_a1.0 as *mut u32 as *mut c_void,
        &mut arg_ad.0 as *mut u32 as *mut c_void,
        &mut arg_b.0 as *mut *mut c_void as *mut c_void,
        &mut arg_b0.0 as *mut u32 as *mut c_void,
        &mut arg_b1.0 as *mut u32 as *mut c_void,
        &mut arg_bd.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut *mut c_void as *mut c_void,
        &mut arg_h.0 as *mut u32 as *mut c_void,
        &mut arg_r.0 as *mut u32 as *mut c_void,
        &mut arg_gr.0 as *mut *mut c_void as *mut c_void,
        &mut arg_go.0 as *mut *mut c_void as *mut c_void,
        &mut arg_gs.0 as *mut u32 as *mut c_void,
        &mut arg_k.0 as *mut u32 as *mut c_void,
        &mut arg_ks.0 as *mut u32 as *mut c_void,
    ];
    unsafe { launch(kernel, device, n_tiles, k_slices, 1, 512, 1, 1, 0, &mut args) }?;
    if std::env::var("DEBUG_OXIDE").map(|v| v == "1").unwrap_or(false) {
        unsafe {
            let rd = |p: *mut c_void| *(p as *const u64);
            let ru = |p: *mut c_void| *(p as *const u32);
            eprintln!(
                "[oxide] args: a={:#x} a_s0={} a_s1={} a={} b={:#x} b_s0={} b_s1={} b={} \
                 scores={:#x} half={} r={} g_raw={:#x} g_ones={:#x} g_s={} K={} k_slices={} grid=({}, {})",
                rd(args[0]),
                ru(args[1]),
                ru(args[2]),
                ru(args[3]),
                rd(args[4]),
                ru(args[5]),
                ru(args[6]),
                ru(args[7]),
                rd(args[8]),
                ru(args[9]),
                ru(args[10]),
                rd(args[11]),
                rd(args[12]),
                ru(args[13]),
                ru(args[14]),
                ru(args[15]),
                n_tiles,
                k_slices
            );
        }
    }
    Ok((g_raw, g_ones))
}

// ===========================================================================
// 具体内核：融合 LIF 扫描（cuda-oxide 编译，PTX 内嵌）
// ===========================================================================

/// LIF 内核 PTX（cuda-oxide 编译：`snn_lif` 示例 → llvm-link libdevice →
/// opt 裁剪 → llc（-fp-contract=fast 融合 FMA）→ PTX，见集成计划文档）。
const LIF_PTX: &[u8] =
    concat!(include_str!("../../hyperscalees-kernels/ptx/lif_fused.ptx"), "\0").as_bytes();

/// 已加载的 LIF 内核（进程级缓存，加载一次）。
fn lif_kernel(device: &CudaDevice) -> &'static OxideKernel {
    static KERNEL: std::sync::OnceLock<OxideKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        load_kernel(device, LIF_PTX, "lif_fused").expect("加载 lif_fused PTX 失败")
    })
}

/// 融合 LIF 扫描（cuda-oxide 内核）：`(T, n, h)` 输入电流 → `(T, n, h)` 0/1 尖峰。
///
/// 与 `hyperscalees_models::snn::run_lif` 语义逐位一致（hard reset）：
/// `charged = v + (cur - v)·(1/tau_m)`，`spike = (charged ≥ v_th)`，`v = charged·(1-spike)`。
/// 每线程处理一个 (n, h) 元素沿 T 顺序扫描（无中间张量/逐时间步 launch）。
///
/// 支持 burn 256B 行对齐 pitch（行 stride ≠ h，如 h=100 f32 → 128；
/// 内核按 row·s + col 寻址，与泊松内核同机制）。
pub fn lif_fused(
    cur: &burn::tensor::Tensor<crate::B, 3>,
    v0: &burn::tensor::Tensor<crate::B, 2>,
    tau_m: f32,
    v_th: f32,
    device: &CudaDevice,
) -> Result<burn::tensor::Tensor<crate::B, 3>, String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [t, n, h] = cur.dims();
    let total = n * h;
    assert_eq!(v0.dims(), [n, h], "LIF v0 形状不匹配: {:?} vs {:?}", v0.dims(), [n, h]);
    assert!(total > 0 && total <= u32::MAX as usize, "LIF total 溢出");

    let out: burn::tensor::Tensor<crate::B, 3> =
        burn::tensor::Tensor::zeros([t, n, h], device);
    let cc = as_cube(cur);
    let cv = as_cube(v0);
    let co = as_cube(&out);
    let pc = raw_ptr(&cc, device);
    let pv = raw_ptr(&cv, device);
    let po = raw_ptr(&co, device);
    // 行 stride（元素单位；burn 256B 对齐 pitch）。
    let s_c = cc.meta.strides()[1] as u32;
    let s_v = cv.meta.strides()[0] as u32;
    let s_o = co.meta.strides()[1] as u32;
    debug_assert_eq!((s_c, s_v), (s_o, s_o), "LIF 行 stride 不一致");

    let kernel = lif_kernel(device);
    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_c = A8(pc as *mut c_void);
    let mut arg_v = A8(pv as *mut c_void);
    let mut arg_o = A8(po as *mut c_void);
    let mut arg_tot = A8(total as u32);
    let mut arg_h = A8(h as u32);
    let mut arg_s = A8(s_c);
    let mut arg_t = A8(t as u32);
    let mut arg_tau = A8(tau_m);
    let mut arg_th = A8(v_th);
    let mut args: [*mut c_void; 9] = [
        &mut arg_c.0 as *mut *mut c_void as *mut c_void,
        &mut arg_v.0 as *mut *mut c_void as *mut c_void,
        &mut arg_o.0 as *mut *mut c_void as *mut c_void,
        &mut arg_tot.0 as *mut u32 as *mut c_void,
        &mut arg_h.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut u32 as *mut c_void,
        &mut arg_t.0 as *mut u32 as *mut c_void,
        &mut arg_tau.0 as *mut f32 as *mut c_void,
        &mut arg_th.0 as *mut f32 as *mut c_void,
    ];
    const BLOCK: u32 = 256;
    let grid = (total as u32).div_ceil(BLOCK);
    unsafe { launch(kernel, device, grid, 1, 1, BLOCK, 1, 1, 0, &mut args) }?;
    Ok(out)
}

// ===========================================================================
// 具体内核：融合泊松编码（cuda-oxide 编译，PTX 内嵌）
// ===========================================================================

/// 泊松内核 PTX（cuda-oxide 编译：`snn_poisson` 示例 → llvm-link libdevice →
/// opt 裁剪 → llc（-fp-contract=fast）→ PTX，见集成计划文档）。
const POISSON_PTX: &[u8] = concat!(
    include_str!("../../hyperscalees-kernels/ptx/poisson_encode_fused.ptx"),
    "\0"
)
.as_bytes();

/// 已加载的泊松内核（进程级缓存，加载一次）。
fn poisson_kernel(device: &CudaDevice) -> &'static OxideKernel {
    static KERNEL: std::sync::OnceLock<OxideKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        load_kernel(device, POISSON_PTX, "poisson_encode_fused")
            .expect("加载 poisson_encode_fused PTX 失败")
    })
}

/// 融合泊松编码（cuda-oxide 内核）：`(batch, in)` 像素强度 → `(t, batch, in)`
/// 0/1 尖峰，u ~ Uniform(0,1)（xorshift32 现场生成）< p 比较，单次启动。
///
/// 统计语义与 `hyperscalees_envs::snn_mnist::poisson_encode` 一致（每元素每
/// 时间步独立 Bernoulli，发放率 ≈ 像素值）；RNG 与 burn 不同源，不要求逐位一致
/// （参考实现即如此）。支持 burn 256B 行对齐 pitch（行 stride ≠ in_dim，
/// 如 784 f32 → 832；内核按 row·s + col 寻址）。
pub fn poisson_encode_fused(
    images: &burn::tensor::Tensor<crate::B, 2>,
    t: usize,
    device: &CudaDevice,
) -> Result<burn::tensor::Tensor<crate::B, 3>, String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [batch, in_dim] = images.dims();
    let total = batch * in_dim;
    assert!(total > 0 && total <= u32::MAX as usize, "poisson total 溢出");
    assert!(t <= 8, "poisson 内核编译期展开上限 T ≤ 8，实际 {t}");

    let out: burn::tensor::Tensor<crate::B, 3> =
        burn::tensor::Tensor::zeros([t, batch, in_dim], device);
    let ci = as_cube(images);
    let co = as_cube(&out);
    let pi = raw_ptr(&ci, device);
    let po = raw_ptr(&co, device);
    // 行 stride（元素单位；burn 256B 对齐 pitch）。
    let s_p = ci.meta.strides()[0] as u32;
    let s_o = co.meta.strides()[1] as u32;
    debug_assert_eq!(s_p, s_o, "probs/out 行 stride 不一致");

    let seeds = next_seeds();
    let kernel = poisson_kernel(device);
    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_p = A8(pi as *mut c_void);
    let mut arg_o = A8(po as *mut c_void);
    let mut arg_tot = A8(total as u32);
    let mut arg_id = A8(in_dim as u32);
    let mut arg_s = A8(s_p);
    let mut arg_t = A8(t as u32);
    let mut arg_s0 = A8(seeds[0]);
    let mut arg_s1 = A8(seeds[1]);
    let mut arg_s2 = A8(seeds[2]);
    let mut arg_s3 = A8(seeds[3]);
    let mut args: [*mut c_void; 10] = [
        &mut arg_p.0 as *mut *mut c_void as *mut c_void,
        &mut arg_o.0 as *mut *mut c_void as *mut c_void,
        &mut arg_tot.0 as *mut u32 as *mut c_void,
        &mut arg_id.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut u32 as *mut c_void,
        &mut arg_t.0 as *mut u32 as *mut c_void,
        &mut arg_s0.0 as *mut u32 as *mut c_void,
        &mut arg_s1.0 as *mut u32 as *mut c_void,
        &mut arg_s2.0 as *mut u32 as *mut c_void,
        &mut arg_s3.0 as *mut u32 as *mut c_void,
    ];
    const BLOCK: u32 = 256;
    let grid = (total as u32).div_ceil(BLOCK);
    unsafe { launch(kernel, device, grid, 1, 1, BLOCK, 1, 1, 0, &mut args) }?;
    Ok(out)
}

/// 诊断封装：dump 首个 chunk 的 acc 槽（tx<2 线程），返回 CPU 拷贝。
pub fn einsum_dump_acc(
    a_t: &burn::tensor::Tensor<crate::B, 3>,
    b_t: &burn::tensor::Tensor<crate::B, 3>,
    scores: &burn::tensor::Tensor<crate::B, 1>,
    device: &CudaDevice,
) -> Result<Vec<f32>, String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [half, r, a] = a_t.dims();
    let [_, _, b] = b_t.dims();
    let ca = as_cube(a_t);
    let cb = as_cube(b_t);
    let cs = as_cube(scores);
    let s_a = ca.meta.strides();
    let s_b = cb.meta.strides();
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let ps = raw_ptr(&cs, device);
    let k_total = half * r;

    let ad: burn::tensor::Tensor<crate::B, 1> =
        burn::tensor::Tensor::zeros([2 * 16 * 112], device);
    let cad = as_cube(&ad);
    let pad = raw_ptr(&cad, device);

    let kernel = load_kernel(device, EINSUM_PTX, "einsum_pair_dump_acc")
        .expect("加载 einsum_pair_dump_acc PTX 失败");
    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_a = A8(pa as *mut c_void);
    let mut arg_a0 = A8(s_a[0] as u32);
    let mut arg_a1 = A8(s_a[1] as u32);
    let mut arg_ad = A8(a as u32);
    let mut arg_b = A8(pb as *mut c_void);
    let mut arg_b0 = A8(s_b[0] as u32);
    let mut arg_b1 = A8(s_b[1] as u32);
    let mut arg_bd = A8(b as u32);
    let mut arg_s = A8(ps as *mut c_void);
    let mut arg_h = A8(half as u32);
    let mut arg_r = A8(r as u32);
    let mut arg_pa = A8(pad as *mut c_void);
    let mut arg_k = A8(k_total as u32);
    let mut arg_ks = A8(1u32);
    let mut args: [*mut c_void; 14] = [
        &mut arg_a.0 as *mut *mut c_void as *mut c_void,
        &mut arg_a0.0 as *mut u32 as *mut c_void,
        &mut arg_a1.0 as *mut u32 as *mut c_void,
        &mut arg_ad.0 as *mut u32 as *mut c_void,
        &mut arg_b.0 as *mut *mut c_void as *mut c_void,
        &mut arg_b0.0 as *mut u32 as *mut c_void,
        &mut arg_b1.0 as *mut u32 as *mut c_void,
        &mut arg_bd.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut *mut c_void as *mut c_void,
        &mut arg_h.0 as *mut u32 as *mut c_void,
        &mut arg_r.0 as *mut u32 as *mut c_void,
        &mut arg_pa.0 as *mut *mut c_void as *mut c_void,
        &mut arg_k.0 as *mut u32 as *mut c_void,
        &mut arg_ks.0 as *mut u32 as *mut c_void,
    ];
    unsafe {
        launch(&kernel, device, 1, 1, 1, 256, 1, 1, 0, &mut args)?;
    }
    Ok(ad.into_data().into_vec::<f32>().map_err(|e| format!("{e:?}"))?)
}

/// 诊断封装：dump A_SH/B_SH 前 2 行（调试用），返回 CPU 拷贝。
pub fn einsum_dump(
    a_t: &burn::tensor::Tensor<crate::B, 3>,
    b_t: &burn::tensor::Tensor<crate::B, 3>,
    scores: &burn::tensor::Tensor<crate::B, 1>,
    device: &CudaDevice,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [half, r, a] = a_t.dims();
    let [_, _, b] = b_t.dims();
    let ca = as_cube(a_t);
    let cb = as_cube(b_t);
    let cs = as_cube(scores);
    let s_a = ca.meta.strides();
    let s_b = cb.meta.strides();
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let ps = raw_ptr(&cs, device);
    let k_total = half * r;

    let ad: burn::tensor::Tensor<crate::B, 1> =
        burn::tensor::Tensor::zeros([2 * 257], device);
    let bd: burn::tensor::Tensor<crate::B, 1> =
        burn::tensor::Tensor::zeros([2 * 112], device);
    let cad = as_cube(&ad);
    let cbd = as_cube(&bd);
    let pad = raw_ptr(&cad, device);
    let pbd = raw_ptr(&cbd, device);

    let kernel = load_kernel(device, EINSUM_PTX, "einsum_pair_dump_tiles")
        .expect("加载 einsum_pair_dump_tiles PTX 失败");
    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_a = A8(pa as *mut c_void);
    let mut arg_a0 = A8(s_a[0] as u32);
    let mut arg_a1 = A8(s_a[1] as u32);
    let mut arg_ad = A8(a as u32);
    let mut arg_b = A8(pb as *mut c_void);
    let mut arg_b0 = A8(s_b[0] as u32);
    let mut arg_b1 = A8(s_b[1] as u32);
    let mut arg_bd = A8(b as u32);
    let mut arg_s = A8(ps as *mut c_void);
    let mut arg_h = A8(half as u32);
    let mut arg_r = A8(r as u32);
    let mut arg_pa = A8(pad as *mut c_void);
    let mut arg_pb = A8(pbd as *mut c_void);
    let mut arg_k = A8(k_total as u32);
    let mut arg_ks = A8(1u32);
    let mut args: [*mut c_void; 15] = [
        &mut arg_a.0 as *mut *mut c_void as *mut c_void,
        &mut arg_a0.0 as *mut u32 as *mut c_void,
        &mut arg_a1.0 as *mut u32 as *mut c_void,
        &mut arg_ad.0 as *mut u32 as *mut c_void,
        &mut arg_b.0 as *mut *mut c_void as *mut c_void,
        &mut arg_b0.0 as *mut u32 as *mut c_void,
        &mut arg_b1.0 as *mut u32 as *mut c_void,
        &mut arg_bd.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut *mut c_void as *mut c_void,
        &mut arg_h.0 as *mut u32 as *mut c_void,
        &mut arg_r.0 as *mut u32 as *mut c_void,
        &mut arg_pa.0 as *mut *mut c_void as *mut c_void,
        &mut arg_pb.0 as *mut *mut c_void as *mut c_void,
        &mut arg_k.0 as *mut u32 as *mut c_void,
        &mut arg_ks.0 as *mut u32 as *mut c_void,
    ];
    unsafe {
        launch(&kernel, device, 1, 1, 1, 256, 1, 1, 0, &mut args)?;
    }
    Ok((
        ad.into_data().into_vec::<f32>().map_err(|e| format!("{e:?}"))?,
        bd.into_data().into_vec::<f32>().map_err(|e| format!("{e:?}"))?,
    ))
}

#[cfg(test)]
mod tests {
    /// 自检：模块可编译、类型/签名存在。
    #[test]
    fn skeleton_compiles() {
        let _ = std::any::type_name::<crate::oxide::OxideKernel>();
    }
}
