//! cuda-oxide 内核的宿主侧加载/启动封装（仅 `gpu` feature）。
//!
//! 集成路径（见 `docs/cuda_oxide_integration_plan.md`）：
//! 用 cuda-oxide 编译器把 `hyperscalees-kernels` 的 Rust 内核编译为 PTX 文本，
//! 宿主经 CUDA driver API 加载（`cuModuleLoadData`）并启动（`cuLaunchKernel`）。
//!
//! 与 [`crate::cublas`] 同一套机制：module/function 挂在 cubecl 的 context 上
//! （复用 `cublas::state` 的 ctx），启动绑到 cubecl 的原始 stream
//! （`raw_stream`）→ 与 burn 算子天然同流有序、零同步（探针实测可用）。
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
    // 注意：cuLaunchKernel 参数顺序是 (..., hStream, kernelParams, extra)——
    // kernelParams 传 args 数组、extra 传 null（曾把两者写反导致
    // CUDA_ERROR_INVALID_VALUE，探针逐字节对比才定位）。
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

/// 半量正态填充：`out` (n/2, r, b) 连续张量，填充 `N(mean, std²)`。
///
/// 与训练热路径的「半噪声」约定配套（配对由消费方隐含施加）。内核为
/// cuda-oxide 编译的 PTX，经 cudarc 在 cubecl 主流上启动（同流有序，零同步）。
pub fn prng_normal_half(
    out: &burn::tensor::Tensor<crate::B, 3>,
    mean: f32,
    std: f32,
    device: &CudaDevice,
) -> Result<(), String> {
    let n_elems = out.shape().dims::<3>().iter().product::<usize>();
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

#[cfg(test)]
mod tests {
    /// 自检：模块可编译、类型/签名存在。
    #[test]
    fn skeleton_compiles() {
        let _ = std::any::type_name::<crate::oxide::OxideKernel>();
    }
}
