//! cuda-oxide 内核的宿主侧加载/启动封装（骨架，仅 `gpu` feature）。
//!
//! 集成路径（见 `docs/cuda_oxide_integration_plan.md`）：
//! 用 cuda-oxide 编译器把 `hyperscalees-kernels` 的 Rust 内核编译为 PTX 文本，
//! 宿主经 CUDA driver API 加载（`cuModuleLoadData`）并启动（`cuLaunchKernel`）。
//!
//! 与 [`crate::cublas`] 同一套机制：
//! - module/function 挂在 cubecl 的 context 上（复用 `cublas::state` 的 ctx）；
//! - 启动时绑到 cubecl 的原始 stream（`raw_stream`）→ 与 burn 算子天然同流有序、
//!   零同步；
//! - 内核参数直接传 burn 张量的原始设备指针（`cublas::raw_ptr` 同款 resolve
//!   机制）与显式 strides（burn 张量 pitched）。
//!
//! 当前状态：PTX 未就位（本机无法获取 cuda-oxide 编译器，见计划文档 §6），
//! 本模块只提供可编译的封装，不参与训练热路径。待 PTX 就位后：
//! ```rust,ignore
//! let ptx = include_bytes!("../kernels/bin/prng_normal_half.ptx");
//! let kernel = oxide::load_kernel(&device, ptx, "prng_normal_half_kernel")?;
//! unsafe { oxide::launch(&kernel, &device, grid, block, 0, &mut args)?; }
//! ```

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
/// `ptx` 为 cuda-oxide 编译产出的 PTX 文本字节；`kernel_name` 为 PTX 内的函数名。
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
        std::ptr::null_mut(), // 使用 kernelParams 数组，不用可变参数形式
        args.as_mut_ptr(),
    );
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuLaunchKernel 失败: {status:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// 骨架自检：PTX 未就位时 load 必然失败（错误路径可用），模块可编译。
    #[test]
    fn skeleton_compiles() {
        // 无 GPU 侧调用；仅确认类型/签名存在。
        let _ = std::any::type_name::<crate::oxide::OxideKernel>();
    }
}
