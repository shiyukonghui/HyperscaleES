//! cuBLAS 集成（仅 `gpu` feature）：把 einsum GEMM 放到 cubecl 的**同一 CUDA stream**
//! 上经 cuBLAS 执行。
//!
//! 为什么：cubecl 的 matmul 内核在「瘦 M + 超长 K」形状（如 einsum 的
//! (2a, half·r) @ (half·r, b)，half·r = 384000）上只有 ~8-12 TFLOP/s（并行度受
//! M×N 平铺块数限制，4090 的 128 个 SM 大量闲置）；cuBLAS 对 split-K / 长 K 形状
//! 有专门的优化内核。einsum 是训练每 epoch ~77ms 的第二大开销。
//!
//! 关键设计：cuBLAS handle 绑定到 vendored cubecl-cuda 暴露的原始 stream
//! （`CudaServer::raw_stream`），因此 cuBLAS 调用与 burn 的其它算子**天然同流有序**，
//! 无需任何同步（逐 chunk 的全量 sync 会打空流水线，得不偿失）。输入指针经
//! `CudaServer::raw_device_ptr` 解析——该路径走 cubecl 的 `resolve` 机制，跨流依赖
//! （多流流水线场景）也会自动插入等待。

use std::sync::OnceLock;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubecl::device::Device;
use cubecl::device_handle::DeviceHandle;
use cubecl::server::Binding;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;

/// 服务器类型（vendored cubecl-cuda 的 `CudaServer`，经 `Runtime::Server` 关联类型可达）。
type Server = <CudaRuntime as Runtime>::Server;

/// 全局 cuBLAS handle（绑定到主线程流的原始 CUDA stream）。
struct CublasState {
    handle: cudarc::cublas::sys::cublasHandle_t,
    ctx: *mut cudarc::driver::sys::CUctx_st,
}

// 原始句柄只在本模块内按序使用，不跨线程共享可变访问。
unsafe impl Send for CublasState {}
unsafe impl Sync for CublasState {}

fn state(device: &CudaDevice) -> &'static CublasState {
    static STATE: OnceLock<CublasState> = OnceLock::new();
    STATE.get_or_init(|| {
        let dh = DeviceHandle::<Server>::new(device.to_id());
        // 闭包结果需 Send：原始指针先转 usize 再还原。
        let ctx = dh
            .submit_blocking(|s| s.raw_context() as usize)
            .expect("取 CUDA 上下文失败") as *mut cudarc::driver::sys::CUctx_st;
        let stream = dh
            .submit_blocking(|s| s.raw_stream(StreamId::current()) as usize)
            .expect("取 CUDA stream 失败") as *mut cudarc::driver::sys::CUstream_st;
        // cuBLAS handle 创建时上下文必须是 current（绑定到主上下文）。
        unsafe {
            cudarc::driver::result::ctx::set_current(ctx).expect("设置 CUDA 上下文失败");
        }
        let handle = cudarc::cublas::result::create_handle().expect("创建 cuBLAS handle 失败");
        // cudarc 的 cublas sys 是独立 bindgen 生成的不透明类型，与 driver 的
        // CUstream_st 布局一致，直接转换。
        unsafe {
            cudarc::cublas::result::set_stream(
                handle,
                stream as *mut cudarc::cublas::sys::CUstream_st,
            )
            .expect("cuBLAS 绑定 stream 失败");
        }
        CublasState { handle, ctx }
    })
}

/// 把 cubecl 张量解析为原始设备指针（走 resolve 机制，跨流依赖自动等待）。
fn raw_ptr(cube: &CubeTensor<CudaRuntime>, device: &CudaDevice) -> *mut std::ffi::c_void {
    let dh = DeviceHandle::<Server>::new(device.to_id());
    let binding: Binding = cube.handle.clone().binding();
    dh.submit_blocking(|s| s.raw_device_ptr(binding, StreamId::current()) as usize)
        .expect("解析设备指针失败") as *mut std::ffi::c_void
}

/// 把 burn 张量解包为后端原始张量（`TensorPrimitive` enum 的 `Float` 变体）。
fn as_cube<const D: usize>(t: &Tensor<Cuda, D>) -> CubeTensor<CudaRuntime> {
    t.clone().into_primitive().tensor()
}

/// `C = A^T @ B`：`A` (k, m) 连续、`B` (k, n) 连续 → 返回 (m, n) 张量。
///
/// 内存布局说明：cuBLAS 的 C 输出是列主序 `C_col(m, n)`，而 burn 张量按行主序解释；
/// 因此输出缓冲区以 (n, m) 行主序承载（恰好等于 C^T），返回前 `transpose()` 视图
/// 即得 (m, n)。下游（切片/加法）对 strided 视图完全支持。
pub fn gemm_atb(a: &Tensor<Cuda, 2>, b: &Tensor<Cuda, 2>, device: &CudaDevice) -> Tensor<Cuda, 2> {
    let [k, m] = a.dims();
    let [k2, n] = b.dims();
    assert_eq!(k, k2, "gemm_atb 的 k 维必须一致：{k} vs {k2}");
    let st = state(device);
    let ca = as_cube(a);
    let cb = as_cube(b);
    // 输出缓冲区（列主序 C 的载体 = (n, m) 行主序 = C^T）。
    let out: Tensor<Cuda, 2> = Tensor::empty([n, m], device);
    let co = as_cube(&out);
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let pc = raw_ptr(&co, device);
    // burn 张量可能带行对齐 padding（PitchedMemoryLayoutPolicy），row stride 未必等于
    // 列数。行主序 (k, m) 按列主序 (m, k) 解释时 leading dim = row stride（元素单位），
    // cuBLAS 原生支持 pitched 输入。
    let sa = ca.meta.strides()[0] as i32;
    let sb = cb.meta.strides()[0] as i32;
    let sc = co.meta.strides()[0] as i32;
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("设置 CUDA 上下文失败");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // 行主序 (k, m) == 列主序 (m, k)（leading dim = row stride）。
        // transa=N（op(A)=A^T）、transb=T（op(B)=(B^T)^T=B）。
        // C = A^T @ B，(m, n) 列主序写入输出缓冲区。
        cudarc::cublas::result::sgemm(
            st.handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            pa as *const f32,
            sa,
            pb as *const f32,
            sb,
            &beta,
            pc as *mut f32,
            sc,
        )
        .expect("cublasSgemm 失败");
    }
    out.transpose()
}

/// 反对称配对 einsum 的 cuBLAS 版：与 `lora_einsum_pair` 数学完全一致
/// （`g_raw = Σ_i (f_i + f_{half+i})·A'_i ⊗ B'_i`，`g_ones = 2·Σ_i A'_i ⊗ B'_i`），
/// 仅把合并 GEMM 从 cubecl matmul 换成 cuBLAS（同流、无同步）。
pub fn lora_einsum_pair_cublas(
    a_t: &Tensor<Cuda, 3>,     // (n, r, a)，A 已乘 sign*base_sigma，反对称配对
    b_t: &Tensor<Cuda, 3>,     // (n, r, b)，反对称配对
    scores: &Tensor<Cuda, 1>,  // (n,)
    device: &CudaDevice,
) -> (Tensor<Cuda, 2>, Tensor<Cuda, 2>) {
    let [n, r, a] = a_t.dims();
    let b = b_t.dims()[2];
    assert!(n % 2 == 0, "配对 einsum 要求 n 为偶数，实际 {n}");
    let half = n / 2;
    let a_half = a_t.clone().slice([0..half, 0..r, 0..a]); // 连续视图
    let b_half = b_t.clone().slice([0..half, 0..r, 0..b]); // 连续视图
    let f_pair = scores
        .clone()
        .slice([0..half])
        .add(scores.clone().slice([half..n])); // (half,)
    let a_w = a_half.clone() * f_pair.reshape([half, 1, 1]); // (half, r, a)
    let a_stack = Tensor::cat(vec![a_w, a_half], 2); // (half, r, 2a)
    let a_flat = a_stack.reshape([half * r, 2 * a]); // (k, m) 连续
    let b_flat = b_half.reshape([half * r, b]); // (k, n) 连续视图
    let g = gemm_atb(&a_flat, &b_flat, device); // (2a, b)
    // 上半行 = g_raw；下半行 = g_ones'（×2 得 g_ones）。
    let g_raw = g.clone().slice([0..a, 0..b]).reshape([a, b]);
    let g_ones = g.slice([a..2 * a, 0..b]).reshape([a, b]).mul_scalar(2.0);
    (g_raw, g_ones)
}

/// 反对称配对的 LoRA 噪声生成（零拷贝版）：返回 `(A' (n,r,a) 已乘 base_sigma, B' (n,r,b))`。
///
/// 直接调用 vendored cubek-random 的 `random_normal_antipodal` 内核：一次内核调用
/// 生成完整张量，后半样本是前半的逐位取负（`out[n/2+i] = -out[i]`），**省去旧实现
/// 的 neg + cat 两次全量拷贝**（fc1 每 chunk 约省 5ms）。A' 用 `std = base_sigma`
/// 直接生成（等价于旧实现的生成后 mul_scalar，舍入差异 ~1e-7，统计无影响）。
///
/// 要求 `n/2 · r · b`（及 `n/2 · r · a`）能被 128 整除（本工作负载恒成立）。
pub fn gen_lora_noise_antipodal(
    n: usize,
    r: usize,
    a: usize,
    b: usize,
    base_sigma: f32,
    device: &CudaDevice,
) -> (Tensor<Cuda, 3>, Tensor<Cuda, 3>) {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    let dtype = StorageType::Scalar(ElemType::Float(FloatKind::F32));

    let b_t: Tensor<Cuda, 3> = Tensor::empty([n, r, b], device);
    let cb = as_cube(&b_t);
    let client = cb.client.clone();
    let binding = cb.binding();
    cubek_random::random_normal_antipodal(&client, 0.0, 1.0, binding, dtype)
        .expect("B' 反对称噪声生成失败");

    let a_t: Tensor<Cuda, 3> = Tensor::empty([n, r, a], device);
    let ca = as_cube(&a_t);
    let client_a = ca.client.clone();
    let binding_a = ca.binding();
    cubek_random::random_normal_antipodal(&client_a, 0.0, base_sigma, binding_a, dtype)
        .expect("A' 反对称噪声生成失败");

    (a_t, b_t)
}
