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
use hyperscalees_models::snn::TrainableVthSnn;

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
        // 默认 TF32 张量核数学模式：与 XLA/JAX 的 fp32 matmul 默认行为一致（XLA 默认
        // 允许 TF32），einsum 长 K 归约实测 0.16s → 0.15s/epoch；梯度噪声 O(1) 远大于
        // TF32 相对误差 1e-3，统计无影响。EINSUM_FP32=1 可切回纯 fp32（对照用）。
        let tf32 = std::env::var("EINSUM_FP32").map(|v| v == "1").unwrap_or(true);
        unsafe {
            let status = cudarc::cublas::sys::cublasSetMathMode(
                handle,
                if tf32 {
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
                } else {
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH
                },
            );
            assert_eq!(
                status,
                cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "设置 cuBLAS 数学模式失败"
            );
            cudarc::cublas::result::set_stream(handle, stream as *mut cudarc::cublas::sys::CUstream_st)
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

/// `C = A @ B^T`：`A` (m, k)、`B` (n, k) 行主序 → 返回 (m, n)。
///
/// 前向 base matmul 用（`xp @ w^T`）：B 直接传权重 `w (a, in)`，**无需转置**——
/// 避免 burn 的 `transpose().reshape()` 对方阵不拷贝（strides 变 (1, k)）的坑。
/// transa=T（A 行主序 (m,k) = (k,m) 列主序）、transb=N（B 行主序 (n,k) =
/// (k,n) 列主序，op(B)=B^T）。
pub fn gemm_abt(
    a: &Tensor<Cuda, 2>,
    b: &Tensor<Cuda, 2>,
    device: &CudaDevice,
) -> Tensor<Cuda, 2> {
    let [m, k] = a.dims();
    let [n, k2] = b.dims();
    assert_eq!(k, k2, "gemm_abt 的 k 维必须一致：{k} vs {k2}");
    let st = state(device);
    let ca = as_cube(a);
    let cb = as_cube(b);
    let out: Tensor<Cuda, 2> = Tensor::empty([n, m], device);
    let co = as_cube(&out);
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let pc = raw_ptr(&co, device);
    let sa = ca.meta.strides()[0] as i32;
    let sb = cb.meta.strides()[0] as i32;
    let sc = co.meta.strides()[0] as i32;
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("设置 CUDA 上下文失败");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cudarc::cublas::result::sgemm(
            st.handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
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

/// `C = A @ B`：`A` (m, k)、`B` (k, n) 行主序（strided 支持）→ 返回 (m, n)。
///
/// 与 [`gemm_atb`] 的转置约定相反：transa=T / transb=T（A (m,k) 行主序 =
/// (k,m) 列主序，op(A)=A；B 同理），输出同样以 (n, m) 行主序承载 C^T 后转置。
pub fn gemm(a: &Tensor<Cuda, 2>, b: &Tensor<Cuda, 2>, device: &CudaDevice) -> Tensor<Cuda, 2> {
    let [m, k] = a.dims();
    let [k2, n] = b.dims();
    assert_eq!(k, k2, "gemm 的 k 维必须一致：{k} vs {k2}");
    let st = state(device);
    let ca = as_cube(a);
    let cb = as_cube(b);
    let out: Tensor<Cuda, 2> = Tensor::empty([n, m], device);
    let co = as_cube(&out);
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let pc = raw_ptr(&co, device);
    let sa = ca.meta.strides()[0] as i32;
    let sb = cb.meta.strides()[0] as i32;
    let sc = co.meta.strides()[0] as i32;
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("设置 CUDA 上下文失败");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cudarc::cublas::result::sgemm(
            st.handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
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

/// 批量 `y = x @ B^T`：`x` (m, n, k)（**批在中间维**，如 (T, n, in)）、
/// `B` (n, r, k) 行主序 → (n, m, r)。
///
/// 每批：op(A) = X_n（stored (k,m) 列主序，transa=T）、op(B) = B_n^T
/// （stored (k,r) 列主序，transb=N）；C (m, r) 列主序，以 (n, r, m) 行主序承载
/// C^T 后转置。批维在中间的好处：直接消费 LIF/泊松输出的 (T, n, *) 布局，
/// 无需 swap/reshape（后者在带 pitch 的张量上会产生畸形 strides）。
pub fn batched_gemm_bt(
    x: &Tensor<Cuda, 3>,
    b: &Tensor<Cuda, 3>,
    device: &CudaDevice,
) -> Tensor<Cuda, 3> {
    let [m, n, k] = x.dims();
    let [n2, r, k2] = b.dims();
    assert_eq!((n, k), (n2, k2), "batched_gemm_bt 形状不匹配");
    let st = state(device);
    let ca = as_cube(x);
    let cb = as_cube(b);
    let out: Tensor<Cuda, 3> = Tensor::empty([n, r, m], device);
    let co = as_cube(&out);
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let pc = raw_ptr(&co, device);
    // x (m, n, k)：每批矩阵 (m, k) 的 row stride = dim0 stride；批间 stride = dim1 stride。
    let lda = ca.meta.strides()[0] as i32;
    let sa = ca.meta.strides()[1] as i32;
    let ldb = cb.meta.strides()[1] as i32; // 行主序 (r,k) 的 row stride = k（或 pitch）
    let sb = cb.meta.strides()[0] as i32; // 批间 stride
    let ldc = co.meta.strides()[1] as i32; // 输出 (n, r, m) 行主序的 row stride
    let sc = co.meta.strides()[0] as i32; // 输出批间 stride
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("设置 CUDA 上下文失败");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cudarc::cublas::result::sgemm_strided_batched(
            st.handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            r as i32,
            k as i32,
            &alpha,
            pa as *const f32,
            lda,
            sa as i64,
            pb as *const f32,
            ldb,
            sb as i64,
            &beta,
            pc as *mut f32,
            ldc,
            sc as i64,
            n as i32,
        )
        .expect("cublasSgemmStridedBatched 失败");
    }
    out.transpose()
}

/// 批量 `y = x @ B^T`（批在第一维，每批连续）：`x` (n, m, k) 行主序连续、
/// `B` (n, r, k) 行主序连续 → (n, m, r)。与 [`batched_gemm_bt`] 数学一致，
/// 但输入按批连续（lda = k、sa = m·k），供需要先 permute 输入为 (n, m, *) 的场景
/// （代价是一次拷贝；批在中间维的 strided 布局在 cuBLAS 上可能显著更慢）。
pub fn batched_gemm_bt_first(
    x: &Tensor<Cuda, 3>,
    b: &Tensor<Cuda, 3>,
    device: &CudaDevice,
) -> Tensor<Cuda, 3> {
    let [n, m, k] = x.dims();
    let [n2, r, k2] = b.dims();
    assert_eq!((n, k), (n2, k2), "batched_gemm_bt_first 形状不匹配");
    let st = state(device);
    let ca = as_cube(x);
    let cb = as_cube(b);
    let out: Tensor<Cuda, 3> = Tensor::empty([n, r, m], device);
    let co = as_cube(&out);
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let pc = raw_ptr(&co, device);
    let lda = ca.meta.strides()[1] as i32; // 每批 (m,k) 行主序的 row stride = k
    let sa = ca.meta.strides()[0] as i32; // 批间 stride = m·k
    let ldb = cb.meta.strides()[1] as i32; // (r,k) 行主序的 row stride = k
    let sb = cb.meta.strides()[0] as i32; // 批间 stride
    let ldc = co.meta.strides()[1] as i32;
    let sc = co.meta.strides()[0] as i32;
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("设置 CUDA 上下文失败");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cudarc::cublas::result::sgemm_strided_batched(
            st.handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            r as i32,
            k as i32,
            &alpha,
            pa as *const f32,
            lda,
            sa as i64,
            pb as *const f32,
            ldb,
            sb as i64,
            &beta,
            pc as *mut f32,
            ldc,
            sc as i64,
            n as i32,
        )
        .expect("cublasSgemmStridedBatched 失败");
    }
    out.transpose()
}

/// 批量 `z = y @ A`：`y` (n, m, k) 与 `A` (n, k, l) 均行主序连续 → (n, m, l)。
///
/// `y` 支持两种每批布局（按 strides 自动选择）：
/// - 连续 (n, m, k)（每批 (m,k) 行主序，`strides[1] = k`）：transa=T、lda = strides[1]；
/// - [`batched_gemm_bt`] 的转置视图（每批 (m,k) 列主序，`strides[1] = 1`、
///   `strides[2] = m`）：transa=N、lda = strides[2]。
/// 每批：op(A) = Y_n、op(B) = A_n（stored (l,k) 列主序，transb=T）；C (m, l)
/// 列主序，以 (n, l, m) 行主序承载 C^T 后转置。用于前向的噪声注入第二步 `z = y @ A'`。
pub fn batched_gemm(
    y: &Tensor<Cuda, 3>,
    a: &Tensor<Cuda, 3>,
    device: &CudaDevice,
) -> Tensor<Cuda, 3> {
    let [n, m, k] = y.dims();
    let [n2, k2, l] = a.dims();
    assert_eq!((n, k), (n2, k2), "batched_gemm 形状不匹配");
    let st = state(device);
    let ca = as_cube(y);
    let cb = as_cube(a);
    let out: Tensor<Cuda, 3> = Tensor::empty([n, l, m], device);
    let co = as_cube(&out);
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let pc = raw_ptr(&co, device);
    let sa = ca.meta.strides()[0] as i32;
    let sb = cb.meta.strides()[0] as i32;
    let sc = co.meta.strides()[0] as i32;
    // y 每批 (m,k) 矩阵：strides[1]==1 时为列主序（batched_gemm_bt 的转置视图），
    // 否则为行主序（连续 (n,m,k)）。
    let (transa, lda) = if ca.meta.strides()[1] == 1 {
        (
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            ca.meta.strides()[2] as i32,
        )
    } else {
        (
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            ca.meta.strides()[1] as i32,
        )
    };
    let ldb = cb.meta.strides()[1] as i32; // 行主序 (k,l) 的 row stride = l
    // 输出 (n, l, m) 行主序：C (m, l) 列主序的 leading dim = 行主序的 row stride。
    let ldc = co.meta.strides()[1] as i32;
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("设置 CUDA 上下文失败");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cudarc::cublas::result::sgemm_strided_batched(
            st.handle,
            transa,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            m as i32,
            l as i32,
            k as i32,
            &alpha,
            pa as *const f32,
            lda,
            sa as i64,
            pb as *const f32,
            ldb,
            sb as i64,
            &beta,
            pc as *mut f32,
            ldc,
            sc as i64,
            n as i32,
        )
        .expect("cublasSgemmStridedBatched 失败");
    }
    out.transpose()
}

/// 单个线性层的 cuBLAS 批量 LoRA 前向：`x` (T, n, in)（批在中间维）-> (T, n, a)。
///
/// 与 [`TrainableVthSnn::forward_batched_lora`] 的 `lora_linear_batched` 数学一致：
/// - base = x @ w^T：一次 2D GEMM（x 展平 (T·n, in)）；
/// - y = x @ B'^T（`batched_gemm_bt`，批在中间维，直接消费 (T, n, *) 布局）；
/// - z = y @ A'（`batched_gemm`）。
/// 全部走 cuBLAS（同流、无同步）；不 permute 输入（swap+reshape 在带 pitch 的
/// 张量上会产生畸形 strides）。
fn lora_linear_cublas(
    x: &Tensor<Cuda, 3>,                       // (T, n, in)
    w: &Tensor<Cuda, 2>,                        // (a, in)
    noise: &(Tensor<Cuda, 3>, Tensor<Cuda, 3>), // (A' (n,r,a), B' (n,r,b))
    device: &CudaDevice,
) -> Tensor<Cuda, 3> {                          // (T, n, a)
    let [t, n, in_dim] = x.dims();
    let [a, _in] = w.dims();
    let (a_ra, b_rb) = noise;
    // base = x @ w^T：B 直接传权重（gemm_abt 内部 transb=N，无需转置）。
    let base = gemm_abt(&x.clone().reshape([t * n, in_dim]), w, device).reshape([t, n, a]);
    // 噪声两步：y = x @ B'^T（批在中间维）；z = y @ A'（(n, T, r) 布局）。
    let y = batched_gemm_bt(x, b_rb, device); // (n, T, r)
    let z = batched_gemm(&y, a_ra, device); // (n, T, a)
    base + z.swap_dims(0, 1) // (T, n, a)
}

/// 前向的 cuBLAS 版（matmul 全走 cuBLAS，LIF 保持 burn），与
/// [`TrainableVthSnn::forward_batched_lora`] 数学一致：`(T, n, in)` -> `(n, C)`。
pub fn forward_batched_lora_cublas(
    model: &TrainableVthSnn,
    x: Tensor<Cuda, 3>,
    th1: f32,
    th2: f32,
    noise: &[(Tensor<Cuda, 3>, Tensor<Cuda, 3>)],
    device: &CudaDevice,
) -> Tensor<Cuda, 2> {
    use hyperscalees_models::snn::{LifParams, run_lif};
    let [t, n, _in] = x.dims();
    let p1 = LifParams { tau_m: model.tau_m, v_th: th1 };
    let p2 = LifParams { tau_m: model.tau_m, v_th: th2 };
    // 第 1 层：x (T, n, in) 直接使用（批在中间维，无需 permute）。
    let cur1 = lora_linear_cublas(&x, &model.fc1.weight, &noise[0], device); // (T, n, h1)
    let v0_1 = Tensor::<Cuda, 2>::zeros([n, model.fc1.weight.dims()[0]], device);
    let spikes1 = run_lif(p1, cur1, v0_1); // (T, n, h1)
    // 第 2 层。
    let cur2 = lora_linear_cublas(&spikes1, &model.fc2.weight, &noise[1], device); // (T, n, h2)
    let v0_2 = Tensor::<Cuda, 2>::zeros([n, model.fc2.weight.dims()[0]], device);
    let spikes2 = run_lif(p2, cur2, v0_2); // (T, n, h2)
    // 读出：mean rate -> fc3（噪声注入同为 batched GEMM，m=1）-> gain。
    let rate = spikes2.mean_dim(0).squeeze_dim::<2>(0); // (n, h2)
    let (a3, b3) = &noise[2];
    let base3 = gemm_abt(&rate.clone(), &model.fc3.weight, device); // (n, C)
    let rate_u = rate.clone().unsqueeze_dim::<3>(0); // (1, n, h2) —— 批在中间维
    let y3 = batched_gemm_bt(&rate_u, b3, device); // (n, 1, r)
    let z3 = batched_gemm(&y3, a3, device); // (n, 1, C)
    let logits = base3 + z3.squeeze_dim::<2>(1);
    let gain = model.out_gain.value.clone().unsqueeze::<2>(); // (1, 1)
    logits * gain
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

/// 反对称配对的 LoRA 噪声生成（半量版）：返回 `(A'_h (n/2,r,a) 已乘 base_sigma, B'_h (n/2,r,b))`。
///
/// 配对隐含：样本 `n/2+i` 的噪声 = 样本 `i` 的噪声取负（由前向
/// [`TrainableVthSnn::forward_batched_lora_half`] 与 `lora_einsum_pair_half` 消费方
/// 施加）。**只生成 n/2 个样本**：fc1 B' 从 2.4GB → 1.2GB，噪声生成阶段实测
/// ~17ms → ~7ms/chunk（plain 内核；此前「一次内核生成完整配对张量」的反对称内核
/// 需两次生成或双写流，均更慢）。A' 用 `std = base_sigma` 直接生成。
///
/// 要求 `n` 为偶数（本工作负载恒成立）。
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
    assert!(n % 2 == 0, "半量噪声生成要求 n 为偶数，实际 {n}");

    let b_h: Tensor<Cuda, 3> = Tensor::empty([n / 2, r, b], device);
    let cbh = as_cube(&b_h);
    let client_h = cbh.client.clone();
    cubek_random::random_normal(&client_h, 0.0, 1.0, cbh.binding(), dtype)
        .expect("B' 前半噪声生成失败");
    let a_h: Tensor<Cuda, 3> = Tensor::empty([n / 2, r, a], device);
    let cah = as_cube(&a_h);
    let client_ah = cah.client.clone();
    cubek_random::random_normal(&client_ah, 0.0, base_sigma, cah.binding(), dtype)
        .expect("A' 前半噪声生成失败");

    (a_h, b_h)
}
