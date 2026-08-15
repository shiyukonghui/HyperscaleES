//! Leaky Integrate-and-Fire (LIF) SNN model, ported from
//! `src/hyperscalees/models/snn.py`.
//!
//! Architecture (two LIF hidden layers + rate readout head):
//!
//! ```text
//!     x: (T, batch, in_dim)  binary Poisson spikes
//!       -> Linear(noised)                  (per timestep)
//!       -> LIF hidden layer 1              (recurrence over T)
//!       -> Linear(noised)
//!       -> LIF hidden layer 2
//!       -> Linear(noised)
//!       -> mean firing rate over time
//!       -> readout logits (batch, num_classes)
//! ```
//!
//! The model is trained with the noiser (evolutionary strategy) abstraction,
//! so it does not rely on gradients through the non-differentiable spike
//! function. The noised matmul is injected as a *closure* so this crate does
//! not (and must not) depend on `hyperscalees-noiser` (which would create a
//! dependency cycle).

use burn::tensor::{Device, Tensor};
use hyperscalees_core::B;

use crate::common::{Mm, Parameter};

/// Frozen LIF hyper-parameters (`tau_m`, `v_th`). In Python these come from
/// `frozen_params` so they are never evolved.
#[derive(Clone, Copy, Debug)]
pub struct LifParams {
    /// Membrane time constant (leak rate = 1 / tau_m).
    pub tau_m: f32,
    /// Firing threshold.
    pub v_th: f32,
}

/// Single LIF update: leak -> charge -> fire -> reset.
///
/// ```text
///   v      = v + (dt / tau_m) * (-v + current)
///   spike  = (v >= v_th).float()
///   v      = v * (1.0 - spike)      # hard reset
/// ```
///
/// `v` and `current` must have the same shape (rank `D`). Returns
/// `(new_v, spike)` where `spike` is the same rank as `v` with 0/1 values.
pub fn lif_step<const D: usize>(
    tau_m: f32,
    v_th: f32,
    v: Tensor<B, D>,
    current: Tensor<B, D>,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let dt = 1.0_f32;
    let leak_scale = dt / tau_m;
    let charged = v.clone() + (v.neg() + current).mul_scalar(leak_scale);
    let spike = charged.clone().greater_equal_elem(v_th).float();
    let reset = charged.clone() * spike.clone().neg().add_scalar(1.0);
    (reset, spike)
}

/// Run LIF dynamics over the time axis.
///
/// `input_current` is `(T, batch, hidden)`; `v0` is `(batch, hidden)`.
/// Returns `spikes` of shape `(T, batch, hidden)` with 0/1 values. This
/// mirrors `jax.lax.scan` over the `T` axis using a plain `for` loop (T is
/// small, e.g. 5).
pub fn run_lif(
    params: LifParams,
    input_current: Tensor<B, 3>,
    v0: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let dims = input_current.dims();
    let t = dims[0];
    let batch = dims[1];
    let hidden = dims[2];

    let mut v = v0;
    let mut spikes: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
    for i in 0..t {
        let current_t = input_current
            .clone()
            .slice([i..i + 1, 0..batch, 0..hidden])
            .squeeze_dim::<2>(0);
        let (new_v, spike) = lif_step(params.tau_m, params.v_th, v, current_t);
        v = new_v;
        // Turn the rank-2 `(batch, hidden)` spike into `(1, batch, hidden)`
        // so concatenation along axis 0 stacks the time dim.
        spikes.push(spike.unsqueeze::<3>());
    }
    Tensor::cat(spikes, 0)
}

/// A noised/clean 2D matmul closure `(x, weight) -> out`, reproducing
/// EggRoll's `do_mm` (clean or with LoRA noise). The closure signature is
/// `(x_t: Tensor<B,2>, weight: Tensor<B,2>) -> Tensor<B,2>`.
pub type NoiseFn<'a> = dyn Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2> + 'a;

/// 批量噪声 matmul 闭包：`(x_t: (n,b), weight: (a,b), tids: &[i32]（该行样本的 thread_id）, epoch) -> (n,a)`。
/// 由 facade（hyperscalees crate，可依赖 noiser）提供实现；models crate 不依赖 noiser，故只定义类型。
pub type BatchedNoiseFn<'a> = dyn Fn(Tensor<B, 2>, Tensor<B, 2>, &[i32], i32) -> Tensor<B, 2> + 'a;

/// Apply the (optionally noised) matmul to a rank-2 tensor.
///
/// When `noise` is `None` this is the clean `x @ weight.T`.
fn matmul_2d(
    x: Tensor<B, 2>,
    weight: &Tensor<B, 2>,
    noise: Option<&NoiseFn>,
) -> Tensor<B, 2> {
    match noise {
        Some(f) => f(x, weight.clone()),
        None => x.matmul(weight.clone().transpose()),
    }
}

/// Apply the (optionally noised) matmul across the leading (time) axis of a
/// rank-3 tensor `(T, batch, in)` -> `(T, batch, out)`.
///
/// Each `x_t` is sliced from the time axis, run through `matmul_2d`, and
/// unsqueezed back to `(1, batch, out)` before concatenation, mirroring
/// `jax.vmap(proj)` over the time axis.
fn matmul_3d(
    x: Tensor<B, 3>,
    weight: &Tensor<B, 2>,
    noise: Option<&NoiseFn>,
) -> Tensor<B, 3> {
    let dims = x.dims();
    let t = dims[0];
    let batch = dims[1];
    let in_dim = dims[2];
    let mut parts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
    for i in 0..t {
        let x_t = x
            .clone()
            .slice([i..i + 1, 0..batch, 0..in_dim])
            .squeeze_dim::<2>(0);
        parts.push(matmul_2d(x_t, weight, noise).unsqueeze::<3>());
    }
    Tensor::cat(parts, 0)
}

/// 批量版 `matmul_3d`：`(T, n, in)` -> `(T, n, out)`。
///
/// 与 [`matmul_3d`] 语义完全一致，但每个时间步的 `x_t (n, in)` 作为**整块**
/// 交给批量噪声闭包一次处理（返回 `(n, a)`），而非逐样本（batch=1）调用；
/// 等价于 Python 的 `jax.vmap` 整批前向，避免 GPU 上大量小 matmul 的开销。
///
/// - `Some(f)` → `f(x_t, weight.clone(), tids, epoch)`；
/// - `None` → 干净路径 `x_t.matmul(weight.clone().transpose())`。
///
/// 逐时间步结果 `unsqueeze` 为 `(1, n, out)` 后沿轴 0 拼接。
fn matmul_3d_batched(
    x: Tensor<B, 3>,        // (T, n, in)
    weight: &Tensor<B, 2>,  // (a, b)
    tids: &[i32],           // 长度 n
    epoch: i32,
    noise: Option<&BatchedNoiseFn>,
) -> Tensor<B, 3> {
    let dims = x.dims();
    let t = dims[0];
    let n = dims[1];
    let in_dim = dims[2];
    let mut parts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
    for i in 0..t {
        let x_t = x
            .clone()
            .slice([i..i + 1, 0..n, 0..in_dim])
            .squeeze_dim::<2>(0);
        let out = match noise {
            Some(f) => f(x_t, weight.clone(), tids, epoch), // (n, a)
            None => x_t.matmul(weight.clone().transpose()), // (n, a)
        };
        parts.push(out.unsqueeze::<3>());
    }
    Tensor::cat(parts, 0)
}

/// A two-LIF-layer SNN classifier, mirroring `SNNModel` in `snn.py`.
///
/// `x` is `(T, batch, in_dim)` binary spikes; `forward` returns readout
/// logits `(batch, num_classes)` scaled by the `out_gain` parameter.
pub struct SnnModel {
    /// fc1 weight, shape `(h1, in_dim)`.
    pub fc1: Mm,
    /// fc2 weight, shape `(h2, h1)`.
    pub fc2: Mm,
    /// fc3 weight, shape `(num_classes, h2)`.
    pub fc3: Mm,
    /// Readout gain parameter, shape `(1,)`.
    pub out_gain: Parameter,
    /// Frozen membrane time constant.
    pub tau_m: f32,
    /// Frozen firing threshold.
    pub v_th: f32,
}

impl SnnModel {
    /// Build an SNN with the given architecture, mirroring
    /// `SNNModel.rand_init` (weights scaled `1/sqrt(fan_in)`, gain = ones).
    pub fn new(
        in_dim: usize,
        hidden1: usize,
        hidden2: usize,
        num_classes: usize,
        device: &Device<B>,
    ) -> Self {
        let fc1 = Mm::new(in_dim, hidden1, device);
        let fc2 = Mm::new(hidden1, hidden2, device);
        let fc3 = Mm::new(hidden2, num_classes, device);
        let out_gain = Parameter::new(Tensor::<B, 1>::ones([1], device));
        Self {
            fc1,
            fc2,
            fc3,
            out_gain,
            tau_m: 20.0,
            v_th: 0.3,
        }
    }

    /// The parameters in trainable order `[fc1, fc2, fc3, out_gain]`.
    pub fn params(&self) -> Vec<Tensor<B, 2>> {
        vec![
            self.fc1.weight.clone(),
            self.fc2.weight.clone(),
            self.fc3.weight.clone(),
            // out_gain is a rank-1 `(1,)`; unsqueeze to rank-2 for the shared
            // matrix parameter plumbing.
            self.out_gain.value.clone().unsqueeze::<2>(),
        ]
    }

    /// The ES map (classification) for each parameter in [`Self::params`]
    /// order: fc1/fc2/fc3 are `MM_PARAM`, out_gain is `PARAM`.
    pub fn es_map(&self) -> Vec<i32> {
        use crate::common::{MM_PARAM, PARAM};
        vec![MM_PARAM, MM_PARAM, MM_PARAM, PARAM]
    }

    /// Forward pass over `(T, batch, in_dim)` spikes -> `(batch, num_classes)`
    /// logits, mirroring the per-sample `SNNModel._forward` vmapped over the
    /// batch axis.
    ///
    /// `noise` is an optional `(x_t, weight) -> out` closure reproducing
    /// EggRoll's `do_mm`. When `None` the forward is the clean, deterministic
    /// `x @ weight.T` path and the result is reproducible.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        noise: Option<&NoiseFn>,
    ) -> Tensor<B, 2> {
        let batch = x.dims()[1];
        let device = x.device().clone();
        let params = LifParams {
            tau_m: self.tau_m,
            v_th: self.v_th,
        };

        // Layer 1: linear projection per timestep, then LIF over T.
        let cur1 = matmul_3d(x, &self.fc1.weight, noise); // (T, batch, h1)
        let v0_1 = Tensor::<B, 2>::zeros([batch, self.fc1.weight.dims()[0]], &device);
        let spikes1 = run_lif(params, cur1, v0_1); // (T, batch, h1)

        // Layer 2.
        let cur2 = matmul_3d(spikes1, &self.fc2.weight, noise); // (T, batch, h2)
        let v0_2 = Tensor::<B, 2>::zeros([batch, self.fc2.weight.dims()[0]], &device);
        let spikes2 = run_lif(params, cur2, v0_2); // (T, batch, h2)

        // Readout: mean firing rate over time -> fc3 -> logits * gain.
        // `mean_dim(0)` keeps the reduced time axis as size 1, so `squeeze_dim`
        // removes only that leading dim -> `(batch, h2)` rates in [0, 1].
        let rate = spikes2.mean_dim(0).squeeze_dim::<2>(0); // (batch, h2)
        let logits = matmul_2d(rate, &self.fc3.weight, noise); // (batch, C)
        let gain = self.out_gain.value.clone().unsqueeze::<2>(); // (1, 1)
        logits * gain
    }
}

/// softplus 激活，数值稳定地实现 ``softplus(x) = ln(1 + exp(x))``。
///
/// 用于把「可训练阈值」的原始参数（raw，可为负）映射为恒正的 LIF 阈值，
/// 与 Python `jax.nn.softplus` 语义一致。`x` 为 rank-1 张量。
fn softplus(x: Tensor<B, 1>) -> Tensor<B, 1> {
    x.clone().exp().log1p()
}

/// 两层 LIF 的可训练阈值（trainable v_th）SNN 变体，
/// 对齐 `llm_experiments/snn_mnist_train_accumulate.py` 中的
/// `TrainableVthSNN`（hidden_dims=[128,128]，in_dim=784，num_classes=10）。
///
/// 与 [`SnnModel`] 的区别：每隐层一个可训练 v_th（以 raw 形式存储，
/// 前向用 `softplus(v_th_i)` 作为该层 LIF 阈值），且 `tau_m` 冻结为 20.0；
/// `out_gain` 及读出逻辑保持一致。
///
/// ```text
///     x: (T, batch, in_dim)  binary Poisson spikes
///       -> Linear(fc1, noised)                  (per timestep)
///       -> LIF hidden layer 1  (v_th = softplus(v_th1))
///       -> Linear(fc2, noised)
///       -> LIF hidden layer 2  (v_th = softplus(v_th2))
///       -> mean firing rate over time
///       -> Linear(fc3, noised) -> * out_gain   -> logits (batch, num_classes)
/// ```
pub struct TrainableVthSnn {
    /// fc1 权重，形状 `(h1, in_dim)`。
    pub fc1: Mm,
    /// fc2 权重，形状 `(h2, h1)`。
    pub fc2: Mm,
    /// fc3 权重（读出层），形状 `(num_classes, h2)`。
    pub fc3: Mm,
    /// 读出增益参数，形状 `(1,)`。
    pub out_gain: Parameter,
    /// 第 1 隐层可训练阈值（raw 形式，前向 softplus），形状 `(1,)`。
    pub v_th1: Parameter,
    /// 第 2 隐层可训练阈值（raw 形式，前向 softplus），形状 `(1,)`。
    pub v_th2: Parameter,
    /// 冻结的膜时间常数（不进 ES 训练）。
    pub tau_m: f32,
}

impl TrainableVthSnn {
    /// 构建两层可训练阈值 SNN，对齐 `TrainableVthSNN.rand_init`：
    /// 权重按 `1/sqrt(fan_in)` 缩放（`Mm::new`），`out_gain` 为 ones，
    /// 每个 `v_th_i` 初始化为 `log(exp(v_th) - 1)`（softplus 后等于 `v_th`，
    /// 默认 `v_th=0.3`）。
    pub fn new(
        in_dim: usize,
        hidden1: usize,
        hidden2: usize,
        num_classes: usize,
        v_th: f32,
        device: &Device<B>,
    ) -> Self {
        let fc1 = Mm::new(in_dim, hidden1, device);
        let fc2 = Mm::new(hidden1, hidden2, device);
        let fc3 = Mm::new(hidden2, num_classes, device);
        let out_gain = Parameter::new(Tensor::<B, 1>::ones([1], device));
        // raw_vth0 = ln(exp(v_th) - 1)，softplus(raw_vth0) == v_th。
        let raw_vth0 = (v_th.exp() - 1.0).ln();
        let v_th1 = Parameter::new(Tensor::<B, 1>::from_data([raw_vth0], device));
        let v_th2 = Parameter::new(Tensor::<B, 1>::from_data([raw_vth0], device));
        Self {
            fc1,
            fc2,
            fc3,
            out_gain,
            v_th1,
            v_th2,
            tau_m: 20.0,
        }
    }

    /// 可训练参数，顺序 `[fc1, fc2, fc3, out_gain(1,1), v_th1(1,1), v_th2(1,1)]`。
    ///
    /// `out_gain`/`v_th` 均为 rank-1 的 `(1,)`，先 unsqueeze 为 `(1,1)`
    /// 进入共享的 rank-2 矩阵参数管线（与 [`SnnModel::params`] 对 out_gain
    /// 的处理一致）。
    pub fn params(&self) -> Vec<Tensor<B, 2>> {
        vec![
            self.fc1.weight.clone(),
            self.fc2.weight.clone(),
            self.fc3.weight.clone(),
            self.out_gain.value.clone().unsqueeze::<2>(),
            self.v_th1.value.clone().unsqueeze::<2>(),
            self.v_th2.value.clone().unsqueeze::<2>(),
        ]
    }

    /// ES 分类（es_map），与 [`Self::params`] 顺序一一对应：
    /// fc1/fc2/fc3 为 `MM_PARAM`，out_gain/v_th1/v_th2 为 `PARAM`。
    pub fn es_map(&self) -> Vec<i32> {
        use crate::common::{MM_PARAM, PARAM};
        vec![MM_PARAM, MM_PARAM, MM_PARAM, PARAM, PARAM, PARAM]
    }

    /// 前向：`(T, batch, in_dim)` 尖峰 -> `(batch, num_classes)` logits。
    ///
    /// 与 [`SnnModel::forward`] 结构一致，仅把每层 LIF 的固定 `v_th` 换成
    /// 可训练的 `softplus(v_th_i)`；`tau_m` 冻结为 `self.tau_m`。
    /// `noise` 为可选的 `(x_t, weight) -> out` 闭包（复现 noiser 的 `do_mm`），
    /// 为 `None` 时走干净确定性的 `x @ weight.T` 路径。
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        noise: Option<&NoiseFn>,
    ) -> Tensor<B, 2> {
        let batch = x.dims()[1];
        let device = x.device().clone();
        // 每层阈值 = softplus(可训练 raw v_th)，恒为正。
        let th1 = softplus(self.v_th1.value.clone());
        let th2 = softplus(self.v_th2.value.clone());
        let p1 = LifParams { tau_m: self.tau_m, v_th: to_scalar(&th1) };
        let p2 = LifParams { tau_m: self.tau_m, v_th: to_scalar(&th2) };

        // 第 1 层：逐时间步线性投影，再 LIF 扫描。
        let cur1 = matmul_3d(x, &self.fc1.weight, noise); // (T, batch, h1)
        let v0_1 = Tensor::<B, 2>::zeros([batch, self.fc1.weight.dims()[0]], &device);
        let spikes1 = run_lif(p1, cur1, v0_1); // (T, batch, h1)

        // 第 2 层。
        let cur2 = matmul_3d(spikes1, &self.fc2.weight, noise); // (T, batch, h2)
        let v0_2 = Tensor::<B, 2>::zeros([batch, self.fc2.weight.dims()[0]], &device);
        let spikes2 = run_lif(p2, cur2, v0_2); // (T, batch, h2)

        // 读出：时间轴上的平均发放率 -> fc3 -> logits * gain。
        let rate = spikes2.mean_dim(0).squeeze_dim::<2>(0); // (batch, h2)
        let logits = matmul_2d(rate, &self.fc3.weight, noise); // (batch, C)
        let gain = self.out_gain.value.clone().unsqueeze::<2>(); // (1, 1)
        logits * gain
    }

    /// 批量前向：`(T, n, in_dim)` 尖峰 -> `(n, num_classes)` logits。
    ///
    /// 与 [`Self::forward`] 数学语义**完全一致**（含阈值 softplus、tau_m、
    /// LIF 扫描与读出 gain），仅把每层 matmul 换成整块批量版
    /// [`matmul_3d_batched`]：每个时间步把整个 `(n, in)` 样本块一次交给
    /// 批量噪声闭包，等价于 Python 的 `jax.vmap` 整批前向。逐样本（batch=1）
    /// 前向在 GPU 上每样本一次内核调用的开销不可接受（如 60000 batch），
    /// 批量版可显著降低调度开销。
    ///
    /// - `tids`：长度 n 的线程 id 数组（`noise` 为 `None` 时可为任意值，如全 0），
    ///   按行透传给噪声闭包；
    /// - `epoch`：透传给噪声闭包；
    /// - `noise`：批量噪声 matmul 闭包 `(x_t, weight, tids, epoch) -> out`，
    ///   为 `None` 时走干净确定性的 `x @ weight.T` 路径。
    pub fn forward_batched(
        &self,
        x: Tensor<B, 3>,        // (T, n, 784)
        tids: &[i32],           // 长度 n（噪声为 None 时可为任意，如全 0）
        epoch: i32,
        noise: Option<&BatchedNoiseFn>,
    ) -> Tensor<B, 2> {         // (n, 10)
        let batch = x.dims()[1];
        let device = x.device().clone();
        // 每层阈值 = softplus(可训练 raw v_th)，恒为正（与 forward 一致）。
        let th1 = softplus(self.v_th1.value.clone());
        let th2 = softplus(self.v_th2.value.clone());
        let p1 = LifParams { tau_m: self.tau_m, v_th: to_scalar(&th1) };
        let p2 = LifParams { tau_m: self.tau_m, v_th: to_scalar(&th2) };

        // 第 1 层：整块批量线性投影，再 LIF 扫描。
        let cur1 = matmul_3d_batched(x, &self.fc1.weight, tids, epoch, noise); // (T, n, h1)
        let v0_1 = Tensor::<B, 2>::zeros([batch, self.fc1.weight.dims()[0]], &device);
        let spikes1 = run_lif(p1, cur1, v0_1); // (T, n, h1)

        // 第 2 层。
        let cur2 = matmul_3d_batched(spikes1, &self.fc2.weight, tids, epoch, noise); // (T, n, h2)
        let v0_2 = Tensor::<B, 2>::zeros([batch, self.fc2.weight.dims()[0]], &device);
        let spikes2 = run_lif(p2, cur2, v0_2); // (T, n, h2)

        // 读出：时间轴上的平均发放率 -> fc3 -> logits * gain。
        let rate = spikes2.mean_dim(0).squeeze_dim::<2>(0); // (n, h2)
        let logits = match noise {
            Some(f) => f(rate, self.fc3.weight.clone(), tids, epoch), // (n, C)
            None => rate.matmul(self.fc3.weight.clone().transpose()), // (n, C)
        };
        let gain = self.out_gain.value.clone().unsqueeze::<2>(); // (1, 1)
        logits * gain
    }
}

/// 提取单元素张量的唯一标量值（用于构造 `LifParams` 的标量 v_th）。
fn to_scalar<const D: usize>(t: &Tensor<B, D>) -> f32 {
    t.clone().into_scalar()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    }

    // -- lif_step ----------------------------------------------------------

    #[test]
    fn lif_step_suprathreshold_current_fires() {
        // Strong constant current exceeds v_th and fires immediately.
        let v = Tensor::<B, 1>::zeros([1], &device());
        let current = Tensor::<B, 1>::from_data([10.0_f32], &device());
        let (_, spike) = lif_step(20.0, 0.3, v, current);
        // v = (1/20)*10 = 5.0 >= 0.3 -> spike = 1.
        assert_eq!(to_vec(spike), vec![1.0_f32]);
    }

    #[test]
    fn lif_step_subthreshold_current_never_fires() {
        // Tiny current keeps v below v_th, so no spike.
        let v = Tensor::<B, 1>::zeros([1], &device());
        let current = Tensor::<B, 1>::from_data([0.1_f32], &device());
        let (_, spike) = lif_step(20.0, 0.3, v, current);
        // v = (1/20)*0.1 = 0.005 < 0.3 -> spike = 0.
        assert_eq!(to_vec(spike), vec![0.0_f32]);
    }

    #[test]
    fn lif_step_accumulates_then_fires() {
        // A moderate current below threshold accumulates over steps and fires.
        let mut v = Tensor::<B, 1>::zeros([1], &device());
        let current = Tensor::<B, 1>::from_data([6.0_f32], &device());
        let mut fired = false;
        for _ in 0..10 {
            let (new_v, spike) = lif_step(20.0, 0.3, v, current.clone());
            v = new_v;
            if to_vec(spike)[0] == 1.0 {
                fired = true;
                break;
            }
        }
        assert!(fired, "suprathreshold current must eventually fire");
    }

    // -- run_lif -----------------------------------------------------------

    #[test]
    fn run_lif_shape_and_firing() {
        let params = LifParams { tau_m: 20.0, v_th: 0.3 };
        // Strong constant current everywhere: T=5, batch=2, hidden=3.
        let current = Tensor::<B, 3>::from_data(
            [[[5.0_f32; 3]; 2]; 5],
            &device(),
        );
        let v0 = Tensor::<B, 2>::zeros([2, 3], &device());
        let spikes = run_lif(params, current, v0);
        assert_eq!(spikes.dims(), [5, 2, 3]);
        // With current=5, tau_m=20, v_th=0.3 the membrane accumulates and
        // fires within a few steps (not necessarily every timestep), so some
        // spikes must appear.
        let vals = to_vec(spikes);
        assert!(
            vals.iter().any(|&s| s == 1.0),
            "strong current must produce some spikes"
        );
    }

    #[test]
    fn run_lif_silent_with_negative_current() {
        let params = LifParams { tau_m: 20.0, v_th: 0.3 };
        // Negative current keeps the membrane well below v_th, so no spikes.
        let current = Tensor::<B, 3>::from_data(
            [[[-10.0_f32; 2]; 1]; 4],
            &device(),
        );
        let v0 = Tensor::<B, 2>::zeros([1, 2], &device());
        let spikes = run_lif(params, current, v0);
        let vals = to_vec(spikes);
        assert!(vals.iter().all(|&s| s == 0.0));
    }

    // -- clean forward: determinism, shape, no-noise -----------------------

    #[test]
    fn snn_forward_clean_deterministic_and_shaped() {
        let model = SnnModel::new(4, 8, 16, 3, &device());
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
             [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
             [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
            &device(),
        );
        let out1 = model.forward(x.clone(), None);
        let out2 = model.forward(x, None);
        assert_eq!(out1.dims(), [2, 3]);
        // Clean no-noise forward is deterministic and reproducible.
        let v1 = to_vec(out1);
        let v2 = to_vec(out2);
        assert_eq!(v1, v2);
        assert!(v1.iter().all(|v| v.is_finite()));
    }

    // -- noised forward ----------------------------------------------------

    #[test]
    fn snn_forward_noised_differs_from_clean() {
        let model = SnnModel::new(4, 8, 16, 3, &device());
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
             [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
             [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
            &device(),
        );
        let clean = model.forward(x.clone(), None);

        // A fixed +1.0 additive perturbation to every matmul output.
        let noise = |xt: Tensor<B, 2>, w: Tensor<B, 2>| {
            xt.matmul(w.transpose()).add_scalar(1.0)
        };
        let noised = model.forward(x, Some(&noise));

        assert_eq!(noised.dims(), [2, 3]);
        let c = to_vec(clean);
        let n = to_vec(noised);
        assert!(
            c.iter().zip(n.iter()).any(|(a, b)| a != b),
            "perturbation must change the output"
        );
    }

    // -- structural --------------------------------------------------------

    #[test]
    fn snn_rand_init_struct_and_es_map() {
        use crate::common::{MM_PARAM, PARAM};
        let model = SnnModel::new(8, 16, 32, 4, &device());
        assert_eq!(model.fc1.weight.dims(), [16, 8]);
        assert_eq!(model.fc2.weight.dims(), [32, 16]);
        assert_eq!(model.fc3.weight.dims(), [4, 32]);
        // out_gain starts as ones.
        assert_eq!(model.out_gain.value.dims(), [1]);
        assert_eq!(to_vec(model.out_gain.value.clone()), vec![1.0_f32]);
        // es_map: fc1/fc2/fc3 are MM_PARAM, out_gain is PARAM.
        assert_eq!(model.es_map(), vec![MM_PARAM, MM_PARAM, MM_PARAM, PARAM]);
    }

    // -- reproducibility ---------------------------------------------------

    #[test]
    fn snn_forward_reproducible_across_calls() {
        // Two independent clean forward calls on the same input must give
        // identical outputs (reproducible, no hidden RNG).
        let model = SnnModel::new(5, 10, 20, 4, &device());
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 1.0, 0.0]]],
            &device(),
        );
        let a = model.forward(x.clone(), None);
        let b = model.forward(x, None);
        assert_eq!(to_vec(a), to_vec(b));
    }

    // -----------------------------------------------------------------------
    // TrainableVthSnn（可训练阈值变体）
    // -----------------------------------------------------------------------

    // -- 结构 & es_map -----------------------------------------------------

    #[test]
    fn trainable_vth_struct_and_es_map() {
        use crate::common::{MM_PARAM, PARAM};
        let model = TrainableVthSnn::new(8, 16, 32, 4, 0.3, &device());
        // 各权重形状正确。
        assert_eq!(model.fc1.weight.dims(), [16, 8]);
        assert_eq!(model.fc2.weight.dims(), [32, 16]);
        assert_eq!(model.fc3.weight.dims(), [4, 32]);
        // out_gain 与 v_th 均为 rank-1 的 (1,)。
        assert_eq!(model.out_gain.value.dims(), [1]);
        assert_eq!(model.v_th1.value.dims(), [1]);
        assert_eq!(model.v_th2.value.dims(), [1]);
        // out_gain 初始为 ones。
        assert_eq!(to_vec(model.out_gain.value.clone()), vec![1.0_f32]);
        // params：6 项，顺序 [fc1, fc2, fc3, out_gain(1,1), v_th1(1,1), v_th2(1,1)]，
        // out_gain/v_th 被 unsqueeze 为 (1,1) 进入 rank-2 管线。
        let params = model.params();
        assert_eq!(params.len(), 6);
        assert_eq!(params[0].dims(), [16, 8]);
        assert_eq!(params[1].dims(), [32, 16]);
        assert_eq!(params[2].dims(), [4, 32]);
        assert_eq!(params[3].dims(), [1, 1]);
        assert_eq!(params[4].dims(), [1, 1]);
        assert_eq!(params[5].dims(), [1, 1]);
        // es_map：fc1/fc2/fc3 为 MM_PARAM，out_gain/v_th1/v_th2 为 PARAM。
        assert_eq!(
            model.es_map(),
            vec![MM_PARAM, MM_PARAM, MM_PARAM, PARAM, PARAM, PARAM]
        );
    }

    // -- softplus(v_th) 恒正且 ≈ 0.3 ---------------------------------------

    #[test]
    fn trainable_vth_softplus_positive_and_near_target() {
        let model = TrainableVthSnn::new(8, 16, 32, 4, 0.3, &device());
        // 从 params 取 v_th1/v_th2 的 raw 值（rank-2 的 (1,1)），还原为 rank-1
        // 后做 softplus，应恒正且 ≈ 0.3（原始初始化保证 softplus(raw)==v_th）。
        let params = model.params();
        let p1 = softplus(params[4].clone().squeeze_dim::<1>(0));
        let p2 = softplus(params[5].clone().squeeze_dim::<1>(0));
        let s1 = to_vec(p1)[0];
        let s2 = to_vec(p2)[0];
        assert!(s1 > 0.0, "softplus(v_th1) 必须为正，实际 {s1}");
        assert!(s2 > 0.0, "softplus(v_th2) 必须为正，实际 {s2}");
        // 与目标 v_th=0.3 接近（浮点误差允许 1e-4）。
        assert!((s1 - 0.3).abs() < 1e-4, "softplus(v_th1)={s1} 应≈0.3");
        assert!((s2 - 0.3).abs() < 1e-4, "softplus(v_th2)={s2} 应≈0.3");
    }

    // -- clean 前向确定性 & 形状 -------------------------------------------

    #[test]
    fn trainable_vth_forward_clean_deterministic_and_shaped() {
        let model = TrainableVthSnn::new(4, 8, 16, 10, 0.3, &device());
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
             [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
             [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
            &device(),
        );
        let out1 = model.forward(x.clone(), None);
        let out2 = model.forward(x, None);
        // 形状 (batch=2, num_classes=10)。
        assert_eq!(out1.dims(), [2, 10]);
        // 相同输入两次 clean 前向逐位一致（确定性、无可复现性问题）。
        let v1 = to_vec(out1);
        let v2 = to_vec(out2);
        assert_eq!(v1, v2);
        assert!(v1.iter().all(|v| v.is_finite()));
    }

    // -- 带噪声前向：形状正确且与 clean 不同 --------------------------------

    #[test]
    fn trainable_vth_forward_noised_differs_from_clean() {
        let model = TrainableVthSnn::new(4, 8, 16, 10, 0.3, &device());
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
             [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
             [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
            &device(),
        );
        let clean = model.forward(x.clone(), None);

        // +1 加法扰动闭包：作用于每个 matmul 输出。
        let noise = |xt: Tensor<B, 2>, w: Tensor<B, 2>| {
            xt.matmul(w.transpose()).add_scalar(1.0)
        };
        let noised = model.forward(x, Some(&noise));

        // 形状正确。
        assert_eq!(noised.dims(), [2, 10]);
        // 扰动必须改变输出（与 clean 不同）。
        let c = to_vec(clean);
        let n = to_vec(noised);
        assert!(
            c.iter().zip(n.iter()).any(|(a, b)| a != b),
            "perturbation must change the output"
        );
    }

    // -- forward_batched（批量前向）-----------------------------------------

    #[test]
    fn forward_batched_clean_matches_forward() {
        // 模型与 Python 对齐：784 -> 128 -> 128 -> 10。
        let model = TrainableVthSnn::new(784, 128, 128, 10, 0.3, &device());
        // (T=3, n=5, 784) 的二元 Poisson 尖峰输入。
        let x = Tensor::<B, 3>::random(
            [3, 5, 784],
            burn::tensor::Distribution::Bernoulli(0.5),
            &device(),
        );
        let tids: Vec<i32> = (0..5).collect();
        // 批量 clean 前向（epoch=7；噪声为 None 时 tids/epoch 不参与计算）。
        let out_batched = model.forward_batched(x.clone(), &tids, 7, None);
        // 逐样本 clean 前向。
        let out_plain = model.forward(x, None);
        // 形状 (n=5, num_classes=10)。
        assert_eq!(out_batched.dims(), [5, 10]);
        assert_eq!(out_plain.dims(), [5, 10]);
        // 数学语义一致：逐位相等（容差 1e-6）。
        let a = to_vec(out_batched);
        let b = to_vec(out_plain);
        assert_eq!(a.len(), b.len());
        for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (u - v).abs() <= 1e-6,
                "批量与逐样本前向在第 {i} 个元素不一致：{u} vs {v}"
            );
        }
    }

    #[test]
    fn forward_batched_noised_differs_from_clean() {
        let model = TrainableVthSnn::new(784, 128, 128, 10, 0.3, &device());
        let x = Tensor::<B, 3>::random(
            [3, 5, 784],
            burn::tensor::Distribution::Bernoulli(0.5),
            &device(),
        );
        let tids: Vec<i32> = (0..5).collect();
        // clean 批量前向。
        let clean = model.forward_batched(x.clone(), &tids, 7, None);
        // +1 加法扰动闭包：匹配 BatchedNoiseFn 签名（4 参，忽略 tids/epoch），
        // 给每个批量 matmul 输出加常数 1。
        let noise = |xt: Tensor<B, 2>, w: Tensor<B, 2>, _tids: &[i32], _epoch: i32| {
            xt.matmul(w.transpose()).add_scalar(1.0)
        };
        let noised = model.forward_batched(x, &tids, 7, Some(&noise));
        // 形状 (n=5, num_classes=10) 正确。
        assert_eq!(noised.dims(), [5, 10]);
        // 扰动必须改变输出（与 clean 不同）。
        let c = to_vec(clean);
        let n = to_vec(noised);
        assert!(
            c.iter().zip(n.iter()).any(|(a, b)| a != b),
            "perturbation must change the output"
        );
    }
}
