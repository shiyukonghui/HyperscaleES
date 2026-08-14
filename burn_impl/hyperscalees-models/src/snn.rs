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
}
