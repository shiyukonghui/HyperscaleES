//! SNN Attention models (Hopfield / Mean-field / Softmax reference), ported
//! from `src/hyperscalees/models/snn_attention.py`.
//!
//! Architecture: a shared Q/K/V front end projects the per-timestep token
//! spikes `x: (T, num_tokens, token_in_dim)` through three `Mm` layers
//! (`jax.vmap` over the time axis) and rate-encodes each into bounded
//! `(0,1)` rate vectors `(num_tokens, d_head)`. An attention route then
//! combines the rates into normalized weights `p` and a value readout
//! `o = p[:,None] * v`; the readout head pools `o` over tokens
//! (`mean(o, axis=0)`), projects through `out`, and scales by `out_gain`.
//!
//! Two trainable routes are implemented (`Route::Hopfield`, `Route::MeanField`),
//! plus a reference `softmax_attention` to measure the SNN/softmax equivalence.

use burn::tensor::activation;
use burn::tensor::{Device, Tensor};
use hyperscalees_core::B;

use crate::common::{Mm, Parameter};
use crate::snn::NoiseFn;

// ---------------------------------------------------------------------------
// softplus (for the trainable inverse of 1/sqrt(d_head))
// ---------------------------------------------------------------------------

/// Mathematically ``softplus(x) = ln(1 + exp(x))``, via the numerically
/// accurate `log1p`. Used to map the raw `beta` parameter into the (positive)
/// attention temperature.
fn softplus(x: Tensor<B, 1>) -> Tensor<B, 1> {
    x.clone().exp().log1p()
}

// ---------------------------------------------------------------------------
// Shared Q/K/V front end: temporal rate encoding
// ---------------------------------------------------------------------------

/// Temporal spike-count rate encoding -> bounded `(0,1)` rates.
///
/// `proj` is `(T, ..., d)` projection currents. The temporal mean over the
/// Poisson window (axis 0) is softsign-normalised and passed through a
/// sigmoid, mirroring `_rate_encode`:
///
/// ```text
///   mean_p   = mean(proj, axis=0)          # (num_tokens, d)
///   out      = sigmoid(gain * mean_p / (1 + |mean_p|))
/// ```
///
/// The leading (time) axis is reduced, so `proj` is `(T, num_tokens, d)` and
/// the result is `(num_tokens, d)`.
pub fn rate_encode(proj: Tensor<B, 3>, gain: f32) -> Tensor<B, 2> {
    let mean_p = proj.mean_dim(0).squeeze_dim::<2>(0);
    let softsign = mean_p.clone() / mean_p.abs().add_scalar(1.0);
    activation::sigmoid(softsign * gain)
}

/// Apply the (optionally noised) `(x_t, weight) -> out` matmul across the
/// leading (time) axis of `x: (T, n_tok, in)` -> `(T, n_tok, out)`, mirroring
/// `jax.vmap(lambda xt: MM(...))(x)`.
fn matmul_3d(
    x: Tensor<B, 3>,
    weight: &Tensor<B, 2>,
    noise: Option<&NoiseFn>,
) -> Tensor<B, 3> {
    let dims = x.dims();
    let t = dims[0];
    let n_tok = dims[1];
    let in_dim = dims[2];
    let mut parts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
    for i in 0..t {
        let x_t = x
            .clone()
            .slice([i..i + 1, 0..n_tok, 0..in_dim])
            .squeeze_dim::<2>(0);
        let out = match noise {
            Some(f) => f(x_t, weight.clone()),
            None => x_t.matmul(weight.clone().transpose()),
        };
        parts.push(out.unsqueeze::<3>());
    }
    Tensor::cat(parts, 0)
}

// ---------------------------------------------------------------------------
// Attention routes
// ---------------------------------------------------------------------------

/// Hopfield energy competition -> softmax-like attention weights.
///
/// ```text
///   q_center = mean(q, axis=0, keepdims)          # (1, d)
///   h        = (beta * (q_center @ k.T))[0]        # (n,)
///   repeat n_iter:                                 # global-inhibitory relax
///     c = mean(u)
///     u = u + (1/tau_a) * (-u + h - g_inh * c)
///   e = exp(u - max(u));  p = e / (sum(e)+1e-6)
///   o = p[:,None] * v
/// ```
///
/// Returns `(p, o)` with `p` in `(n,)` and `o` in `(n, d)`.
pub fn hopfield_attention(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    g_inh: f32,
    tau_a: f32,
    beta: Tensor<B, 1>,
    n_iter: usize,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let q_center = q.mean_dim(0); // (1, d)
    let qk = q_center.matmul(k.transpose()); // (1, n)
    let h = (beta.unsqueeze_dim::<2>(0) * qk).squeeze_dim::<1>(0); // (n,)

    let mut u = h.clone();
    for _ in 0..n_iter {
        let c = u.clone().mean(); // [1] global activity
        let inner = u.clone().neg().add(h.clone()) - c.mul_scalar(g_inh);
        u = u.clone() + inner.mul_scalar(1.0 / tau_a);
    }

    let e = (u.clone() - u.clone().max()).exp(); // stable Boltzmann readout
    let p = e.clone() / e.clone().sum().add_scalar(1e-6); // (n,)
    let o = p.clone().unsqueeze_dim::<2>(1) * v; // (n,1)*(n,d) -> (n,d)
    (p, o)
}

/// Mean-field (Wilson–Cowan) population approach to the attention weights.
///
/// ```text
///   h = (beta * (q_center @ k.T))[0]
///   r = relu(h)
///   repeat n_iter:                                 # divisive normalization
///     R      = sum(r)
///     r      = relu(h - gamma*R)
///     r      = r / max(sum(r), 1e-6)
///   r = r / (sum(r) + 1e-6)
///   e = exp(beta * q_center @ k.T)                 # (1, n)
///   numer = e[0] * r;  A = numer / (sum(numer)+1e-6)
///   o = A[:,None] * v
/// ```
///
/// Returns `(A, o)` with `A` in `(n,)` and `o` in `(n, d)`.
pub fn meanfield_attention(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    gamma: f32,
    beta: Tensor<B, 1>,
    n_iter: usize,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let q_center = q.mean_dim(0); // (1, d)
    let qk = q_center.matmul(k.transpose()); // (1, n)
    let h = (beta.clone().unsqueeze_dim::<2>(0) * qk.clone()).squeeze_dim::<1>(0); // (n,)

    let mut r = activation::relu(h.clone());
    for _ in 0..n_iter {
        let r_sum = r.clone().sum(); // [1]
        let r_new = activation::relu(h.clone() - r_sum.mul_scalar(gamma)); // (n,)
        let denom = r_new.clone().sum().clamp_min(1e-6); // [1]
        r = r_new / denom;
    }
    r = r.clone() / r.clone().sum().add_scalar(1e-6); // (n,)

    let e = (beta.unsqueeze_dim::<2>(0) * qk).exp().squeeze_dim::<1>(0); // (n,)
    let numer = e * r; // (n,)
    let a = numer.clone() / numer.clone().sum().add_scalar(1e-6); // (n,)
    let o = a.clone().unsqueeze_dim::<2>(1) * v; // (n,d)
    (a, o)
}

/// Reference softmax self-attention used as the equivalence target.
///
/// ```text
///   e = exp(beta * (q_center @ k.T))[0]
///   p = e / (sum(e)+1e-6);  o = p[:,None] * v
/// ```
pub fn softmax_attention(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    beta: Tensor<B, 1>,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let q_center = q.mean_dim(0); // (1, d)
    let qk = q_center.matmul(k.transpose()); // (1, n)
    let e = (beta.unsqueeze_dim::<2>(0) * qk).exp().squeeze_dim::<1>(0); // (n,)
    let p = e.clone() / e.clone().sum().add_scalar(1e-6); // (n,)
    let o = p.clone().unsqueeze_dim::<2>(1) * v; // (n,d)
    (p, o)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Attention route selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Graded Hopfield attractor competition (`hopfield_attention`).
    Hopfield,
    /// Wilson–Cowan population dynamics (`meanfield_attention`).
    MeanField,
}

/// An SNN attention classifier, mirroring `SNNAttentionModel` /
/// `HopfieldAttnSNN` / `MeanFieldAttnSNN` in `snn_attention.py`.
///
/// `x` is `(T, num_tokens, token_in_dim)` binary Poisson spikes; `forward`
/// returns logits `(num_classes,)` scaled by the `out_gain` parameter.
pub struct SnnAttentionModel {
    /// Q projection weight, shape `(d_head, token_in_dim)`.
    pub q: Mm,
    /// K projection weight, shape `(d_head, token_in_dim)`.
    pub k: Mm,
    /// V projection weight, shape `(d_head, token_in_dim)`.
    pub v: Mm,
    /// Readout weight, shape `(num_classes, d_head)`.
    pub out: Mm,
    /// Readout gain parameter, shape `(1,)`.
    pub out_gain: Parameter,
    /// Raw (inverse-softplus) attention temperature parameter, shape `(1,)`.
    pub beta: Parameter,
    /// Frozen membrane time constant (carried, not used by attention).
    pub tau_m: f32,
    /// Frozen projection gain for the rate encoder.
    pub proj_gain: f32,
    /// Whether beta is trained (softplus) or frozen at `1/sqrt(d_head)`.
    pub trainable_beta: bool,
    /// Hopfield global-inhibition strength.
    pub g_inh: f32,
    /// Hopfield attractor relaxation time constant.
    pub tau_a: f32,
    /// Mean-field divisive normalization strength.
    pub gamma: f32,
    /// Number of recurrent / population iterations.
    pub n_iter: usize,
    /// Attention route selector.
    pub route: Route,
}

impl SnnAttentionModel {
    /// Build an SNN attention model, mirroring `rand_init`: q/k/v `Mm`
    /// `(token_in_dim -> d_head)`, `out` `Mm` `(d_head -> num_classes)`, an
    /// `out_gain` of ones, and `beta` initialized so that
    /// `softplus(beta) == 1/sqrt(d_head)`.
    pub fn new(
        token_in_dim: usize,
        num_classes: usize,
        d_head: usize,
        route: Route,
        device: &Device<B>,
    ) -> Self {
        let q = Mm::new(token_in_dim, d_head, device);
        let k = Mm::new(token_in_dim, d_head, device);
        let v = Mm::new(token_in_dim, d_head, device);
        let out = Mm::new(d_head, num_classes, device);
        let out_gain = Parameter::new(Tensor::<B, 1>::ones([1], device));
        // raw_beta = ln(exp(1/sqrt(d_head)) - 1)
        let raw_beta = ((1.0 / (d_head as f32).sqrt()).exp() - 1.0).ln();
        let beta = Parameter::new(Tensor::<B, 1>::from_data([raw_beta], device));
        Self {
            q,
            k,
            v,
            out,
            out_gain,
            beta,
            tau_m: 20.0,
            proj_gain: 2.0,
            trainable_beta: true,
            g_inh: 0.5,
            tau_a: 5.0,
            gamma: 0.5,
            n_iter: 8,
            route,
        }
    }

    /// Project spikes and rate-encode into per-token Q/K/V rate vectors
    /// `(num_tokens, d_head)`, mirroring `_mk_qkv`.
    fn mk_qkv(
        &self,
        x: Tensor<B, 3>,
        noise: Option<&NoiseFn>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let q_proj = matmul_3d(x.clone(), &self.q.weight, noise); // (T, n, d)
        let k_proj = matmul_3d(x.clone(), &self.k.weight, noise); // (T, n, d)
        let v_proj = matmul_3d(x, &self.v.weight, noise); // (T, n, d)
        let gain = self.proj_gain;
        let q_rate = rate_encode(q_proj, gain);
        let k_rate = rate_encode(k_proj, gain);
        let v_rate = rate_encode(v_proj, gain);
        (q_rate, k_rate, v_rate)
    }

    /// Dispatch the configured route's attention.
    fn attention(
        &self,
        q: Tensor<B, 2>,
        k: Tensor<B, 2>,
        v: Tensor<B, 2>,
        beta: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 2>) {
        match self.route {
            Route::Hopfield => {
                hopfield_attention(q, k, v, self.g_inh, self.tau_a, beta, self.n_iter)
            }
            Route::MeanField => meanfield_attention(q, k, v, self.gamma, beta, self.n_iter),
        }
    }

    /// Forward pass over `(T, num_tokens, token_in_dim)` spikes ->
    /// `(num_classes,)` logits, mirroring `_forward`.
    ///
    /// `noise` is an optional `(x_t, weight) -> out` closure reproducing
    /// EggRoll's `do_mm`, applied to the q/k/v projections and the readout
    /// projection. When `None` the forward is the clean, deterministic
    /// `x @ weight.T` path.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        noise: Option<&NoiseFn>,
    ) -> Tensor<B, 1> {
        let (q_rate, k_rate, v_rate) = self.mk_qkv(x, noise); // (n, d) each

        let beta: Tensor<B, 1> = if self.trainable_beta {
            softplus(self.beta.value.clone())
        } else {
            let d_head = q_rate.dims()[1];
            Tensor::<B, 1>::from_data([1.0 / (d_head as f32).sqrt()], &q_rate.device())
        };

        let (_, o) = self.attention(q_rate, k_rate, v_rate, beta); // (n, d)

        // pooled = mean(o, axis=0) -> (d_head,)
        let pooled = o.mean_dim(0).squeeze_dim::<1>(0);
        // logits = out(pooled) -> (num_classes,)
        let logits = self.out.forward(pooled.unsqueeze::<2>()).squeeze_dim::<1>(0);
        // logits * out_gain
        logits * self.out_gain.value.clone()
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

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Deterministic 3-token / 2-head test tensors for the attention routes.
    /// `q_center = mean(q, 0) = [3, 4]`, so `q_center @ k.T = [3, 4, 7]`.
    fn test_qkv() -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let q = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device());
        let k = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device());
        let v = Tensor::<B, 2>::from_data([[1.0_f32, 10.0], [2.0, 20.0], [3.0, 30.0]], &device());
        (q, k, v)
    }

    fn beta_one() -> Tensor<B, 1> {
        Tensor::<B, 1>::from_data([1.0_f32], &device())
    }

    // -- softmax_attention ------------------------------------------------

    #[test]
    fn softmax_attention_matches_hand_computed() {
        let (q, k, v) = test_qkv();
        let (p, o) = softmax_attention(q, k, v, beta_one());

        // Hand-computed p = softmax([3, 4, 7]).
        let exp = [20.08553692_f32, 54.59815003, 1096.63315843];
        let sum = exp.iter().sum::<f32>() + 1e-6;
        let p_expected = [exp[0] / sum, exp[1] / sum, exp[2] / sum];

        let pv = to_vec(p.clone());
        assert!(pv.iter().sum::<f32>() > 0.999 && pv.iter().sum::<f32>() < 1.001);
        for (a, b) in pv.iter().zip(p_expected.iter()) {
            assert!(near(*a, *b, 1e-4), "expected {b}, got {a}");
        }

        // o = p[:,None] * v
        let ov = to_vec(o);
        let pv2 = to_vec(p);
        for i in 0..3 {
            for j in 0..2 {
                let expected = pv2[i] * [1.0_f32, 10.0][j] * [1.0_f32, 2.0, 3.0][i];
                assert!(near(ov[i * 2 + j], expected, 1e-4));
            }
        }
    }

    // -- hopfield_attention ----------------------------------------------

    #[test]
    fn hopfield_attention_matches_hand_computed() {
        let (q, k, v) = test_qkv();
        let (p, o) = hopfield_attention(q, k, v, 0.5, 5.0, beta_one(), 8);

        // After 8 relaxations u -> h minus a constant, so p ~ softmax(h).
        // Hand-computed p = softmax([3, 4, 7]).
        let exp = [20.08553692_f32, 54.59815003, 1096.63315843];
        let sum = exp.iter().sum::<f32>() + 1e-6;
        let p_expected = [exp[0] / sum, exp[1] / sum, exp[2] / sum];

        let pv = to_vec(p.clone());
        assert!(pv.iter().sum::<f32>() > 0.999 && pv.iter().sum::<f32>() < 1.001);
        for (a, b) in pv.iter().zip(p_expected.iter()) {
            assert!(near(*a, *b, 1e-3), "expected {b}, got {a}");
        }

        // o = p[:,None] * v with v = [[1,10],[2,20],[3,30]].
        let ov = to_vec(o);
        let pv2 = to_vec(p);
        for i in 0..3 {
            for j in 0..2 {
                let expected = pv2[i] * [1.0_f32, 10.0][j] * [1.0_f32, 2.0, 3.0][i];
                assert!(near(ov[i * 2 + j], expected, 1e-4));
            }
        }
    }

    #[test]
    fn hopfield_attention_concentrates_with_large_beta() {
        let (q, k, v) = test_qkv();
        let big_beta = Tensor::<B, 1>::from_data([8.0_f32], &device());
        let (p, _o) = hopfield_attention(q, k, v, 0.5, 5.0, big_beta, 8);
        let pv = to_vec(p);
        // The argmax token (index 2) should dominate.
        let argmax = (0..pv.len()).max_by(|&a, &b| pv[a].partial_cmp(&pv[b]).unwrap()).unwrap();
        assert_eq!(argmax, 2);
        assert!(pv[2] > 0.99, "expected concentration on token 2, got {pv:?}");
    }

    // -- meanfield_attention ---------------------------------------------

    #[test]
    fn meanfield_attention_matches_hand_computed() {
        let (q, k, v) = test_qkv();
        // gamma = 0.1 keeps the population from collapsing to zero for h = [3,4,7].
        let (a, o) = meanfield_attention(q, k, v, 0.1, beta_one(), 8);

        let av = to_vec(a.clone());
        assert!(av.iter().sum::<f32>() > 0.999 && av.iter().sum::<f32>() < 1.001);

        // Hand-computed fixed point: r ~ [2.9, 3.9, 6.9]/13.7 (normalized),
        // then A = softmax(h)*r / sum(softmax(h)*r).
        let exp = [20.08553692_f32, 54.59815003, 1096.63315843];
        let r = [2.9_f32 / 13.7, 3.9 / 13.7, 6.9 / 13.7];
        let numer: Vec<f32> = (0..3).map(|i| exp[i] * r[i]).collect();
        let numer_sum: f32 = numer.iter().sum();
        let a_expected: Vec<f32> = numer.iter().map(|n| n / numer_sum).collect();
        for (a, b) in av.iter().zip(a_expected.iter()) {
            assert!(near(*a, *b, 1e-3), "expected {b}, got {a}");
        }

        // o = A[:,None] * v
        let ov = to_vec(o);
        let av2 = to_vec(a);
        for i in 0..3 {
            for j in 0..2 {
                let expected = av2[i] * [1.0_f32, 10.0][j] * [1.0_f32, 2.0, 3.0][i];
                assert!(near(ov[i * 2 + j], expected, 1e-4));
            }
        }
    }

    // -- rate_encode ------------------------------------------------------

    #[test]
    fn rate_encode_is_bounded_and_matches_hand_computed() {
        // proj: (T=2, n=1, d=1), so mean over axis 0 of each entry.
        let proj = Tensor::<B, 3>::from_data([[[2.0_f32]], [[4.0]]], &device());
        let out = rate_encode(proj, 2.0_f32);
        let val = to_vec(out)[0];
        // mean_p = 3.0; sigmoid(2.0 * 3.0 / (1 + 3.0)) = sigmoid(1.5)
        let expected = 1.0 / (1.0 + (-1.5_f32).exp());
        assert!(near(val, expected, 1e-5), "expected {expected}, got {val}");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn rate_encode_monotone_in_magnitude() {
        // Larger projection magnitude -> larger rate (sigmoid is monotone).
        let small = Tensor::<B, 3>::from_data([[[1.0_f32]], [[1.0]]], &device());
        let large = Tensor::<B, 3>::from_data([[[3.0_f32]], [[3.0]]], &device());
        let s = to_vec(rate_encode(small, 1.0_f32))[0];
        let l = to_vec(rate_encode(large, 1.0_f32))[0];
        assert!(s > 0.0 && s < 1.0 && l > 0.0 && l < 1.0);
        assert!(l > s, "rate must be monotone in |proj|");

        // Negative projections also fall inside (0,1).
        let neg = Tensor::<B, 3>::from_data([[[-5.0_f32]], [[5.0]]], &device());
        let nv = to_vec(rate_encode(neg, 1.0_f32));
        assert!(nv.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    // -- SnnAttentionModel::forward --------------------------------------

    #[test]
    fn forward_clean_deterministic_and_shaped() {
        let model = SnnAttentionModel::new(4, 3, 5, Route::Hopfield, &device());
        // (T=4, n_tok=3, token_in_dim=4)
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
             [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
             [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
             [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]],
            &device(),
        );
        let out1 = model.forward(x.clone(), None);
        let out2 = model.forward(x, None);
        assert_eq!(out1.dims(), [3]); // (num_classes,)
        let v1 = to_vec(out1);
        let v2 = to_vec(out2);
        assert_eq!(v1, v2); // reproducible, no hidden RNG
        assert!(v1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_trainable_and_frozen_beta_both_work() {
        let mut hop = SnnAttentionModel::new(4, 3, 5, Route::Hopfield, &device());
        let mut mean = SnnAttentionModel::new(4, 3, 5, Route::MeanField, &device());
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
             [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]],
            &device(),
        );

        // Trainable (default) path.
        let trainable_h = to_vec(hop.forward(x.clone(), None));
        let trainable_m = to_vec(mean.forward(x.clone(), None));
        assert_eq!(trainable_h.len(), 3);
        assert_eq!(trainable_m.len(), 3);

        // Frozen beta path.
        hop.trainable_beta = false;
        mean.trainable_beta = false;
        let frozen_h = to_vec(hop.forward(x.clone(), None));
        let frozen_m = to_vec(mean.forward(x, None));
        assert!(frozen_h.iter().all(|v| v.is_finite()));
        assert!(frozen_m.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_out_gain_multiplies_logits() {
        let mut model = SnnAttentionModel::new(4, 3, 5, Route::Hopfield, &device());
        let x = Tensor::<B, 3>::from_data(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
             [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]],
            &device(),
        );

        // Same-model approach: all weights/beta are fixed between the two runs,
        // only `out_gain` changes, so logits must scale by exactly the gain.
        let base = to_vec(model.forward(x.clone(), None));
        model.out_gain.value = Tensor::<B, 1>::from_data([2.0_f32], &device());
        let doubled = to_vec(model.forward(x, None));
        assert_eq!(base.len(), doubled.len());
        for (a, b) in base.iter().zip(doubled.iter()) {
            assert!(near(*b, 2.0 * a, 1e-4), "expected 2*{a}, got {b}");
        }
    }
}
