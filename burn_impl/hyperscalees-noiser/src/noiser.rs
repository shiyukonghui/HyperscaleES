//! Noiser abstraction and the identity `BaseNoiser`, ported from
//! `src/hyperscalees/noiser/base_noiser.py`.
//!
//! Because burn tensors are statically ranked with concrete shapes (unlike
//! JAX pytrees), the `Noiser` trait operates on concrete [`Tensor`] values
//! rather than generic pytree parameters. For this crate the ES update target
//! is represented as a slice of rank-2 tensors (`Vec<Tensor<B, 2>>`), matching
//! `Mm.weight` / `Tmm.weight`. Rank-1 `Parameter` values can be handled by a
//! later task / different noiser on the same plumbing.
//!
//! This module also owns the *shared* optimizer plumbing ([`Solver`] /
//! [`OptimizerState`]) so that downstream noisers (OpenES, Sparse, AltEggRoll,
//! EggRollBS, SNN) reuse the exact same update semantics.

use burn::tensor::{Device, Int, Tensor, TensorData};
use hyperscalees_core::B;

// ---------------------------------------------------------------------------
// Shared ES types
// ---------------------------------------------------------------------------

/// A single ES iteration: the sweep index (`epoch`) and the environment /
/// worker index (`thread_id`). Mirrors the Python `iterinfo = (epoch, thread_id)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterInfo {
    pub epoch: i32,
    pub thread_id: i32,
}

/// Optimizer configuration, mirroring the optax `solver` selected in
/// `EggRoll.init_noiser`. The semantics below reproduce optax exactly:
///
/// * `Sgd`    : `updates = -lr * grad`
/// * `Adam`   : `updates = -lr * m_hat / (sqrt(v_hat) + eps)`
/// * `AdamW`  : `updates = -lr * m_hat / (sqrt(v_hat) + eps) + wd * param`
///
/// where `m_hat` / `v_hat` are the bias-corrected first/second moments.
///
/// Note that the `grad` fed to [`Solver::update`] is the *already-negated*
/// value returned by EggRoll's `_do_update` (`-g * sqrt(N)`), so the net SGD
/// step is `param_new = param + lr * g * sqrt(N)`.
#[derive(Clone, Debug)]
pub enum Solver {
    /// Plain stochastic gradient descent (no momentum).
    Sgd { lr: f32 },
    /// optax Adam.
    Adam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
    },
    /// optax AdamW (decoupled weight decay).
    AdamW {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    },
    /// optax `chain(clip_by_global_norm(max_norm), inner)` as used by
    /// `EggRollBS.init_noiser`: the combined gradient vector (over the whole
    /// flat-params slice) is clipped by its global L2 norm to `max_norm`
    /// *before* the inner [`Solver`] step. Mirrors optax `clip_by_global_norm`:
    /// if `global_norm > max_norm` (and `global_norm != 0`) every per-param
    /// gradient is scaled by `max_norm / global_norm`; otherwise a no-op.
    TrustRegion {
        /// The global-clip bound (`trust_region_norm`).
        max_norm: f32,
        /// The solver applied after the clip (sgd / adam / adamw / ...).
        inner: Box<Solver>,
    },
}

/// Per-parameter optimizer moments (only used by `Adam` / `AdamW`).
#[derive(Clone, Debug)]
pub struct MomentState {
    /// First moment estimate `m`.
    pub m: Tensor<B, 2>,
    /// Second moment estimate `v`.
    pub v: Tensor<B, 2>,
}

/// Mutable optimizer state, parallel to the parameter list. `step` is shared
/// across all parameters (matching optax's single global count).
#[derive(Clone, Debug)]
pub struct OptimizerState {
    /// Global optimization step counter (`t`).
    pub step: i32,
    /// One [`MomentState`] per parameter; empty for `Sgd`.
    pub moments: Vec<MomentState>,
}

impl Solver {
    /// Construct a plain SGD solver.
    pub fn sgd(lr: f32) -> Self {
        Solver::Sgd { lr }
    }

    /// Construct an Adam solver with optax defaults.
    pub fn adam(lr: f32) -> Self {
        Solver::Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Construct an AdamW solver with optax defaults.
    pub fn adamw(lr: f32) -> Self {
        Solver::AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        }
    }

    /// Initialise the optimizer state for the given parameter shapes.
    ///
    /// For `Sgd` no per-parameter state is needed; `Adam` / `AdamW` initialise
    /// `m = v = 0` for every parameter.
    pub fn init_state(&self, params: &[Tensor<B, 2>], device: &Device<B>) -> OptimizerState {
        match self {
            Solver::Sgd { .. } => OptimizerState {
                step: 0,
                moments: Vec::new(),
            },
            Solver::Adam { .. } | Solver::AdamW { .. } => {
                let moments = params
                    .iter()
                    .map(|p| {
                        let shape = p.dims();
                        MomentState {
                            m: Tensor::<B, 2>::zeros(shape, device),
                            v: Tensor::<B, 2>::zeros(shape, device),
                        }
                    })
                    .collect();
                OptimizerState {
                    step: 0,
                    moments,
                }
            }
            // The clip wrapper shares the inner solver's state layout.
            Solver::TrustRegion { inner, .. } => inner.init_state(params, device),
        }
    }

    /// Apply one optimizer step to all parameters, mutating `state` in place
    /// and returning the updated parameter tensors (input params are left
    /// untouched). The `grads` are the (already negated) ES gradients.
    pub fn update(
        &self,
        params: &[Tensor<B, 2>],
        grads: &[Tensor<B, 2>],
        state: &mut OptimizerState,
    ) -> Vec<Tensor<B, 2>> {
        match self {
            Solver::Sgd { lr } => params
                .iter()
                .zip(grads)
                .map(|(p, g)| p.clone() - g.clone().mul_scalar(*lr))
                .collect(),
            Solver::Adam {
                lr,
                beta1,
                beta2,
                eps,
            }
            | Solver::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                ..
            } => {
                state.step += 1;
                let t = state.step as f32;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);
                let wd = match self {
                    Solver::AdamW { weight_decay, .. } => *weight_decay,
                    _ => 0.0,
                };
                params
                    .iter()
                    .zip(grads)
                    .enumerate()
                    .map(|(i, (p, g))| {
                        // mu = b1*mu + (1-b1)*g ; nu = b2*nu + (1-b2)*g^2
                        let mom = &mut state.moments[i];
                        let m_new = mom.m.clone().mul_scalar(*beta1) + g.clone().mul_scalar(1.0 - beta1);
                        let v_new = mom
                            .v
                            .clone()
                            .mul_scalar(*beta2)
                            + g.clone().powf_scalar(2.0).mul_scalar(1.0 - beta2);
                        mom.m = m_new;
                        mom.v = v_new;
                        // Bias-corrected moments.
                        let m_hat = mom.m.clone().mul_scalar(1.0 / bc1);
                        let v_hat = mom.v.clone().mul_scalar(1.0 / bc2);
                        let denom = v_hat.sqrt().add_scalar(*eps);
                        let adam_term = (m_hat / denom).mul_scalar(-*lr);
                        if wd == 0.0 {
                            p.clone() + adam_term
                        } else {
                            p.clone() + adam_term + p.clone().mul_scalar(wd)
                        }
                    })
                    .collect()
            }
            // optax `chain(clip_by_global_norm(max_norm), inner)`: clip the
            // combined gradient vector by its global L2 norm, then let the
            // inner solver update.
            Solver::TrustRegion { max_norm, inner } => {
                let clipped = clip_grads_global_norm(grads, *max_norm);
                inner.update(params, &clipped, state)
            }
        }
    }
}

/// optax `clip_by_global_norm(max_norm)` over a flat-params gradient slice.
///
/// Computes `global_norm = sqrt(sum of squared norms of every per-param
/// grad)`; if `global_norm > max_norm` (and `global_norm != 0`), every
/// per-param grad is scaled by `max_norm / global_norm`; otherwise a no-op.
pub(crate) fn clip_grads_global_norm(grads: &[Tensor<B, 2>], max_norm: f32) -> Vec<Tensor<B, 2>> {
    let global_sq: f32 = grads
        .iter()
        .map(|g| g.clone().powf_scalar(2.0).sum().into_scalar())
        .sum();
    let global_norm = global_sq.sqrt();
    if global_norm > max_norm && global_norm > 0.0 {
        let scale = max_norm / global_norm;
        grads.iter().map(|g| g.clone().mul_scalar(scale)).collect()
    } else {
        grads.to_vec()
    }
}

/// Frozen (configuration) parameters of a noiser, mirroring the Python
/// `frozen_noiser_params` dict in `EggRoll.init_noiser`.
#[derive(Clone, Debug)]
pub struct FrozenNoiserParams {
    /// Fitness grouping for `convert_fitnesses`; `0` means global z-score.
    pub group_size: i32,
    /// If `true`, skip the dense (`nonlora`) perturb/update path entirely.
    pub freeze_nonlora: bool,
    /// Noise reuse factor for the epoch mapping.
    pub noise_reuse: i32,
    /// LoRA rank.
    pub rank: usize,
    /// Optimizer configuration.
    pub solver: Solver,
}

/// Mutable parameters of a noiser, mirroring the Python `noiser_params` dict.
#[derive(Clone, Debug)]
pub struct NoiserParams {
    /// ES noise scale (`sigma`).
    pub sigma: f32,
    /// Mutable optimizer state.
    pub opt_state: OptimizerState,
}

// ---------------------------------------------------------------------------
// Noiser trait
// ---------------------------------------------------------------------------

/// The shared noiser interface.
///
/// Methods operate on concrete rank-2 parameters (matrix weights), with the
/// current [`IterInfo`] passed as `Option` — a `None` value means "no noise"
/// (identity forward).
pub trait Noiser {
    /// Noisy matrix-multiply forward: `x @ param.T` (+ LoRA noise).
    fn do_mm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2>;

    /// Noisy transposed matrix-multiply forward: `x @ param`.
    #[allow(non_snake_case)]
    fn do_Tmm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2>;

    /// Noisy embedding forward: `param[indices]`.
    fn do_emb(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        indices: Tensor<B, 1, Int>,
    ) -> Tensor<B, 2>;

    /// The noisy parameter used as the "standard" (unperturbed-ish) value.
    fn get_noisy_standard(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
    ) -> Tensor<B, 2>;

    /// Convert raw scores into normalised fitnesses.
    fn convert_fitnesses(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        raw: Tensor<B, 1>,
    ) -> Tensor<B, 1>;

    /// Compute the ES gradients and apply the optimizer to every parameter.
    ///
    /// Returns the updated parameters and mutates `noiser.opt_state`. The
    /// input parameter slice is left untouched.
    fn do_updates(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &mut NoiserParams,
        params: &[Tensor<B, 2>],
        base_keys: &[u64],
        fitnesses: Tensor<B, 1>,
        iterinfos: &[IterInfo],
        es_classes: &[i32],
    ) -> Vec<Tensor<B, 2>>;
}

/// The identity noiser: forwards are noiseless and updates are no-ops.
#[derive(Clone, Copy, Debug, Default)]
pub struct BaseNoiser;

impl Noiser for BaseNoiser {
    fn do_mm(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        _base_key: u64,
        _iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        x.matmul(param.clone().transpose())
    }

    fn do_Tmm(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        _base_key: u64,
        _iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        x.matmul(param.clone())
    }

    fn do_emb(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        _base_key: u64,
        _iterinfo: Option<&IterInfo>,
        indices: Tensor<B, 1, Int>,
    ) -> Tensor<B, 2> {
        param.clone().select(0, indices)
    }

    fn get_noisy_standard(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        _base_key: u64,
        _iterinfo: Option<&IterInfo>,
    ) -> Tensor<B, 2> {
        param.clone()
    }

    fn convert_fitnesses(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        raw: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        raw
    }

    fn do_updates(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &mut NoiserParams,
        params: &[Tensor<B, 2>],
        _base_keys: &[u64],
        _fitnesses: Tensor<B, 1>,
        _iterinfos: &[IterInfo],
        _es_classes: &[i32],
    ) -> Vec<Tensor<B, 2>> {
        params.to_vec()
    }
}

// ---------------------------------------------------------------------------
// Deterministic seeded normal sampler
// ---------------------------------------------------------------------------

/// A small deterministic normal sampler used to make ES noise reproducible in
/// tests (and to give later noisers exact, seedable noise without relying on
/// burn's internal RNG, which is not seedable per-call).
///
/// `burn :: Tensor :: random` uses burn's global RNG and is not trivially
/// seedable per call, so we generate i.i.d. standard normals ourselves via
/// Box-Muller on a seeded xorshift64 and load them through
/// [`Tensor::from_data`].
#[derive(Clone, Debug)]
pub struct DeterministicNoise {
    state: u64,
}

impl DeterministicNoise {
    /// Create a sampler with the given seed. A zero seed is promoted to `1`
    /// so the xorshift state never collapses to zero.
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_unit(&mut self) -> f32 {
        // Map into [0, 1). 24 fractional bits is plenty for Box-Muller.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A single `U(0, 1)` draw, exposed for noisers that need to derive a
    /// discrete index (e.g. Sparse's uniform `idxjoint`) from the same seeded
    /// stream.
    pub(crate) fn unit(&mut self) -> f32 {
        self.next_unit()
    }

    /// A single `N(0, 1)` draw via Box-Muller.
    pub fn standard_normal(&mut self) -> f32 {
        // Keep u1 strictly positive so `ln(0)` never happens.
        let u1 = self.next_unit().max(1e-9);
        let u2 = self.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        radius * theta.cos()
    }

    /// Fill a `D`-dimensional tensor of shape `shape` with i.i.d. standard
    /// normals (row-major ordering).
    pub fn normal_tensor<const D: usize>(
        &mut self,
        shape: [usize; D],
        device: &Device<B>,
    ) -> Tensor<B, D> {
        let n = shape.iter().product::<usize>();
        let vals: Vec<f32> = (0..n).map(|_| self.standard_normal()).collect();
        Tensor::from_data(TensorData::new(vals, shape.to_vec()), device)
    }
}

/// Derive a deterministic RNG seed from a per-parameter key and the
/// `(true_epoch, true_thread_idx)` pair. This is the analogue of
/// `jax.random.fold_in(fold_in(key, true_epoch), true_thread_idx)`.
pub fn noise_seed(key: u64, true_epoch: i32, true_thread: i32) -> u64 {
    let mut h = key ^ 0x9E37_79B9_7F4A_7C15u64;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= (true_epoch as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h = h.rotate_left(17) ^ (true_thread as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 32;
    h
}

/// Shared fitness z-score conversion used by [`crate::eggroll::EggRoll`],
/// [`crate::open_es::OpenES`] and [`crate::sparse::Sparse`], mirroring the
/// Python `convert_fitnesses` in each noiser.
///
/// The denominator always uses the *global* variance of the whole `raw`
/// tensor (as the Python does: `sqrt(var(raw_scores, keepdims=True) + 1e-5)`),
/// even in the per-group branch. A `group_size == 0` computes a plain global
/// z-score; otherwise each row of `group_size` is centred by its own mean but
/// scaled by the global std.
pub fn convert_fitnesses_impl(group_size: i32, raw: Tensor<B, 1>) -> Tensor<B, 1> {
    let gs = group_size as usize;
    let n = raw.dims()[0];

    // Global mean / var of the full raw array (matches Python's use of
    // `jnp.var(raw_scores, keepdims=True)` in both branches).
    let mean = raw.clone().mean().into_scalar();
    let var = raw.clone().powf_scalar(2.0).mean().into_scalar() - mean * mean;
    let std = (var + 1e-5).sqrt();

    if gs == 0 {
        // Global z-score.
        raw.add_scalar(-mean).mul_scalar(1.0 / std)
    } else {
        // Per-group mean (keepdims), global std.
        let n_groups = n / gs;
        let groups = raw.reshape([n_groups, gs]); // (groups, gs)
        let gmean = groups.clone().mean_dim(-1); // (groups, 1)
        (groups - gmean).mul_scalar(1.0 / std).reshape([n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    }

    // -- BaseNoiser identity ----------------------------------------------

    #[test]
    fn base_noiser_do_mm_is_x_at_param_t() {
        // param (out=2, in=3) = [[1,2,3],[4,5,6]]
        let param = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &device(),
        );
        // x (batch=2, in=3)
        let x = Tensor::<B, 2>::from_data(
            [[1.0_f32, 1.0, 1.0], [2.0, 2.0, 2.0]],
            &device(),
        );
        let frozen = FrozenNoiserParams {
            group_size: 0,
            freeze_nonlora: false,
            noise_reuse: 0,
            rank: 1,
            solver: Solver::sgd(0.1),
        };
        let noiser = NoiserParams {
            sigma: 1.0,
            opt_state: solver_state(&frozen, &[param.clone()]),
        };
        let out = to_vec(BaseNoiser.do_mm(
            &frozen,
            &noiser,
            &param,
            1,
            None,
            x.clone(),
        ));
        // x @ param.T = [[6, 15],[12,30]]
        assert_eq!(out, vec![6.0, 15.0, 12.0, 30.0]);
        // With an iterinfo present it must still be identity.
        let out2 = to_vec(BaseNoiser.do_mm(
            &frozen,
            &noiser,
            &param,
            1,
            Some(&IterInfo { epoch: 0, thread_id: 0 }),
            x,
        ));
        assert_eq!(out2, vec![6.0, 15.0, 12.0, 30.0]);
    }

    #[test]
    fn base_noiser_do_tmm_is_x_at_param() {
        let param = Tensor::<B, 2>::from_data(
            [[1.0_f32, 4.0], [2.0, 5.0], [3.0, 6.0]],
            &device(),
        );
        // x (batch=1, in=3)
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 1.0, 1.0]], &device());
        let frozen = FrozenNoiserParams {
            group_size: 0,
            freeze_nonlora: false,
            noise_reuse: 0,
            rank: 1,
            solver: Solver::sgd(0.1),
        };
        let noiser = NoiserParams {
            sigma: 1.0,
            opt_state: solver_state(&frozen, &[param.clone()]),
        };
        let out = to_vec(BaseNoiser.do_Tmm(&frozen, &noiser, &param, 1, None, x));
        // x @ param = [[6, 15]]
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn base_noiser_do_emb_selects_rows() {
        let param = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            &device(),
        );
        let idx = Tensor::<B, 1, Int>::from_data([2, 0], &device());
        let frozen = FrozenNoiserParams {
            group_size: 0,
            freeze_nonlora: false,
            noise_reuse: 0,
            rank: 1,
            solver: Solver::sgd(0.1),
        };
        let noiser = NoiserParams {
            sigma: 1.0,
            opt_state: solver_state(&frozen, &[param.clone()]),
        };
        let out = to_vec(BaseNoiser.do_emb(&frozen, &noiser, &param, 1, None, idx));
        assert_eq!(out, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn base_noiser_get_noisy_standard_is_identity() {
        let param = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let frozen = FrozenNoiserParams {
            group_size: 0,
            freeze_nonlora: false,
            noise_reuse: 0,
            rank: 1,
            solver: Solver::sgd(0.1),
        };
        let noiser = NoiserParams {
            sigma: 1.0,
            opt_state: solver_state(&frozen, &[param.clone()]),
        };
        let out = BaseNoiser.get_noisy_standard(&frozen, &noiser, &param, 1, None);
        assert_eq!(to_vec(out), vec![1.0, 2.0]);
    }

    #[test]
    fn base_noiser_convert_fitnesses_is_identity_and_do_updates_noop() {
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0], &device());
        let frozen = FrozenNoiserParams {
            group_size: 0,
            freeze_nonlora: false,
            noise_reuse: 0,
            rank: 1,
            solver: Solver::sgd(0.1),
        };
        let noiser = NoiserParams {
            sigma: 1.0,
            opt_state: solver_state(&frozen, &[]),
        };
        let out = BaseNoiser.convert_fitnesses(&frozen, &noiser, raw.clone());
        assert_eq!(to_vec(out), vec![1.0, 2.0, 3.0]);

        let p = Tensor::<B, 2>::from_data([[1.0_f32, -2.0]], &device());
        let updated = BaseNoiser.do_updates(&frozen, &mut noiser.clone(), &[p.clone()], &[7u64], raw, &[IterInfo { epoch: 0, thread_id: 0 }], &[1]);
        assert_eq!(to_vec(updated[0].clone()), vec![1.0, -2.0]);
    }

    // -- DeterministicNoise ------------------------------------------------

    #[test]
    fn deterministic_noise_is_repeatable_and_varies_with_seed() {
        let mut a = DeterministicNoise::new(1234);
        let mut b = DeterministicNoise::new(1234);
        let mut c = DeterministicNoise::new(1235);
        for _ in 0..32 {
            assert_eq!(a.standard_normal(), b.standard_normal());
        }
        // Different seed -> different in general (with overwhelming
        // probability at least one element differs).
        let s1: Vec<f32> = DeterministicNoise::new(1).normal_tensor([2, 3], &device()).into_data().into_vec().unwrap();
        let s2: Vec<f32> = DeterministicNoise::new(2).normal_tensor([2, 3], &device()).into_data().into_vec().unwrap();
        let _ = c.standard_normal();
        assert!(s1 != s2);
    }

    fn solver_state(frozen: &FrozenNoiserParams, params: &[Tensor<B, 2>]) -> OptimizerState {
        frozen.solver.init_state(params, &device())
    }
}
