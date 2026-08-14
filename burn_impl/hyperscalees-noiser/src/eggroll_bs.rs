//! EggRollBS: Eggs with baseline subtraction + group masking + a trust-region
//! clipped solver, ported from
//! `src/hyperscalees/noiser/eggroll_baseline_subtraction.py`.
//!
//! Differences from [`crate::eggroll::EggRoll`]:
//!
//! * **Masked noise** (`get_nonlora_update_params` / `get_lora_update_params`):
//!   a scalar `mask = (thread_id % G >= 2)` (G = `group_size`) zeroes the
//!   noise for direction indices `0` and `1` (threads whose `tid` is 0 or 1).
//!   Non-masked threads get the usual `sign * base_sigma * eps`.
//! * **Baseline-subtraction `convert_fitnesses`**: reshapes raw scores to
//!   `(Q, G)`, subtracts the first column (the "baseline"), divides by the
//!   per-row std (+ `1e-8`), zeroes columns 0 and 1, and returns the mean over
//!   rows (length `G`).
//! * **Trust-region solver**: `init_noiser` wires the solver as
//!   `optax.chain(clip_by_global_norm(trust_region_norm), base_solver)`, i.e.
//!   the combined gradient vector is clipped by its global L2 norm to
//!   `trust_region_norm` before applying the base Sgd/Adam step. This is
//!   implemented by the [`Solver::TrustRegion`] variant.

use burn::tensor::{Device, Int, Tensor, TensorData};
use hyperscalees_core::B;

use crate::eggroll::epoch_thread_sign;
use crate::noiser::{
    noise_seed, DeterministicNoise, FrozenNoiserParams, IterInfo, Noiser, NoiserParams, Solver,
};

/// The EggRollBS noiser. A zero-sized marker implementing [`Noiser`].
#[derive(Clone, Copy, Debug, Default)]
pub struct EggRollBS;

/// Build the frozen + mutable noiser parameters, mirroring
/// `EggRollBS.init_noiser`. Wraps the base `solver` in a
/// [`Solver::TrustRegion`] clip (optax `clip_by_global_norm`). `params` is
/// used only to size the optimizer state.
pub fn init_noiser(
    params: &[Tensor<B, 2>],
    sigma: f32,
    // Kept for API parity with `EggRollBS.init_noiser`; the learning rate
    // already lives inside `solver`.
    _lr: f32,
    group_size: i32,
    freeze_nonlora: bool,
    noise_reuse: i32,
    rank: usize,
    trust_region_norm: f32,
    solver: Solver,
    device: &Device<B>,
) -> (FrozenNoiserParams, NoiserParams) {
    let solver = Solver::TrustRegion {
        max_norm: trust_region_norm,
        inner: Box::new(solver),
    };
    let frozen = FrozenNoiserParams {
        group_size,
        freeze_nonlora,
        noise_reuse,
        rank,
        solver,
    };
    let opt_state = frozen.solver.init_state(params, device);
    let noiser = NoiserParams { sigma, opt_state };
    (frozen, noiser)
}

// ---------------------------------------------------------------------------
// Masked noise helpers
// ---------------------------------------------------------------------------

/// `mask = (thread_id % G >= 2)` as a float (1.0 keeps noise, 0.0 zeroes it).
/// Direction indices 0 and 1 (tid 0/1) are always silenced.
fn mask(info: &IterInfo, group_size: i32) -> f32 {
    if group_size > 0 && info.thread_id % group_size >= 2 {
        1.0
    } else {
        0.0
    }
}

/// Masked dense (`nonlora`) noise: `eps * sigma * mask`, where
/// `sigma = base_sigma * sign` and `sign = +1/-1` by `thread_id % 2`.
pub fn get_nonlora_update_params(
    base_sigma: f32,
    key_seed: u64,
    info: &IterInfo,
    shape: [usize; 2],
    noise_reuse: i32,
    group_size: i32,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let (true_epoch, true_thread, sign) = epoch_thread_sign(info, noise_reuse);
    let mut rng = DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));
    let eps = rng.normal_tensor(shape, device);
    let m = mask(info, group_size);
    eps.mul_scalar(sign * base_sigma * m)
}

/// Masked LoRA noise: returns `(A, B)` with both `A` and `B` multiplied by the
/// mask, and `A` additionally scaled by `sign * base_sigma`. `A` has shape
/// `(a, r)`, `B` has shape `(b, r)`.
pub fn get_lora_update_params(
    base_sigma: f32,
    key_seed: u64,
    rank: usize,
    info: &IterInfo,
    a: usize,
    b: usize,
    noise_reuse: i32,
    group_size: i32,
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let (true_epoch, true_thread, sign) = epoch_thread_sign(info, noise_reuse);
    let mut rng = DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));
    let lora = rng.normal_tensor([a + b, rank], device);
    let b_raw = lora.clone().slice([0..b, 0..rank]); // b x r
    let a_raw = lora.slice([b..a + b, 0..rank]); // a x r
    let m = mask(info, group_size);
    (a_raw.mul_scalar(sign * base_sigma * m), b_raw.mul_scalar(m))
}

// ---------------------------------------------------------------------------
// Per-parameter update functions (same shape as EggRoll)
// ---------------------------------------------------------------------------

/// `_simple_full_update`: `1/N * sum_i f_i * masked_noise_i` or zeros when
/// frozen.
fn simple_full_update(
    sigma: f32,
    key: u64,
    shape: [usize; 2],
    scores: &[f32],
    iterinfos: &[IterInfo],
    frozen: &FrozenNoiserParams,
    device: &Device<B>,
) -> Tensor<B, 2> {
    if frozen.freeze_nonlora {
        return Tensor::<B, 2>::zeros(shape, device);
    }
    let n = scores.len() as f32;
    let mut acc = Tensor::<B, 2>::zeros(shape, device);
    for (i, info) in iterinfos.iter().enumerate() {
        let up = get_nonlora_update_params(
            sigma,
            key,
            info,
            shape,
            frozen.noise_reuse,
            frozen.group_size,
            device,
        );
        acc = acc + up.mul_scalar(scores[i]);
    }
    acc.mul_scalar(1.0 / n)
}

/// `_simple_lora_update`: `1/N * sum_i f_i * (A_i @ B_i^T)` with the masked
/// `(A, B)`.
fn simple_lora_update(
    sigma: f32,
    key: u64,
    shape: [usize; 2],
    scores: &[f32],
    iterinfos: &[IterInfo],
    frozen: &FrozenNoiserParams,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let n = scores.len() as f32;
    let rank = frozen.rank;
    let [a, b] = shape;
    let base_sigma = sigma / (rank as f32).sqrt();
    let mut acc = Tensor::<B, 2>::zeros([a, b], device);
    for (i, info) in iterinfos.iter().enumerate() {
        let (a_t, b_t) = get_lora_update_params(
            base_sigma,
            key,
            rank,
            info,
            a,
            b,
            frozen.noise_reuse,
            frozen.group_size,
            device,
        );
        acc = acc + a_t.matmul(b_t.transpose()).mul_scalar(scores[i]);
    }
    acc.mul_scalar(1.0 / n)
}

/// `_noop_update`: zeros of the parameter's shape.
fn noop_update(shape: [usize; 2], device: &Device<B>) -> Tensor<B, 2> {
    Tensor::<B, 2>::zeros(shape, device)
}

/// `_do_update`: choose the update fn by `map_classification` and return the
/// *negated* gradient scaled by `sqrt(N)`. Lookup
/// `[_simple_full_update, _simple_lora_update, _noop_update, _noop_update]`.
fn do_update(
    param: &Tensor<B, 2>,
    base_key: u64,
    fitnesses: &Tensor<B, 1>,
    iterinfos: &[IterInfo],
    map_class: i32,
    sigma: f32,
    frozen: &FrozenNoiserParams,
) -> Tensor<B, 2> {
    let device = param.device();
    let shape = param.dims();
    let scores = fitnesses.clone().into_data().into_vec::<f32>().unwrap();
    let g = match map_class {
        0 => simple_full_update(sigma, base_key, shape, &scores, iterinfos, frozen, &device),
        1 => simple_lora_update(sigma, base_key, shape, &scores, iterinfos, frozen, &device),
        _ => noop_update(shape, &device),
    };
    let n = scores.len() as f32;
    g.mul_scalar(n.sqrt()).neg()
}

/// `convert_fitnesses` (EggRollBS): baseline subtraction over `(Q, G)` rows.
///
/// * `S = raw.reshape(Q, G)`, `Q = size / G`
/// * `b = S[:, :1]` (baseline = first column per row)
/// * `Z = (S - b) / (std(S, axis=1, keepdims) + 1e-8)`
/// * `Z[:, 0] = 0`, `Z[:, 1] = 0`
/// * `return mean(Z, axis=0)` (length `G`)
pub fn convert_fitnesses(group_size: i32, raw: Tensor<B, 1>) -> Tensor<B, 1> {
    let g = group_size as usize;
    let n = raw.dims()[0];
    let q = n / g;
    let device = raw.device();
    let s = raw.reshape([q, g]); // (Q, G)
    let b = s.clone().slice([0..q, 0..1]); // (Q, 1) baseline = first column
    // Per-row std (keepdims): sqrt(var over axis=1).
    let row_mean = s.clone().mean_dim(1); // (Q, 1)
    let row_var = s.clone().powf_scalar(2.0).mean_dim(1) - row_mean.clone().powf_scalar(2.0);
    let row_std = row_var.sqrt(); // (Q, 1)
    let z = (s - b) / row_std.add_scalar(1e-8); // (Q, G)
    // Zero out the first two columns (directions 0 and 1) via a broadcast
    // mask: (Q, G) * (1, G).
    let mask_vals: Vec<f32> = std::iter::once(0.0)
        .chain(std::iter::once(0.0))
        .chain(std::iter::repeat(1.0))
        .take(g)
        .collect();
    let mask_t =
        Tensor::<B, 2>::from_data(TensorData::new(mask_vals, vec![1, g]), &device);
    let z = z * mask_t;
    // per_dir_fitness = mean(Z, axis=0) -> (1, G) -> (G,).
    z.mean_dim(0).reshape([g])
}

impl Noiser for EggRollBS {
    fn do_mm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let base = x.clone().matmul(param.clone().transpose());
        match iterinfo {
            None => base,
            Some(info) => {
                let [a, b] = param.dims();
                let (a_t, b_t) = get_lora_update_params(
                    noiser.sigma / (frozen.rank as f32).sqrt(),
                    base_key,
                    frozen.rank,
                    info,
                    a,
                    b,
                    frozen.noise_reuse,
                    frozen.group_size,
                    &param.device(),
                );
                // base + x @ B @ A.T
                base + x.matmul(b_t).matmul(a_t.transpose())
            }
        }
    }

    #[allow(non_snake_case)]
    fn do_Tmm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let base = x.clone().matmul(param.clone());
        match iterinfo {
            None => base,
            Some(info) => {
                let [a, b] = param.dims();
                let (a_t, b_t) = get_lora_update_params(
                    noiser.sigma / (frozen.rank as f32).sqrt(),
                    base_key,
                    frozen.rank,
                    info,
                    a,
                    b,
                    frozen.noise_reuse,
                    frozen.group_size,
                    &param.device(),
                );
                // base + x @ A @ B.T
                base + x.matmul(a_t).matmul(b_t.transpose())
            }
        }
    }

    fn do_emb(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        _param: &Tensor<B, 2>,
        _base_key: u64,
        _iterinfo: Option<&IterInfo>,
        _indices: Tensor<B, 1, Int>,
    ) -> Tensor<B, 2> {
        // Python raises NotImplementedError("Embedding is not implemented").
        unimplemented!("EggRollBS embedding is not implemented")
    }

    fn get_noisy_standard(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
    ) -> Tensor<B, 2> {
        match iterinfo {
            None => param.clone(),
            Some(_) if frozen.freeze_nonlora => param.clone(),
            Some(info) => {
                let shape = param.dims();
                let noise = get_nonlora_update_params(
                    noiser.sigma,
                    base_key,
                    info,
                    shape,
                    frozen.noise_reuse,
                    frozen.group_size,
                    &param.device(),
                );
                param.clone() + noise
            }
        }
    }

    fn convert_fitnesses(
        &self,
        frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        raw: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        convert_fitnesses(frozen.group_size, raw)
    }

    fn do_updates(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &mut NoiserParams,
        params: &[Tensor<B, 2>],
        base_keys: &[u64],
        fitnesses: Tensor<B, 1>,
        iterinfos: &[IterInfo],
        es_classes: &[i32],
    ) -> Vec<Tensor<B, 2>> {
        if params.is_empty() {
            return Vec::new();
        }
        let grads: Vec<Tensor<B, 2>> = params
            .iter()
            .zip(base_keys.iter())
            .zip(es_classes.iter())
            .map(|((p, k), c)| do_update(p, *k, &fitnesses, iterinfos, *c, noiser.sigma, frozen))
            .collect();
        frozen.solver.update(params, &grads, &mut noiser.opt_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noiser::clip_grads_global_norm;

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    }

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn frozen_with(
        group_size: i32,
        freeze_nonlora: bool,
        noise_reuse: i32,
        rank: usize,
        solver: Solver,
    ) -> FrozenNoiserParams {
        FrozenNoiserParams {
            group_size,
            freeze_nonlora,
            noise_reuse,
            rank,
            solver,
        }
    }

    fn noiser_with(sigma: f32, frozen: &FrozenNoiserParams, params: &[Tensor<B, 2>]) -> NoiserParams {
        NoiserParams {
            sigma,
            opt_state: frozen.solver.init_state(params, &device()),
        }
    }

    // -- masked nonlora noise -------------------------------------------

    #[test]
    fn nonlora_mask_zeroes_dirs_0_and_1() {
        // G = 4: tid 0 and 1 masked (-> all zero), tid 2 kept; tids 6/7 kept
        // with the same true_thread_idx but opposite signs.
        let g = 4i32;
        let sigma = 0.5_f32;
        let key = 9u64;
        let shape = [2, 2];
        // tid 0: mask 0 -> all zeros.
        let z0 = get_nonlora_update_params(sigma, key, &IterInfo { epoch: 0, thread_id: 0 }, shape, 0, g, &device());
        assert!(to_vec(z0).iter().all(|&v| v == 0.0));
        // tid 1: mask 0 -> all zeros.
        let z1 = get_nonlora_update_params(sigma, key, &IterInfo { epoch: 0, thread_id: 1 }, shape, 0, g, &device());
        assert!(to_vec(z1).iter().all(|&v| v == 0.0));
        // tid 2: mask = 2%4 = 2 >= 2 -> kept; generally non-zero.
        let k2 = get_nonlora_update_params(sigma, key, &IterInfo { epoch: 0, thread_id: 2 }, shape, 0, g, &device());
        assert!(to_vec(k2.clone()).iter().any(|&v| v != 0.0));
        // tid 6 (6%4=2 kept, tt=3, sign+), tid 7 (7%4=3 kept, tt=3, sign-):
        // same eps, opposite signs, both kept.
        let k6a = get_nonlora_update_params(sigma, key, &IterInfo { epoch: 0, thread_id: 6 }, shape, 0, g, &device());
        let k6b = get_nonlora_update_params(sigma, key, &IterInfo { epoch: 0, thread_id: 6 }, shape, 0, g, &device());
        let k7 = get_nonlora_update_params(sigma, key, &IterInfo { epoch: 0, thread_id: 7 }, shape, 0, g, &device());
        assert!(to_vec(k6a.clone()).iter().any(|&v| v != 0.0));
        assert_eq!(to_vec(k6a.clone()), to_vec(k6b)); // deterministic
        let v6 = to_vec(k6a);
        let v7 = to_vec(k7);
        assert!(v6.iter().zip(v7.iter()).all(|(a, b)| near(*a, -*b, 1e-5)));
    }

    // -- masked lora noise ----------------------------------------------

    #[test]
    fn lora_mask_zeroes_dirs_0_and_1() {
        let g = 3i32;
        let key = 4u64;
        let a = 2usize;
        let b = 2usize;
        let r = 2usize;
        // tid 0 masked -> A and B both all zeros.
        let (a0, b0) = get_lora_update_params(0.5, key, r, &IterInfo { epoch: 0, thread_id: 0 }, a, b, 0, g, &device());
        assert!(to_vec(a0).iter().all(|&v| v == 0.0));
        assert!(to_vec(b0).iter().all(|&v| v == 0.0));
        // tid 2 kept -> non-zero in general.
        let (a2, b2) = get_lora_update_params(0.5, key, r, &IterInfo { epoch: 0, thread_id: 2 }, a, b, 0, g, &device());
        assert_eq!(a2.dims(), [a, r]);
        assert_eq!(b2.dims(), [b, r]);
        assert!(to_vec(a2.clone()).iter().any(|&v| v != 0.0));
        assert!(to_vec(b2.clone()).iter().any(|&v| v != 0.0));
    }

    // -- convert_fitnesses baseline subtraction ---------------------------

    #[test]
    fn convert_fitnesses_baseline_subtracts_and_zeroes_first_cols() {
        // Hand-built matrix: S = [[1,2,5],[3,4,7]] (Q=2, G=3).
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 5.0, 3.0, 4.0, 7.0], &device());
        let g = 3i32;
        let out: Vec<f32> = to_vec(convert_fitnesses(g, raw));
        assert_eq!(out.len(), 3);
        // First two directions are zeroed.
        assert!(near(out[0], 0.0, 1e-6));
        assert!(near(out[1], 0.0, 1e-6));
        // Third: mean over rows of (S[:,2] - baseline)/std + zeroed cols.
        // row0 baseline=1, row1 baseline=3.
        // Recompute with the same per-row std formula (+1e-8).
        let row0 = [1.0_f32, 2.0, 5.0];
        let row1 = [3.0_f32, 4.0, 7.0];
        let std0 = (row0.iter().map(|x| x * x).sum::<f32>() / 3.0
            - (row0.iter().sum::<f32>() / 3.0).powi(2))
            .sqrt();
        let std1 = (row1.iter().map(|x| x * x).sum::<f32>() / 3.0
            - (row1.iter().sum::<f32>() / 3.0).powi(2))
            .sqrt();
        let z02 = (5.0 - 1.0) / (std0 + 1e-8);
        let z12 = (7.0 - 3.0) / (std1 + 1e-8);
        let expected = (z02 + z12) / 2.0;
        assert!(near(out[2], expected, 1e-4), "got {} exp {}", out[2], expected);
    }

    // -- trust-region clip ----------------------------------------------

    #[test]
    fn clip_grads_global_norm_clips_to_max_norm_and_noops_small() {
        // Big grad, norm 5 -> clipped to 1.
        let g = Tensor::<B, 2>::from_data([[3.0_f32, 4.0]], &device());
        let out = clip_grads_global_norm(&[g.clone()], 1.0);
        let n: f32 = out[0].clone().powf_scalar(2.0).sum().into_scalar().sqrt();
        assert!(near(n, 1.0, 1e-5), "norm {n}");
        // Small grad (norm 0.5) -> no-op (returns identical).
        let small = Tensor::<B, 2>::from_data([[0.5_f32, 0.0]], &device());
        let out2 = clip_grads_global_norm(&[small.clone()], 1.0);
        assert_eq!(to_vec(out2[0].clone()), to_vec(small));
    }

    #[test]
    fn trust_region_solver_clips_global_grad_norm() {
        // A single huge negated-ES-grad; after the TrustRegion SGD step the
        // recovered clipped gradient has global L2 norm == trust_region_norm.
        let max_norm = 1.0_f32;
        let lr = 0.1_f32;
        let solver = Solver::TrustRegion {
            max_norm,
            inner: Box::new(Solver::sgd(lr)),
        };
        let p = Tensor::<B, 2>::from_data([[0.0_f32, 0.0]], &device());
        let g = Tensor::<B, 2>::from_data([[100.0_f32, 0.0]], &device()); // norm 100
        let mut state = solver.init_state(&[p.clone()], &device());
        let new = solver.update(&[p.clone()], &[g.clone()], &mut state);
        // new = p - lr * clipped_grad  =>  clipped_grad = (p - new)/lr.
        let clipped = (p - new[0].clone()).mul_scalar(1.0 / lr);
        let norm: f32 = clipped.clone().powf_scalar(2.0).sum().into_scalar().sqrt();
        assert!(near(norm, max_norm, 1e-4), "norm {norm}");
        // The clipped gradient must point the same direction as g (positive x).
        let cv = to_vec(clipped);
        assert!(cv[0] > 0.0 && near(cv[1], 0.0, 1e-5));
    }

    // -- do_updates runs ------------------------------------------------

    #[test]
    fn do_updates_runs_and_changes_params() {
        let frozen = frozen_with(3, false, 0, 2, Solver::TrustRegion {
            max_norm: 1.0,
            inner: Box::new(Solver::sgd(0.1)),
        });
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let mut noiser = noiser_with(0.5, &frozen, &[p.clone()]);
        let fitness = Tensor::<B, 1>::from_data([0.5_f32, 0.5, 0.5], &device());
        let infos = [
            IterInfo { epoch: 0, thread_id: 0 },
            IterInfo { epoch: 0, thread_id: 1 },
            IterInfo { epoch: 0, thread_id: 2 },
        ];
        let updated = EggRollBS.do_updates(&frozen, &mut noiser, &[p.clone()], &[7u64], fitness, &infos, &[1]);
        let uv = to_vec(updated[0].clone());
        assert!(uv.iter().all(|v| v.is_finite()));
        // Masked dirs 0/1 zero the noise for those envs, but env 2 (tid 2)
        // contributes LoRA noise, so the params should move somewhere.
        let pv = to_vec(p);
        assert!(uv.iter().zip(pv.iter()).any(|(u, q)| !near(*u, *q, 1e-6)), "no change {uv:?}");
    }

    // -- init_noiser ------------------------------------------------------

    #[test]
    fn init_noiser_wraps_in_trust_region() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let (frozen, noiser) = init_noiser(&[p], 0.4, 0.01, 3, false, 0, 2, 0.9, Solver::adam(0.01), &device());
        assert!(matches!(frozen.solver, Solver::TrustRegion { max_norm: 0.9, .. }));
        assert_eq!(frozen.group_size, 3);
        assert_eq!(frozen.rank, 2);
        assert_eq!(noiser.sigma, 0.4);
        assert_eq!(noiser.opt_state.moments.len(), 1);
    }

    // -- determinism ------------------------------------------------------

    #[test]
    fn same_seed_gives_same_noise() {
        let frozen = frozen_with(3, false, 0, 2, Solver::TrustRegion {
            max_norm: 1.0,
            inner: Box::new(Solver::sgd(0.1)),
        });
        let noiser = noiser_with(0.7, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let info = IterInfo { epoch: 1, thread_id: 2 };
        let a = EggRollBS.get_noisy_standard(&frozen, &noiser, &p, 42, Some(&info));
        let b = EggRollBS.get_noisy_standard(&frozen, &noiser, &p, 42, Some(&info));
        assert_eq!(to_vec(a), to_vec(b));
    }
}
