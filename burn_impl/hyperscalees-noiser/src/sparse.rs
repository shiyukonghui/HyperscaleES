//! Sparse: a sparse-noise ES noiser, ported from
//! `src/hyperscalees/noiser/sparse.py`.
//!
//! Instead of a dense or LoRA perturbation, `Sparse` adds noise at `k =
//! max(a, b)` random sparse coordinates. The coordinates (`idxa, idxb`) and a
//! `(k,)` normal vector are drawn deterministically from a seed; the forward
//! `do_mm`/`do_Tmm` scatter-add `x[:, idxb] * sparse_vector` into the columns
//! `idxa` of the base matmul result.
//!
//! The update path uses `_simple_sparse_update`: build a per-env `(a, b)`
//! sparse matrix `E` by scatter-adding `scores[j] * sparse_vector` at the
//! `(idxa[i], idxb[i])` positions, average over envs and scale by
//! `q = k / (a*b)`, then apply the shared [`crate::noiser::Solver`] plumbing.
//!
//! burn 0.21 has no single `index_add`, so the multi-dimensional scatter-adds
//! are done with [`Tensor::scatter_nd`] (building `E`) and [`Tensor::select_assign`]
//! with [`IndexingUpdateOp::Add`] (the 2D `do_mm` scatter-add), both of which
//! support sum-reduction.

use burn::tensor::{Device, IndexingUpdateOp, Int, Tensor, TensorData};
use hyperscalees_core::B;

use crate::eggroll::get_nonlora_update_params;
use crate::noiser::{
    convert_fitnesses_impl, noise_seed, DeterministicNoise, FrozenNoiserParams, IterInfo, Noiser,
    NoiserParams, Solver,
};

/// The Sparse noiser. A zero-sized marker implementing [`Noiser`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Sparse;

/// Build the frozen + mutable noiser parameters, mirroring
/// `Sparse.init_noiser`. `q_multiplier` is stored in `frozen` for API parity
/// with Python but is not used in the math. `params` is used only to size the
/// optimizer state.
pub fn init_noiser(
    params: &[Tensor<B, 2>],
    sigma: f32,
    // Kept for API parity; the learning rate lives inside `solver`.
    _lr: f32,
    group_size: i32,
    freeze_nonlora: bool,
    noise_reuse: i32,
    rank: usize,
    // Kept for parity with `Sparse.init_noiser(q_multiplier=...)`; unused in math.
    _q_multiplier: f32,
    solver: Solver,
    device: &Device<B>,
) -> (FrozenNoiserParams, NoiserParams) {
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

/// Derive the `(true_epoch, true_thread_idx, sign)` triple from an
/// [`IterInfo`] and the noise-reuse factor, as in `get_sparse_update_params`.
fn epoch_thread_sign(info: &IterInfo, noise_reuse: i32) -> (i32, i32, f32) {
    let true_epoch = if noise_reuse == 0 { 0 } else { info.epoch / noise_reuse };
    let true_thread = info.thread_id / 2;
    let sign = if info.thread_id % 2 == 0 { 1.0 } else { -1.0 };
    (true_epoch, true_thread, sign)
}

/// `get_sparse_update_params`: returns the scaled sparse normal vector and the
/// two coordinate index vectors.
///
/// `idxjoint = floor(uniform(k) * (a*b))`, `idxa = idxjoint // b`,
/// `idxb = idxjoint % b`, so `idxa` lies in `[0, a)` and `idxb` in `[0, b)`.
pub fn get_sparse_update_params(
    base_sigma: f32,
    key_seed: u64,
    info: &IterInfo,
    shape: [usize; 2],
    noise_reuse: i32,
) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let [a, b] = shape;
    let k = a.max(b);
    let (true_epoch, true_thread, sign) = epoch_thread_sign(info, noise_reuse);
    // `deterministic_key = fold_in(fold_in(key, true_epoch), true_thread_idx)`,
    // then split into two streams (key1 -> idx, key2 -> sparse_vector).
    let mut rng = DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));

    let ab = (a * b) as f32;
    let mut idxa = Vec::with_capacity(k);
    let mut idxb = Vec::with_capacity(k);
    for _ in 0..k {
        let idxjoint = (rng.unit() * ab).floor() as i32;
        idxa.push(idxjoint / b as i32);
        idxb.push(idxjoint % b as i32);
    }

    let sv: Vec<f32> = (0..k).map(|_| rng.standard_normal() * sign * base_sigma).collect();
    (sv, idxa, idxb)
}

/// `_simple_full_update`: `1/N * sum_i f_i * noise_i` or zeros when frozen.
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
        let up = get_nonlora_update_params(sigma, key, info, shape, frozen.noise_reuse, device);
        acc = acc + up.mul_scalar(scores[i]);
    }
    acc.mul_scalar(1.0 / n)
}

/// `_simple_sparse_update`: build per-env sparse (a, b) matrices by
/// scatter-adding `scores[j] * sparse_vector` at `(idxa[i], idxb[i])`, average
/// over envs and scale by `q = k / (a*b)`.
fn simple_sparse_update(
    sigma: f32,
    key: u64,
    shape: [usize; 2],
    scores: &[f32],
    iterinfos: &[IterInfo],
    frozen: &FrozenNoiserParams,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [a, b] = shape;
    let k = a.max(b);
    let n = scores.len() as f32;
    let q = k as f32 / (a * b) as f32;

    let mut acc = Tensor::<B, 2>::zeros([a, b], device);
    for (j, info) in iterinfos.iter().enumerate() {
        let (sv, idxa, idxb) = get_sparse_update_params(sigma, key, info, shape, frozen.noise_reuse);
        // (k, 2) index tensor: [idxa[i], idxb[i]]
        let idx_pairs: Vec<i32> = idxa
            .iter()
            .zip(idxb.iter())
            .flat_map(|(x, y)| [*x, *y])
            .collect();
        let idx_t = Tensor::<B, 2, Int>::from_data(
            TensorData::new(idx_pairs, vec![k, 2]),
            device,
        );
        // values (k,): scores[j] * sparse_vector[i]
        let vals: Vec<f32> = sv.iter().map(|v| scores[j] * v).collect();
        let vals_t = Tensor::<B, 1>::from_data(TensorData::new(vals, vec![k]), device);
        // E[a_idx, b_idx] += value ; scatter_nd on an (a,b) target with a
        // (k,2) index tensor whose last axis (size 2) indexes the two dims.
        let e = Tensor::<B, 2>::zeros([a, b], device).scatter_nd::<2, 1>(idx_t, vals_t, IndexingUpdateOp::Add);
        acc = acc + e;
    }
    acc.mul_scalar(q / n)
}

/// `_noop_update`: zeros of the parameter's shape.
fn noop_update(shape: [usize; 2], device: &Device<B>) -> Tensor<B, 2> {
    Tensor::<B, 2>::zeros(shape, device)
}

/// `_do_update`: choose the update fn by `map_classification` and return the
/// *negated* gradient scaled by `sqrt(N)`. Lookup
/// `[_simple_full_update, _simple_sparse_update, _noop_update, _noop_update]`.
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
        1 => simple_sparse_update(sigma, base_key, shape, &scores, iterinfos, frozen, &device),
        _ => noop_update(shape, &device),
    };
    let n = scores.len() as f32;
    g.mul_scalar(n.sqrt()).neg()
}

impl Noiser for Sparse {
    fn do_mm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // x: (batch, b), param: (a, b) -> base_ans (batch, a).
        let base_ans = x.clone().matmul(param.clone().transpose());
        match iterinfo {
            None => base_ans,
            Some(info) => {
                let (sv, idxa, idxb) = get_sparse_update_params(
                    noiser.sigma,
                    base_key,
                    info,
                    param.dims(),
                    frozen.noise_reuse,
                );
                let k = sv.len();
                // x_prod = x[:, idxb] * sparse_vector  (batch, k)
                let idxb_t = Tensor::<B, 1, Int>::from_data(TensorData::new(idxb, vec![k]), &param.device());
                let x_sel = x.select(1, idxb_t); // (batch, k)
                let sv_t = Tensor::<B, 1>::from_data(TensorData::new(sv, vec![k]), &param.device());
                // broadcast (1, k) -> (batch, k) so both factors are rank 2.
                let sv_2d = sv_t.unsqueeze_dim::<2>(0).expand(x_sel.dims());
                let x_prod = x_sel * sv_2d; // (batch, k)
                // base_ans[j, idxa[i]] += x_prod[j, i]  (scatter-add along dim 1).
                let idxa_t = Tensor::<B, 1, Int>::from_data(TensorData::new(idxa, vec![k]), &param.device());
                base_ans.select_assign(1, idxa_t, x_prod, IndexingUpdateOp::Add)
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
        // x: (batch, b), param: (b, a) -> base_ans (batch, a).
        let base_ans = x.clone().matmul(param.clone());
        match iterinfo {
            None => base_ans,
            Some(info) => {
                let (sv, idxa, idxb) = get_sparse_update_params(
                    noiser.sigma,
                    base_key,
                    info,
                    param.dims(),
                    frozen.noise_reuse,
                );
                let k = sv.len();
                // x_prod = x[:, idxb] * sparse_vector  (batch, k)
                let idxb_t = Tensor::<B, 1, Int>::from_data(TensorData::new(idxb, vec![k]), &param.device());
                let x_sel = x.select(1, idxb_t); // (batch, k)
                let sv_t = Tensor::<B, 1>::from_data(TensorData::new(sv, vec![k]), &param.device());
                // broadcast (1, k) -> (batch, k) so both factors are rank 2.
                let sv_2d = sv_t.unsqueeze_dim::<2>(0).expand(x_sel.dims());
                let x_prod = x_sel * sv_2d; // (batch, k)
                // base_ans[j, idxa[i]] += x_prod[j, i]  (scatter-add along dim 1).
                let idxa_t = Tensor::<B, 1, Int>::from_data(TensorData::new(idxa, vec![k]), &param.device());
                base_ans.select_assign(1, idxa_t, x_prod, IndexingUpdateOp::Add)
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
        unimplemented!("Sparse embedding is not implemented")
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
                let device = param.device();
                let shape = param.dims();
                let noise = get_nonlora_update_params(
                    noiser.sigma,
                    base_key,
                    info,
                    shape,
                    frozen.noise_reuse,
                    &device,
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
        convert_fitnesses_impl(frozen.group_size, raw)
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

    // -- get_sparse_update_params -----------------------------------------

    #[test]
    fn get_sparse_update_params_shapes_and_bounds() {
        let shape = [3, 5]; // a=3, b=5, k=5
        let info = IterInfo { epoch: 1, thread_id: 2 };
        let (sv, idxa, idxb) = get_sparse_update_params(0.5, 42, &info, shape, 0);
        // sparse_vector (k,), idxa (k,) in [0,a), idxb (k,) in [0,b).
        let k = 5usize;
        assert_eq!(sv.len(), k);
        assert_eq!(idxa.len(), k);
        assert_eq!(idxb.len(), k);
        assert!(idxa.iter().all(|&i| (0..shape[0] as i32).contains(&i)), "idxa {idxa:?}");
        assert!(idxb.iter().all(|&i| (0..shape[1] as i32).contains(&i)), "idxb {idxb:?}");
        // Relation: idxa = idxjoint // b, idxb = idxjoint % b.
        for i in 0..k {
            let idxjoint = idxa[i] * shape[1] as i32 + idxb[i];
            assert!(idxjoint >= 0 && idxjoint < (shape[0] * shape[1]) as i32);
        }
    }

    #[test]
    fn get_sparse_update_params_is_deterministic() {
        let info = IterInfo { epoch: 3, thread_id: 0 };
        let (s1, a1, b1) = get_sparse_update_params(0.7, 9, &info, [2, 4], 0);
        let (s2, a2, b2) = get_sparse_update_params(0.7, 9, &info, [2, 4], 0);
        assert_eq!(s1, s2);
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
    }

    // -- do_mm ------------------------------------------------------------

    #[test]
    fn do_mm_no_noise_is_x_at_param_t() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        let param = Tensor::<B, 2>::from_data([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device());
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 1.0, 1.0], [2.0, 2.0, 2.0]], &device());
        let out = to_vec(Sparse.do_mm(&frozen, &noiser, &param, 7, None, x));
        assert_eq!(out, vec![6.0, 15.0, 12.0, 30.0]);
    }

    #[test]
    fn do_mm_noised_matches_scatter_add_semantics() {
        // Small hand-computed example, a=2, b=2, k=2, batch=3.
        let sigma = 0.5_f32;
        let key = 5u64;
        let info = IterInfo { epoch: 0, thread_id: 0 }; // sign +1
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(sigma, &frozen, &[]);

        // param (a=2, b=2), x (batch=3, b=2).
        let param = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0]], &device());
        let x = Tensor::<B, 2>::from_data(
            [[1.0_f32, 10.0], [2.0, 20.0], [30.0, 3.0]],
            &device(),
        );
        let base = to_vec(x.clone().matmul(param.clone().transpose()));
        // base = x @ I = x itself: [1,10],[2,20],[30,3].

        // Hand-compute the sparse indices + vector with the same sampler.
        let (sv, idxa, idxb) = get_sparse_update_params(sigma, key, &info, [2, 2], 0);
        // x_prod[j, i] = x[j, idxb[i]] * sv[i].
        // base[j, idxa[i]] += x_prod[j, i].
        let mut expected = base.clone();
        for j in 0..3 {
            for i in 0..2 {
                let col = idxb[i] as usize;
                let target_col = idxa[i] as usize;
                expected[j * 2 + target_col] += x.clone().into_data().into_vec::<f32>().unwrap()[j * 2 + col] * sv[i];
            }
        }

        let got = to_vec(Sparse.do_mm(&frozen, &noiser, &param, key, Some(&info), x.clone()));
        assert!(got.iter().zip(expected.iter()).all(|(g, e)| near(*g, *e, 1e-4)), "got {got:?} exp {expected:?}");
    }

    // -- _simple_sparse_update --------------------------------------------

    #[test]
    fn simple_sparse_update_matches_hand_computed() {
        // a=2, b=3, k=3, N=2.
        let sigma = 0.4_f32;
        let key = 8u64;
        let shape = [2, 3];
        let infos = [IterInfo { epoch: 0, thread_id: 0 }, IterInfo { epoch: 0, thread_id: 1 }];
        let scores = [1.0_f32, 2.0];
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let g = simple_sparse_update(sigma, key, shape, &scores, &infos, &frozen, &device());
        let gv = to_vec(g.clone());

        // Hand-compute: q = k/(a*b) = 3/6 = 0.5. For each env build E and accumulate.
        let n = 2usize;
        let k = 3usize;
        let q = k as f32 / (shape[0] * shape[1]) as f32;
        let mut acc = [0.0_f32; 6];
        for (j, _) in infos.iter().enumerate() {
            let (sv, idxa, idxb) = get_sparse_update_params(sigma, key, &infos[j], shape, 0);
            let mut e = [0.0_f32; 6];
            for i in 0..k {
                let ai = idxa[i] as usize;
                let bi = idxb[i] as usize;
                e[ai * shape[1] + bi] += scores[j] * sv[i];
            }
            for (acc_v, e_v) in acc.iter_mut().zip(e.iter()) {
                *acc_v += e_v;
            }
        }
        let expected: Vec<f32> = acc.iter().map(|v| v * q / n as f32).collect();
        assert!(gv.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "got {gv:?} exp {expected:?}");
    }

    // -- _do_update sign --------------------------------------------------

    #[test]
    fn do_update_is_neg_grad_times_sqrt_n() {
        // N=1 sparse update: g = q * E (E built from scores=1). do_update = -g*sqrt(1).
        let sigma = 0.3_f32;
        let key = 4u64;
        let shape = [2, 2];
        let info = IterInfo { epoch: 0, thread_id: 0 };
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let scores = [1.0_f32];
        let g = simple_sparse_update(sigma, key, shape, &scores, &[info], &frozen, &device());
        let gv = to_vec(g.clone());
        let param = Tensor::<B, 2>::zeros(shape, &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32], &device());
        let got = do_update(&param, key, &fitness, &[info], 1, sigma, &frozen);
        let gotv = to_vec(got);
        // do_update = -g * sqrt(N)  with N=1 => -g.
        assert!(gotv.iter().zip(gv.iter()).all(|(x, y)| near(*x, -y, 1e-5)), "got {gotv:?} exp {:?}", gv.iter().map(|v| -v).collect::<Vec<_>>());
    }

    // -- do_updates full pipeline ----------------------------------------

    #[test]
    fn do_updates_sgd_sparse_pipeline_smoke() {
        let lr = 0.1_f32;
        let sigma = 0.5_f32;
        let key = 22u64;
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(lr));
        let mut noiser = noiser_with(sigma, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[0.5_f32, -1.0], [2.0, 0.0]], &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32, -1.0], &device());
        let infos = [IterInfo { epoch: 0, thread_id: 0 }, IterInfo { epoch: 0, thread_id: 1 }];
        let updated = Sparse.do_updates(&frozen, &mut noiser, &[p], &[key], fitness, &infos, &[1]);
        let uv = to_vec(updated[0].clone());
        assert!(uv.iter().all(|v| v.is_finite()));
    }

    // -- get_noisy_standard ----------------------------------------------

    #[test]
    fn get_noisy_standard_identity_when_frozen_or_no_iter() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let frozen = frozen_with(0, true, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        let id = Sparse.get_noisy_standard(&frozen, &noiser, &p, 1, None);
        assert_eq!(to_vec(id), vec![1.0, 2.0]);
        let id2 = Sparse.get_noisy_standard(&frozen, &noiser, &p, 1, Some(&IterInfo { epoch: 0, thread_id: 1 }));
        assert_eq!(to_vec(id2), vec![1.0, 2.0]);
    }

    // -- convert_fitnesses (shared helper) -------------------------------

    #[test]
    fn convert_fitnesses_uses_global_denominator() {
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0, 4.0], &device());
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(1.0, &frozen, &[]);
        let g: Vec<f32> = to_vec(Sparse.convert_fitnesses(&frozen, &noiser, raw.clone()));
        assert!(g.iter().map(|x| x * x).sum::<f32>() / 4.0 - 1.0 < 1e-3);
        assert!(g.iter().sum::<f32>() / 4.0 < 1e-3);
        // Group denominators use global variance too.
        let frozen_g = frozen_with(2, false, 0, 1, Solver::sgd(0.1));
        let noiser_g = noiser_with(1.0, &frozen_g, &[]);
        let out: Vec<f32> = to_vec(Sparse.convert_fitnesses(&frozen_g, &noiser_g, raw));
        let e = 0.5 / 1.11803399_f32;
        assert!(out.iter().zip([-e, e, -e, e].iter()).all(|(a, b)| near(*a, *b, 1e-4)), "{out:?}");
    }

    // -- init_noiser ------------------------------------------------------

    #[test]
    fn init_noiser_builds_frozen_and_noiser_params() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let (frozen, noiser) = init_noiser(&[p], 0.4, 0.01, 0, false, 0, 2, 1.0, Solver::adamw(0.01), &device());
        assert_eq!(frozen.rank, 2);
        assert_eq!(noiser.sigma, 0.4);
        assert_eq!(noiser.opt_state.moments.len(), 1);
    }

    // -- determinism ------------------------------------------------------

    #[test]
    fn same_seed_gives_same_sparse_result() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.7, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let info = IterInfo { epoch: 1, thread_id: 2 };
        let a = Sparse.do_mm(&frozen, &noiser, &p, 42, Some(&info), x.clone());
        let b = Sparse.do_mm(&frozen, &noiser, &p, 42, Some(&info), x);
        assert_eq!(to_vec(a), to_vec(b));
    }
}
