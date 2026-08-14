//! EggRoll: a LoRA-noised matmul noiser plus ES parameter updates, ported
//! from `src/hyperscalees/noiser/eggroll.py`.
//!
//! The math mirrors the Python exactly:
//!
//! * `get_lora_update_params` : seedable `N(0, 1)` noise of shape `(a+b, r)`,
//!   split into `B` (`b x r`) and `A` (`a x r`), with `A` scaled by
//!   `sign * base_sigma` (sign from `thread_id % 2`).
//! * `get_nonlora_update_params` : seedable `N(0, 1)` noise of `param.shape`
//!   scaled by `sign * base_sigma`.
//! * `convert_fitnesses` : global or per-group z-score.
//! * `do_updates` : `new_grad = -(update_fn * sqrt(N))`, then one optimizer
//!   step with the shared [`Solver`] plumbing.

use burn::tensor::{Device, Int, Tensor};
use hyperscalees_core::B;

use crate::noiser::{
    noise_seed, DeterministicNoise, FrozenNoiserParams, IterInfo, Noiser, NoiserParams, Solver,
};

/// The EggRoll noiser. A zero-sized marker implementing [`Noiser`].
#[derive(Clone, Copy, Debug, Default)]
pub struct EggRoll;

/// Build the frozen + mutable noiser parameters, mirroring
/// `EggRoll.init_noiser`. `params` is used only to size the optimizer state.
pub fn init_noiser(
    params: &[Tensor<B, 2>],
    sigma: f32,
    // Kept for API parity with `EggRoll.init_noiser`; the learning rate already
    // lives inside `solver`.
    _lr: f32,
    group_size: i32,
    freeze_nonlora: bool,
    noise_reuse: i32,
    rank: usize,
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

// ---------------------------------------------------------------------------
// Per-parameter noise helpers
// ---------------------------------------------------------------------------

/// Derive the `(true_epoch, true_thread_idx, sign)` triple from an
/// [`IterInfo`] and the noise-reuse factor, as in
/// `get_lora_update_params` / `get_nonlora_update_params`.
pub(crate) fn epoch_thread_sign(info: &IterInfo, noise_reuse: i32) -> (i32, i32, f32) {
    let true_epoch = if noise_reuse == 0 { 0 } else { info.epoch / noise_reuse };
    let true_thread = info.thread_id / 2;
    let sign = if info.thread_id % 2 == 0 { 1.0 } else { -1.0 };
    (true_epoch, true_thread, sign)
}

/// Signature of a per-parameter `_simple_lora_update` variant. Used so that
/// [`eggroll::EggRoll`] and [`crate::alteggroll::AltEggRoll`] can share the
/// whole [`crate::noiser::Noiser`] plumbing while differing only in the sign of
/// the LoRA gradient (EggRoll: `A @ B.T`; AltEggRoll: `sign(A) @ sign(B).T`).
pub(crate) type LoraUpdateFn = fn(
    sigma: f32,
    key: u64,
    shape: [usize; 2],
    scores: &[f32],
    iterinfos: &[IterInfo],
    frozen: &FrozenNoiserParams,
    device: &Device<B>,
) -> Tensor<B, 2>;

/// Deterministic LoRA noise parameters: returns `(A, B)`.
///
/// `A` has shape `(a, r)` and `B` has shape `(b, r)`. `A` is scaled by
/// `sign * base_sigma` (the caller is expected to pass
/// `sigma / sqrt(rank)` as `base_sigma`).
pub fn get_lora_update_params(
    base_sigma: f32,
    key_seed: u64,
    rank: usize,
    info: &IterInfo,
    a: usize,
    b: usize,
    noise_reuse: i32,
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let (true_epoch, true_thread, sign) = epoch_thread_sign(info, noise_reuse);
    let mut rng = DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));
    // Shape (a + b, r) with A taking the lower rows and B the upper rows.
    let lora = rng.normal_tensor([a + b, rank], device);
    let b_t = lora.clone().slice([0..b, 0..rank]); // b x r
    let a_t = lora.slice([b..a + b, 0..rank]); // a x r
    (a_t.mul_scalar(sign * base_sigma), b_t)
}

/// Deterministic dense (`nonlora`) noise of the parameter's shape.
pub fn get_nonlora_update_params(
    base_sigma: f32,
    key_seed: u64,
    info: &IterInfo,
    shape: [usize; 2],
    noise_reuse: i32,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let (true_epoch, true_thread, sign) = epoch_thread_sign(info, noise_reuse);
    let mut rng = DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));
    let updates = rng.normal_tensor(shape, device);
    updates.mul_scalar(sign * base_sigma)
}

// ---------------------------------------------------------------------------
// Per-parameter update functions
// ---------------------------------------------------------------------------

/// `_simple_full_update`: `1/N * sum_i f_i * noise_i` or zeros when frozen.
pub(crate) fn simple_full_update(
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

/// `_simple_lora_update`: `1/N * sum_i f_i * (A_i @ B_i^T)`.
pub(crate) fn simple_lora_update(
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
        let (a_t, b_t) =
            get_lora_update_params(base_sigma, key, rank, info, a, b, frozen.noise_reuse, device);
        acc = acc + a_t.matmul(b_t.transpose()).mul_scalar(scores[i]);
    }
    acc.mul_scalar(1.0 / n)
}

/// `_noop_update`: zeros of the parameter's shape.
pub(crate) fn noop_update(shape: [usize; 2], device: &Device<B>) -> Tensor<B, 2> {
    Tensor::<B, 2>::zeros(shape, device)
}

/// `_do_update`: choose the update fn by `map_classification` and return the
/// *negated* gradient scaled by `sqrt(N)`. `lora_update` is the
/// `_simple_lora_update` variant (EggRoll uses `A @ B.T`; AltEggRoll uses
/// `sign(A) @ sign(B).T`).
pub(crate) fn do_update_with(
    lora_update: LoraUpdateFn,
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
        1 => lora_update(sigma, base_key, shape, &scores, iterinfos, frozen, &device),
        _ => noop_update(shape, &device),
    };
    let n = scores.len() as f32;
    g.mul_scalar(n.sqrt()).neg()
}

// ---------------------------------------------------------------------------
// Shared `Noiser` forward + update machinery
// ---------------------------------------------------------------------------
//
// These `pub(crate)` bodies let [`EggRoll`] and [`crate::alteggroll::AltEggRoll`]
// share the whole forward/update plumbing. The two noisers differ only in
// `_simple_lora_update` (EggRoll: `A @ B.T`; AltEggRoll: `sign(A) @ sign(B).T`),
// so `do_mm`/`do_Tmm`/`get_noisy_standard` are shared verbatim and only
// `do_updates` is parameterised by the `_simple_lora_update` variant.
// ([`crate::eggroll_bs::EggRollBS`] keeps its own module because it needs
// masked noise params and a baseline-subtraction `convert_fitnesses`.)

/// Shared `do_mm` (EggRoll/AltEggRoll): `base + x @ B @ A.T` under LoRA noise.
pub(crate) fn do_mm_impl(
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
                &param.device(),
            );
            // base + x @ B @ A.T
            base + x.matmul(b_t).matmul(a_t.transpose())
        }
    }
}

/// Shared `do_Tmm` (EggRoll/AltEggRoll): `base + x @ A @ B.T` under LoRA noise.
#[allow(non_snake_case)]
pub(crate) fn do_Tmm_impl(
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
                &param.device(),
            );
            // base + x @ A @ B.T
            base + x.matmul(a_t).matmul(b_t.transpose())
        }
    }
}

/// Shared `get_noisy_standard` (EggRoll/AltEggRoll): `param + nonlora` noise.
pub(crate) fn get_noisy_standard_impl(
    frozen: &FrozenNoiserParams,
    noiser: &NoiserParams,
    param: &Tensor<B, 2>,
    base_key: u64,
    iterinfo: Option<&IterInfo>,
) -> Tensor<B, 2> {
    match iterinfo {
        None => param.clone(),
        Some(_) if frozen.freeze_nonlora => param.clone(),
        Some(ref info) => {
            let shape = param.dims();
            let noise = get_nonlora_update_params(
                noiser.sigma,
                base_key,
                info,
                shape,
                frozen.noise_reuse,
                &param.device(),
            );
            param.clone() + noise
        }
    }
}

/// Shared `do_updates`: compute per-param grad via `_do_update` and apply the
/// solver. `lora_update` is the `_simple_lora_update` variant (differs between
/// EggRoll and AltEggRoll).
pub(crate) fn do_updates_impl(
    lora_update: LoraUpdateFn,
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
        .map(|((p, k), c)| do_update_with(lora_update, p, *k, &fitnesses, iterinfos, *c, noiser.sigma, frozen))
        .collect();
    frozen.solver.update(params, &grads, &mut noiser.opt_state)
}

impl Noiser for EggRoll {
    fn do_mm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        do_mm_impl(frozen, noiser, param, base_key, iterinfo, x)
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
        do_Tmm_impl(frozen, noiser, param, base_key, iterinfo, x)
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
        unimplemented!("EggRoll embedding is not implemented")
    }

    fn get_noisy_standard(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
    ) -> Tensor<B, 2> {
        get_noisy_standard_impl(frozen, noiser, param, base_key, iterinfo)
    }

    fn convert_fitnesses(
        &self,
        frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        raw: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let group_size = frozen.group_size as usize;
        let n = raw.dims()[0];

        // Global mean / var of the full raw array (matches Python's use of
        // `jnp.var(raw_scores, keepdims=True)` in both branches).
        let mean = raw.clone().mean().into_scalar();
        let var = raw.clone().powf_scalar(2.0).mean().into_scalar() - mean * mean;
        let std = (var + 1e-5).sqrt();

        if group_size == 0 {
            // Global z-score.
            raw.add_scalar(-mean).mul_scalar(1.0 / std)
        } else {
            // Per-group mean (keepdims), global std.
            let n_groups = n / group_size;
            let groups = raw.reshape([n_groups, group_size]); // (groups, gs)
            let gmean = groups.clone().mean_dim(-1); // (groups, 1)
            (groups - gmean).mul_scalar(1.0 / std).reshape([n])
        }
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
        do_updates_impl(simple_lora_update, frozen, noiser, params, base_keys, fitnesses, iterinfos, es_classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noiser::NoiserParams;

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

    // -- convert_fitnesses -------------------------------------------------

    #[test]
    fn convert_fitnesses_global_is_zero_mean_unit_std() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(1.0, &frozen, &[]);
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0, 4.0], &device());
        let out: Vec<f32> = to_vec(EggRoll.convert_fitnesses(&frozen, &noiser, raw));
        // mean=2.5, var=1.25, std=sqrt(1.25+1e-5)=1.118034.
        assert!(out.iter().map(|x| x * x).sum::<f32>() / 4.0 - 1.0 < 1e-3);
        assert!(out.iter().sum::<f32>() / 4.0 < 1e-3);
    }

    #[test]
    fn convert_fitnesses_group_matches_hand_computed() {
        let frozen = frozen_with(2, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(1.0, &frozen, &[]);
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0, 4.0], &device());
        let out: Vec<f32> = to_vec(EggRoll.convert_fitnesses(&frozen, &noiser, raw));
        // groups [[1,2],[3,4]]; global std = sqrt(1.25+1e-5)=1.118034;
        // group means 1.5 and 3.5 -> each pair (dx/1.118034).
        let e = 0.5 / 1.11803399;
        assert!(out.iter().zip([-e, e, -e, e].iter()).all(|(a, b)| near(*a, *b, 1e-4)), "{out:?}");
    }

    // -- get_lora_update_params -------------------------------------------

    #[test]
    fn lora_update_params_shape_determinism_and_thread_var() {
        let a = 2usize;
        let b = 3usize;
        let r = 2usize;
        let key = 42u64;
        let info = IterInfo { epoch: 1, thread_id: 0 };

        let (a1, b1) = get_lora_update_params(0.5, key, r, &info, a, b, 0, &device());
        assert_eq!(a1.dims(), [a, r]);
        assert_eq!(b1.dims(), [b, r]);

        // Same inputs -> identical.
        let (a2, b2) = get_lora_update_params(0.5, key, r, &info, a, b, 0, &device());
        assert_eq!(to_vec(a1.clone()), to_vec(a2));
        assert_eq!(to_vec(b1.clone()), to_vec(b2));

        // Different true thread_idx (thread 0 vs thread 4 -> 0 vs 2) differs.
        let other = IterInfo { epoch: 1, thread_id: 4 };
        let (a3, _b3) = get_lora_update_params(0.5, key, r, &other, a, b, 0, &device());
        assert!(to_vec(a1).iter().zip(to_vec(a3).iter()).any(|(x, y)| x != y));
    }

    // -- _simple_lora_update (via do_update path pieces) -------------------

    #[test]
    fn simple_lora_update_matches_hand_computed() {
        let a = 2usize;
        let b = 2usize;
        let r = 1usize;
        let key = 7u64;
        let sigma = 0.5_f32;
        let info = IterInfo { epoch: 0, thread_id: 0 }; // sign +1
        let frozen = frozen_with(0, false, 0, r, Solver::sgd(0.1));

        // Hand-compute the expected update with the same deterministic RNG.
        let (true_epoch, true_thread, sign) = epoch_thread_sign(&info, frozen.noise_reuse);
        let mut rng = DeterministicNoise::new(noise_seed(key, true_epoch, true_thread));
        let lora = rng.normal_tensor([a + b, r], &device());
        let lv = to_vec(lora); // length a+b = 4
        let b_raw = [lv[0], lv[1]]; // rows 0..2
        let a_raw = [lv[2], lv[3]]; // rows 2..4
        let base_sigma = sigma / (r as f32).sqrt();
        let sc = sign * base_sigma;
        // A = sc * a_raw (2x1); B = b_raw (2x1); expected = A @ B^T (2x2).
        let mut expected = [0.0_f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                expected[i * 2 + j] = (sc * a_raw[i]) * b_raw[j];
            }
        }

        let scores = [1.0_f32]; // N=1, fitness weight 1.
        let shape = [a, b];
        let got = simple_lora_update(sigma, key, shape, &scores, &[info], &frozen, &device());
        let gv = to_vec(got);
        assert!(gv.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "got {gv:?} exp {expected:?}");
    }

    // -- _do_update sign ---------------------------------------------------

    #[test]
    fn do_update_is_neg_grad_times_sqrt_n() {
        // N=1 dense update: g = noise; do_update = -g * sqrt(1) = -g.
        let sigma = 0.3_f32;
        let key = 5u64;
        let shape = [2, 1];
        let info = IterInfo { epoch: 0, thread_id: 0 };
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));

        // Hand-computed g.
        let (te, tt, sign) = epoch_thread_sign(&info, frozen.noise_reuse);
        let mut rng = DeterministicNoise::new(noise_seed(key, te, tt));
        let nv = to_vec(rng.normal_tensor(shape, &device()));
        let expected_g: Vec<f32> = nv.iter().map(|v| sign * sigma * v).collect();

        let param = Tensor::<B, 2>::zeros(shape, &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32], &device());
        let got = do_update_with(simple_lora_update, &param, key, &fitness, &[info], 0, sigma, &frozen);
        let gv = to_vec(got);
        let neg: Vec<f32> = expected_g.iter().map(|v| -v).collect();
        // got = -expected_g * sqrt(1)
        assert!(gv.iter().zip(expected_g.iter()).all(|(x, y)| near(*x, -y, 1e-5)), "got {gv:?} exp {neg:?}");
    }

    // -- do_updates SGD full pipeline -------------------------------------

    #[test]
    fn do_updates_sgd_full_pipeline_matches_formula() {
        // Two envs, dense update, shape [1, 2], lr=0.1, sigma=0.5.
        let lr = 0.1_f32;
        let sigma = 0.5_f32;
        let key = 99u64;
        let shape = [1, 2];
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(lr));
        let mut noiser = noiser_with(sigma, &frozen, &[]);

        let p = Tensor::<B, 2>::from_data([[1.0_f32, -1.0]], &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32, 2.0], &device());
        let infos = [IterInfo { epoch: 0, thread_id: 0 }, IterInfo { epoch: 0, thread_id: 1 }];

        // Hand-compute g = 1/2 * (f0*u0 + f1*u1) with deterministic noise.
        let mut g = [0.0_f32; 2];
        for (i, info) in infos.iter().enumerate() {
            let (te, tt, sign) = epoch_thread_sign(info, 0);
            let mut rng = DeterministicNoise::new(noise_seed(key, te, tt));
            let nv = to_vec(rng.normal_tensor(shape, &device()));
            let f = fitness.clone().into_data().into_vec::<f32>().unwrap()[i];
            for (k, v) in nv.iter().enumerate() {
                g[k] += f * sign * sigma * v;
            }
        }
        for x in g.iter_mut() {
            *x /= 2.0; // /N
        }
        // param_new = param + lr * g * sqrt(N), N = 2.
        let n_root = (2.0_f32).sqrt();
        let expected: Vec<f32> = vec![
            1.0 + lr * g[0] * n_root,
            -1.0 + lr * g[1] * n_root,
        ];

        let updated = EggRoll.do_updates(&frozen, &mut noiser, &[p], &[key], fitness, &infos, &[0]);
        let uv = to_vec(updated[0].clone());
        assert!(uv.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "got {uv:?} exp {expected:?}");
    }

    // -- do_updates Adam runs without panic --------------------------------

    #[test]
    fn do_updates_adam_runs_without_panic() {
        let frozen = frozen_with(0, false, 0, 2, Solver::adam(0.01));
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let mut noiser = noiser_with(0.5, &frozen, &[p.clone()]);
        let fitness = Tensor::<B, 1>::from_data([1.0_f32, 2.0], &device());
        let infos = [IterInfo { epoch: 0, thread_id: 0 }, IterInfo { epoch: 0, thread_id: 2 }];
        // LoRA update path (map_class = 1).
        let updated = EggRoll.do_updates(&frozen, &mut noiser, &[p.clone()], &[3u64], fitness.clone(), &infos, &[1]);
        assert!(to_vec(updated[0].clone()).iter().all(|x| x.is_finite()));
        // Dense path as well.
        let updated2 = EggRoll.do_updates(&frozen, &mut noiser, &[p.clone()], &[4u64], fitness, &infos, &[0]);
        assert!(to_vec(updated2[0].clone()).iter().all(|x| x.is_finite()));
        // Running a second Adam step must not panic either.
        let fitness2 = Tensor::<B, 1>::from_data([0.5_f32, -1.0], &device());
        let updated3 = EggRoll.do_updates(&frozen, &mut noiser, &[p], &[5u64], fitness2, &infos, &[1]);
        assert!(to_vec(updated3[0].clone()).iter().all(|x| x.is_finite()));
    }

    // -- init_noiser -------------------------------------------------------

    #[test]
    fn init_noiser_builds_frozen_and_noiser_params() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let (frozen, noiser) = init_noiser(&[p], 0.4, 0.01, 0, false, 0, 4, Solver::adamw(0.01), &device());
        assert_eq!(frozen.rank, 4);
        assert_eq!(frozen.group_size, 0);
        assert_eq!(noiser.sigma, 0.4);
        assert_eq!(noiser.opt_state.moments.len(), 1);
    }

    // -- get_noisy_standard & do_mm ---------------------------------------

    #[test]
    fn get_noisy_standard_adds_dense_noise_with_iterinfo() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        // Without iterinfo -> identity.
        let id = EggRoll.get_noisy_standard(&frozen, &noiser, &p, 1, None);
        assert_eq!(to_vec(id), vec![1.0, 2.0]);
        // With freeze_nonlora -> identity.
        let frz = frozen_with(0, true, 0, 1, Solver::sgd(0.1));
        let noiser2 = noiser_with(0.5, &frz, &[]);
        let id2 = EggRoll.get_noisy_standard(&frz, &noiser2, &p, 1, Some(&IterInfo { epoch: 0, thread_id: 1 }));
        assert_eq!(to_vec(id2), vec![1.0, 2.0]);
    }

    #[test]
    fn do_mm_adds_lora_noise_when_iterinfo_present() {
        let frozen = frozen_with(0, false, 0, 2, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        // param (out=1, in=2) -> a=1, b=2.
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let base = EggRoll.do_mm(&frozen, &noiser, &p, 8, None, x.clone());
        // x @ p.T = [[5]]
        assert_eq!(to_vec(base), vec![5.0]);
        let noisy = EggRoll.do_mm(&frozen, &noiser, &p, 8, Some(&IterInfo { epoch: 0, thread_id: 0 }), x);
        // x @ p.T + x @ B @ A.T ; must differ from base in general.
        assert!(to_vec(noisy)[0] != 5.0);
    }
}
