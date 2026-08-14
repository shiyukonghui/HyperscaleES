//! OpenES: a dense-noise ES noiser, ported from
//! `src/hyperscalees/noiser/open_es.py`.
//!
//! Unlike EggRoll's LoRA formulation, OpenES adds dense (`nonlora`) noise
//! directly to the *weight* and then does the full matmul:
//!
//! ```text
//! do_mm  : x @ (param + nonlora).T
//! do_Tmm : x @ (param + nonlora)
//! ```
//!
//! The update path is the same shared machinery: `_simple_full_update` /
//! `_simple_lora_update` mean over envs of `scores_j * nonlora_j`, then
//! `-grad * sqrt(N)` through the [`crate::noiser::Solver`] plumbing.

use burn::tensor::{Device, Int, Tensor};
use hyperscalees_core::B;

use crate::eggroll::get_nonlora_update_params;
use crate::noiser::{
    convert_fitnesses_impl, FrozenNoiserParams, IterInfo, Noiser, NoiserParams, Solver,
};

/// The OpenES noiser. A zero-sized marker implementing [`Noiser`].
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenES;

/// Build the frozen + mutable noiser parameters, mirroring
/// `OpenES.init_noiser`. `params` is used only to size the optimizer state.
pub fn init_noiser(
    params: &[Tensor<B, 2>],
    sigma: f32,
    // Kept for API parity with `OpenES.init_noiser`; the learning rate already
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

/// `_simple_full_update` / `_simple_lora_update`: `1/N * sum_i f_i * noise_i`
/// or zeros when frozen. Both OpenES map classes (0 = full, 1 = lora) reduce to
/// the same dense (`nonlora`) mean in the Python source, so they share this
/// helper.
fn simple_dense_update(
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
        0 | 1 => simple_dense_update(sigma, base_key, shape, &scores, iterinfos, frozen, &device),
        _ => noop_update(shape, &device),
    };
    let n = scores.len() as f32;
    g.mul_scalar(n.sqrt()).neg()
}

impl Noiser for OpenES {
    fn do_mm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        match iterinfo {
            None => x.matmul(param.clone().transpose()),
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
                let new_param = param.clone() + noise;
                x.matmul(new_param.transpose())
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
        match iterinfo {
            None => x.matmul(param.clone()),
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
                let new_param = param.clone() + noise;
                x.matmul(new_param)
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
        unimplemented!("OpenES embedding is not implemented")
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
    use crate::noiser::{noise_seed, DeterministicNoise};

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

    // -- do_mm ------------------------------------------------------------

    #[test]
    fn do_mm_no_noise_is_x_at_param_t() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        // param (a=2, b=3), x (batch=2, b=3).
        let param = Tensor::<B, 2>::from_data([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device());
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 1.0, 1.0], [2.0, 2.0, 2.0]], &device());
        let out = to_vec(OpenES.do_mm(&frozen, &noiser, &param, 7, None, x.clone()));
        // x @ param.T = [[6,15],[12,30]]
        assert_eq!(out, vec![6.0, 15.0, 12.0, 30.0]);
    }

    #[test]
    fn do_mm_noised_is_x_at_param_plus_nonlora() {
        let sigma = 0.5_f32;
        let key = 11u64;
        let info = IterInfo { epoch: 0, thread_id: 0 }; // sign +1
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(sigma, &frozen, &[]);

        let param = Tensor::<B, 2>::from_data([[1.0_f32, 2.0, 3.0]], &device()); // (a=1, b=3)
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0, 3.0]], &device()); // (1, 3)

        let noisy = to_vec(OpenES.do_mm(
            &frozen,
            &noiser,
            &param,
            key,
            Some(&info),
            x.clone(),
        ));

        // Hand-compute noise with the same deterministic sampler.
        let shape = param.dims();
        let mut rng = DeterministicNoise::new(noise_seed(key, 0, 0));
        let nv = to_vec(rng.normal_tensor(shape, &device()));
        let noise: Vec<f32> = nv.iter().map(|v| sigma * v).collect(); // sign +1
        let new_param: Vec<f32> = [1.0, 2.0, 3.0]
            .iter()
            .zip(noise.iter())
            .map(|(p, n)| p + n)
            .collect();
        // x @ new_param.T  (x is 1x3, new_param is (1,3) -> [sum_i x_i * new_param_i])
        let expected: f32 = (0..3).map(|i| [1.0, 2.0, 3.0][i] * new_param[i]).sum();
        assert!(near(noisy[0], expected, 1e-4), "got {:?} exp {expected}", noisy[0]);
    }

    // -- get_noisy_standard ----------------------------------------------

    #[test]
    fn get_noisy_standard_no_iterinfo_or_frozen_is_identity() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        // No iterinfo -> identity.
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        let id = OpenES.get_noisy_standard(&frozen, &noiser, &p, 1, None);
        assert_eq!(to_vec(id), vec![1.0, 2.0]);
        // freeze_nonlora + iterinfo -> identity.
        let frz = frozen_with(0, true, 0, 1, Solver::sgd(0.1));
        let noiser2 = noiser_with(0.5, &frz, &[]);
        let id2 = OpenES.get_noisy_standard(&frz, &noiser2, &p, 1, Some(&IterInfo { epoch: 0, thread_id: 1 }));
        assert_eq!(to_vec(id2), vec![1.0, 2.0]);
    }

    #[test]
    fn get_noisy_standard_adds_dense_noise_when_not_frozen() {
        let sigma = 0.4_f32;
        let key = 3u64;
        let info = IterInfo { epoch: 0, thread_id: 0 };
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(sigma, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let out = to_vec(OpenES.get_noisy_standard(&frozen, &noiser, &p, key, Some(&info)));
        // expected = param + sign*sigma*noise
        let mut rng = DeterministicNoise::new(noise_seed(key, 0, 0));
        let nv = to_vec(rng.normal_tensor([1, 2], &device()));
        let expected = [1.0, 2.0]
            .iter()
            .zip(nv.iter())
            .map(|(p, n)| p + sigma * n)
            .collect::<Vec<_>>();
        assert!(out.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "{out:?} exp {expected:?}");
    }

    // -- convert_fitnesses ------------------------------------------------

    #[test]
    fn convert_fitnesses_global_and_group_use_global_var() {
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0, 4.0], &device());
        // Global: mean=2.5, var=1.25, std=sqrt(1.25+1e-5).
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(1.0, &frozen, &[]);
        let g: Vec<f32> = to_vec(OpenES.convert_fitnesses(&frozen, &noiser, raw.clone()));
        assert!(g.iter().map(|x| x * x).sum::<f32>() / 4.0 - 1.0 < 1e-3);
        assert!(g.iter().sum::<f32>() / 4.0 < 1e-3);

        // Group (groups of 2): group means 1.5 / 3.5, global std = 1.11803399.
        let frozen_g = frozen_with(2, false, 0, 1, Solver::sgd(0.1));
        let noiser_g = noiser_with(1.0, &frozen_g, &[]);
        let out: Vec<f32> = to_vec(OpenES.convert_fitnesses(&frozen_g, &noiser_g, raw));
        let e = 0.5 / 1.11803399_f32;
        assert!(out.iter().zip([-e, e, -e, e].iter()).all(|(a, b)| near(*a, *b, 1e-4)), "{out:?}");
    }

    // -- _do_update / do_updates -----------------------------------------

    #[test]
    fn do_update_is_neg_grad_times_sqrt_n() {
        let sigma = 0.3_f32;
        let key = 5u64;
        let shape = [2, 1];
        let info = IterInfo { epoch: 0, thread_id: 0 };
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        // Hand-computed g = noise (N=1).
        let mut rng = DeterministicNoise::new(noise_seed(key, 0, 0));
        let nv = to_vec(rng.normal_tensor(shape, &device()));
        let expected_g: Vec<f32> = nv.iter().map(|v| sigma * v).collect();
        let param = Tensor::<B, 2>::zeros(shape, &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32], &device());
        let got = do_update(&param, key, &fitness, &[info], 0, sigma, &frozen);
        let gv = to_vec(got);
        assert!(gv.iter().zip(expected_g.iter()).all(|(x, y)| near(*x, -y, 1e-5)), "got {gv:?}");
    }

    #[test]
    fn do_updates_sgd_pipeline_matches_formula() {
        let lr = 0.1_f32;
        let sigma = 0.5_f32;
        let key = 99u64;
        let shape = [1, 2];
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(lr));
        let mut noiser = noiser_with(sigma, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, -1.0]], &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32, 2.0], &device());
        let infos = [IterInfo { epoch: 0, thread_id: 0 }, IterInfo { epoch: 0, thread_id: 1 }];
        let fvec = fitness.clone().into_data().into_vec::<f32>().unwrap();
        // Hand-compute g = 1/N * sum_j f_j * (sign_j*sigma*noise_j).
        let mut g = [0.0_f32; 2];
        for (i, info) in infos.iter().enumerate() {
            let (te, tt, sign) = (0i32, info.thread_id / 2, if info.thread_id % 2 == 0 { 1.0 } else { -1.0 });
            let mut rng = DeterministicNoise::new(noise_seed(key, te, tt));
            let nv = to_vec(rng.normal_tensor(shape, &device()));
            for (k, v) in nv.iter().enumerate() {
                g[k] += fvec[i] * sign * sigma * v;
            }
        }
        for x in g.iter_mut() {
            *x /= 2.0;
        }
        let n_root = (2.0_f32).sqrt();
        let expected = [1.0 + lr * g[0] * n_root, -1.0 + lr * g[1] * n_root];
        let updated = OpenES.do_updates(&frozen, &mut noiser, &[p], &[key], fitness, &infos, &[0]);
        let uv = to_vec(updated[0].clone());
        assert!(uv.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "got {uv:?} exp {expected:?}");
    }

    #[test]
    fn init_noiser_builds_frozen_and_noiser_params() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let (frozen, noiser) = init_noiser(&[p], 0.4, 0.01, 0, false, 0, 4, Solver::adamw(0.01), &device());
        assert_eq!(frozen.rank, 4);
        assert_eq!(frozen.group_size, 0);
        assert_eq!(noiser.sigma, 0.4);
        assert_eq!(noiser.opt_state.moments.len(), 1);
    }

    // -- determinism ------------------------------------------------------

    #[test]
    fn same_seed_gives_same_noise() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.7, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device());
        let info = IterInfo { epoch: 1, thread_id: 2 };
        let a = OpenES.get_noisy_standard(&frozen, &noiser, &p, 42, Some(&info));
        let b = OpenES.get_noisy_standard(&frozen, &noiser, &p, 42, Some(&info));
        assert_eq!(to_vec(a), to_vec(b));
    }
}
