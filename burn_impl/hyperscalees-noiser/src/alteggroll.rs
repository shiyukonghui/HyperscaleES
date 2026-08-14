//! AltEggRoll: EggRoll with an alternate LoRA gradient, ported from
//! `src/hyperscalees/noiser/alteggroll.py`.
//!
//! AltEggRoll is *identical* to [`crate::eggroll::EggRoll`] everywhere except
//! `_simple_lora_update`: instead of `einsum('nir,njr->ij', A, B)` (the plain
//! `A @ B.T` LoRA grad), it uses the *sign* of the noise:
//!
//! ```text
//! A = base_sigma * broadcasted_scores * sign(A) / sqrt(rank)
//! return einsum('nir,njr->ij', A, sign(B)) / num_envs
//! ```
//!
//! where `A, B` come from the shared `get_lora_update_params` (one per env,
//! with `A` already scaled by `sigma/sqrt(rank)`). So the LoRA gradient is
//! driven purely by the sign pattern of the perturbation. Everything else
//! (forwards, dense update, `convert_fitnesses`, solver plumbing) reuses the
//! shared [`crate::eggroll`] helpers.

use burn::tensor::{Device, Int, Tensor};
use hyperscalees_core::B;

use crate::eggroll::{
    do_mm_impl, do_Tmm_impl, do_updates_impl, get_lora_update_params, get_noisy_standard_impl,
};
use crate::noiser::{
    convert_fitnesses_impl, FrozenNoiserParams, IterInfo, Noiser, NoiserParams, Solver,
};

/// The AltEggRoll noiser. A zero-sized marker implementing [`Noiser`].
#[derive(Clone, Copy, Debug, Default)]
pub struct AltEggRoll;

/// Build the frozen + mutable noiser parameters, mirroring
/// `AltEggRoll.init_noiser`. AltEggRoll reuses `EggRoll.init_noiser` exactly.
pub fn init_noiser(
    params: &[Tensor<B, 2>],
    sigma: f32,
    lr: f32,
    group_size: i32,
    freeze_nonlora: bool,
    noise_reuse: i32,
    rank: usize,
    solver: Solver,
    device: &Device<B>,
) -> (FrozenNoiserParams, NoiserParams) {
    crate::eggroll::init_noiser(
        params,
        sigma,
        lr,
        group_size,
        freeze_nonlora,
        noise_reuse,
        rank,
        solver,
        device,
    )
}

/// `_simple_lora_update` (AltEggRoll): `1/N * sum_i (A_i @ sign(B_i).T)`
/// where `A_i = sigma * scores[i] * sign(A_raw_i) / sqrt(rank)`.
///
/// This is the single difference from EggRoll, whose `_simple_lora_update`
/// uses the raw `A @ B.T`. Reuses EggRoll's `get_lora_update_params` to get
/// the per-env raw `(A, B)` pair; only the contraction differs.
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
        let (a_t, b_t) =
            get_lora_update_params(base_sigma, key, rank, info, a, b, frozen.noise_reuse, device);
        // A_new = sigma * scores[i] * sign(A) / sqrt(rank)  (elementwise sign)
        let a_new = a_t
            .sign()
            .mul_scalar(sigma * scores[i] / (rank as f32).sqrt());
        let b_new = b_t.sign();
        acc = acc + a_new.matmul(b_new.transpose());
    }
    acc.mul_scalar(1.0 / n)
}

impl Noiser for AltEggRoll {
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
        unimplemented!("AltEggRoll embedding is not implemented")
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
        do_updates_impl(simple_lora_update, frozen, noiser, params, base_keys, fitnesses, iterinfos, es_classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eggroll::{epoch_thread_sign, LoraUpdateFn};
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

    // -- _simple_lora_update uses sign(A)/sign(B) -------------------------

    #[test]
    fn simple_lora_update_matches_hand_computed_sign_version() {
        // a=2, b=2, rank=1, N=1 env, sign +1.
        let a = 2usize;
        let b = 2usize;
        let r = 1usize;
        let key = 7u64;
        let sigma = 0.5_f32;
        let info = IterInfo { epoch: 0, thread_id: 0 }; // sign +1
        let frozen = frozen_with(0, false, 0, r, Solver::sgd(0.1));

        // Hand-compute expected: A,B from get_lora_update_params, then the
        // sign contraction: A_new = sigma*scores*sign(A)/sqrt(rank);
        // B_new = sign(B); out = A_new @ B_new.T / N.
        let (true_epoch, true_thread, sign) = epoch_thread_sign(&info, frozen.noise_reuse);
        let mut rng = DeterministicNoise::new(noise_seed(key, true_epoch, true_thread));
        let lora = rng.normal_tensor([a + b, r], &device());
        let lv = to_vec(lora);
        let b_raw = [lv[0], lv[1]]; // rows 0..b
        let a_raw = [lv[2], lv[3]]; // rows b..a+b
        let base_sigma = sigma / (r as f32).sqrt();
        let sc_a = sign * base_sigma; // scale applied to A by get_lora_update_params
        // sign(A) == sign(a_raw) (because sc_a > 0).
        assert!(sc_a > 0.0);
        let scores = [1.0_f32]; // N=1
        let mut expected = [0.0_f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                // A_new[i] = sigma*scores[0]*sign(a_raw[i])/sqrt(r)
                let a_new_i = sigma * scores[0] * a_raw[i].signum() / (r as f32).sqrt();
                let b_new_j = b_raw[j].signum();
                expected[i * 2 + j] = a_new_i * b_new_j;
            }
        }

        let shape = [a, b];
        let got = simple_lora_update(sigma, key, shape, &scores, &[info], &frozen, &device());
        let gv = to_vec(got);
        assert!(gv.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "got {gv:?} exp {expected:?}");
    }

    #[test]
    fn alt_lora_update_differs_from_eggroll_lora_update() {
        // Same seed/inputs, AltEggRoll (sign) vs EggRoll (raw A @ B.T) must
        // generally differ because sign() collapses magnitudes to {-1,0,1}.
        use crate::eggroll::simple_lora_update as eggroll_simple_lora_update;

        let a = 3usize;
        let b = 2usize;
        let r = 2usize;
        let key = 123u64;
        let sigma = 0.4_f32;
        let info = IterInfo { epoch: 1, thread_id: 2 };
        let frozen = frozen_with(0, false, 0, r, Solver::sgd(0.1));
        let scores = [1.0_f32, 2.0];

        let alt = to_vec(simple_lora_update(sigma, key, [a, b], &scores, &[info], &frozen, &device()));
        let egg = to_vec(eggroll_simple_lora_update(sigma, key, [a, b], &scores, &[info], &frozen, &device()));
        // There must be at least one difference (signs differ from magnitudes).
        assert!(alt.iter().zip(egg.iter()).any(|(x, y)| x != y), "alt {alt:?} egg {egg:?}");
    }

    // -- _do_update returns -g*sqrt(N) -----------------------------------

    #[test]
    fn do_update_is_neg_grad_times_sqrt_n() {
        // Dense path (map_class 0) is shared with EggRoll: g = noise, N=1.
        let sigma = 0.3_f32;
        let key = 5u64;
        let shape = [2, 1];
        let info = IterInfo { epoch: 0, thread_id: 0 };
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));

        let (te, tt, sign) = epoch_thread_sign(&info, frozen.noise_reuse);
        let mut rng = DeterministicNoise::new(noise_seed(key, te, tt));
        let nv = to_vec(rng.normal_tensor(shape, &device()));
        let expected_g: Vec<f32> = nv.iter().map(|v| sign * sigma * v).collect();

        let param = Tensor::<B, 2>::zeros(shape, &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32], &device());
        let got = crate::eggroll::do_update_with(
            simple_lora_update,
            &param,
            key,
            &fitness,
            &[info],
            0,
            sigma,
            &frozen,
        );
        let gv = to_vec(got);
        // got = -expected_g * sqrt(1)
        assert!(gv.iter().zip(expected_g.iter()).all(|(x, y)| near(*x, -y, 1e-5)), "got {gv:?}");
    }

    // -- do_mm forwards ----------------------------------------------------

    #[test]
    fn do_mm_no_noise_is_x_at_param_t() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(1.0, &frozen, &[]);
        let param = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &device(),
        );
        let x = Tensor::<B, 2>::from_data(
            [[1.0_f32, 1.0, 1.0], [2.0, 2.0, 2.0]],
            &device(),
        );
        let out = to_vec(AltEggRoll.do_mm(&frozen, &noiser, &param, 1, None, x));
        assert_eq!(out, vec![6.0, 15.0, 12.0, 30.0]);
    }

    #[test]
    fn do_mm_noised_is_base_plus_x_at_b_at_a_t() {
        let frozen = frozen_with(0, false, 0, 2, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        let param = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device()); // a=1, b=2
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let base = to_vec(x.clone().matmul(param.clone().transpose())); // [[5]]
        assert_eq!(base, vec![5.0]);
        let noisy = to_vec(AltEggRoll.do_mm(
            &frozen,
            &noiser,
            &param,
            8,
            Some(&IterInfo { epoch: 0, thread_id: 0 }),
            x.clone(),
        ));
        // noisy = base + x @ B @ A.T; matches EggRoll's shared do_mm_impl.
        let eggroll_noisy = to_vec(crate::eggroll::do_mm_impl(
            &frozen,
            &noiser,
            &param,
            8,
            Some(&IterInfo { epoch: 0, thread_id: 0 }),
            x,
        ));
        assert_eq!(noisy, eggroll_noisy);
        assert_ne!(noisy[0], 5.0);
    }

    // -- init_noiser -------------------------------------------------------

    #[test]
    fn init_noiser_reuses_eggroll_signature() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let (frozen, noiser) = init_noiser(&[p], 0.4, 0.01, 0, false, 0, 4, Solver::adamw(0.01), &device());
        assert_eq!(frozen.rank, 4);
        assert_eq!(frozen.group_size, 0);
        assert_eq!(noiser.sigma, 0.4);
        assert_eq!(noiser.opt_state.moments.len(), 1);
    }

    // -- determinism -----------------------------------------------------

    #[test]
    fn same_seed_gives_same_noise() {
        let frozen = frozen_with(0, false, 0, 2, Solver::sgd(0.1));
        let noiser = noiser_with(0.7, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let info = IterInfo { epoch: 1, thread_id: 2 };
        let a = AltEggRoll.do_mm(&frozen, &noiser, &p, 42, Some(&info), x.clone());
        let b = AltEggRoll.do_mm(&frozen, &noiser, &p, 42, Some(&info), x);
        assert_eq!(to_vec(a), to_vec(b));
    }

    // Ensure the LoraUpdateFn alias is satisfied by the sign variant (so it
    // can be passed to do_updates_impl).
    const _: LoraUpdateFn = simple_lora_update;
}
