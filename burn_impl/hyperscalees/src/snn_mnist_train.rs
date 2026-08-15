//! End-to-end "阶段 A 收敛冒烟" (convergence smoke) driver.
//!
//! This module wires the already-ported components into a full evolutionary
//! training loop and verifies that (a) the loop completes, (b) EggRoll's
//! `do_updates` actually changes the parameters, and (c) accuracy stays at or
//! above chance (~10% for 10 classes) rather than collapsing.
//!
//! The [`hyperscalees`](crate) facade is the *only* crate in the workspace
//! allowed to depend on both `hyperscalees-models` and `hyperscalees-noiser`
//! (this avoids the models <-> noiser dependency cycle). It therefore owns the
//! glue that injects EggRoll's noised matmul into [`SnnModel::forward`] via the
//! `NoiseFn` closure.
//!
//! # Noise-closure design (no models <-> noiser dependency)
//!
//! [`SnnModel::forward`] takes an optional `NoiseFn`
//! `(x: Tensor<B,2>, w: Tensor<B,2>) -> Tensor<B,2>`. Because the models crate
//! must NOT depend on the noiser crate, the facade builds a closure that calls
//! the *public* `hyperscalees_noiser::eggroll::get_lora_update_params` helper
//! to reproduce EggRoll's `do_mm` exactly:
//!
//! ```text
//!   do_mm = x @ w^T  +  x @ B @ A^T
//! ```
//!
//! where `A` (scaled by `sign * sigma / sqrt(rank)`) and `B` are the seedable
//! LoRA perturbation factors derived from the `(epoch, thread)` [`IterInfo`].
//! The per-parameter `base_key` is resolved inside the closure by matching the
//! weight's `[out, in]` dims against the parameter shapes captured at init
//! (each layer's shape — `fc1`, `fc2`, `fc3`, `out_gain` — is distinct, so the
//! mapping is unambiguous). The closure only captures shared references / `Copy`
//! values, so it satisfies the `dyn Fn` bound of `NoiseFn`.
//!
//! The ES *update* (`convert_fitnesses` + `do_updates`) is performed by the real
//! [`EggRoll`] with per-environment `iterinfos`. To keep the gradient estimate
//! consistent, each environment is evaluated via its OWN noised forward closure
//! (one `SnnModel::forward` call with `batch = 1` and that environment's
//! `IterInfo`), and the resulting per-environment logits are stacked. This
//! mirrors the Python loop's `vmap` of `forward` over the `(epoch, thread)`
//! iterinfo arrays and gives `do_updates` a coherent ES gradient.
//!
//! # Synthetic-data approach
//!
//! No external MNIST files are required (so the smoke test has no external
//! dependency). We generate `NUM_ENVS` random images in `[0, 1]^in_dim` plus a
//! small linear discriminator: `NUM_CLASSES` fixed class-prototype vectors, and
//! each image's label is `argmax(image . prototypes^T)`. This yields a
//! learnable, low-rank signal so the ES can move accuracy at/above chance
//! without needing the real dataset.

use burn::tensor::{Device, Int, Tensor, TensorData};
use hyperscalees_core::B;
use hyperscalees_envs::snn_mnist::{
    accuracy_from_logits, fitness_from_logits, poisson_encode,
};
use hyperscalees_models::snn::SnnModel;
use hyperscalees_noiser::eggroll::{get_lora_update_params, init_noiser, EggRoll};
use hyperscalees_noiser::{IterInfo, Noiser, Solver};

/// Input dimensionality of the synthetic data. Large enough that the summed
/// LIF input currents (dense spikes * weights) exceed `v_th`, so neurons fire
/// and the readout is non-degenerate (the LIF integrates `dt / tau_m = 1/20`
/// per step, so it needs a sizeable current to fire over `SMALL_T` steps).
const IN_DIM: usize = 128;
/// Hidden layer 1 size.
const HIDDEN1: usize = 64;
/// Hidden layer 2 size.
const HIDDEN2: usize = 64;
/// Number of output classes (mirrors the 10 MNIST digits).
const NUM_CLASSES: usize = 10;
/// Number of evolutionary environments (samples per generation).
const NUM_ENVS: usize = 128;
/// Number of ES generations / epochs.
const NUM_EPOCHS: usize = 24;
/// Number of Poisson spike timesteps.
const SMALL_T: usize = 4;

/// ES noise scale (`sigma`).
const SIGMA: f32 = 0.25;
/// Optimizer learning rate.
const LR: f32 = 0.05;
/// LoRA rank for the EggRoll update / noise.
const RANK: usize = 4;

/// Outcome of one smoke run, used by both the public helper and the test so the
/// loop is defined once.
#[derive(Debug, Clone, Copy)]
pub struct SmokeStats {
    /// Best (max over epochs) accuracy on the training batch, in `[0, 1]`.
    pub best_acc: f32,
    /// Max absolute element change across all parameters between the snapshot
    /// taken before the loop and the final parameters.
    pub max_param_delta: f32,
}

/// Run the end-to-end evolutionary SNN smoke loop and return the best accuracy.
///
/// Convenience wrapper: equivalent to [`train_smoke_stats`]`.best_acc`.
pub fn train_smoke() -> f32 {
    train_smoke_stats().best_acc
}

/// Run the end-to-end evolutionary SNN smoke loop on synthetic data.
///
/// Returns both the best accuracy and the max parameter delta (used to prove
/// that `do_updates` actually moved the weights).
pub fn train_smoke_stats() -> SmokeStats {
    let device = Device::<B>::default();

    // --- Model + ES plumbing -------------------------------------------
    let mut model = SnnModel::new(IN_DIM, HIDDEN1, HIDDEN2, NUM_CLASSES, &device);
    // The default LIF in `SnnModel::new` integrates very slowly
    // (`dt / tau_m = 1/20` per step), so with a small `SMALL_T` the membrane
    // never reaches `v_th` and the network stays silent (readout == 0). This
    // smoke driver overrides the frozen LIF hyper-parameters on the model
    // instance (they are `pub` fields) to a fast-integrating regime so the
    // hidden neurons actually fire and the ES sees a real, non-degenerate
    // fitness signal. No change to the models crate is required.
    model.tau_m = 1.0; // dt / tau_m = 1.0 -> membrane follows the current
    model.v_th = 0.3;
    let es_classes = model.es_map();
    let params0 = model.params();

    // One deterministic base_key per parameter leaf (fc1, fc2, fc3, out_gain).
    let base_keys: Vec<u64> = (0..params0.len())
        .map(|i| (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();

    // EggRoll noiser (LoRA rank 4, AdamW). `noise_reuse = 1` so the per-thread
    // noise actually changes across epochs.
    let (frozen, mut noiser) = init_noiser(
        &params0,
        SIGMA,
        LR,
        0,          // group_size: global z-score in convert_fitnesses
        false,      // freeze_nonlora: allow the dense out_gain path too
        1,          // noise_reuse
        RANK,
        Solver::adamw(LR),
        &device,
    );

    // Map each parameter's [out, in] shape -> base_key, so the noise closure
    // can pick the right key by the weight it is called with (shapes are
    // preserved by SGD/Adam, so the initial map stays valid).
    let dim_key_pairs: Vec<([usize; 2], u64)> = params0
        .iter()
        .zip(base_keys.iter())
        .map(|(p, k)| (p.dims(), *k))
        .collect();

    // --- Synthetic data -------------------------------------------------
    let (images, labels) = synthetic_data(NUM_ENVS, &device);

    // Snapshot to prove `do_updates` actually moved the parameters.
    let before = model.params();

    let mut best_acc = 0.0_f32;
    for epoch in 0..NUM_EPOCHS {
        // Poisson-encode the batch -> (T, num_envs, in_dim).
        let spikes = poisson_encode(images.clone(), SMALL_T);

        // One iterinfo per environment (globally unique thread id), mirroring
        // `jnp.arange(num_envs)` in the Python loop. Each environment is
        // evaluated with its OWN `(epoch, thread)` perturbation, and
        // `do_updates` consumes the exact same `iterinfos` — so the ES
        // gradient is consistent (fitness_i reflects the perturbation_i used
        // in the update), mirroring Python's `vmap` of `forward` over the
        // iterinfo arrays.
        let iterinfos: Vec<IterInfo> = (0..NUM_ENVS)
            .map(|i| IterInfo {
                epoch: epoch as i32,
                thread_id: i as i32,
            })
            .collect();

        // Build a noised-matmul closure for ONE environment. It reproduces
        // EggRoll's `do_mm` (`x @ w^T + x @ B @ A^T`) using the *public*
        // `get_lora_update_params` helper, keyed per-parameter by weight dims.
        let sigma = noiser.sigma;
        let rank = frozen.rank;
        let nreuse = frozen.noise_reuse;
        let noise_device = device.clone();
        let dim_keys = &dim_key_pairs;
        let mut logit_rows: Vec<Tensor<B, 2>> = Vec::with_capacity(NUM_ENVS);
        for i in 0..NUM_ENVS {
            // One sample: (T, 1, in_dim).
            let sample = spikes.clone().slice([0..SMALL_T, i..i + 1, 0..IN_DIM]);
            let info = iterinfos[i];
            // CudaDevice 非 Copy：每个闭包单独克隆一份设备（flex 下克隆是无代价的）。
            let nd = noise_device.clone();
            let noise_helper = move |x: Tensor<B, 2>, w: Tensor<B, 2>| -> Tensor<B, 2> {
                let dims = w.dims();
                let key = dim_keys
                    .iter()
                    .find(|(d, _)| *d == dims)
                    .map(|(_, k)| *k)
                    .unwrap_or(0);
                let base = x.clone().matmul(w.clone().transpose());
                let [a, b] = dims;
                let (a_t, b_t) = get_lora_update_params(
                    sigma / (rank as f32).sqrt(),
                    key,
                    rank,
                    &info,
                    a,
                    b,
                    nreuse,
                    &nd,
                );
                base + x.matmul(b_t).matmul(a_t.transpose())
            };
            let noise: &dyn Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2> = &noise_helper;
            logit_rows.push(model.forward(sample, Some(noise))); // (1, C)
        }
        // Stack the per-environment logits -> (num_envs, C).
        let logits = Tensor::cat(logit_rows, 0);

        // Fitness -> ES update.
        let raw = fitness_from_logits(logits.clone(), labels.clone());
        let fitnesses = EggRoll.convert_fitnesses(&frozen, &noiser, raw);
        let new_params = EggRoll.do_updates(
            &frozen,
            &mut noiser,
            &model.params(),
            &base_keys,
            fitnesses,
            &iterinfos,
            &es_classes,
        );

        // Write the updated parameters back into the model for the next epoch.
        write_params(&mut model, new_params);

        // Track peak accuracy on the (noised) training batch.
        let acc = accuracy_from_logits(logits, labels.clone());
        if acc > best_acc {
            best_acc = acc;
        }
    }

    let after = model.params();
    let max_param_delta = max_abs_delta(&before, &after);

    SmokeStats {
        best_acc,
        max_param_delta,
    }
}

/// Write the parameter vector from [`EggRoll::do_updates`] back into the model,
/// in [`SnnModel::params`] order `[fc1, fc2, fc3, out_gain]`.
fn write_params(model: &mut SnnModel, new_params: Vec<Tensor<B, 2>>) {
    let mut it = new_params.into_iter();
    model.fc1.weight = it.next().expect("missing fc1");
    model.fc2.weight = it.next().expect("missing fc2");
    model.fc3.weight = it.next().expect("missing fc3");
    // out_gain is stored unsqueezed as (1, 1) in the ES plumbing; squeeze back
    // to the model's rank-1 (1,) representation.
    model.out_gain.value = it.next().expect("missing out_gain").squeeze_dim::<1>(0);
}

/// Largest absolute elementwise difference between two parameter lists.
fn max_abs_delta(a: &[Tensor<B, 2>], b: &[Tensor<B, 2>]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x.clone().sub(y.clone()).abs();
            d.into_data()
                .into_vec::<f32>()
                .unwrap()
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Generate synthetic `(images, labels)` for the smoke test.
///
/// `images` is `(num_envs, in_dim)` in `[0, 1]`, generated from a small
/// discriminator pattern: `num_classes` fixed random prototype vectors in
/// `[0, 1]^in_dim`, and every image is a prototype plus uniform noise in
/// `[0, NOISE]`. The label of an image is the class whose prototype it most
/// resembles, i.e. `label = argmax(image . prototypes^T)`. This is:
///
/// * **Dense** — most input pixels are active, so the summed LIF input currents
///   are large enough for the membrane to cross `v_th` and fire (unlike an
///   ultra-sparse one-hot input, which leaves the LIF silent and the readout
///   degenerate). This matches why the Python reference uses 784-pixel MNIST.
/// * **Learnable** — a low-rank, well-separated linear signal, so a small,
///   noisy pure-ES run can move accuracy at/above chance.
///
/// The data is generated with a small deterministic LCG so the smoke test is
/// reproducible and has no external dependency.
fn synthetic_data(
    num_envs: usize,
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
    // Deterministic xorshift-style PRNG (no external deps, reproducible).
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Fractional noise added to each pixel: `prototype + U(0, NOISE)`.
    const NOISE: f32 = 0.3;

    // Random prototypes per class in [0, 1]^IN_DIM.
    let mut proto = vec![0.0_f32; NUM_CLASSES * IN_DIM];
    for (k, p) in proto.iter_mut().enumerate() {
        *p = (next() >> 40) as f32 / (1u64 << 24) as f32; // U(0,1)
        let _ = k;
    }

    // Build images (prototype + noise) and labels (argmax of image . proto^T).
    let mut img_vec = vec![0.0_f32; num_envs * IN_DIM];
    let mut lab_vec = vec![0_i32; num_envs];
    for i in 0..num_envs {
        let c = (next() as usize) % NUM_CLASSES;
        lab_vec[i] = c as i32;
        for d in 0..IN_DIM {
            let noise = (next() >> 40) as f32 / (1u64 << 24) as f32 * NOISE;
            img_vec[i * IN_DIM + d] = proto[c * IN_DIM + d] + noise;
        }
    }

    let images = Tensor::<B, 2>::from_data(
        TensorData::new(img_vec, [num_envs, IN_DIM].to_vec()),
        device,
    );
    let labels: Tensor<B, 1, Int> =
        Tensor::from_data(TensorData::new(lab_vec, [num_envs].to_vec()), device);
    (images, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_smoke_loop_updates_params_and_stays_at_chance() {
        let stats = train_smoke_stats();

        // (a) The loop completed and `do_updates` actually changed the params.
        assert!(
            stats.max_param_delta > 0.0,
            "noiser did not change any parameter: max_param_delta={}",
            stats.max_param_delta
        );

        // (b) Accuracy is at least at chance (10 classes => ~10%),
        // never collapsed to zero.
        assert!(
            stats.best_acc >= 0.10,
            "accuracy collapsed below chance: {:.3}",
            stats.best_acc
        );

        eprintln!(
            "OK train_smoke_loop (best_acc={:.4}, max_param_delta={:.6})",
            stats.best_acc, stats.max_param_delta
        );
    }
}
