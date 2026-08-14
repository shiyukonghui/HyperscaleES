"""Train SNN Attention models (Hopfield / Mean-field routes) on patched-MNIST
classification using HyperscaleES' evolutionary noiser (no back-prop through spikes).

The MNIST 28x28 image is cut into ``P x P`` patches, each patch becomes a token whose
pixels are the token feature vector. Self-attention is taken over the patch tokens
(global query = mean token query), and the attention-weighted value readout is pooled
and classified. Two routes from ``docs/注意力机制数学等价snn迁移.md`` are implemented
and compared:

  - ``hopfield`` : LIF attention neurons + global inhibition + synaptic trace.
  - ``meanfield``: Wilson-Cowan population + divisive normalization.

During evaluation the script also computes the Section 五 equivalence metrics between
the SNN attention weights and the reference Softmax attention (weight error
``||p_hat - p*||`` and output cosine similarity) so the "近似等效注意力" claim is
quantified rather than asserted.

Run:
    .\\.venv\\Scripts\\python.exe -m llm_experiments.snn_attention_train \
        --route hopfield --batch 256 --epochs 200 --T 8
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.snn_attention import (
    HopfieldAttnSNN,
    MeanFieldAttnSNN,
    model_rand_init,
)
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays,
    poisson_encode,
    accuracy_from_logits,
)

NOISER = hs.noiser.eggroll.EggRoll
DTYPE = jnp.float32
NUM_CLASSES = 10
# local MNIST IDX directory (same as the SNN-MNIST experiments)
MNIST_DIR = os.environ.get("MNIST_DIR", r"D:\Rust\snn_t1\mnist_data")


def reward_from_logits(logits, labels, reward="loglik"):
    """Per-sample ES reward.

    ``loglik`` (default): log-probability of the true class — a dense, smooth fitness
    landscape on which the project's pure-ES noiser is more stable and accurate than
    hard 0/1 (see ``docs/snn_es_mnist_experiment.md`` 7.5 & ``snn_mnist_train_multi_gpu.py``).
    ``binary``: 1.0 if argmax==label else 0.0 (hard reward).
    """
    if reward == "binary":
        pred = jnp.argmax(logits, axis=-1)
        return (pred == labels).astype(jnp.float32)
    if reward == "loglik":
        ll = jax.nn.log_softmax(logits, axis=-1)
        return ll[jnp.arange(labels.shape[0]), labels]
    raise ValueError(f"unknown reward: {reward}")


# ---------------------------------------------------------------------------
# Patch the 28x28 image into P x P tokens (each token = one flattened patch)
# ---------------------------------------------------------------------------
def patch_images(images, patch_px):
    """images: (batch, 28*28) float in [0,1] -> (batch, P*P, patch_px*patch_px).
    28 must be divisible by patch_px."""
    assert 28 % patch_px == 0
    side = 28 // patch_px
    images = images.reshape(images.shape[0], side, patch_px, side, patch_px)
    # (batch, P, P, patch_px, patch_px) -> (batch, P*P, patch_px*patch_px)
    images = images.transpose(0, 1, 3, 2, 4).reshape(
        images.shape[0], side * side, patch_px * patch_px
    )
    return images


def parse_args():
    p = argparse.ArgumentParser(description="SNN Attention + ES patched-MNIST training")
    p.add_argument("--route", choices=["hopfield", "meanfield"], default="hopfield")
    p.add_argument("--patch-px", type=int, default=7,
                   help="patch side (28//patch_px tokens, e.g. 7 -> 4x4=16 tokens)")
    p.add_argument("--d-head", type=int, default=16, help="Q/K/V projection dim")
    p.add_argument("--T", type=int, default=8, help="SNN time steps / poisson length")
    p.add_argument("--batch", type=int, default=256, help="ES population (num_envs)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--rank", type=int, default=8, help="LoRA noise rank")
    p.add_argument("--n-iter", type=int, default=8,
                   help="attention recurrent/population iterations")
    p.add_argument("--tau-m", type=float, default=20.0)
    p.add_argument("--proj-gain", type=float, default=2.0,
                   help="sigmoid slope of the Q/K/V rate encoder")
    p.add_argument("--reward", choices=["loglik", "binary"], default="loglik",
                   help="ES reward: loglik (default, smoother) or hard 0/1 binary")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate-every", type=int, default=20)
    p.add_argument("--val-batch", type=int, default=512)
    p.add_argument("--mnist-dir", default=MNIST_DIR)
    p.add_argument("--csv-out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    assert 28 % args.patch_px == 0, "patch_px must divide 28"
    num_tokens = (28 // args.patch_px) ** 2
    token_in_dim = args.patch_px ** 2

    key = jax.random.key(args.seed)
    model_key, es_key, data_key = jax.random.split(key, 3)

    MODEL = HopfieldAttnSNN if args.route == "hopfield" else MeanFieldAttnSNN
    frozen_params, params, scan_map, es_map = MODEL.rand_init(
        model_key, token_in_dim=token_in_dim, num_tokens=num_tokens,
        num_classes=NUM_CLASSES, d_head=args.d_head, tau_m=args.tau_m,
        proj_gain=args.proj_gain, n_iter=args.n_iter, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=max(20, args.epochs // 10),
        decay_steps=args.epochs, end_value=args.lr * 0.1,
    )
    frozen_noiser, noiser_params = NOISER.init_noiser(
        params, args.sigma, lr_schedule,
        solver=optax.adamw, solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=args.rank,
    )

    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: MODEL.forward(
            NOISER, frozen_noiser, n, frozen_params, p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0),
    ))
    jit_forward_eval = jax.jit(jax.vmap(
        lambda n, p, x: MODEL.forward(
            NOISER, frozen_noiser, n, frozen_params, p, es_tree_key, None, x),
        in_axes=(None, None, 0),
    ))
    jit_update = jax.jit(
        lambda n, p, f, i: NOISER.do_updates(frozen_noiser, n, p, es_tree_key, f, i, es_map)
    )

    # --- data ---------------------------------------------------------------
    x_train, y_train = get_mnist_arrays("train", data_dir=args.mnist_dir)
    x_test, y_test = get_mnist_arrays("test", data_dir=args.mnist_dir)
    n_train = x_train.shape[0]
    print(f"[data] train={n_train} patched-{num_tokens} tokens, "
          f"token_dim={token_in_dim}, route={args.route}")

    def next_batch(rng):
        rng, sub = jax.random.split(rng)
        idx = jax.random.permutation(sub, n_train)[:args.batch]
        imgs = patch_images(jnp.asarray(x_train[idx], dtype=DTYPE), args.patch_px)
        labels = jnp.asarray(y_train[idx], dtype=jnp.int32)
        rng, enc = jax.random.split(rng)
        # poisson over (batch, num_tokens, token_in_dim) -> (T, batch, N, D)
        spikes = poisson_encode(imgs, args.T, enc).transpose(1, 0, 2, 3)
        return rng, spikes, labels

    def evaluate(sample_size=64):
        """Return (acc, w_err, cos_o) on a held-out subset."""
        idx = jax.random.permutation(data_key, x_test.shape[0])[:args.val_batch]
        imgs = patch_images(jnp.asarray(x_test[idx], dtype=DTYPE), args.patch_px)
        labels = jnp.asarray(y_test[idx], dtype=jnp.int32)
        _, enc = jax.random.split(jax.random.key(1234))
        spikes = poisson_encode(imgs, args.T, enc)          # (T, val_batch, N, D)
        spikes = spikes.transpose(1, 0, 2, 3)               # (val_batch, T, N, D)

        logits = jit_forward_eval(noiser_params, params, spikes)
        acc = accuracy_from_logits(logits, labels)

        # --- equivalence metrics on a small subset (Section 五) --------------
        sub_spikes = spikes[:sample_size]                   # (S, T, N, D)
        sub_imgs = sub_spikes.mean(axis=1, keepdims=True)   # (S, 1, N, D) mean-rate
        w_err, cos_o = compute_equivalence(
            frozen_params, params, es_tree_key, sub_imgs, route=args.route,
            frozen_noiser=frozen_noiser, noiser_params=noiser_params)
        return float(acc), float(w_err), float(cos_o)

    # --- warm-up ------------------------------------------------------------
    print("Compiling...")
    t0 = time.time()
    warm = jnp.zeros((args.batch, args.T, num_tokens, token_in_dim), dtype=DTYPE)
    warm_iter = (jnp.zeros(args.batch, dtype=jnp.int32),
                 jnp.arange(args.batch, dtype=jnp.int32))
    _ = jit_forward(noiser_params, params, warm_iter, warm)
    _ = jit_forward_eval(noiser_params, params,
                         jnp.zeros((args.val_batch, args.T, num_tokens, token_in_dim)))
    jit_update(noiser_params, params, jnp.zeros(args.batch), warm_iter)
    print(f"Warm-up done in {time.time() - t0:.1f}s")

    # --- training loop ------------------------------------------------------
    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            f.write("epoch,train_acc,val_acc,w_err,cos_o\n")

    best = 0.0
    for epoch in range(args.epochs):
        data_key, spikes, labels = next_batch(data_key)
        iterinfo = (jnp.full(args.batch, epoch, dtype=jnp.int32),
                    jnp.arange(args.batch, dtype=jnp.int32))
        logits = jit_forward(noiser_params, params, iterinfo, spikes)
        raw = reward_from_logits(logits, labels, args.reward)
        fitnesses = NOISER.convert_fitnesses(frozen_noiser, noiser_params, raw)
        noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)

        train_acc = float(accuracy_from_logits(logits, labels))
        msg = f"epoch {epoch:3d} | train_acc {train_acc:.3f}"

        if epoch % args.validate_every == 0 or epoch == args.epochs - 1:
            acc, w_err, cos_o = evaluate()
            best = max(best, acc)
            msg += (f" | val_acc {acc:.3f} (best {best:.3f}) | "
                    f"w_err {w_err:.3f} | cos_o {cos_o:.3f}")
            if args.csv_out:
                with open(args.csv_out, "a", newline="") as f:
                    f.write(f"{epoch},{float(train_acc):.6f},{acc:.6f},"
                            f"{w_err:.6f},{cos_o:.6f}\n")
        print(msg, flush=True)

    print(f"Done. route={args.route} best_val={best:.3f}")


def compute_equivalence(frozen_params, params, es_tree_key, mean_rate,
                        route="hopfield", frozen_noiser=None, noiser_params=None):
    """Compare SNN attention weights vs reference Softmax on mean-rate tokens.

    ``mean_rate``: (S, 1, num_tokens, d) — a single-timestep rate input so that the
    same (iterinfo=None) noised Q/K/V projections feed both the SNN core and the
    reference. Returns float (weight_error, output_cosine).
    """
    from hyperscalees.models.base_model import CommonParams
    from hyperscalees.models.snn_attention import (
        _mk_qkv, softmax_attention, hopfield_attention, meanfield_attention,
    )

    def attn_core(q, k, v, beta):
        if route == "hopfield":
            return hopfield_attention(q, k, v, g_inh=frozen_params["g_inh"],
                                      tau_a=frozen_params["tau_a"],
                                      beta=beta, n_iter=frozen_params["n_iter"])
        return meanfield_attention(q, k, v, gamma=frozen_params["gamma"],
                                   beta=beta, n_iter=frozen_params["n_iter"])

    def one(x):
        # iterinfo=None => no LoRA noise; identical q/k/v for SNN core and reference.
        common = CommonParams(NOISER, frozen_noiser, noiser_params,
                              frozen_params, params, es_tree_key, None)
        q, k, v = _mk_qkv(common, x)
        beta = jax.nn.softplus(params["beta"])
        p_snn, o_snn = attn_core(q, k, v, beta)
        p_ref, o_ref = softmax_attention(q, k, v, beta)
        w_err = jnp.mean(jnp.abs(p_snn - p_ref))
        cos_num = jnp.sum(o_snn * o_ref)
        cos_den = jnp.maximum(
            jnp.sqrt(jnp.sum(o_snn ** 2) * jnp.sum(o_ref ** 2)), 1e-6)
        return w_err, cos_num / cos_den

    w_errs, cos_os = jax.vmap(one)(mean_rate)
    return jnp.mean(w_errs), jnp.mean(cos_os)


if __name__ == "__main__":
    main()
