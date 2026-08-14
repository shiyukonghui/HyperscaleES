"""Tests for the SNN Attention models (Hopfield / Mean-field routes) trained on
patched-MNIST with the HyperscaleES noiser.

Run (from repo root):
    python tests/snn_attention_test.py
"""

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.snn_attention import (
    HopfieldAttnSNN,
    MeanFieldAttnSNN,
    lif_step,
    run_lif,
    softmax_attention,
)
from hyperscalees.environments.snn_mnist import (
    poisson_encode,
    fitness_from_logits,
    accuracy_from_logits,
)

NOISER = hs.noiser.eggroll.EggRoll
TOKEN_IN = 49          # 7x7 patch
NUM_TOKENS = 16        # 4x4 patches
NUM_CLASSES = 10
D_HEAD = 16
T = 6
DTYPE = jnp.float32


def _factory(variant):
    if variant == "hopfield":
        return HopfieldAttnSNN
    return MeanFieldAttnSNN


def _build(variant="hopfield"):
    key = jax.random.key(0)
    model_key, es_key = jax.random.split(key)
    MODEL = _factory(variant)
    frozen_params, params, scan_map, es_map = MODEL.rand_init(
        model_key, token_in_dim=TOKEN_IN, num_tokens=NUM_TOKENS,
        num_classes=NUM_CLASSES, d_head=D_HEAD, n_iter=6, tau_m=20.0,
        proj_gain=2.0, dtype=DTYPE)
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser, noiser_params = NOISER.init_noiser(
        params, sigma=0.1, lr=0.01, solver=optax.adamw, rank=4)
    return (frozen_params, params, scan_map, es_map, es_tree_key,
            frozen_noiser, noiser_params)


def test_rand_init_structure():
    for variant in ("hopfield", "meanfield"):
        frozen_params, params, *_ = _build(variant)
        assert set(params.keys()) == {"q", "k", "v", "out", "out_gain", "beta"}
        assert set(frozen_params.keys()) >= {"tau_m", "proj_gain", "n_iter"}
        # Q/K/V/readout are MM_PARAM (1); scalars are PARAM (0)
        assert all(params[m] is not None for m in ("q", "k", "v", "out"))
    print("OK test_rand_init_structure")


def test_lif_dynamics():
    params = {"tau_m": jnp.asarray(20.0, DTYPE), "v_th": jnp.asarray(1.0, DTYPE)}
    cur = jnp.full((8,), 2.0, dtype=DTYPE)
    v = jnp.zeros((8,), dtype=DTYPE)
    fired = False
    for _ in range(50):
        v, spike = lif_step(params, v, cur)
        if jnp.any(spike):
            fired = True
            break
    assert fired, "LIF should fire under strong current"
    print("OK test_lif_dynamics")


def test_forward_shapes_and_reproducibility():
    frozen_params, params, _sm, _em, es_tree_key, frozen_noiser, noiser_params = _build("hopfield")
    key = jax.random.key(1)

    def spikes_batch(batch=4):
        imgs = jax.random.uniform(key, (batch, NUM_TOKENS, TOKEN_IN), dtype=DTYPE)
        sp = poisson_encode(imgs, T, key)            # (T, batch, N, D)
        return sp.transpose(1, 0, 2, 3)

    xb = spikes_batch(4)
    eval_logits = jax.vmap(
        lambda x: HopfieldAttnSNN.forward(NOISER, frozen_noiser, noiser_params,
                                          frozen_params, params, es_tree_key, None, x),
        in_axes=0,
    )(xb)
    assert eval_logits.shape == (4, NUM_CLASSES) and eval_logits.dtype == DTYPE

    eval_logits2 = jax.vmap(
        lambda x: HopfieldAttnSNN.forward(NOISER, frozen_noiser, noiser_params,
                                          frozen_params, params, es_tree_key, None, x),
        in_axes=0,
    )(xb)
    assert jnp.array_equal(eval_logits, eval_logits2)

    iterinfo = (jnp.full(4, 0, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32))
    noisy = jax.vmap(
        lambda i, x: HopfieldAttnSNN.forward(NOISER, frozen_noiser, noiser_params,
                                             frozen_params, params, es_tree_key, i, x),
        in_axes=(0, 0),
    )(iterinfo, xb)
    assert noisy.shape == (4, NUM_CLASSES)
    print("OK test_forward_shapes_and_reproducibility")


def test_attention_weights_normalize():
    """SNN attention weights from both routes must sum to 1."""
    key = jax.random.key(3)
    q = jax.random.normal(key, (NUM_TOKENS, D_HEAD), dtype=DTYPE)
    k = jax.random.normal(jax.random.split(key)[1], (NUM_TOKENS, D_HEAD), dtype=DTYPE)
    v = jax.random.normal(jax.random.split(key)[2], (NUM_TOKENS, D_HEAD), dtype=DTYPE)
    beta = 0.5
    frozen = {"g_inh": 0.5, "tau_a": 5.0, "gamma": 0.5, "n_iter": 8}

    from hyperscalees.models.snn_attention import hopfield_attention, meanfield_attention
    p_h, _ = hopfield_attention(q, k, v, g_inh=frozen["g_inh"], tau_a=frozen["tau_a"],
                                beta=beta, n_iter=frozen["n_iter"])
    a_m, _ = meanfield_attention(q, k, v, gamma=frozen["gamma"], beta=beta,
                                 n_iter=frozen["n_iter"])
    assert abs(float(jnp.sum(p_h)) - 1.0) < 1e-3
    assert abs(float(jnp.sum(a_m)) - 1.0) < 1e-3
    print("OK test_attention_weights_normalize")


def test_training_smoke():
    """Both routes must train end-to-end with the ES noiser (params move, acc >= chance)."""
    from hyperscalees.environments.snn_mnist import get_mnist_arrays
    try:
        x_tr, y_tr = get_mnist_arrays("train", data_dir=r"D:\Rust\snn_t1\mnist_data")
    except Exception as e:  # pragma: no cover
        print("SKIP test_training_smoke (MNIST unavailable):", e)
        return

    def patch(images):
        side = 28 // 7
        images = images.reshape(images.shape[0], side, 7, side, 7)
        return images.transpose(0, 1, 3, 2, 4).reshape(images.shape[0], 16, 49)

    rng = jax.random.key(7)
    num_envs = 64
    num_epochs = 8
    small_T = 4

    for variant in ("hopfield", "meanfield"):
        MODEL = _factory(variant)
        frozen_params, params, _sm, es_map, es_tree_key, frozen_noiser, noiser_params = _build(variant)

        jit_forward = jax.jit(jax.vmap(
            lambda n, p, i, x: MODEL.forward(
                NOISER, frozen_noiser, n, frozen_params, p, es_tree_key, i, x),
            in_axes=(None, None, 0, 0)))
        jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
            frozen_noiser, n, p, es_tree_key, f, i, es_map))

        before = {k: jnp.copy(v) for k, v in params.items()}
        best_acc = 0.0
        for epoch in range(num_epochs):
            rng, enc = jax.random.split(rng)
            idx = jax.random.permutation(rng, x_tr.shape[0])[:num_envs]
            imgs = patch(jnp.asarray(x_tr[idx], dtype=DTYPE))
            labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
            spikes = poisson_encode(imgs, small_T, enc).transpose(1, 0, 2, 3)
            iterinfo = (jnp.full(num_envs, epoch, dtype=jnp.int32),
                        jnp.arange(num_envs, dtype=jnp.int32))
            logits = jit_forward(noiser_params, params, iterinfo, spikes)
            raw = fitness_from_logits(logits, labels)
            fitnesses = NOISER.convert_fitnesses(frozen_noiser, noiser_params, raw)
            noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)
            best_acc = max(best_acc, float(accuracy_from_logits(logits, labels)))

        max_delta = max(float(jnp.max(jnp.abs(params[k] - before[k]))) for k in params)
        assert max_delta > 0.0, f"{variant}: noiser did not change any parameter"
        assert best_acc >= 0.10, f"{variant}: accuracy collapsed: {best_acc:.3f}"
        print(f"OK test_training_smoke ({variant}, best_acc={best_acc:.3f})")


def test_softmax_reference():
    """Reference softmax attention returns normalized weights and valid readout."""
    q = jnp.ones((NUM_TOKENS, D_HEAD), dtype=DTYPE)
    k = jnp.ones((NUM_TOKENS, D_HEAD), dtype=DTYPE)
    v = jnp.ones((NUM_TOKENS, D_HEAD), dtype=DTYPE)
    p, o = softmax_attention(q, k, v, beta=1.0)
    assert abs(float(jnp.sum(p)) - 1.0) < 1e-4
    assert o.shape == v.shape
    print("OK test_softmax_reference")


if __name__ == "__main__":
    test_rand_init_structure()
    test_lif_dynamics()
    test_forward_shapes_and_reproducibility()
    test_attention_weights_normalize()
    test_softmax_reference()
    test_training_smoke()
    print("ALL SNN ATTENTION TESTS PASSED")
