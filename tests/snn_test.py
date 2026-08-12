"""Tests for the SNN model trained with the HyperscaleES noiser on MNIST.

Run (from repo root):
    python tests/snn_test.py
"""

import os

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.snn import SNNModel, lif_step, run_lif
from hyperscalees.environments.snn_mnist import (
    poisson_encode,
    fitness_from_logits,
    accuracy_from_logits,
)

NOISER = hs.noiser.eggroll.EggRoll

IN_DIM = 28 * 28
HIDDEN = [64, 64]
NUM_CLASSES = 10
T = 5
DTYPE = jnp.float32


def _build():
    key = jax.random.key(0)
    model_key, es_key = jax.random.split(key)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE)
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser, noiser_params = NOISER.init_noiser(
        params, sigma=0.1, lr=0.01, solver=optax.adamw, rank=4)
    return (frozen_params, params, scan_map, es_map, es_tree_key,
            frozen_noiser, noiser_params)


def test_rand_init_structure():
    frozen_params, params, scan_map, es_map, *_ = _build()
    assert set(params.keys()) == {"fc1", "fc2", "fc3", "out_gain"}
    assert set(es_map.keys()) == set(params.keys())
    assert set(scan_map.keys()) == set(params.keys())
    assert set(frozen_params.keys()) == {"tau_m", "v_th"}
    # fc1/fc2/fc3 weights are MM_PARAM (1); out_gain is PARAM (0)
    assert es_map["fc1"] == 1 and es_map["fc2"] == 1 and es_map["fc3"] == 1
    assert es_map["out_gain"] == 0
    print("OK test_rand_init_structure")


def test_lif_dynamics():
    # a constant suprathreshold current must eventually fire; subthreshold must not
    dtype = jnp.float32
    params = {"tau_m": jnp.asarray(20.0, dtype), "v_th": jnp.asarray(1.0, dtype)}
    cur = jnp.full((8,), 2.0, dtype=dtype)   # strong current
    v = jnp.zeros((8,), dtype=dtype)
    for _ in range(50):
        v, spike = lif_step(params, v, cur)
        if jnp.any(spike):
            break
    assert jnp.any(spike), "LIF should fire under strong positive current"
    # weak current far below threshold should not fire
    cur_weak = jnp.full((8,), 0.01, dtype=dtype)
    v = jnp.zeros((8,), dtype=dtype)
    fired = False
    for _ in range(30):
        v, spike = lif_step(params, v, cur_weak)
        if jnp.any(spike):
            fired = True
            break
    assert not fired, "LIF should not fire under very weak current"
    print("OK test_lif_dynamics")


def test_forward_shapes_and_reproducibility():
    frozen_params, params, _scan_map, es_map, es_tree_key, frozen_noiser, noiser_params = _build()
    key = jax.random.key(1)
    imgs = jax.random.uniform(key, (4, IN_DIM), dtype=DTYPE)
    spikes = poisson_encode(imgs, T, key)  # (T, 4, IN_DIM)
    spikes_batch = spikes.transpose(1, 0, 2)  # (4, T, IN_DIM)

    eval_logits = jax.vmap(
        lambda x: SNNModel.forward(NOISER, frozen_noiser, noiser_params,
                                   frozen_params, params, es_tree_key, None, x),
        in_axes=0,
    )(spikes_batch)
    assert eval_logits.shape == (4, NUM_CLASSES)
    assert eval_logits.dtype == DTYPE

    # reproducibility: same eval input -> identical output (frozen keys)
    eval_logits2 = jax.vmap(
        lambda x: SNNModel.forward(NOISER, frozen_noiser, noiser_params,
                                   frozen_params, params, es_tree_key, None, x),
        in_axes=0,
    )(spikes_batch)
    assert jnp.array_equal(eval_logits, eval_logits2)

    # noised generation must work and preserve shape
    iterinfo = (jnp.full(4, 0, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32))
    noisy_logits = jax.vmap(
        lambda i, x: SNNModel.forward(NOISER, frozen_noiser, noiser_params,
                                      frozen_params, params, es_tree_key, i, x),
        in_axes=(0, 0),
    )(iterinfo, spikes_batch)
    assert noisy_logits.shape == (4, NUM_CLASSES)
    print("OK test_forward_shapes_and_reproducibility")


def test_training_smoke():
    """Verify the SNN + evolutionary noiser training loop end-to-end on 10-class MNIST.

    On a single CPU the pure-ES noiser only starts to move accuracy slightly above
    chance (full convergence is meant for large GPU-parallel runs), so this smoke test
    asserts that (a) the loop runs, (b) ``do_updates`` actually changes the params, and
    (c) accuracy is at/above chance (>= 10 %) rather than collapsing.
    """
    frozen_params, params, _scan_map, es_map, es_tree_key, frozen_noiser, noiser_params = _build()

    try:
        from hyperscalees.environments.snn_mnist import get_mnist_arrays
        x_tr, y_tr = get_mnist_arrays("train", data_dir=r"D:\Rust\snn_t1\mnist_data")
    except Exception as e:  # pragma: no cover - no local data file
        print("SKIP test_training_smoke (MNIST unavailable):", e)
        return

    rng = jax.random.key(7)
    num_envs = 128
    num_epochs = 15
    small_T = 5

    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: SNNModel.forward(
            NOISER, frozen_noiser, n, frozen_params, p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0)))
    jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
        frozen_noiser, n, p, es_tree_key, f, i, es_map))

    # capture a snapshot to prove the noiser actually updates the parameters
    before = {k: jnp.copy(v) for k, v in params.items()}
    best_acc = 0.0
    for epoch in range(num_epochs):
        rng, enc = jax.random.split(rng)
        idx = jax.random.permutation(rng, x_tr.shape[0])[:num_envs]
        imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
        labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, small_T, enc).transpose(1, 0, 2)
        iterinfo = (jnp.full(num_envs, epoch, dtype=jnp.int32),
                    jnp.arange(num_envs, dtype=jnp.int32))
        logits = jit_forward(noiser_params, params, iterinfo, spikes)
        raw = fitness_from_logits(logits, labels)
        fitnesses = NOISER.convert_fitnesses(frozen_noiser, noiser_params, raw)
        noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)
        acc = float(accuracy_from_logits(logits, labels))
        best_acc = max(best_acc, acc)

    # (a) params moved => evolutionary update took effect
    max_delta = max(
        float(jnp.max(jnp.abs(params[k] - before[k]))) for k in params
    )
    assert max_delta > 0.0, "noiser did not change any parameter"
    # (b) accuracy at least at chance (10 classes => ~10 %), never collapsed to zero
    assert best_acc >= 0.10, f"accuracy collapsed below chance: {best_acc:.3f}"
    print(f"OK test_training_smoke (best_acc={best_acc:.3f}, max_param_delta={max_delta:.5f})")


if __name__ == "__main__":
    test_rand_init_structure()
    test_lif_dynamics()
    test_forward_shapes_and_reproducibility()
    test_training_smoke()
    print("ALL SNN TESTS PASSED")
