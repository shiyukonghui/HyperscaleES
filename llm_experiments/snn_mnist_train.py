"""Single-device (CPU/GPU) training of an SNN on MNIST using HyperscaleES' evolutionary
noiser (no backprop through the spike function).

This mirrors the minimal loop in tests/end_to_end_test.py:

    batch -> poisson encode -> noised forward (vmap over batch, iterinfo)
         -> fitness -> convert_fitnesses -> do_updates -> params

Run:
    python -m llm_experiments.snn_mnist_train
"""

import time

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.snn import SNNModel
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays,
    poisson_encode,
    fitness_from_logits,
    accuracy_from_logits,
)

# --- configuration ----------------------------------------------------------
NOISER = hs.noiser.eggroll.EggRoll
seed = 0
sigma = 0.2
lr = 0.03

num_epochs = 1000
num_envs = 128          # parallel samples ("generations") per epoch
T = 8                  # timesteps for the SNN recurrence / poisson length

in_dim = 28 * 28
hidden_dims = [128, 128]
num_classes = 10
tau_m = 20.0
v_th = 0.3
noise_rank = 8

validate_every = 10
val_batch = 1024

# --- initialise model / noiser ---------------------------------------------
key = jax.random.key(seed)
model_key, es_key, data_key = jax.random.split(key, 3)

MODEL = SNNModel
frozen_params, params, scan_map, es_map = MODEL.rand_init(
    model_key, in_dim=in_dim, hidden_dims=hidden_dims,
    num_classes=num_classes, tau_m=tau_m, v_th=v_th, dtype=jnp.float32,
)
es_tree_key = simple_es_tree_key(params, es_key, scan_map)
# 最优学习率调度（docs 7.6）：warmup + cosine 退火，显著优于固定 LR。
# 如需固定 LR，把 lr_schedule 换成 lr 即可。
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=lr, warmup_steps=200,
    decay_steps=num_epochs, end_value=lr * 0.1,
)
frozen_noiser_params, noiser_params = NOISER.init_noiser(
    params, sigma, lr_schedule,
    solver=optax.adamw, solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=noise_rank,
)

# noised generation (iterinfo per sample)
jit_forward = jax.jit(
    jax.vmap(
        lambda n, p, i, x: MODEL.forward(
            NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x
        ),
        in_axes=(None, None, 0, 0),
    )
)
# clean evaluation (iterinfo=None)
jit_forward_eval = jax.jit(
    jax.vmap(
        lambda n, p, x: MODEL.forward(
            NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, None, x
        ),
        in_axes=(None, None, 0),
    )
)
jit_update = jax.jit(
    lambda n, p, f, i: NOISER.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map)
)

# --- data -------------------------------------------------------------------
mnist_data_dir = r"D:\Rust\snn_t1\mnist_data"  # local IDX MNIST files (None -> HF download)
x_train, y_train = get_mnist_arrays("train", data_dir=mnist_data_dir)
x_test, y_test = get_mnist_arrays("test", data_dir=mnist_data_dir)
n_train = x_train.shape[0]


def next_batch(rng_key):
    """Return (spikes (num_envs, T, in_dim), labels (num_envs,)) from the training set,
    freshly poisson-encoded with an independent key."""
    rng_key, sub = jax.random.split(rng_key)
    idx = jax.random.permutation(sub, n_train)[:num_envs]
    imgs = jnp.asarray(x_train[idx], dtype=jnp.float32)
    labels = jnp.asarray(y_train[idx], dtype=jnp.int32)
    _, enc = jax.random.split(rng_key)
    spikes = poisson_encode(imgs, T, enc)
    return rng_key, spikes, labels


def evaluate():
    idx = jax.random.permutation(key, x_test.shape[0])[:val_batch]
    imgs = jnp.asarray(x_test[idx], dtype=jnp.float32)
    labels = jnp.asarray(y_test[idx], dtype=jnp.int32)
    _, enc = jax.random.split(jax.random.key(1234))
    spikes = poisson_encode(imgs, T, enc)  # (T, val_batch, in_dim)
    spikes = spikes.transpose(1, 0, 2)     # (val_batch, T, in_dim)
    logits = jit_forward_eval(noiser_params, params, spikes)
    return accuracy_from_logits(logits, labels)


# --- training loop ----------------------------------------------------------
print("Compiling...")
t0 = time.time()
# warm-up compilations
warm_spikes = jnp.zeros((num_envs, T, in_dim), dtype=jnp.float32)
warm_iter = (jnp.zeros(num_envs, dtype=jnp.int32), jnp.arange(num_envs, dtype=jnp.int32))
_print = jit_forward(noiser_params, params, warm_iter, warm_spikes)
_print_eval = jit_forward_eval(noiser_params, params, jnp.zeros((val_batch, T, in_dim)))
jit_update(noiser_params, params, jnp.zeros(num_envs), warm_iter)
print(f"Warm-up done in {time.time() - t0:.1f}s")

for epoch in range(num_epochs):
    data_key, spikes, labels = next_batch(data_key)
    # forward wants (batch, T, in_dim)
    spikes = spikes.transpose(1, 0, 2)
    iterinfo = (jnp.full(num_envs, epoch, dtype=jnp.int32), jnp.arange(num_envs, dtype=jnp.int32))

    logits = jit_forward(noiser_params, params, iterinfo, spikes)
    raw_scores = fitness_from_logits(logits, labels)
    fitnesses = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
    noiser_params, new_params = jit_update(noiser_params, params, fitnesses, iterinfo)

    grad_norm = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree.map(
            lambda old_p, new_p: jnp.mean((old_p - new_p) ** 2), params, new_params
        ),
    )
    params = new_params

    train_acc = jnp.mean(raw_scores)
    msg = (f"epoch {epoch:3d} | train_acc {train_acc:.3f} | "
           f"param_delta {grad_norm:.6f}")
    if epoch % validate_every == 0:
        val_acc = evaluate()
        msg += f" | val_acc {val_acc:.3f}"
    print(msg)

print("Done.")
