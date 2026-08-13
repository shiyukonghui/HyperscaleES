"""固定 LR 下，进化批次（并行候选数 num_envs）对学习能力的放大效应实验。

固定：LR=0.03（固定，不衰减）、500 次进化更新（epoch）、T=8、硬 0/1 奖励。
变量：num_envs ∈ {32, 64, 128, 256, 512}（每次进化更新使用的并行样本/候选数）。

ES 理论：噪声梯度误差 ∝ σ / sqrt(N)。批次越大，单次更新的梯度估计越准，
预期在同样 500 次更新下 val_acc / best_train 越高 —— 即"批次放大学习能力"。

用法：
    .\\.venv\\Scripts\\python.exe exp_batch_effect.py [epochs]
"""

import sys
import time

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.snn import SNNModel
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays, poisson_encode, fitness_from_logits, accuracy_from_logits,
)

NOISER = hs.noiser.eggroll.EggRoll
DTYPE = jnp.float32
IN_DIM = 28 * 28
HIDDEN = [128, 128]
NUM_CLASSES = 10
MNIST_DIR = r"D:\Rust\snn_t1\mnist_data"

num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
T = 8
sigma = 0.2
lr = 0.03
seed = 0
BATCHES = [1024, 2048, 4096]

# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]
VAL = 1024

results = {}
for num_envs in BATCHES:
    print(f"\n===== num_envs (批次) = {num_envs} =====", flush=True)
    t0 = time.time()

    key = jax.random.key(seed)
    model_key, es_key, data_key = jax.random.split(key, 3)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser, noiser_params = NOISER.init_noiser(
        params, sigma, lr, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=8,
    )
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: SNNModel.forward(NOISER, frozen_noiser, n, frozen_params,
                                            p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0)))
    jit_forward_eval = jax.jit(jax.vmap(
        lambda n, p, x: SNNModel.forward(NOISER, frozen_noiser, n, frozen_params,
                                         p, es_tree_key, None, x),
        in_axes=(None, None, 0)))
    jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
        frozen_noiser, n, p, es_tree_key, f, i, es_map))

    def eval_acc():
        idx = jax.random.permutation(jax.random.key(1), x_te.shape[0])[:VAL]
        imgs = jnp.asarray(x_te[idx], dtype=DTYPE)
        labels = jnp.asarray(y_te[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, T, jax.random.key(1)).transpose(1, 0, 2)
        logits = jit_forward_eval(noiser_params, params, spikes)
        return float(accuracy_from_logits(logits, labels))

    # warm-up
    ws = jnp.zeros((num_envs, T, IN_DIM), dtype=DTYPE)
    wi = (jnp.zeros(num_envs, dtype=jnp.int32), jnp.arange(num_envs, dtype=jnp.int32))
    jit_forward(noiser_params, params, wi, ws)
    jit_update(noiser_params, params, jnp.zeros(num_envs), wi)

    best = 0.0
    last = 0.0
    for epoch in range(num_epochs):
        data_key, enc, per = jax.random.split(data_key, 3)
        idx = jax.random.permutation(per, n_train)[:num_envs]
        imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
        labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, T, enc).transpose(1, 0, 2)
        it = (jnp.full(num_envs, epoch, dtype=jnp.int32),
              jnp.arange(num_envs, dtype=jnp.int32))
        logits = jit_forward(noiser_params, params, it, spikes)
        raw = fitness_from_logits(logits, labels)
        fits = NOISER.convert_fitnesses(frozen_noiser, noiser_params, raw)
        noiser_params, params = jit_update(noiser_params, params, fits, it)
        acc = float(accuracy_from_logits(logits, labels))
        best = max(best, acc)
        if epoch == num_epochs - 1 or epoch % 125 == 0:
            last = eval_acc()
            print(f"  epoch {epoch:4d} | val {last:.3f} | best_train {best:.3f}", flush=True)

    elapsed = time.time() - t0
    results[num_envs] = (last, best, elapsed)
    print(f"  [num_envs={num_envs}] val_acc={last:.3f} best_train={best:.3f} ({elapsed:.0f}s)")

print("\n===== 批次放大效应（固定 LR=0.03, 500 次更新） =====")
print(f"{'num_envs(批次)':>16} | {'val_acc':>8} | {'best_train':>10} | {'用时(s)':>8}")
for n, (last, best, el) in results.items():
    print(f"{n:16d} | {last:8.3f} | {best:10.3f} | {el:8.1f}")
