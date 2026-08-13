"""硬 0/1 奖励下，结合大批次(默认 2048)的多学习率调度对比实验。

对每种 optax 学习率调度，用硬 0/1 奖励在大批次 num_envs 下跑 num_epochs 次更新，
比较终端 val_acc，研究"批次放大 + 学习率调度"的叠加效应。

用法：
    .\\.venv\\Scripts\\python.exe exp_lr_schedule.py [epochs] [num_envs] [base_lr]
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

num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
base_lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
T = 8
sigma = 0.2
seed = 0


def build_schedule(kind):
    """Return an optax.Schedule and a label for the given learning-rate strategy."""
    if kind == "fixed":
        return base_lr, f"fixed_lr={base_lr}"
    if kind == "linear":
        s = optax.linear_schedule(base_lr, base_lr * 0.1, transition_steps=num_epochs)
        return s, f"linear {base_lr}->{base_lr*0.1}"
    if kind == "cosine":
        s = optax.cosine_decay_schedule(base_lr, decay_steps=num_epochs)
        return s, f"cosine {base_lr}"
    if kind == "exp":
        s = optax.exponential_decay(base_lr, transition_steps=max(10, num_epochs // 5),
                                    decay_rate=0.995)
        return s, f"exp-decay {base_lr} x0.995/{max(10, num_epochs // 5)}"
    if kind == "warmcos":
        s = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=base_lr, warmup_steps=max(10, num_epochs // 10),
            decay_steps=num_epochs, end_value=base_lr * 0.05)
        return s, f"warmup+cosine {base_lr}"
    raise ValueError(kind)


SCHEDULES = ["fixed", "linear", "cosine", "exp", "warmcos"]

# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]
VAL = 1024

results = {}
for kind in SCHEDULES:
    lr_sched, label = build_schedule(kind)
    print(f"\n===== LR schedule: {label} =====", flush=True)
    t0 = time.time()

    # fresh model per schedule (same seed for fairness)
    key = jax.random.key(seed)
    model_key, es_key, data_key = jax.random.split(key, 3)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser, noiser_params = NOISER.init_noiser(
        params, sigma, lr_sched, solver=optax.adamw,
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
        acc = float(accuracy_from_logits(logits, labels))  # hard-01, = raw mean
        best = max(best, acc)
        if epoch == num_epochs - 1 or epoch % 250 == 0:
            last = eval_acc()
            print(f"  epoch {epoch:4d} | val {last:.3f} | best_train {best:.3f}", flush=True)

    elapsed = time.time() - t0
    results[label] = (last, best, elapsed)
    print(f"  [{label}] 4000-step val_acc={last:.3f} (best_train={best:.3f}, {elapsed:.0f}s)")

print("\n===== 学习率调度对比汇总（硬 0/1 奖励） =====")
print(f"{'调度':28s} | {'val_acc@4000步':>14} | {'best_train':>10} | {'用时(s)':>8}")
for label, (last, best, el) in results.items():
    print(f"{label:28s} | {last:14.3f} | {best:10.3f} | {el:8.1f}")
