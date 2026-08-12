"""长时间 CPU 训练实验：观察 10 类 MNIST 准确率与训练时间的关系。

每隔若干 epoch 记录一次累计训练时间(wall-clock)与 train/val 准确率，
结束后打印 'epoch | 累计用时(s) | train_acc | val_acc' 对照表，
用于考察“准确率是否随训练时间上升（近似成正比/单调）”。

用法：
    .\\.venv\\Scripts\\python.exe exp_train_time.py [num_epochs] [num_envs]
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

# ---- config ---------------------------------------------------------------
NOISER = hs.noiser.eggroll.EggRoll
DTYPE = jnp.float32
IN_DIM = 28 * 28
HIDDEN = [128, 128]
NUM_CLASSES = 10
MNIST_DIR = r"D:\Rust\snn_t1\mnist_data"

num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 128
T = int(sys.argv[3]) if len(sys.argv) > 3 else 8
eval_every = 25

sigma = 0.2
lr = 0.03
seed = 0

key = jax.random.key(seed)
model_key, es_key, data_key = jax.random.split(key, 3)

MODEL = SNNModel
frozen_params, params, scan_map, es_map = MODEL.rand_init(
    model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
    num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
)
es_tree_key = simple_es_tree_key(params, es_key, scan_map)
frozen_noiser, noiser_params = NOISER.init_noiser(
    params, sigma, lr, solver=optax.adamw,
    solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=8,
)

jit_forward = jax.jit(jax.vmap(
    lambda n, p, i, x: MODEL.forward(NOISER, frozen_noiser, n, frozen_params,
                                      p, es_tree_key, i, x),
    in_axes=(None, None, 0, 0)))
jit_forward_eval = jax.jit(jax.vmap(
    lambda n, p, x: MODEL.forward(NOISER, frozen_noiser, n, frozen_params,
                                   p, es_tree_key, None, x),
    in_axes=(None, None, 0)))
jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
    frozen_noiser, n, p, es_tree_key, f, i, es_map))

# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]
VAL = 1024


def eval_acc():
    idx = jax.random.permutation(jax.random.key(1234), x_te.shape[0])[:VAL]
    imgs = jnp.asarray(x_te[idx], dtype=DTYPE)
    labels = jnp.asarray(y_te[idx], dtype=jnp.int32)
    spikes = poisson_encode(imgs, T, jax.random.key(1234)).transpose(1, 0, 2)
    logits = jit_forward_eval(noiser_params, params, spikes)
    return float(accuracy_from_logits(logits, labels))


# warm-up compilation
print("warmup...")
t0 = time.time()
ws = jnp.zeros((num_envs, T, IN_DIM), dtype=DTYPE)
wi = (jnp.zeros(num_envs, dtype=jnp.int32), jnp.arange(num_envs, dtype=jnp.int32))
jit_forward(noiser_params, params, wi, ws)
jit_update(noiser_params, params, jnp.zeros(num_envs), wi)
print(f"warmup {time.time()-t0:.1f}s | start {num_epochs} epochs, num_envs={num_envs}, T={T}")

start_all = time.time()
rows = []
for epoch in range(num_epochs):
    data_key, enc, perm_key = jax.random.split(data_key, 3)
    idx = jax.random.permutation(perm_key, n_train)[:num_envs]
    imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
    labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
    spikes = poisson_encode(imgs, T, enc).transpose(1, 0, 2)
    it = (jnp.full(num_envs, epoch, dtype=jnp.int32),
          jnp.arange(num_envs, dtype=jnp.int32))
    logits = jit_forward(noiser_params, params, it, spikes)
    raw = fitness_from_logits(logits, labels)
    fits = NOISER.convert_fitnesses(frozen_noiser, noiser_params, raw)
    noiser_params, params = jit_update(noiser_params, params, fits, it)
    train_acc = float(accuracy_from_logits(logits, labels))

    if epoch % eval_every == 0 or epoch == num_epochs - 1:
        spent = time.time() - start_all
        va = eval_acc()
        rows.append((epoch, spent, train_acc, va))
        print(f"epoch {epoch:4d} | {spent:8.1f}s | train {train_acc:.3f} | val {va:.3f}")
        sys.stdout.flush()

total = time.time() - start_all
print("\n==== 时间-准确率 汇总 ====")
print(f"总训练时间: {total:.1f}s, num_epochs={num_epochs}, num_envs={num_envs}, T={T}")
print(f"{'epoch':>6} | {'用时(s)':>10} | {'train_acc':>10} | {'val_acc':>8}")
for (e, s, ta, va) in rows:
    print(f"{e:6d} | {s:10.1f} | {ta:10.3f} | {va:8.3f}")

if len(rows) >= 4:
    # 分析曲线形态：对比“前 1/3 段”与“后 1/3 段”的 val_acc 提升速度
    e0, s0, ta0, va0 = rows[0]
    e1, s1, ta1, va1 = rows[-1]
    n = len(rows)
    third = max(1, n // 3)
    va_low = rows[third][3]          # 前 1/3 末点 val_acc
    va_high = rows[-third][3]        # 后 1/3 首点 val_acc
    s_low = rows[third][1]
    s_high = rows[-third][1]
    early_rate = (va_low - va0) / max(1e-6, s_low - s0)      # val/s（前段）
    late_rate = (va1 - va_high) / max(1e-6, s1 - s_high)     # val/s（后段）
    print("\n[对比] 曲线形态分析:")
    print(f"  val_acc: {va0:.3f} -> {va_low:.3f} (前段 {s_low-s0:.1f}s, 速率 {early_rate*100:.2f}%/s)")
    print(f"           {va_high:.3f} -> {va1:.3f} (后段 {s1-s_high:.1f}s, 速率 {late_rate*100:.3f}%/s)")
    if late_rate < early_rate * 0.5:
        verdict = "准确率早期快速上升，后期趋于平台/饱和 —— 与训练时间【不成正比】(边际收益递减)"
    elif va1 > va0 + 0.02:
        verdict = "准确率持续上升（近似正比/近线性）"
    else:
        verdict = "准确率未见明显上升/停滞"
    print(f"  结论: {verdict}")
