"""LoRA rank 单变量扫描实验：固定 batch 比例、固定 LR，比较不同 rank 的效果（确定收益边际点）。

设计：
  - 固定批次比例 BATCH_RATIO（默认 0.2 × 训练集 = 12000），固定 LR（0.03）、硬 0/1 奖励。
  - 变量 = LoRA rank（EggRoll 每次更新使用的低秩扰动/更新子空间维度）。
  - 每个 rank 用相同 seed / sigma / epoch 数（默认 1000）跑固定次数更新，横向比较。
  - rank 会影响扰动幅度（EggRoll 用 sigma/sqrt(rank) 归一化），因此 rank 是完整单变量。
  - 结果追加写盘（results_rank_sweep.csv），便于确认"收益边际点"。

用法：
    python exp_rank_sweep.py [rank0 rank1 ...]
示例：
    python exp_rank_sweep.py
    python exp_rank_sweep.py 4 8 16 32 64 128
"""

import csv
import os
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
MNIST_DIR = os.environ.get("MNIST_DIR") or r"D:\Rust\snn_t1\mnist_data"
# WSL 下 Windows 盘符路径不生效，自动回退到 /mnt/d 挂载路径（原生 Windows 行为不变）
if not os.path.isdir(MNIST_DIR) and os.path.isdir("/mnt/d/Rust/snn_t1/mnist_data"):
    MNIST_DIR = "/mnt/d/Rust/snn_t1/mnist_data"

T = 8
sigma = 0.2
lr = 0.03          # 固定 LR —— 隔离 rank 变量
seed = 0
VAL = 1024
EVAL_EVERY = 50    # 周期性测试集评估间隔（epoch）

# 单变量配置
BATCH_RATIO = 0.2          # 固定批次比例（相对训练集大小，0.2 × 60000 = 12000）
MAX_EPOCHS = 1000          # 每个 rank 固定运行的进化更新次数（含 10 次标定更新）
RANKS = [int(r) for r in sys.argv[1:]] or [4, 8, 16, 32, 64, 128]  # rank 扫描点

CSV_PATH = "results_rank_sweep.csv"

# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]
num_envs = int(round(BATCH_RATIO * n_train))

# ---- model（rank 无关的公共结构） -------------------------------------------
key = jax.random.key(seed)
model_key, es_key, data_key = jax.random.split(key, 3)
frozen_params, params, scan_map, es_map = SNNModel.rand_init(
    model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
    num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
)
total_params = sum(p.size for p in jax.tree.leaves(params))


def build(rank):
    """按 rank 构建 noiser 与 JIT 函数。

    rank 决定 LoRA 扰动/更新子空间维度（且扰动幅度 sigma/sqrt(rank) 随 rank 变化），
    因此不同 rank 必须重建 noiser 并重新 JIT 编译。
    """
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser, noiser_params = NOISER.init_noiser(
        params, sigma, lr, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=rank,
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
    return frozen_noiser, noiser_params, jit_forward, jit_forward_eval, jit_update


def eval_acc(frozen_noiser, jit_forward_eval):
    """用固定测试子集评估当前参数（每个 rank 独立的 noiser 状态）。"""
    global noiser_params, params
    idx = jax.random.permutation(jax.random.key(1), x_te.shape[0])[:VAL]
    imgs = jnp.asarray(x_te[idx], dtype=DTYPE)
    labels = jnp.asarray(y_te[idx], dtype=jnp.int32)
    spikes = poisson_encode(imgs, T, jax.random.key(1)).transpose(1, 0, 2)
    logits = jit_forward_eval(noiser_params, params, spikes)
    return float(accuracy_from_logits(logits, labels))


def make_step(data_key, num_envs, frozen_noiser, jit_forward, jit_update):
    """返回 (step(epoch) -> epoch_idx, acc_train) 闭包，每次采样新批次。"""
    def step(epoch):
        global data_key, noiser_params, params
        data_key2, enc, per = jax.random.split(data_key, 3)
        data_key = data_key2
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
        return epoch, float(accuracy_from_logits(logits, labels))
    return step


def run_point(rank, num_envs, max_epochs):
    """在固定 batch 上，用给定 rank 跑 max_epochs 次进化更新。"""
    global data_key, noiser_params, params
    print(f"\n===== [rank sweep] rank={rank} batch={num_envs} "
          f"(lr={lr}, sigma={sigma}) max_epochs={max_epochs} =====", flush=True)
    t0 = time.time()
    frozen_noiser, noiser_params, jit_forward, jit_forward_eval, jit_update = build(rank)
    step = make_step(data_key, num_envs, frozen_noiser, jit_forward, jit_update)

    # 标定 s/epoch（10 次真实更新，含 JIT）
    cal = 10
    start = time.time()
    best = 0.0
    for e in range(cal):
        _, acc = step(e)
        best = max(best, acc)
    cal_s = (time.time() - start) / cal
    print(f"  calib: {cal_s:.3f}s/epoch -> est {cal_s * num_envs:.1f} ms/sample/update",
          flush=True)

    epoch = cal
    last_val = 0.0
    while epoch < max_epochs:
        _, acc = step(epoch)
        best = max(best, acc)
        epoch += 1
        if epoch % EVAL_EVERY == 0:
            last_val = eval_acc(frozen_noiser, jit_forward_eval)
            print(f"  epoch {epoch:5d} | val {last_val:.3f} | best_train {best:.3f} "
                  f"| {(time.time()-t0):.0f}s", flush=True)

    last_val = eval_acc(frozen_noiser, jit_forward_eval) if last_val == 0.0 and epoch > cal else last_val
    elapsed = time.time() - t0
    return epoch, last_val, best, elapsed, cal_s


def main():
    print(f"total_params = {total_params}", flush=True)
    print(f"batch = {num_envs} ({BATCH_RATIO}x train), lr = {lr}, sigma = {sigma}, "
          f"max_epochs/point = {MAX_EPOCHS}", flush=True)
    print(f"rank points: {RANKS}", flush=True)

    # 初始化结果 CSV（含表头）
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["rank", "batch", "epochs", "val_acc", "best_train",
                        "s_per_epoch", "time_s"])
        f.flush()

    results = []
    for rank in RANKS:
        epochs, last_val, best, elapsed, cal_s = run_point(rank, num_envs, MAX_EPOCHS)
        results.append((rank, num_envs, epochs, last_val, best, elapsed))
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([rank, num_envs, epochs, last_val, best,
                                    round(cal_s, 5), round(elapsed)])
        print(f"  [rank={rank}] epochs={epochs} val_acc={last_val:.3f} "
              f"best_train={best:.3f} ({elapsed:.0f}s)", flush=True)

    print("\n===== LoRA rank 扫描汇总（固定 batch/LR/epoch） =====")
    print(f"{'rank':>6} | {'batch':>6} | {'epochs':>6} | {'val_acc':>8} | {'best_train':>10} | {'用时(s)':>8}")
    for rank_r, batch_n, epochs, last_val, best, elapsed in results:
        print(f"{rank_r:6d} | {batch_n:6d} | {epochs:6d} | {last_val:8.3f} | "
              f"{best:10.3f} | {elapsed:8.0f}")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
