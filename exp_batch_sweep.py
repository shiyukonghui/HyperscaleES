"""按"批次大小 / 总参数量"比例取多个线性点做批次扫描实验（整夜、预算上限驱动版）。

设计：
  变量 = 批次大小 num_envs（每次进化更新使用的并行候选/样本数）。
  - 取"线性点"：ratio_i ∈ RATIOS（默认 0.1~0.5 均匀 5 点），
    batch_i = round(ratio_i × total_params)，其中 total_params 由模型实时计算。
  - 每个点用相同固定 LR（默认 0.03）、硬 0/1 奖励，跑尽可能多的更新（epoch）；
    epoch 数由"每点时间片 = 总预算 / 点数"自适应决定：先标定 s/epoch，再填满时间片，
    对 CPU 速度波动 / 大批次非线性耗时鲁棒。
  - 全局硬截止时间（默认 15 小时，可传 budget_hours）到时自动停止并输出汇总，
    保证整夜不超时、明早 9 点前结束。
  - 每个点结束后把结果追加写盘（results_batch_sweep.csv）、进度写 progress 文件，
    后台崩溃也能保留已完成点的数据。

用法：
    .\\.venv\\Scripts\\python.exe exp_batch_sweep.py [budget_hours] [r0 r1 ...]
示例：
    .\\.venv\\Scripts\\python.exe exp_batch_sweep.py 15
    .\\.venv\\Scripts\\python.exe exp_batch_sweep.py 15 0.1 0.2 0.3 0.4 0.5
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
MNIST_DIR = r"D:\Rust\snn_t1\mnist_data"

T = 8
sigma = 0.2
lr = 0.03          # 固定 LR —— 隔离"批次大小"变量（与 exp_batch_effect.py 同口径）
seed = 0
VAL = 1024
EVAL_EVERY = 50    # 周期性测试集评估间隔（epoch）

# 命令行参数
budget_hours = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
RATIOS = [float(r) for r in sys.argv[2:]] or [0.1, 0.2, 0.3, 0.4, 0.5]

BUDGET_S = budget_hours * 3600.0
CSV_PATH = "results_batch_sweep.csv"

# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]

# ---- model + total params -------------------------------------------------
key = jax.random.key(seed)
model_key, es_key, data_key = jax.random.split(key, 3)
frozen_params, params, scan_map, es_map = SNNModel.rand_init(
    model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
    num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
)
total_params = sum(p.size for p in jax.tree.leaves(params))

# 批次线性点（截断到训练集大小）
batch_points = []
for r in RATIOS:
    b = int(round(r * total_params))
    b = max(1, min(b, n_train))
    batch_points.append((r, b, r * total_params))

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


def make_step(data_key, num_envs):
    """Return (step(epoch) -> epoch_idx, acc_train) closure sampling a fresh batch."""
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


def run_point(ratio, num_envs, point_slice_s, deadline):
    """在一个批次点上，自适应地跑满 point_slice_s（但不越过全局 deadline）。"""
    print(f"\n===== [batch sweep] ratio={ratio:.3f} batch={num_envs} "
          f"(target {num_envs / total_params:.3f}x) slice={point_slice_s:.0f}s =====",
          flush=True)
    t0 = time.time()
    step = make_step(data_key, num_envs)

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

    est_each = int(max(10, (point_slice_s - (time.time() - t0)) / cal_s)) if cal_s > 0 else 100
    print(f"  plan: ~{est_each} more epochs to fill slice", flush=True)

    epoch = cal
    last_val = 0.0
    while True:
        if time.time() >= deadline:
            print("  global deadline reached, stopping point.", flush=True)
            break
        if time.time() - t0 >= point_slice_s:
            break
        _, acc = step(epoch)
        best = max(best, acc)
        epoch += 1
        if epoch % EVAL_EVERY == 0:
            last_val = eval_acc()
            print(f"  epoch {epoch:5d} | val {last_val:.3f} | best_train {best:.3f} "
                  f"| {(time.time()-t0):.0f}s", flush=True)

    last_val = eval_acc() if last_val == 0.0 and epoch > cal else last_val
    elapsed = time.time() - t0
    return epoch, last_val, best, elapsed, cal_s


def main():
    deadline = time.time() + BUDGET_S
    n_points = len(batch_points)
    point_slice_s = BUDGET_S / n_points

    print(f"total_params = {total_params}", flush=True)
    print(f"budget = {budget_hours}h ({BUDGET_S:.0f}s), points = {n_points}, "
          f"per-point slice = {point_slice_s:.0f}s", flush=True)
    print(f"batch points (ratio -> batch): "
          f"{[(r, b) for r, b, _ in batch_points]}", flush=True)

    # 初始化结果 CSV（含表头）
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["ratio", "batch", "epochs", "val_acc", "best_train",
                        "s_per_epoch", "time_s"])
        f.flush()

    results = []
    for ratio, num_envs, raw_target in batch_points:
        if time.time() >= deadline:
            print("global deadline reached before next point.", flush=True)
            break
        epochs, last_val, best, elapsed, cal_s = run_point(
            ratio, num_envs, point_slice_s, deadline)
        results.append((ratio, num_envs, epochs, last_val, best, elapsed))
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([ratio, num_envs, epochs, last_val, best,
                                    round(cal_s, 5), round(elapsed)])
        print(f"  [{ratio:.3f} x {num_envs}] epochs={epochs} "
              f"val_acc={last_val:.3f} best_train={best:.3f} ({elapsed:.0f}s)",
              flush=True)

    print("\n===== 批次扫描汇总（按批次/总参数量比例取线性点，固定 LR） =====")
    print(f"{'ratio':>6} | {'batch':>6} | {'epochs':>6} | {'val_acc':>8} | {'best_train':>10} | {'用时(s)':>8}")
    for ratio, num_envs, epochs, last_val, best, elapsed in results:
        print(f"{ratio:6.3f} | {num_envs:6d} | {epochs:6d} | {last_val:8.3f} | "
              f"{best:10.3f} | {elapsed:8.0f}")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
