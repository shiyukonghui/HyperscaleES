"""rank 参数扫描：单卡累积 batch=60000，扫描 rank，绘制准确率 vs rank。

数学/工程背景（es_batch_equivalence_experiment.md）：
  - 单卡 24GB 用梯度累积（全局 z-score + chunked einsum）等价大批次；
  - 更新/前向显存由 B=(chunk, 784, rank) 主导，随 rank 线性增长；
  - 高 rank 需缩小 chunk（增大 accumulate）以保持 B ≈ 4.8GB，从而在 24GB 内运行。

用法（WSL venv 内运行）：
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' \
    /root/hyperscalees-venv/bin/python pythonScript/exp_rank_sweep_accumulate.py
"""
import json
import os
import subprocess
import sys

import numpy as np

BATCH = 60000
NUM_EPOCHS = 3000
MNIST_DIR = "/mnt/f/PythonProject/HyperscaleES/data/MNIST/raw"
OUT_DIR = "records/rank_sweep"
RANKS = [64, 96, 128, 256, 512, 1024]

# 内存安全映射：chunk × 784 × rank × 4 ≤ ~4.8GB（B 矩阵），即 chunk ≈ 1.53e6/rank
def mem_safe_accumulate(rank):
    max_chunk = max(1, int(1.53e6 / rank))       # 保持 B ≈ 4.8GB
    max_chunk = min(max_chunk, 12000)            # 低 rank 时 cap 到 12000（已知稳定配置）
    accumulate = BATCH // max_chunk
    if accumulate < 1:
        accumulate = 1
    while BATCH % accumulate != 0:                # 保证 batch 可被 accumulate 整除
        accumulate += 1
    return accumulate


def parse_best(csv_path):
    """从训练 CSV 读取最终的 best_val / best_train（最后一行）。"""
    if not os.path.exists(csv_path):
        return 0.0, 0.0
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return 0.0, 0.0
    last = lines[-1].strip().split(",")
    # 列：epoch,train_acc,val_acc,best_val,best_train,epoch_time,cum_time
    return float(last[3]), float(last[4])


def plot(results, out_png):
    """准确率 vs rank 折线图（rank 用对数坐标）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib 不可用，跳过绘图")
        return

    ranks = [r["rank"] for r in results]
    vals = [r["best_val"] for r in results]
    trains = [r["best_train"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ranks, vals, "o-", lw=2, label="best_val", color="#d62728")
    plt.plot(ranks, trains, "s--", lw=2, label="best_train", color="#1f77b4")
    for x, y in zip(ranks, vals):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    plt.xscale("log", base=2)
    plt.xticks(ranks, [str(r) for r in ranks])
    plt.xlabel("LoRA rank (log2)")
    plt.ylabel("accuracy")
    plt.title("单卡 24GB 梯度累积 batch=60000：accuracy vs rank")
    plt.ylim(min(vals + trains) - 0.02, max(vals + trains) + 0.02)
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"[plot] saved {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    results_json = os.path.join(OUT_DIR, "results.json")
    plot_png = os.path.join(OUT_DIR, "accuracy_vs_rank.png")

    for rank in RANKS:
        acc = mem_safe_accumulate(rank)
        chunk = BATCH // acc
        csv_path = os.path.join(OUT_DIR, f"r{rank}.csv")
        log_path = os.path.join(OUT_DIR, f"r{rank}.log")
        print(f"[sweep] rank={rank:4d}  accumulate={acc:2d}  chunk={chunk:5d}  ...", flush=True)

        cmd = [
            sys.executable, "-m", "llm_experiments.snn_mnist_train_accumulate",
            "--batch", str(BATCH), "--accumulate", str(acc), "--rank", str(rank),
            "--num-epochs", str(NUM_EPOCHS),
            "--mnist-dir", MNIST_DIR, "--csv-out", csv_path,
        ]
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"
        with open(log_path, "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)

        best_val, best_train = parse_best(csv_path)
        results.append({"rank": rank, "accumulate": acc,
                        "best_val": best_val, "best_train": best_train})
        print(f"  -> best_val={best_val:.4f}  best_train={best_train:.4f}", flush=True)

        # 增量保存 + 增量绘图（便于中途查看）
        with open(results_json, "w") as f:
            json.dump(results, f, indent=2)
        plot(results, plot_png)

    print("\n=== rank 扫描完成 ===")
    for r in results:
        print(f"  rank={r['rank']:4d}  best_val={r['best_val']:.4f}  best_train={r['best_train']:.4f}")
    print(f"结果: {results_json}")
    print(f"图:   {plot_png}")


if __name__ == "__main__":
    main()
