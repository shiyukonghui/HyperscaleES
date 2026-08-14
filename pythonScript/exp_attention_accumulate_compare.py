"""Hopfield vs Mean-field：大批次累积训练两路对比驱动。

顺序运行 snn_attention_train_accumulate 的两种路由（同 batch/rank/种子/配置）：
  --route hopfield  与  --route meanfield
各自完成累积训练后，聚合两个 CSV 的 best_val / best_train / 终末 w_err / cos_o /
每 epoch 平均耗时，写入 records/attention_accumulate/comparison.csv 并在终端打印对比表。

本驱动用于验证"大批次（batch=60000）累积训练下的注意力等价性与两路性能对比"。

用法（WSL venv 内，GPU）：
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /root/hyperscalees-venv/bin/python pythonScript/exp_attention_accumulate_compare.py \
      --batch 60000 --rank 64 --num-epochs 2000 --mnist-dir /mnt/d/Rust/snn_t1/mnist_data
"""
import argparse
import csv
import os
import subprocess
import sys
import time

OUT_DIR = "records/attention_accumulate"
# 每个 route CSV 的列（与 snn_attention_train_accumulate.py 输出一致）
ROUTES = ["hopfield", "meanfield"]
CSV_COLS = ["epoch", "train_acc", "val_acc", "best_val", "best_train",
            "w_err", "cos_o", "epoch_time", "cum_time"]


def parse_csv(path):
    """读取训练 CSV，返回 {best_val, best_train, final_w_err, final_cos_o, avg_epoch_time}。

    best_val/best_train 取全局最大值；终末 w_err/cos_o 取最后一次非空验证行；
    avg_epoch_time 对所有 epoch 的 epoch_time 取均值。
    """
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            parts = [p.strip() for p in line.split(",")]
            row = dict(zip(CSV_COLS, parts))
            rows.append(row)

    if not rows:
        return None

    # best_val / best_train：CSV 已维护 running max，取最后一非空即可；但保守取其列最大值
    best_val = max((float(r["best_val"]) for r in rows if r["best_val"]), default=0.0)
    best_train = max((float(r["best_train"]) for r in rows if r["best_train"]), default=0.0)

    # 终末等价性指标：最后一次出现非空 w_err/cos_o 的行
    final_w_err = final_cos_o = None
    for r in rows:
        if r["w_err"]:
            final_w_err = float(r["w_err"])
        if r["cos_o"]:
            final_cos_o = float(r["cos_o"])

    times = [float(r["epoch_time"]) for r in rows if r["epoch_time"]]
    avg_t = sum(times) / len(times) if times else 0.0

    return {"best_val": best_val, "best_train": best_train,
            "final_w_err": final_w_err, "final_cos_o": final_cos_o,
            "avg_epoch_time": avg_t, "rows": len(rows)}


def parse_args():
    p = argparse.ArgumentParser(description="Hopfield vs Mean-field 大批次累积对比")
    p.add_argument("--batch", type=int, default=60000)
    p.add_argument("--accumulate", type=int, default=0, help="0=自动按显存公式")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=2000)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--n-iter", type=int, default=8)
    p.add_argument("--patch-px", type=int, default=7)
    p.add_argument("--d-head", type=int, default=16)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--val-batch", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mnist-dir", default="/mnt/d/Rust/snn_t1/mnist_data")
    p.add_argument("--skip-trained", action="store_true",
                   help="跳过已完成（best_val>0）的 route，断点续跑")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "comparison.csv")

    results = {}
    for route in ROUTES:
        csv_path = os.path.join(OUT_DIR, f"{route}.csv")
        if args.skip_trained and parse_csv(csv_path) and parse_csv(csv_path)["best_val"] > 0:
            print(f"[{route}] 已训练完成，跳过", flush=True)
            results[route] = parse_csv(csv_path)
            continue

        log_path = os.path.join(OUT_DIR, f"{route}.log")
        cmd = [sys.executable, "-m", "llm_experiments.snn_attention_train_accumulate",
               "--route", route,
               "--batch", str(args.batch),
               "--rank", str(args.rank),
               "--num-epochs", str(args.num_epochs),
               "--T", str(args.T),
               "--sigma", str(args.sigma),
               "--lr", str(args.lr),
               "--n-iter", str(args.n_iter),
               "--patch-px", str(args.patch_px),
               "--d-head", str(args.d_head),
               "--validate-every", str(args.validate_every),
               "--val-batch", str(args.val_batch),
               "--seed", str(args.seed),
               "--mnist-dir", args.mnist_dir,
               "--csv-out", csv_path]
        if args.accumulate:
            cmd += ["--accumulate", str(args.accumulate)]

        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"
        print(f"[{route}] batch={args.batch} rank={args.rank} "
              f"epochs={args.num_epochs} ...", flush=True)
        t0 = time.time()
        with open(log_path, "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
        dt = time.time() - t0
        stat = parse_csv(csv_path)
        if stat:
            stat["wall"] = dt
            results[route] = stat
            print(f"  [{route}] best_val={stat['best_val']:.4f} "
                  f"best_train={stat['best_train']:.4f} "
                  f"w_err={stat['final_w_err']:.4f} cos_o={stat['final_cos_o']:.4f} "
                  f"avg_ep={stat['avg_epoch_time']:.3f}s wall={dt:.0f}s", flush=True)
        else:
            print(f"  [{route}] 无结果（CSV 缺失）", flush=True)

    # 写对比 CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["route", "best_val", "best_train", "final_w_err", "final_cos_o",
                    "avg_epoch_time", "epochs", "wall_time"])
        for route in ROUTES:
            s = results.get(route)
            if s:
                w.writerow([route, f"{s['best_val']:.6f}", f"{s['best_train']:.6f}",
                            f"{s['final_w_err']:.6f}", f"{s['final_cos_o']:.6f}",
                            f"{s['avg_epoch_time']:.4f}", s["rows"],
                            f"{s.get('wall', 0):.1f}"])

    # 终端打印对比表
    print("\n=== Hopfield vs Mean-field 大批次累积对比 ===")
    hdr = f"{'route':<12}{'best_val':>10}{'best_train':>12}{'w_err':>9}{'cos_o':>9}{'avg_e/ep':>10}"
    print(hdr)
    print("-" * len(hdr))
    for route in ROUTES:
        s = results.get(route)
        if not s:
            print(f"{route:<12}{'NA':>10}")
            continue
        w_err = "-" if s["final_w_err"] is None else f"{s['final_w_err']:.4f}"
        cos_o = "-" if s["final_cos_o"] is None else f"{s['final_cos_o']:.4f}"
        print(f"{route:<12}{s['best_val']:>10.4f}{s['best_train']:>12.4f}"
              f"{w_err:>9}{cos_o:>9}{s['avg_epoch_time']:>10.4f}")
    print(f"\n结果: {out_csv}")


if __name__ == "__main__":
    main()
