"""小批次等效大批次的数学验证（ES 梯度估计器视角）。

1) 验证 ES 梯度估计方差 ∝ 1/N 律
2) 四种方差缩减手段的方差与"等效批次倍数"：基线 / 对偶 / 梯度累积 / 回归式ES
3) 固定总样本预算下，小批次 + 各技巧 vs 大批次的优化轨迹对比

用法: .venv/Scripts/python.exe pythonScript/exp_small_batch_equiv.py
"""
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

rng = np.random.default_rng(42)

d = 64                 # 参数维度
sigma = 0.2            # 扰动幅度
x_star = rng.normal(size=d)   # 最优解
x0 = np.ones(d) * 0.3         # 起始点


def f(x):
    """二次 fitness（最大化）：f(x) = -0.5||x - x_star||^2，∇f = x_star - x。"""
    return -0.5 * np.sum((x - x_star) ** 2)


def grad_true(x):
    return x_star - x


# ---------------------------------------------------------------------------
# ES 梯度估计器（均在固定点 x 处估计，目标是让 ĝ ≈ ∇f）
# ---------------------------------------------------------------------------
def est_vanilla(x, N, reps):
    """ĝ = (1/N)Σ f(x+σε)·ε/σ"""
    out = np.empty((reps, d))
    for r in range(reps):
        eps = rng.normal(size=(N, d))
        fs = np.array([f(x + sigma * e) for e in eps])
        out[r] = (fs[:, None] * eps).mean(axis=0) / sigma
    return out


def est_baseline(x, N, reps):
    """ĝ = (1/N)Σ (f_i - f̄)·ε/σ（减去均值 = z-score 中心化）"""
    out = np.empty((reps, d))
    for r in range(reps):
        eps = rng.normal(size=(N, d))
        fs = np.array([f(x + sigma * e) for e in eps])
        out[r] = ((fs - fs.mean())[:, None] * eps).mean(axis=0) / sigma
    return out


def est_antithetic(x, N, reps):
    """对偶对 (ε, -ε)：ĝ = (1/(N/2))Σ [f(+)-f(-)]·ε/(2σ)"""
    assert N % 2 == 0
    M = N // 2
    out = np.empty((reps, d))
    for r in range(reps):
        eps = rng.normal(size=(M, d))
        fp = np.array([f(x + sigma * e) for e in eps])
        fm = np.array([f(x - sigma * e) for e in eps])
        out[r] = ((fp - fm)[:, None] * eps).mean(axis=0) / (2 * sigma)
    return out


def est_regression(x, N, reps):
    """回归式/guided ES：最小二乘拟合 f ≈ a + bᵀε，梯度估计 = b/σ。
    Gauss-Markov：对"线性于 ε"的部分，OLS 系数是方差最小的无偏估计。"""
    out = np.empty((reps, d))
    for r in range(reps):
        eps = rng.normal(size=(N, d))
        fs = np.array([f(x + sigma * e) for e in eps])
        X = np.column_stack([np.ones(N), eps])
        coef, *_ = np.linalg.lstsq(X, fs, rcond=None)
        out[r] = coef[1:] / sigma
    return out


def est_accumulated(x, N, K, reps):
    """梯度累积：K 个大小为 N 的分块，每块用不同噪声、参数冻结（都在 x 处评估），
    各块梯度平均 —— 数学上 = 单个 batch 大小 K*N 的同一估计器（精确等价）。"""
    out = np.empty((reps, d))
    for r in range(reps):
        g_acc = np.zeros(d)
        for _ in range(K):
            eps = rng.normal(size=(N, d))
            fs = np.array([f(x + sigma * e) for e in eps])
            g_acc += ((fs - fs.mean())[:, None] * eps).mean(axis=0) / sigma
        out[r] = g_acc / K
    return out


def report(name, g_ests):
    g0 = grad_true(x0)
    errs = np.linalg.norm(g_ests - g0, axis=1)
    bias = np.linalg.norm(g_ests.mean(axis=0) - g0)
    return dict(name=name, mse=float(errs.mean() ** 2),
                bias=float(bias), var=float(errs.var()))


# ---------------------------------------------------------------------------
# 1) 验证 1/N 律 + 2) 各方法方差/等效批次倍数
# ---------------------------------------------------------------------------
print("=" * 78)
print(f"d={d}, sigma={sigma}, x0 处真梯度‖∇f‖={np.linalg.norm(grad_true(x0)):.3f}")
print("=" * 78)
print("\n[1] 方差 ∝ 1/N 律验证（vanilla，reps=400）")
N_scan = [50, 100, 200, 400, 800]
prev_v = None
for N in N_scan:
    g = est_vanilla(x0, N, 400)
    v = g.var(axis=0).sum()
    ratio = prev_v * (N / 2) / (v * N) if prev_v else float("nan")
    print(f"  N={N:4d}  Var(ĝ)={v:.4f}   Var·N={v * N:.2f}  (应近似常数={v * N:.2f})")
    prev_v = v

print("\n[2] 各方法梯度估计统计（N=200，reps=2000，真梯度方差作基准）")
methods = {
    "vanilla  (原始)": est_vanilla,
    "baseline (减均值)": est_baseline,
    "antithetic(对偶对)": est_antithetic,
    "regression(回归式ES)": est_regression,
}
base_var = None
rows = []
for name, fn in methods.items():
    g = fn(x0, 200, 2000)
    var = g.var(axis=0).sum()
    rows.append((name, var))
    if base_var is None:
        base_var = var
print(f"  {'方法':<28}{'Var(ĝ)':>12}{'等效批次倍数F':>16}")
for name, var in rows:
    print(f"  {name:<28}{var:>12.4f}{base_var / var:>16.1f}x")

print("\n[3] 梯度累积 vs 单个大批次（精确等价验证，reps=500）")
for N_s, K, N_l in [(100, 6, 600), (200, 4, 800), (50, 10, 500)]:
    g_acc = est_accumulated(x0, N_s, K, 500)
    g_big = est_vanilla(x0, N_l, 500)
    g_bl = est_baseline(x0, N_l, 500)
    print(f"  累积 {N_s}x{K}={N_s * K} 个样本 vs 单批 {N_l}: "
          f"Var累积={g_acc.var(axis=0).sum():.4f}, "
          f"Var单批(带基线)={g_bl.var(axis=0).sum():.4f}, "
          f"Var单批(无基线)={g_big.var(axis=0).sum():.4f}")

# ---------------------------------------------------------------------------
# 3) 固定总样本预算下的优化轨迹（小批次 + 技巧 vs 大批次）
# ---------------------------------------------------------------------------
S = 400_000          # 总样本预算
lr = 0.02            # 对单位曲率二次面稳定（过大步长会发散）


def run_traj(est_fn, N, x_init):
    """用估计器更新参数 x += lr·ĝ，直到样本预算用完。返回 (样本数, 距离)。"""
    x = x_init.copy()
    steps = S // N
    samples, dists = [], []
    for k in range(steps):
        g = est_fn(x, N, 1)[0]
        x = x + lr * g
        if k % max(1, steps // 60) == 0:
            samples.append((k + 1) * N)
            dists.append(np.linalg.norm(x - x_star))
    return samples, dists


trajs = {
    "large-batch N=2000 (200 steps)": lambda: run_traj(est_baseline, 2000, x0),
    "small-batch N=100 (4000 steps)": lambda: run_traj(est_baseline, 100, x0),
    "small-batch+antithetic N=100": lambda: run_traj(est_antithetic, 100, x0),
    "small-batch+regression N=100": lambda: run_traj(est_regression, 100, x0),
}
print("\n[4] 固定样本预算 S={} 的优化轨迹（终态距离）".format(S))
for name, fn in trajs.items():
    samples, dists = fn()
    print(f"  {name:<34} final ||x-x*|| = {dists[-1]:.4f}  (start {dists[0]:.4f})")

if HAS_MPL:
    plt.figure(figsize=(8, 5))
    for name, fn in trajs.items():
        samples, dists = fn()
        plt.plot(samples, dists, label=name, lw=2)
    plt.axhline(np.linalg.norm(x0 - x_star), ls="--", color="gray",
                label="initial distance")
    plt.xlabel("samples consumed (fair budget)")
    plt.ylabel("||x - x*||")
    plt.title("Fixed sample budget: small-batch + tricks vs large-batch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("records/exp_small_batch_equiv.png", dpi=130)
    print("\nfigure saved: records/exp_small_batch_equiv.png")
