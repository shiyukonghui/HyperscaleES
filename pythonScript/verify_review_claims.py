"""复核《审查报告一》中的关键数学断言（精确计算 + 蒙特卡洛交叉验证）。

验证对象：
  A. 同批均值中心化的偏差：E[ĝ_N] = (1 - 1/N)∇F_σ   （不是精确无偏）
  B. 梯度项协方差：Cov[(εᵀg)ε] = ‖g‖² I + g gᵀ（trace=(d+1)‖g‖²，每分量=(1+1/d)‖g‖²）
  C. 对偶采样梯度项方差 = 独立采样的 2 倍（方向数减半）
  D. 局部中心化下，小批次多次 vs 大批次的漂移差 = η'(1/N_L - 1/N_s)∇F_σ（O(η')）

纯 numpy，运行: .venv/Scripts/python.exe pythonScript/verify_review_claims.py
"""
import numpy as np

D = 8
SIGMA = 0.2
X_STAR = np.linspace(-1.0, 1.0, D)          # 二次面 ∇f = X_STAR - x
G = X_STAR                                   # 在 x=0 处 ∇F_σ = ∇f = X_STAR


def f_vec(z):
    """二次 fitness f(x) = -0.5||x-x*||²，z: (N, D)。"""
    return -0.5 * np.sum((z - X_STAR) ** 2, axis=1)


def es_local_centered(x, eps):
    """同批均值中心化的 ES 估计器（EggRoll 的 mean 部分，不含 /std）。"""
    fs = f_vec(x + SIGMA * eps)
    fbar = fs.mean()
    return ((fs - fbar)[:, None] * eps).mean(axis=0) / SIGMA


def es_unbiased(x, eps):
    """独立基线（总体真均值 F_σ 已知）→ 严格无偏。"""
    fs = f_vec(x + SIGMA * eps)
    fbar = F_sigma_exact(x)                  # 用真均值（二次面下 = f(x) - 0.5 σ² d）
    return ((fs - fbar)[:, None] * eps).mean(axis=0) / SIGMA


def F_sigma_exact(x):
    """二次面高斯平滑的精确值 F_σ(x) = f(x) - 0.5 σ² d。"""
    return f_vec(x[None, :])[0] - 0.5 * SIGMA ** 2 * D


print("=" * 78)
print(f"D={D}, σ={SIGMA}, x=0 处 ∇F_σ = ∇f = X_STAR, ‖∇f‖={np.linalg.norm(G):.4f}")
print("=" * 78)

# ---------------------------------------------------------------------------
# A. 同批均值中心化的 1/N 偏差
# ---------------------------------------------------------------------------
print("\n[A] 同批均值中心化偏差 E[ĝ_N] = (1-1/N)∇F_σ")
rng = np.random.default_rng(0)
N, reps = 50, 30000
acc = np.empty((reps, D))
acc_unb = np.empty((reps, D))
for r in range(reps):
    eps = rng.normal(size=(N, D))
    acc[r] = es_local_centered(np.zeros(D), eps)
    acc_unb[r] = es_unbiased(np.zeros(D), eps)

mean_local = acc.mean(axis=0)
mean_unb = acc_unb.mean(axis=0)
pred_local = (1 - 1 / N) * G          # 审查报告预测：有 1/N 偏差
print(f"  E[ĝ_N](同批均值)  = {np.round(mean_local, 4)}")
print(f"  预测 (1-1/N)∇F_σ   = {np.round(pred_local, 4)}")
print(f"  E[ĝ_N](独立基线)  = {np.round(mean_unb, 4)}   (应≈∇F_σ={np.round(G, 4)})")
err_local = np.linalg.norm(mean_local - pred_local)
err_unb = np.linalg.norm(mean_unb - G)
print(f"  ‖同批均值 − 预测‖ = {err_local:.5f}   ‖独立基线 − ∇F_σ‖ = {err_unb:.5f}")
print(f"  → 同批均值偏差量 ‖mean − ∇F_σ‖ = {np.linalg.norm(mean_local - G):.5f} "
      f"(理论 (1/N)‖∇F_σ‖ = {np.linalg.norm(G)/N:.5f})")

# ---------------------------------------------------------------------------
# B. 梯度项协方差精确式（Isserlis/Wick）
# ---------------------------------------------------------------------------
print("\n[B] Cov[(εᵀg)ε] = ‖g‖² I + g gᵀ")
rng = np.random.default_rng(1)
reps = 200000
eps = rng.normal(size=(reps, D))
grad_term = (eps @ G)[:, None] * eps            # (reps, D)：每个样本的 (εᵀg)ε
cov_emp = np.cov(grad_term, rowvar=False)
cov_exact = np.linalg.norm(G) ** 2 * np.eye(D) + np.outer(G, G)
print(f"  trace 经验值 = {np.trace(cov_emp):.4f}   理论 (d+1)‖g‖² = {(D+1)*np.linalg.norm(G)**2:.4f}")
print(f"  每分量经验均值 = {np.trace(cov_emp)/D:.4f}   理论 (1+1/d)‖g‖² = {(1+1/D)*np.linalg.norm(G)**2:.4f}")
print(f"  ‖Cov经验 − Cov精确‖/‖Cov精确‖ = {np.linalg.norm(cov_emp-cov_exact)/np.linalg.norm(cov_exact):.5f}")

# ---------------------------------------------------------------------------
# C. 对偶采样梯度项方差 = 2x
# ---------------------------------------------------------------------------
print("\n[C] 对偶采样梯度项方差 = 独立采样 2 倍（方向数减半）")
rng = np.random.default_rng(2)
reps = 100000
N_ind = 100
# 独立采样（N 个独立方向）的梯度项估计
eps_ind = rng.normal(size=(reps, N_ind, D))
g_ind = np.mean((eps_ind @ G)[:, :, None] * eps_ind, axis=1) / 1.0   # 均值即方向数归一，梯度项= (εᵀg)ε
# 对偶采样（N/2 个方向，成对 ±）
M = N_ind // 2
eps_a = rng.normal(size=(reps, M, D))
fp = (eps_a @ G)       # 正向 (εᵀg)
g_anti = np.mean((fp - (-fp))[:, :, None] * eps_a / 2.0, axis=1)  # [(εᵀg)-(-εᵀg)]/(2) ε = (εᵀg)ε
# 注意：梯度项对独立/对偶都是 (εᵀg)ε，但对偶只有 M=N/2 个独立方向
var_ind = np.trace(np.cov(g_ind, rowvar=False))
var_anti = np.trace(np.cov(g_anti, rowvar=False))
print(f"  Var(独立 N=100) = {var_ind:.4f}   Var(对偶 N/2=50方向) = {var_anti:.4f}")
print(f"  比值 Var(对偶)/Var(独立) = {var_anti/var_ind:.3f}   (理论≈2.0)")

# ---------------------------------------------------------------------------
# D. 局部中心化下的漂移差（精确分离两个误差源）
# ---------------------------------------------------------------------------
print("\n[D] 局部中心化漂移差 = 中心化偏差项 η'(1/N_L-1/N_s)∇F_σ  +  移动点项 O(η'²)")
rng = np.random.default_rng(3)
x = np.zeros(D)
eta_prime = 0.1
K, N_s = 4, 50                       # N_L = K*N_s = 200
eta = eta_prime / K
N_L = K * N_s
reps = 40000

# D1) 冻结点 x=0：K 次小批次(局部) vs 1 次大批次(局部) —— 只有中心化偏差，无移动点误差
def frozen_k_small_local():
    g = np.zeros(D)
    for _ in range(K):
        eps = rng.normal(size=(N_s, D))
        g += es_local_centered(x, eps)     # 都在 x 处评估
    return eta * g                          # K 个微步共 η·Σ(1/N_s 中心化) ≈ η'(1-1/N_s)∇F_σ

def frozen_large_local():
    eps = rng.normal(size=(N_L, D))
    return eta_prime * es_local_centered(x, eps)   # η'(1-1/N_L)∇F_σ

d1 = np.empty((reps, D)); d1b = np.empty((reps, D))
for r in range(reps):
    d1[r] = frozen_k_small_local()
    d1b[r] = frozen_large_local()
centering_diff = np.linalg.norm(d1.mean(0) - d1b.mean(0))
pred_centering = abs(eta_prime * (1/N_L - 1/N_s)) * np.linalg.norm(G)

# D2) 独立基线(无偏)下：K 次小批次(移动点) vs 1 次大批次 —— 只剩移动点 O(η'²)
def macro_large_unb():
    eps = rng.normal(size=(N_L, D))
    return eta_prime * es_unbiased(x, eps)

def macro_small_unb():
    xx = x.copy()
    for _ in range(K):
        eps = rng.normal(size=(N_s, D))
        xx = xx + eta * es_unbiased(xx, eps)
    return xx - x

d2 = np.empty((reps, D)); d2b = np.empty((reps, D))
for r in range(reps):
    d2[r] = macro_small_unb()
    d2b[r] = macro_large_unb()
moving_diff = np.linalg.norm(d2.mean(0) - d2b.mean(0))

print(f"  [D1 冻结点] 中心化偏差项(经验) = {centering_diff:.6f}   预测 η'|1/N_L-1/N_s|‖∇F_σ‖ = {pred_centering:.6f}")
print(f"  [D2 移动点] 独立基线漂移差(经验) = {moving_diff:.6f}   理论 O(η'²)≈{eta_prime**2:.6f}")
print(f"  → 局部中心化下总漂移差 ≈ 中心化偏差({pred_centering:.6f}) + 移动点({moving_diff:.6f})，量级 O(η') 而非 O(η'²)")
print("=" * 78)
print("DONE")
