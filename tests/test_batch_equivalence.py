"""小批次多次训练 ⇔ 大批次训练的数学等价性测试用例。

逐条数值验证 docs/es_batch_equivalence_proof.md 中的定理：

  定理1 (Stein/热核):  E[ĝ_N] = ∇F_σ（ES 估计器无偏）
  定理2 (代数精确):    参数冻结时，K 段梯度累积 = 单个大批次（逐样本相等）
  定理3 (分布等价):    冻结点上两者同均值、同协方差
  定理4 (噪声协方差):  动态下 K 微步累积噪声协方差 = 大批次单步噪声协方差（主阶精确相等）
  定理5 (漂移一致):    动态下漂移差为 O(η'²)（一阶求积误差）
  定理6 (极限同一):    η' → 0 时两者收敛到同一条梯度流（同一极限点）
  定理7 (误差阶):      轨迹差（均值意义）∝ η'²

纯 numpy 实现，不依赖 JAX，可在任意 Python 环境运行。

运行（从仓库根目录）:
    .venv/Scripts/python.exe tests/test_batch_equivalence.py
"""
import numpy as np

# ---------------------------------------------------------------------------
# 全局设定
# ---------------------------------------------------------------------------
D = 16                 # 参数维度
SIGMA = 0.2            # 扰动幅度
X_STAR = np.linspace(-1.0, 1.0, D)   # 二次 fitness 的最优点（∇f 的目标）
A = np.linspace(0.5, 1.5, D)         # 线性 fitness 的常梯度 a


# ---------------------------------------------------------------------------
# fitness 与真梯度（对二次面，∇F_σ = ∇f；对线性面，∇F_σ = a，均精确）
# ---------------------------------------------------------------------------
def f_quad_vec(z):
    """二次 fitness f(x) = -0.5||x - x*||²，z: (N, D)。"""
    return -0.5 * np.sum((z - X_STAR) ** 2, axis=1)


def grad_quad(x):
    """∇f(x) = x* - x（也是 ∇F_σ，见定理1的二次面注记）。"""
    return X_STAR - x


def f_lin_vec(z):
    """线性 fitness f(x) = a·x，z: (N, D)。"""
    return z @ A


# ---------------------------------------------------------------------------
# ES 梯度估计器（z-score 全局中心化，等价于 EggRoll 的 convert_fitnesses）
# ---------------------------------------------------------------------------
def es_centered(x, eps, f_vec):
    """ĝ = (1/N) Σ (f(x+σε_i) - f̄) ε_i / σ，f̄ 为全局均值。eps: (N, D)。"""
    fs = f_vec(x + SIGMA * eps)
    fbar = fs.mean()
    return ((fs - fbar)[:, None] * eps).mean(axis=0) / SIGMA


def es_accumulated_chunks(x, chunks, f_vec):
    """梯度累积：K 个 chunk（各含 N_s 个 ε），全局中心化后逐段求和再平均。

    数学上应精确等于对全部样本一次性求 ĝ（定理2）。
    """
    all_eps = np.concatenate(chunks, axis=0)
    all_fs = f_vec(x + SIGMA * all_eps)
    fbar = all_fs.mean()          # 全局均值（跨所有 chunk）
    g = np.zeros_like(x)
    start = 0
    for eps in chunks:
        n = eps.shape[0]
        fs = all_fs[start:start + n]
        g += ((fs - fbar)[:, None] * eps).mean(axis=0) / SIGMA  # 每 chunk: (1/n)Σ
        start += n
    return g / len(chunks)        # 再对 K 个 chunk 平均


# ---------------------------------------------------------------------------
# 动态更新：大批次单步 vs 小批次多次
# ---------------------------------------------------------------------------
def macro_step_large(x, eta_prime, N_L, rng, f_vec):
    """大批次：一个宏观步 x += η' ĝ_{N_L}(x)。"""
    eps = rng.normal(size=(N_L, D))
    return x + eta_prime * es_centered(x, eps, f_vec)


def macro_step_small(x, eta, N_s, K, rng, f_vec):
    """小批次多次：K 个微步 x += η ĝ_{N_s}(x)。"""
    for _ in range(K):
        eps = rng.normal(size=(N_s, D))
        x = x + eta * es_centered(x, eps, f_vec)
    return x


# 确定性漂移（无噪声，用于精确刻画定理5/6/7的 O(η'²) 误差阶）
def drift_large_det(x, eta_prime, grad_fn):
    return x + eta_prime * grad_fn(x)


def drift_small_det(x, eta, K, grad_fn):
    for _ in range(K):
        x = x + eta * grad_fn(x)
    return x


# ---------------------------------------------------------------------------
# 定理1：ES 估计器无偏（E[ĝ] = ∇F_σ = ∇f）
# ---------------------------------------------------------------------------
def test_stein_unbiasedness():
    rng = np.random.default_rng(0)
    x = np.zeros(D)
    N, reps = 200, 4000
    ests = np.empty((reps, D))
    for r in range(reps):
        eps = rng.normal(size=(N, D))
        ests[r] = es_centered(x, eps, f_quad_vec)

    g_true = grad_quad(x)
    bias = np.linalg.norm(ests.mean(axis=0) - g_true)
    scale = np.linalg.norm(g_true)
    # 无偏：均值偏差应远小于真梯度范数（蒙特卡洛采样误差 ~ 1/√reps）
    assert bias < 0.02 * scale, f"定理1失败: bias={bias:.5f}, scale={scale:.5f}"
    print(f"OK 定理1 (Stein无偏)   bias={bias:.5f}  ‖∇f‖={scale:.5f}  bias/‖∇f‖={bias/scale:.5f}")


# ---------------------------------------------------------------------------
# 定理2：参数冻结时梯度累积 = 大批次（代数精确）
# ---------------------------------------------------------------------------
def test_gradient_accumulation_exact():
    rng = np.random.default_rng(1)
    x = rng.normal(size=D)
    K, N_s = 4, 50                      # 小批次 50，累积 4 段 = 大批次 200
    chunks = [rng.normal(size=(N_s, D)) for _ in range(K)]

    g_acc = es_accumulated_chunks(x, chunks, f_quad_vec)
    g_big = es_centered(x, np.concatenate(chunks, axis=0), f_quad_vec)

    err = np.linalg.norm(g_acc - g_big)
    assert err < 1e-9, f"定理2失败: 累积与大单批不相等, err={err:.3e}"
    print(f"OK 定理2 (代数精确)    累积==大批次, 最大逐分量误差={np.max(np.abs(g_acc - g_big)):.3e}")


# ---------------------------------------------------------------------------
# 定理3：冻结点上两者同分布（同均值、同协方差）
# ---------------------------------------------------------------------------
def test_frozen_distribution_equivalence():
    rng = np.random.default_rng(2)
    x = rng.normal(size=D)
    K, N_s, reps = 4, 50, 4000
    acc, big = np.empty((reps, D)), np.empty((reps, D))
    for r in range(reps):
        chunks = [rng.normal(size=(N_s, D)) for _ in range(K)]
        acc[r] = es_accumulated_chunks(x, chunks, f_quad_vec)
        big[r] = es_centered(x, np.concatenate(chunks, axis=0), f_quad_vec)

    mean_err = np.linalg.norm(acc.mean(0) - big.mean(0))
    cov_acc = np.cov(acc, rowvar=False)
    cov_big = np.cov(big, rowvar=False)
    cov_err = np.linalg.norm(cov_acc - cov_big) / max(np.linalg.norm(cov_big), 1e-12)

    assert mean_err < 1e-3, f"定理3失败: 均值不一致 mean_err={mean_err:.3e}"
    assert cov_err < 0.05, f"定理3失败: 协方差不一致 rel_err={cov_err:.4f}"
    print(f"OK 定理3 (分布等价)    mean_err={mean_err:.3e}  cov_rel_err={cov_err:.4f}")


# ---------------------------------------------------------------------------
# 定理4：动态下累积噪声协方差 = 大批次噪声协方差（线性 fitness，漂移恒为 a）
# ---------------------------------------------------------------------------
def test_dynamic_noise_covariance_match():
    rng = np.random.default_rng(3)
    x = np.zeros(D)
    eta_prime = 0.05
    K, N_s = 4, 50                       # N_L = K*N_s = 200
    eta = eta_prime / K
    reps = 16000

    dl = np.empty((reps, D))
    ds = np.empty((reps, D))
    for r in range(reps):
        # 线性面下漂移恒为 η'·a，故 Δx 的唯一随机来源是噪声 → 可干净测协方差
        xl = macro_step_large(x, eta_prime, K * N_s, rng, f_lin_vec)
        xs = macro_step_small(x, eta, N_s, K, rng, f_lin_vec)
        dl[r] = xl - x
        ds[r] = xs - x

    # 漂移也应对齐（线性面下精确相等）
    drift_err = np.linalg.norm(dl.mean(0) - ds.mean(0)) / max(np.linalg.norm(eta_prime * A), 1e-12)
    cov_l = np.cov(dl, rowvar=False)
    cov_s = np.cov(ds, rowvar=False)
    # 总噪声能量（trace）是"噪声强度逐阶相等"最稳健的标量判据
    trace_err = abs(np.trace(cov_l) - np.trace(cov_s)) / max(np.trace(cov_l), 1e-12)

    assert drift_err < 0.05, f"定理4失败: 漂移不一致 drift_err={drift_err:.4f}"
    assert trace_err < 0.05, f"定理4失败: 噪声能量不一致 trace_rel_err={trace_err:.4f}"
    print(f"OK 定理4 (噪声协方差)  drift_err={drift_err:.4f}  trace_rel_err={trace_err:.4f}")


# ---------------------------------------------------------------------------
# 定理5：漂移差为 O(η'²)（确定性漂移，二次面，一阶求积误差）
# ---------------------------------------------------------------------------
def test_drift_error_second_order():
    x = np.zeros(D)
    eta1, eta2 = 0.2, 0.1          # 宏观步长减半
    K = 10

    e1 = np.linalg.norm(drift_small_det(x, eta1 / K, K, grad_quad)
                        - drift_large_det(x, eta1, grad_quad))
    e2 = np.linalg.norm(drift_small_det(x, eta2 / K, K, grad_quad)
                        - drift_large_det(x, eta2, grad_quad))

    # O(η'²)：η' 减半 → 误差应降为约 1/4
    ratio = e2 / e1
    assert 0.15 < ratio < 0.40, f"定理5失败: 误差比应≈0.25, 实际={ratio:.4f}"
    print(f"OK 定理5 (漂移O(η'²))  e(η'=0.2)={e1:.6f}  e(η'=0.1)={e2:.6f}  比值={ratio:.4f} (应≈0.25)")


# ---------------------------------------------------------------------------
# 定理6：η' → 0 时两者收敛到同一极限（确定性梯度流 + 随机弱收敛）
# ---------------------------------------------------------------------------
def test_weak_convergence_same_limit():
    x = np.zeros(D)
    # 6a) 确定性：η' 越小，小批次多次与大批次轨迹终点越接近（都趋近同一梯度流）
    K = 20
    gaps = []
    for eta_prime in [0.5, 0.25, 0.125]:
        xl = drift_large_det(x, eta_prime, grad_quad)
        xs = drift_small_det(x, eta_prime / K, K, grad_quad)
        gaps.append(np.linalg.norm(xl - xs))
    assert gaps[-1] < gaps[0] * 0.3, f"定理6失败: 轨迹差未随 η'→0 缩小, gaps={gaps}"

    # 6b) 随机（弱收敛）：η' 减半，大批次与小批次终点的均值差随之二次方衰减
    rng = np.random.default_rng(4)
    N_s, K, reps = 50, 4, 3000
    mean_gaps = []
    for eta_prime in [0.08, 0.04]:
        eta = eta_prime / K
        dl = np.empty((reps, D)); ds = np.empty((reps, D))
        for r in range(reps):
            dl[r] = macro_step_large(x, eta_prime, K * N_s, rng, f_quad_vec)
            ds[r] = macro_step_small(x, eta, N_s, K, rng, f_quad_vec)
        mean_gaps.append(np.linalg.norm(dl.mean(0) - ds.mean(0)))
    ratio = mean_gaps[1] / mean_gaps[0]
    assert ratio < 0.5, f"定理6失败: 均值差未按 η'² 衰减, 比值={ratio:.4f}"
    print(f"OK 定理6 (极限同一)    确定性gaps={[f'{g:.4f}' for g in gaps]}  "
          f"随机均值差比={ratio:.4f} (应<0.5)")


# ---------------------------------------------------------------------------
# 定理7：轨迹误差（均值意义）∝ η'²
# ---------------------------------------------------------------------------
def test_trajectory_error_second_order():
    x = np.zeros(D)
    rng = np.random.default_rng(5)
    N_s, K, reps = 50, 4, 4000
    errs = []
    for eta_prime in [0.08, 0.04]:
        eta = eta_prime / K
        dl = np.empty((reps, D)); ds = np.empty((reps, D))
        for r in range(reps):
            dl[r] = macro_step_large(x, eta_prime, K * N_s, rng, f_quad_vec)
            ds[r] = macro_step_small(x, eta, N_s, K, rng, f_quad_vec)
        # 均值意义下随机项相消，仅剩 O(η'²) 漂移差
        errs.append(np.linalg.norm(dl.mean(0) - ds.mean(0)))
    ratio = errs[1] / errs[0]
    assert 0.15 < ratio < 0.45, f"定理7失败: 误差比应≈0.25, 实际={ratio:.4f}"
    print(f"OK 定理7 (误差O(η'²))  e(η'=0.08)={errs[0]:.6f}  e(η'=0.04)={errs[1]:.6f}  比值={ratio:.4f} (应≈0.25)")


if __name__ == "__main__":
    test_stein_unbiasedness()
    test_gradient_accumulation_exact()
    test_frozen_distribution_equivalence()
    test_dynamic_noise_covariance_match()
    test_drift_error_second_order()
    test_weak_convergence_same_limit()
    test_trajectory_error_second_order()
    print("ALL BATCH-EQUIVALENCE TESTS PASSED")
