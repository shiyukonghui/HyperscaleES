"""STDP（脉冲时序依赖可塑性）+ 监督读出 突破实验。

背景：
  纯 ES + LoRA 框架下 best_train 封顶 0.833，根因是 ES 无梯度、仅靠全局随机扰动 +
  适应度加权更新，对训练集的精确拟合能力有限。STDP 是 SNN 的天然**局部**学习规则，
  基于突触前(pre)/后(post)神经元的**发放时序**做精确、因果的权重更新，是比 ES 更强的
  局部信号。

本实验方案（自包含，不依赖 NOISER）：
  - fc1（784->128）、fc2（128->128）用**非对称 STDP** 无监督更新：
      ΔW = A_plus * post_spike * pre_trace - A_minus * pre_spike * post_trace
      pre_trace / post_trace 为发放痕迹（指数衰减）。
  - fc3（128->10）用 **softmax 回归梯度下降**（rate->logits 可微）做监督分类。
  - v_th / tau 固定（0.3 / 20.0），LIF 复用 run_lif。

用法：
    python exp_stdp.py
"""

import csv
import os

import jax
import jax.numpy as jnp
import optax

from hyperscalees.models.snn import run_lif
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays, poisson_encode, accuracy_from_logits,
)

DTYPE = jnp.float32
IN_DIM = 28 * 28
HIDDEN = [256, 256]
NUM_CLASSES = 10
MNIST_DIR = os.environ.get("MNIST_DIR") or r"D:\Rust\snn_t1\mnist_data"
# WSL 下 Windows 盘符路径不生效，自动回退到 /mnt/d 挂载路径（原生 Windows 行为不变）
if not os.path.isdir(MNIST_DIR) and os.path.isdir("/mnt/d/Rust/snn_t1/mnist_data"):
    MNIST_DIR = "/mnt/d/Rust/snn_t1/mnist_data"

# ---- 配置 ----------------------------------------------------------------
T = 8
BATCH = 6000
MAX_EPOCHS = 10000
TAU_M = 20.0
V_TH = 0.3

# STDP 超参
A_PLUS = 0.005       # LTP（post 在 pre 后发放，增强）幅度
A_MINUS = 0.003      # LTD（post 在 pre 前发放，抑制）幅度
TRACE_DECAY = 0.9    # 发放痕迹衰减因子
STDP_LR = 1.0        # STDP 更新缩放
W_CLIP = 3.0         # 权重裁剪，防止 STDP 发散

# 输出层监督超参
LR_FC3 = 0.01        # softmax 读出层学习率

seed = 42
VAL = 1024
EVAL_EVERY = 100
CSV_PATH = "results_stdp.csv"


def compute_trace(spikes, decay):
    """发放痕迹：trace[t] = trace[t-1]*decay + spikes[t]（含当前时刻尖峰）。

    spikes: (T, ...) 0/1。返回 (T, ...) 同形状的痕迹。
    """
    def step(carry, s):
        carry = carry * decay + s
        return carry, carry
    init = jnp.zeros(spikes.shape[1:], dtype=spikes.dtype)
    _, traces = jax.lax.scan(step, init, spikes)
    return traces


def stdp_update(pre_spikes, post_spikes, a_plus, a_minus, decay, fitness=None):
    """计算 STDP 权重更新 ΔW（post x pre），可选按 fitness 做监督调制。

    pre_spikes:  (T, batch, in_dim)
    post_spikes: (T, batch, out_dim)
    fitness:     可选 (batch,) 每样本标量权重（监督信号），None 表示无监督均匀权重。
    返回 ΔW: (out_dim, in_dim)
    """
    pre_trace = compute_trace(pre_spikes, decay)    # (T, batch, in_dim)
    post_trace = compute_trace(post_spikes, decay)  # (T, batch, out_dim)
    batch = pre_spikes.shape[1]
    if fitness is not None:
        # 用 fitness 调制每个样本的贡献（正确样本强化、错误样本抑制）
        ltp = jnp.einsum('tbi,tbj,b->ij', post_spikes, pre_trace, fitness) / batch
        ltd = jnp.einsum('tbj,tbi,b->ij', pre_spikes, post_trace, fitness) / batch
    else:
        ltp = jnp.einsum('tbi,tbj->ij', post_spikes, pre_trace) / batch
        ltd = jnp.einsum('tbj,tbi->ij', pre_spikes, post_trace) / batch
    return a_plus * ltp - a_minus * ltd


# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]


def init_weights(key):
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = (jax.random.normal(k1, (HIDDEN[0], IN_DIM)) / jnp.sqrt(IN_DIM)).astype(DTYPE)
    w2 = (jax.random.normal(k2, (HIDDEN[1], HIDDEN[0])) / jnp.sqrt(HIDDEN[0])).astype(DTYPE)
    w3 = (jax.random.normal(k3, (NUM_CLASSES, HIDDEN[1])) / jnp.sqrt(HIDDEN[1])).astype(DTYPE)
    return w1, w2, w3


def forward(w1, w2, w3, x):
    """前向：返回 logits 与中间尖峰（用于 STDP）。

    x: (T, batch, in_dim)
    返回 (logits, spikes1, spikes2)
    """
    lif_params = {"tau_m": jnp.asarray(TAU_M, dtype=DTYPE), "v_th": jnp.asarray(V_TH, dtype=DTYPE)}

    batch = x.shape[1]
    cur1 = jnp.einsum('tbi,oi->tbo', x, w1)            # (T, batch, h1)
    spikes1 = run_lif(lif_params, cur1, jnp.zeros((batch, HIDDEN[0]), dtype=DTYPE))

    cur2 = jnp.einsum('tbi,oi->tbo', spikes1, w2)      # (T, batch, h2)
    spikes2 = run_lif(lif_params, cur2, jnp.zeros((batch, HIDDEN[1]), dtype=DTYPE))

    rate = jnp.mean(spikes2, axis=0)                   # (batch, h2)
    logits = rate @ w3.T                              # (batch, C)
    return logits, spikes1, spikes2


def main():
    print(f"STDP: batch={BATCH}, T={T}, A+={A_PLUS}, A-={A_MINUS}, decay={TRACE_DECAY}, "
          f"lr_fc3={LR_FC3}, {MAX_EPOCHS} epochs, [128,128], seed={seed}", flush=True)

    key = jax.random.key(seed)
    w_key, dkey = jax.random.split(key, 2)
    w1, w2, w3 = init_weights(w_key)
    # softmax 回归优化器（作用于 fc3）
    fc3_opt = optax.adam(LR_FC3)
    fc3_state = fc3_opt.init(w3)

    @jax.jit
    def train_step(w1, w2, w3, fc3_state, spikes, labels):
        # 前向
        logits, spikes1, spikes2 = forward(w1, w2, w3, spikes)
        # 监督信号：loglik 的 z-score（正确样本为正、错误样本为负），用于调制 STDP
        loglik = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels]
        fitness = (loglik - jnp.mean(loglik)) / (jnp.sqrt(jnp.var(loglik)) + 1e-5)
        # 监督 STDP 更新 fc1 / fc2
        dw1 = stdp_update(spikes, spikes1, A_PLUS, A_MINUS, TRACE_DECAY, fitness)
        dw2 = stdp_update(spikes1, spikes2, A_PLUS, A_MINUS, TRACE_DECAY, fitness)
        w1 = jnp.clip(w1 + STDP_LR * dw1, -W_CLIP, W_CLIP)
        w2 = jnp.clip(w2 + STDP_LR * dw2, -W_CLIP, W_CLIP)
        # fc3 softmax 回归梯度（rate -> logits 可微，直接计算 softmax 交叉熵对 w3 的梯度）
        prob = jax.nn.softmax(logits, axis=-1)
        onehot = jax.nn.one_hot(labels, NUM_CLASSES)
        rate = jnp.mean(spikes2, axis=0)
        grad_w3 = jnp.einsum('bc,bh->ch', (prob - onehot), rate) / labels.shape[0]
        updates, fc3_state = fc3_opt.update(grad_w3, fc3_state, w3)
        w3 = optax.apply_updates(w3, updates)
        return w1, w2, w3, fc3_state, logits

    @jax.jit
    def eval_forward(w1, w2, w3, spikes):
        logits, _, _ = forward(w1, w2, w3, spikes)
        return logits

    state = {"data_key": dkey, "w1": w1, "w2": w2, "w3": w3, "fc3_state": fc3_state}

    def step(epoch):
        dk, enc, per = jax.random.split(state["data_key"], 3)
        state["data_key"] = dk
        idx = jax.random.permutation(per, n_train)[:BATCH]
        imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
        labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, T, enc)  # (T, batch, in_dim)
        # 改用确定性 rate 输入：每个时间步输入相同像素值（持续电流），
        # 使 pre/post 尖峰时序关系稳定可学（Poisson 随机尖峰无时间结构，STDP 学到噪声）。
        x = jnp.tile(imgs[None], (T, 1, 1))    # (T, batch, in_dim)
        w1, w2, w3, fc3_state, logits = train_step(
            state["w1"], state["w2"], state["w3"], state["fc3_state"], x, labels)
        state.update(w1=w1, w2=w2, w3=w3, fc3_state=fc3_state)
        return float(accuracy_from_logits(logits, labels))

    def eval_acc():
        idx = jax.random.permutation(jax.random.key(1), x_te.shape[0])[:VAL]
        imgs = jnp.asarray(x_te[idx], dtype=DTYPE)
        labels = jnp.asarray(y_te[idx], dtype=jnp.int32)
        x = jnp.tile(imgs[None], (T, 1, 1))    # 与训练一致的确定性 rate 输入
        logits = eval_forward(state["w1"], state["w2"], state["w3"], x)
        return float(accuracy_from_logits(logits, labels))

    # 标定 + JIT
    best = 0.0
    for e in range(10):
        acc = step(e)
        best = max(best, acc)
    last_val = 0.0
    epoch = 10
    while epoch < MAX_EPOCHS:
        acc = step(epoch)
        best = max(best, acc)
        epoch += 1
        if epoch % EVAL_EVERY == 0:
            last_val = eval_acc()
            print(f"  epoch {epoch:5d} | val {last_val:.4f} | best_train {best:.4f}",
                  flush=True)
    if last_val == 0.0 or MAX_EPOCHS % EVAL_EVERY != 0:
        last_val = eval_acc()

    print(f"\n===== STDP 结果 =====")
    print(f"  A+={A_PLUS} A-={A_MINUS} decay={TRACE_DECAY} lr_fc3={LR_FC3} {MAX_EPOCHS}ep")
    print(f"  best_train={best:.4f}  val_acc={last_val:.4f}")

    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["batch", "T", "a_plus", "a_minus", "decay", "lr_fc3",
                        "max_epochs", "val_acc", "best_train"])
        w.writerow([BATCH, T, A_PLUS, A_MINUS, TRACE_DECAY, LR_FC3, MAX_EPOCHS,
                    round(last_val, 5), round(best, 5)])
    print(f"results appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
