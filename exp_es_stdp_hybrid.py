"""ES + STDP 混合突破实验：低秩 ES 全局搜索 + STDP 满秩局部精调。

核心思想（发挥想象力结合两种理论）：
  - **ES(LoRA)**：低秩子空间的全局适应度搜索（loglik fitness），样本效率高，
    但低秩子空间限制了权重更新的表达力（best_train 封顶 0.833）。
  - **STDP**：基于 pre/post 尖峰时序的**满秩**局部权重更新，能沿任意方向精调权重
    （突破低秩限制），但纯 STDP 缺乏全局分类方向（仅 0.42）。
  - **混合**：每步总更新 = ES 低秩梯度（全局方向）+ STDP 满秩更新（局部精调），
    两者互补，期望突破 0.833 逼近 0.9。

实现要点（自包含，参考 eggroll.py 的 LoRA ES 公式）：
  - LoRA ES：采样低秩噪声 A@B.T，fitness(z-score) 加权 -> einsum(fitness*A, B)/N。
  - STDP：确定性 rate 输入（有时间结构），trace-based 非对称 STDP，fitness 调制。
  - 输出层：softmax 回归梯度（rate->logits 可微）。
  - 三者梯度用 optax.adam 统一更新。

用法：
    python exp_es_stdp_hybrid.py
"""

import csv
import os

import jax
import jax.numpy as jnp
import optax

from hyperscalees.models.snn import run_lif
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays, accuracy_from_logits,
)

DTYPE = jnp.float32
IN_DIM = 28 * 28
HIDDEN = [128, 128]
NUM_CLASSES = 10
MNIST_DIR = os.environ.get("MNIST_DIR") or r"D:\Rust\snn_t1\mnist_data"
# WSL 下 Windows 盘符路径不生效，自动回退到 /mnt/d 挂载路径（原生 Windows 行为不变）
if not os.path.isdir(MNIST_DIR) and os.path.isdir("/mnt/d/Rust/snn_t1/mnist_data"):
    MNIST_DIR = "/mnt/d/Rust/snn_t1/mnist_data"

# ---- 配置 ----------------------------------------------------------------
T = 8
BATCH = 2000
MAX_EPOCHS = 5000
TAU_M = 20.0
V_TH = 0.3

# LoRA ES 超参
RANK = 72
SIGMA = 0.2
LR_ES = 0.01        # ES 梯度学习率

# STDP 超参
A_PLUS = 0.005
A_MINUS = 0.003
TRACE_DECAY = 0.9
STDP_LR = 1.0       # STDP 满秩更新缩放
W_CLIP = 3.0

# 输出层 softmax 回归超参
LR_FC3 = 0.01

seed = 42
VAL = 1024
EVAL_EVERY = 100
CSV_PATH = "results_es_stdp_hybrid.csv"


def compute_trace(spikes, decay):
    """发放痕迹：trace[t] = trace[t-1]*decay + spikes[t]。"""
    def step(carry, s):
        carry = carry * decay + s
        return carry, carry
    init = jnp.zeros(spikes.shape[1:], dtype=spikes.dtype)
    _, traces = jax.lax.scan(step, init, spikes)
    return traces


def stdp_update(pre_spikes, post_spikes, a_plus, a_minus, decay, fitness):
    """STDP 满秩权重更新 ΔW（post x pre），按 fitness 调制。

    pre_spikes:  (T, batch, in_dim)；post_spikes: (T, batch, out_dim)
    fitness:     (batch,) 每样本标量（监督调制）
    返回 ΔW: (out_dim, in_dim)
    """
    pre_trace = compute_trace(pre_spikes, decay)
    post_trace = compute_trace(post_spikes, decay)
    batch = pre_spikes.shape[1]
    ltp = jnp.einsum('tbi,tbj,b->ij', post_spikes, pre_trace, fitness) / batch
    ltd = jnp.einsum('tbj,tbi,b->ij', pre_spikes, post_trace, fitness) / batch
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


def sample_lora(key, shape, rank, sigma):
    """采样低秩扰动 A@B.T（shape: (a,b)），噪声 std = sigma/sqrt(rank)。

    返回 (A, B)：A (a,r), B (b,r)。
    """
    a, b = shape
    lora = jax.random.normal(key, (a + b, rank), dtype=DTYPE) * (sigma / jnp.sqrt(rank))
    return lora[b:], lora[:b]  # A (a,r), B (b,r)


def forward(w1, w2, w3, x, v_th):
    """裸前向（无噪声），返回 logits 与中间尖峰。x: (T, batch, in_dim)。"""
    lif_params = {"tau_m": jnp.asarray(TAU_M, dtype=DTYPE), "v_th": v_th}
    batch = x.shape[1]
    cur1 = jnp.einsum('tbi,oi->tbo', x, w1)
    spikes1 = run_lif(lif_params, cur1, jnp.zeros((batch, HIDDEN[0]), dtype=DTYPE))
    cur2 = jnp.einsum('tbi,oi->tbo', spikes1, w2)
    spikes2 = run_lif(lif_params, cur2, jnp.zeros((batch, HIDDEN[1]), dtype=DTYPE))
    rate = jnp.mean(spikes2, axis=0)
    logits = rate @ w3.T
    return logits, spikes1, spikes2, rate


def forward_perturbed(w1, w2, w3, x, A1, B1, A2, B2, A3, B3, v_th):
    """带低秩扰动的批量前向（在线低秩计算，不物化满秩扰动）。

    A1/B1: (batch, h1, r) / (batch, in_dim, r)；扰动 ΔW1[b] = A1[b] @ B1[b].T。
    返回 logits: (batch, C)。
    """
    lif_params = {"tau_m": jnp.asarray(TAU_M, dtype=DTYPE), "v_th": v_th}
    batch = x.shape[1]
    # fc1: x @ (W1 + A1@B1.T).T = x @ W1.T + (x @ B1) @ A1.T
    base1 = jnp.einsum('tbi,oi->tbo', x, w1)            # (T,batch,h1)
    xb1 = jnp.einsum('tbi,bir->tbr', x, B1)             # (T,batch,r)
    pert1 = jnp.einsum('tbr,bor->tbo', xb1, A1)         # (T,batch,h1)
    spikes1 = run_lif(lif_params, base1 + pert1, jnp.zeros((batch, HIDDEN[0]), dtype=DTYPE))

    # fc2 同理
    base2 = jnp.einsum('tbi,oi->tbo', spikes1, w2)
    s1b2 = jnp.einsum('tbi,bir->tbr', spikes1, B2)
    pert2 = jnp.einsum('tbr,bor->tbo', s1b2, A2)
    spikes2 = run_lif(lif_params, base2 + pert2, jnp.zeros((batch, HIDDEN[1]), dtype=DTYPE))

    rate = jnp.mean(spikes2, axis=0)                    # (batch, h2)
    # fc3: rate @ (W3 + A3@B3.T).T
    base3 = rate @ w3.T                                 # (batch, C)
    rb3 = jnp.einsum('bh,bhr->br', rate, B3)            # (batch, r)
    pert3 = jnp.einsum('br,bcr->bc', rb3, A3)           # (batch, C)
    return base3 + pert3


def main():
    print(f"ES+STDP混合: batch={BATCH}, rank={RANK}, sigma={SIGMA}, lr_es={LR_ES}, "
          f"stdp_lr={STDP_LR}, lr_fc3={LR_FC3}, {MAX_EPOCHS}ep, [128,128], seed={seed}",
          flush=True)

    key = jax.random.key(seed)
    w_key, dkey = jax.random.split(key, 2)
    w1, w2, w3 = init_weights(w_key)
    v_th = jnp.asarray(V_TH, dtype=DTYPE)
    # optax adam（统一处理三个梯度）
    opt = optax.adam(1.0)  # 学习率在梯度缩放里体现
    opt_state = opt.init((w1, w2, w3))

    @jax.jit
    def train_step(w1, w2, w3, opt_state, x, labels, es_key):
        batch = x.shape[1]
        # ---- 1. 采样 LoRA 噪声因子 A/B（每个样本独立，低秩，不物化满秩扰动）----
        def sample_all(k):
            kk1, kk2, kk3 = jax.random.split(k, 3)
            A1, B1 = sample_lora(kk1, w1.shape, RANK, SIGMA)
            A2, B2 = sample_lora(kk2, w2.shape, RANK, SIGMA)
            A3, B3 = sample_lora(kk3, w3.shape, RANK, SIGMA)
            return A1, B1, A2, B2, A3, B3
        keys = jax.random.split(es_key, batch)
        A1, B1, A2, B2, A3, B3 = jax.vmap(sample_all)(keys)
        # ---- 2. 扰动前向 -> fitness ----
        logits_pert = forward_perturbed(w1, w2, w3, x, A1, B1, A2, B2, A3, B3, v_th)
        loglik = jax.nn.log_softmax(logits_pert, axis=-1)[jnp.arange(batch), labels]
        fitness = (loglik - jnp.mean(loglik)) / (jnp.sqrt(jnp.var(loglik)) + 1e-5)
        # ---- 3. LoRA ES 梯度（参考 eggroll _simple_lora_update）----
        f = fitness[:, None, None]
        g1 = jnp.einsum('bir,bjr->ij', f * A1, B1) / batch  # (h1, in_dim)
        g2 = jnp.einsum('bir,bjr->ij', f * A2, B2) / batch
        g3 = jnp.einsum('bir,bjr->ij', f * A3, B3) / batch
        # 缩放 sqrt(N)（对齐 eggroll 的 _do_update 约定）
        g1 = g1 * jnp.sqrt(batch)
        g2 = g2 * jnp.sqrt(batch)
        g3 = g3 * jnp.sqrt(batch)
        # ---- 4. 裸前向 -> 中间尖峰（STDP + softmax 读出）----
        logits, spikes1, spikes2, rate = forward(w1, w2, w3, x, v_th)
        # STDP 满秩更新
        dw1_stdp = stdp_update(x, spikes1, A_PLUS, A_MINUS, TRACE_DECAY, fitness)
        dw2_stdp = stdp_update(spikes1, spikes2, A_PLUS, A_MINUS, TRACE_DECAY, fitness)
        # softmax 回归梯度（输出层）
        prob = jax.nn.softmax(logits, axis=-1)
        onehot = jax.nn.one_hot(labels, NUM_CLASSES)
        dw3_softmax = jnp.einsum('bc,bh->ch', (prob - onehot), rate) / batch
        # ---- 5. 总梯度（ES 梯度 + STDP/softmax 局部梯度）----
        total_g1 = LR_ES * g1 + STDP_LR * dw1_stdp
        total_g2 = LR_ES * g2 + STDP_LR * dw2_stdp
        total_g3 = LR_ES * g3 + LR_FC3 * dw3_softmax
        # clip 防发散
        total_g1 = jnp.clip(total_g1, -1.0, 1.0)
        total_g2 = jnp.clip(total_g2, -1.0, 1.0)
        total_g3 = jnp.clip(total_g3, -1.0, 1.0)
        grads = (total_g1, total_g2, total_g3)
        updates, opt_state = opt.update(grads, opt_state)
        w1, w2, w3 = optax.apply_updates((w1, w2, w3), updates)
        w1 = jnp.clip(w1, -W_CLIP, W_CLIP)
        w2 = jnp.clip(w2, -W_CLIP, W_CLIP)
        w3 = jnp.clip(w3, -W_CLIP, W_CLIP)
        return w1, w2, w3, opt_state, logits

    @jax.jit
    def eval_forward(w1, w2, w3, x):
        logits, _, _, _ = forward(w1, w2, w3, x, v_th)
        return logits

    state = {"data_key": dkey, "w1": w1, "w2": w2, "w3": w3, "opt_state": opt_state}

    def step(epoch):
        dk, enc, per = jax.random.split(state["data_key"], 3)
        state["data_key"] = dk
        idx = jax.random.permutation(per, n_train)[:BATCH]
        imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
        labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
        x = jnp.tile(imgs[None], (T, 1, 1))  # 确定性 rate 输入
        w1, w2, w3, opt_state, logits = train_step(
            state["w1"], state["w2"], state["w3"], state["opt_state"], x, labels, enc)
        state.update(w1=w1, w2=w2, w3=w3, opt_state=opt_state)
        return float(accuracy_from_logits(logits, labels))

    def eval_acc():
        idx = jax.random.permutation(jax.random.key(1), x_te.shape[0])[:VAL]
        imgs = jnp.asarray(x_te[idx], dtype=DTYPE)
        labels = jnp.asarray(y_te[idx], dtype=jnp.int32)
        x = jnp.tile(imgs[None], (T, 1, 1))
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

    print(f"\n===== ES+STDP 混合结果 =====")
    print(f"  rank={RANK} sigma={SIGMA} lr_es={LR_ES} stdp_lr={STDP_LR} {MAX_EPOCHS}ep")
    print(f"  best_train={best:.4f}  val_acc={last_val:.4f}")

    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["rank", "sigma", "lr_es", "stdp_lr", "lr_fc3", "max_epochs",
                        "val_acc", "best_train"])
        w.writerow([RANK, SIGMA, LR_ES, STDP_LR, LR_FC3, MAX_EPOCHS,
                    round(last_val, 5), round(best, 5)])
    print(f"results appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
