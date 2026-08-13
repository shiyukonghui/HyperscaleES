"""2 层 SNN 奖励函数对比（重新测试文档 7.5 节，改用最新配置 + 固定 LR=0.01）。

背景：
  文档 7.5 只在 num_envs=128 / lr=0.03 的老配置下对比了 硬0/1 / log-likelihood / sigmoid
  margin 三种奖励。本脚本改用最近的 2 层可训练 v_th 配置（batch=0.2、rank=32、T=8、
  sigma=0.2、3000 epochs、group_size=0），学习率固定小 LR=0.01，系统对比更多种类的
  奖励/适应度函数（含从网络检索到的温度缩放 softmax 概率、原始 margin 等设计）。

奖励函数（per-sample, 与 fitness_from_logits(logits, labels) 同接口）：
  - binary          硬 0/1 离散奖励（7.5 基线）
  - loglik          正确类 log_softmax（连续、无界，( -inf, 0 ]）——7.5 曾崩溃
  - sigmoid_margin  sigmoid(正确类 logit - 最大其他类 logit)（连续、有界 (0,1)）——7.5 基线
  - softmax_prob    正确类温度缩放 softmax 概率（连续、有界 (0,1)，归一化置信度）
  - margin          正确类 logit - 最大其他类 logit 原始边距（连续、有符号、无界）
  - binary_conf     硬 0/1 * softmax 置信度（稀疏但按置信度加权，兼顾稳定性与密度）

用法：
    python exp_reward_sweep.py
"""

import csv
import os

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.base_model import Model, CommonInit
from hyperscalees.models.common import (
    merge_inits, call_submodule, simple_es_tree_key, MM, Parameter,
)
from hyperscalees.models.snn import run_lif
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays, poisson_encode, fitness_from_logits, accuracy_from_logits,
)

NOISER = hs.noiser.eggroll.EggRoll
DTYPE = jnp.float32
IN_DIM = 28 * 28
HIDDEN = [128, 128]      # 固定 2 层
NUM_CLASSES = 10
MNIST_DIR = os.environ.get("MNIST_DIR") or r"D:\Rust\snn_t1\mnist_data"
# WSL 下 Windows 盘符路径不生效，自动回退到 /mnt/d 挂载路径（原生 Windows 行为不变）
if not os.path.isdir(MNIST_DIR) and os.path.isdir("/mnt/d/Rust/snn_t1/mnist_data"):
    MNIST_DIR = "/mnt/d/Rust/snn_t1/mnist_data"

# ---- 固定配置 ----------------------------------------------------------------
BATCH_RATIO = 0.2
LR = 0.01            # 固定小学习率（用户要求调低到固定 lr=0.01）
RANK = 32
MAX_EPOCHS = 3000
TRAIN_TAU = False
seed = 0
VAL = 1024
EVAL_EVERY = 50
# 其他超参用 7.16 基线：T=8, sigma=0.2, noise_reuse=0, group_size=0
T = 8
SIGMA = 0.2
NOISE_REUSE = 0
GROUP_SIZE = 0

CSV_PATH = "results_reward_sweep_lr001.csv"


# ---- 奖励函数 ----------------------------------------------------------------
def reward_binary(logits, labels):
    """硬 0/1 离散奖励（稀疏）。(batch,) ∈ {0,1}。"""
    pred = jnp.argmax(logits, axis=-1)
    return (pred == labels).astype(jnp.float32)


def reward_loglik(logits, labels):
    """正确类 log_softmax（连续、无界，( -inf, 0 ]）。"""
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), labels]


def reward_sigmoid_margin(logits, labels):
    """sigmoid(正确类 logit - 最大其他类 logit)，连续、有界 (0,1)。"""
    logits_l = logits - jnp.max(logits, axis=-1, keepdims=True)
    correct = logits_l[jnp.arange(logits.shape[0]), labels]
    # 屏蔽正确类后取最大其他类
    mask = jnp.arange(logits.shape[-1])[None, :] == labels[:, None]
    other_max = jnp.where(mask, -jnp.inf, logits_l).max(axis=-1)
    return jax.nn.sigmoid(correct - other_max)


def reward_softmax_prob(logits, labels):
    """正确类温度缩放 softmax 概率（连续、有界 (0,1)，归一化置信度）。"""
    p = jax.nn.softmax(logits, axis=-1)
    return p[jnp.arange(logits.shape[0]), labels]


def reward_margin(logits, labels):
    """正确类 logit - 最大其他类 logit 原始边距（连续、有符号、无界）。"""
    logits_l = logits - jnp.max(logits, axis=-1, keepdims=True)
    correct = logits_l[jnp.arange(logits.shape[0]), labels]
    mask = jnp.arange(logits.shape[-1])[None, :] == labels[:, None]
    other_max = jnp.where(mask, -jnp.inf, logits_l).max(axis=-1)
    return correct - other_max


def reward_binary_conf(logits, labels):
    """硬 0/1 * softmax 置信度：稀疏但按置信度加权，兼顾稳定性与梯度密度。"""
    binary = reward_binary(logits, labels)
    p = jax.nn.softmax(logits, axis=-1)
    conf = p[jnp.arange(logits.shape[0]), labels]
    return binary * conf


# 扫描的奖励函数（保留 7.5 的 3 个基线 + 网络检索/新设计的 3 个）
REWARDS = {
    "binary": reward_binary,
    "loglik": reward_loglik,
    "sigmoid_margin": reward_sigmoid_margin,
    "softmax_prob": reward_softmax_prob,
    "margin": reward_margin,
    "binary_conf": reward_binary_conf,
}


# ---- 模型 ---------------------------------------------------------------------
class TrainableVthSNN2L(Model):
    """2 层 SNN，v_th 可训练（softplus 参数化）。与 exp_params_sweep.py 同款固定 2 层。"""

    @classmethod
    def rand_init(cls, key, in_dim, hidden_dims, num_classes, tau_m=20.0,
                  v_th=0.3, dtype=jnp.float32):
        in_key, h1_key, h2_key, out_key = jax.random.split(key, 4)
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        layers = merge_inits(
            fc1=MM.rand_init(in_key, in_dim, hidden_dims[0], dtype),
            fc2=MM.rand_init(h1_key, hidden_dims[0], hidden_dims[1], dtype),
            fc3=MM.rand_init(h2_key, hidden_dims[1], num_classes, dtype),
            out_gain=Parameter.rand_init(out_key, None, None, jnp.ones((1,)), dtype=dtype),
            v_th=Parameter.rand_init(None, None, None, raw_vth0, dtype=dtype),
        )
        return CommonInit({"tau_m": jnp.asarray(tau_m, dtype=dtype)},
                          layers.params, layers.scan_map, layers.es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        x = x.astype(common_params.params["fc1"].dtype)
        v_th = jax.nn.softplus(call_submodule(Parameter, "v_th", common_params))
        tau = common_params.frozen_params["tau_m"]
        lif_params = {"tau_m": tau, "v_th": v_th}

        def proj1(x_t):
            return call_submodule(MM, "fc1", common_params, x_t)
        cur1 = jax.vmap(proj1)(x)
        spikes1 = run_lif(lif_params, cur1, jnp.zeros((HIDDEN[0],), dtype=cur1.dtype))

        def proj2(x_t):
            return call_submodule(MM, "fc2", common_params, x_t)
        cur2 = jax.vmap(proj2)(spikes1)
        spikes2 = run_lif(lif_params, cur2, jnp.zeros((HIDDEN[1],), dtype=cur2.dtype))

        rate = jnp.mean(spikes2, axis=0)
        logits = call_submodule(MM, "fc3", common_params, rate)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]
num_envs = int(round(BATCH_RATIO * n_train))


# 每个奖励函数独立建模型+noiser，跑满 MAX_EPOCHS，返回 (last_val, best_train)
def run_reward(reward_fn):
    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    # 固定小学习率（用户指定 lr=0.01）
    lr_sched = LR
    frozen_noiser, n0 = NOISER.init_noiser(
        p0, SIGMA, lr_sched, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=RANK,
        noise_reuse=NOISE_REUSE, group_size=GROUP_SIZE,
    )
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: TrainableVthSNN2L.forward(
            NOISER, frozen_noiser, n, frozen_p, p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0)))
    jit_forward_eval = jax.jit(jax.vmap(
        lambda n, p, x: TrainableVthSNN2L.forward(
            NOISER, frozen_noiser, n, frozen_p, p, es_tree_key, None, x),
        in_axes=(None, None, 0)))
    jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
        frozen_noiser, n, p, es_tree_key, f, i, es_map))

    # 用可变容器保存步间状态，避免依赖模块级 global
    state = {"data_key": dkey, "noiser_params": n0, "params": p0}

    def eval_acc():
        idx = jax.random.permutation(jax.random.key(1), x_te.shape[0])[:VAL]
        imgs = jnp.asarray(x_te[idx], dtype=DTYPE)
        labels = jnp.asarray(y_te[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, T, jax.random.key(1)).transpose(1, 0, 2)
        logits = jit_forward_eval(state["noiser_params"], state["params"], spikes)
        return float(accuracy_from_logits(logits, labels))

    def step(epoch):
        dk, enc, per = jax.random.split(state["data_key"], 3)
        state["data_key"] = dk
        idx = jax.random.permutation(per, n_train)[:num_envs]
        imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
        labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, T, enc).transpose(1, 0, 2)
        it = (jnp.full(num_envs, epoch, dtype=jnp.int32),
              jnp.arange(num_envs, dtype=jnp.int32))
        logits = jit_forward(state["noiser_params"], state["params"], it, spikes)
        raw = reward_fn(logits, labels)
        fits = NOISER.convert_fitnesses(frozen_noiser, state["noiser_params"], raw)
        state["noiser_params"], state["params"] = jit_update(
            state["noiser_params"], state["params"], fits, it)
        return float(accuracy_from_logits(logits, labels))

    # 标定 10 次（含 JIT）
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
    if last_val == 0.0:
        last_val = eval_acc()
    return last_val, best


def main():
    print(f"固定: batch={num_envs}, lr={LR}, rank={RANK}, epochs={MAX_EPOCHS}, "
          f"2 层 [128,128], v_th 可训练, T={T}, sigma={SIGMA}, group_size={GROUP_SIZE}", flush=True)

    results = []
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["reward", "val_acc", "best_train"])
        f.flush()

    for name, fn in REWARDS.items():
        print(f"[{name}] ...", flush=True)
        val, best = run_reward(fn)
        results.append((name, val, best))
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([name, round(val, 5), round(best, 5)])
            f.flush()
        print(f"  [{name}] val_acc={val:.4f} best_train={best:.4f}", flush=True)

    print("\n===== 奖励函数对比汇总（固定 lr=0.01） =====")
    for name, val, best in results:
        print(f"  {name:<16}  val_acc={val:.4f}  best_train={best:.4f}")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
