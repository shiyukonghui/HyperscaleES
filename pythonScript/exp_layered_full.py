"""分层自由度突破实验：输出层 fc3 满秩 + fc1/fc2 LoRA。

背景（框架级探索）：
  - 全 LoRA（rank72）：best_train 封顶 0.83——梯度限低秩子空间，表达力不足。
  - 全满秩 ES：best_train 仅 0.59——150k 维纯 ES 搜索维度爆炸，无法收敛。
  本实验尝试**按层分配自由度**：输出层 fc3（10x128=1280 参数，ES 完全可搜索）用满秩
  满秩更新 + 满秩扰动（FullMM），给分类头最大表达力；fc1/fc2 保持 LoRA（低维可探索）。
  期望：分类头能精确拟合标签，又不引入大维度的搜索爆炸。

实现：复用 exp_fullrank_es.py 的 FullMM 模块；模型为 fc1/fc2 = MM(LoRA rank72)、
  fc3 = FullMM(满秩)。batch=6000（LoRA 兼容 + fc3 满秩显存小），loglik + 线性衰减。

用法：
    python exp_layered_full.py
"""

import csv
import os

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.base_model import Model, CommonInit
from hyperscalees.models.common import (
    merge_inits, call_submodule, PARAM, simple_es_tree_key, MM, Parameter,
)
from hyperscalees.models.snn import run_lif
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays, poisson_encode, accuracy_from_logits,
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

# ---- 配置 ----------------------------------------------------------------
BATCH_RATIO = 0.1          # 6000（LoRA rank72 兼容）
RANK = 72                  # fc1/fc2 的 LoRA 秩
LR_START = 0.01
LR_END = 0.001             # 线性衰减（best_train 最优调度）
SIGMA = 0.2
MAX_EPOCHS = 6000
T = 8
seed = 42
VAL = 1024
EVAL_EVERY = 100
CSV_PATH = "results_layered_full.csv"


def reward_loglik(logits, labels):
    """正确类 log_softmax（连续、无界）。"""
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), labels]


class FullMM(Model):
    """满秩矩阵乘模块：forward 用满秩高斯扰动，更新走满秩（es_map=PARAM）。"""
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = (jax.random.normal(key, (out_dim, in_dim)) * scale).astype(dtype=dtype)
        return CommonInit(None, params, (), PARAM)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        w = common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params, common_params.noiser_params,
            common_params.params, common_params.es_tree_key, common_params.iterinfo,
        )
        return x @ jnp.asarray(w).T


class LayeredSNN(Model):
    """2 层 SNN：fc1/fc2 = LoRA(MM)，fc3 = 满秩(FullMM)，v_th 可训练。"""

    @classmethod
    def rand_init(cls, key, in_dim, hidden_dims, num_classes, tau_m=20.0,
                  v_th=0.3, dtype=jnp.float32):
        in_key, h1_key, h2_key, out_key = jax.random.split(key, 4)
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        layers = merge_inits(
            fc1=MM.rand_init(in_key, in_dim, hidden_dims[0], dtype),
            fc2=MM.rand_init(h1_key, hidden_dims[0], hidden_dims[1], dtype),
            fc3=FullMM.rand_init(h2_key, hidden_dims[1], num_classes, dtype),
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
        logits = call_submodule(FullMM, "fc3", common_params, rate)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]
num_envs = int(round(BATCH_RATIO * n_train))


def main():
    print(f"分层自由度: batch={num_envs}, fc1/fc2 LoRA rank={RANK}, fc3 满秩, "
          f"lr 线性 {LR_START}->{LR_END}, sigma={SIGMA}, epochs={MAX_EPOCHS}, "
          f"2层[128,128], v_th 可训练, T={T}, loglik, seed={seed}", flush=True)

    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = LayeredSNN.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    lr_sched = optax.linear_schedule(LR_START, LR_END, transition_steps=MAX_EPOCHS)
    frozen_noiser, n0 = NOISER.init_noiser(
        p0, SIGMA, lr_sched, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=RANK,
        noise_reuse=0, group_size=0,
    )
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: LayeredSNN.forward(
            NOISER, frozen_noiser, n, frozen_p, p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0)))
    jit_forward_eval = jax.jit(jax.vmap(
        lambda n, p, x: LayeredSNN.forward(
            NOISER, frozen_noiser, n, frozen_p, p, es_tree_key, None, x),
        in_axes=(None, None, 0)))
    jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
        frozen_noiser, n, p, es_tree_key, f, i, es_map))

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
        raw = reward_loglik(logits, labels)
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
            print(f"  epoch {epoch:5d} | val {last_val:.4f} | best_train {best:.4f}",
                  flush=True)
    if last_val == 0.0 or MAX_EPOCHS % EVAL_EVERY != 0:
        last_val = eval_acc()

    print(f"\n===== 分层自由度结果 =====")
    print(f"  fc1/fc2 LoRA rank={RANK}, fc3 满秩, batch={num_envs}, "
          f"lr 线性 {LR_START}->{LR_END} {MAX_EPOCHS}ep loglik")
    print(f"  best_train={best:.4f}  val_acc={last_val:.4f}")

    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["rank", "batch", "lr_start", "lr_end", "max_epochs",
                        "val_acc", "best_train"])
        w.writerow([RANK, num_envs, LR_START, LR_END, MAX_EPOCHS,
                    round(last_val, 5), round(best, 5)])
    print(f"results appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
