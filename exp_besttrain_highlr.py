"""best_train 强化实验（修正版）：严格复现文档最优配置并延长训练。

背景与教训：
  文档 7.16/7.17 全项目最高 best_train=0.840 的配置是：
  **batch=12000, 固定 LR=0.03, group_size=50, rank=32, T=8, sigma=0.2, 硬 0/1 奖励,
  v_th 可训练, 2 层 [128,128], 3000 epochs**。
  初版脚本误把 batch 降到 6000，破坏了 group_size 的有效前提（整除 + 组内归一化统计），
  gs30/gs50 在 6000 下 val 崩到 0.72，说明 **必须保持 batch=12000 才能复现 0.840**。

  本脚本严格保持上述最优配置（batch=12000, rank=32 避免 OOM），唯一变化维度为
  **训练时长 MAX_EPOCHS（3000 -> 更长）**，检验 best_train 是否在 0.840 基础上随
  训练继续爬升，向 0.9+ 逼近。best_train 是取峰值，长训练更易累积更高值。

用法：
    python exp_besttrain_highlr.py
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

# ---- 严格复现文档 0.840 最优配置，仅延长训练 ----------------------------------
BATCH_RATIO = 0.2          # 12000 —— 必须保持，group_size 有效前提
LR = 0.03                  # 固定大学习率（文档 0.840 的关键）
GROUP_SIZE = 50            # 文档 0.840 的最优组大小
RANK = 32                  # 与文档一致（rank>32 在 batch=12000 会 OOM）
T = 8
SIGMA = 0.2
NOISE_REUSE = 0
MAX_EPOCHS = 12000         # 从文档 3000 延长到 12000，观察 best_train 是否继续爬升
seed = 0                   # 与文档 7.16 同 seed，保证可复现 0.840
VAL = 1024
EVAL_EVERY = 100

CSV_PATH = "results_besttrain_highlr.csv"


def reward_binary(logits, labels):
    """硬 0/1 离散奖励（文档 0.840 用的奖励）。"""
    pred = jnp.argmax(logits, axis=-1)
    return (pred == labels).astype(jnp.float32)


class TrainableVthSNN2L(Model):
    """2 层 SNN，v_th 可训练（softplus 参数化）。"""

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


def main():
    print(f"配置: batch={num_envs}, 固定LR={LR}, group_size={GROUP_SIZE}, rank={RANK}, "
          f"T={T}, sigma={SIGMA}, 硬0/1, {MAX_EPOCHS} epochs, v_th 可训练, "
          f"2层[128,128], seed={seed}", flush=True)

    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    lr_sched = LR  # 固定大学习率
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
        raw = reward_binary(logits, labels)
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
            print(f"  epoch {epoch:6d} | val {last_val:.4f} | best_train {best:.4f}",
                  flush=True)
    if last_val == 0.0 or MAX_EPOCHS % EVAL_EVERY != 0:
        last_val = eval_acc()

    print(f"\n===== 结果 =====")
    print(f"  固定LR={LR} gs={GROUP_SIZE} rank={RANK} batch={num_envs} "
          f"{MAX_EPOCHS}ep 硬0/1")
    print(f"  best_train={best:.4f}  val_acc={last_val:.4f}")

    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["lr", "group_size", "rank", "batch", "max_epochs",
                        "val_acc", "best_train"])
        w.writerow([LR, GROUP_SIZE, RANK, num_envs, MAX_EPOCHS,
                    round(last_val, 5), round(best, 5)])
    print(f"results appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
