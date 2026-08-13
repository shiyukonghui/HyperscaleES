"""满秩 ES 突破实验：解除 LoRA 低秩限制，用满秩高斯扰动 + 满秩更新训练 SNN。

背景（框架级分析）：
  现有训练中 fc1/fc2/fc3 权重全部走 `MM`（es_map=MM_PARAM -> _simple_lora_update），
  梯度被限制在 rank 子空间（LoRA A@B.T，rank 越大单次扰动 sigma/sqrt(rank) 越小）。
  这是 best_train 封顶 0.83 的根因——低秩子空间无法表达拟合训练集所需的高秩更新。

本实验框架改动：
  定义 `FullMM` 模块替代 `MM`：
    - es_map = PARAM  -> 更新走 _simple_full_update（满秩梯度，mean(scores*noise)）
    - _forward 用 noiser.get_noisy_standard（满秩高斯扰动 param + noise）
  从而 forward 与 update 都满秩，完全绕开 LoRA，让 ES 能沿任意方向调整权重，
  期望大幅提升 best_train（训练集拟合上限）。

显存考量：满秩噪声张量 (num_envs, out_dim, in_dim) 很大，故 batch 需调小（默认 512）。
  满秩扰动 std = sigma（LoRA 为 sigma/sqrt(rank)，故满秩扰动本身更大）。

用法：
    python exp_fullrank_es.py
"""

import csv
import os

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.base_model import Model, CommonInit
from hyperscalees.models.common import (
    merge_inits, call_submodule, PARAM, simple_es_tree_key, Parameter,
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
BATCH = 512              # 满秩噪声显存大，batch 调小
LR_START = 0.01
LR_END = 0.001           # 线性衰减（best_train 最优调度）
SIGMA = 0.2              # 满秩扰动 std（探索幅度）
MAX_EPOCHS = 3000
T = 8
seed = 42
VAL = 1024
EVAL_EVERY = 100
CSV_PATH = "results_fullrank_es.csv"


def reward_loglik(logits, labels):
    """正确类 log_softmax（连续、无界）。"""
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), labels]


class FullMM(Model):
    """满秩矩阵乘模块：forward 用满秩高斯扰动，更新走满秩（es_map=PARAM）。

    替代 MM（LoRA 低秩），解除梯度子空间限制。
    """
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = (jax.random.normal(key, (out_dim, in_dim)) * scale).astype(dtype=dtype)
        frozen_params = None
        scan_map = ()
        es_map = PARAM          # 满秩更新
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        # 满秩扰动权重的矩阵乘（training 时 iterinfo 非 None -> param+noise；eval 时原 param）
        w = common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params, common_params.noiser_params,
            common_params.params, common_params.es_tree_key, common_params.iterinfo,
        )
        return x @ jnp.asarray(w).T


class TrainableVthSNN2L_Full(Model):
    """2 层满秩 SNN，v_th 可训练（softplus），权重用 FullMM（满秩 ES）。"""

    @classmethod
    def rand_init(cls, key, in_dim, hidden_dims, num_classes, tau_m=20.0,
                  v_th=0.3, dtype=jnp.float32):
        in_key, h1_key, h2_key, out_key = jax.random.split(key, 4)
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        layers = merge_inits(
            fc1=FullMM.rand_init(in_key, in_dim, hidden_dims[0], dtype),
            fc2=FullMM.rand_init(h1_key, hidden_dims[0], hidden_dims[1], dtype),
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
            return call_submodule(FullMM, "fc1", common_params, x_t)
        cur1 = jax.vmap(proj1)(x)
        spikes1 = run_lif(lif_params, cur1, jnp.zeros((HIDDEN[0],), dtype=cur1.dtype))

        def proj2(x_t):
            return call_submodule(FullMM, "fc2", common_params, x_t)
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


def main():
    print(f"满秩ES: batch={BATCH}, lr 线性 {LR_START}->{LR_END}, sigma={SIGMA}, "
          f"epochs={MAX_EPOCHS}, 2层[128,128], v_th 可训练, T={T}, loglik, seed={seed}",
          flush=True)

    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L_Full.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    lr_sched = optax.linear_schedule(LR_START, LR_END, transition_steps=MAX_EPOCHS)
    frozen_noiser, n0 = NOISER.init_noiser(
        p0, SIGMA, lr_sched, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=1,
        noise_reuse=0, group_size=0,
    )
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: TrainableVthSNN2L_Full.forward(
            NOISER, frozen_noiser, n, frozen_p, p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0)))
    jit_forward_eval = jax.jit(jax.vmap(
        lambda n, p, x: TrainableVthSNN2L_Full.forward(
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
        idx = jax.random.permutation(per, n_train)[:BATCH]
        imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
        labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, T, enc).transpose(1, 0, 2)
        it = (jnp.full(BATCH, epoch, dtype=jnp.int32),
              jnp.arange(BATCH, dtype=jnp.int32))
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

    print(f"\n===== 满秩 ES 结果 =====")
    print(f"  batch={BATCH} sigma={SIGMA} lr 线性 {LR_START}->{LR_END} {MAX_EPOCHS}ep loglik")
    print(f"  best_train={best:.4f}  val_acc={last_val:.4f}")

    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["batch", "sigma", "lr_start", "lr_end", "max_epochs",
                        "val_acc", "best_train"])
        w.writerow([BATCH, SIGMA, LR_START, LR_END, MAX_EPOCHS,
                    round(last_val, 5), round(best, 5)])
    print(f"results appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
