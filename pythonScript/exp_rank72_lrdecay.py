"""log-likelihood 奖励 + 高 rank(72) + 线性递减学习率 + 长训练(10000 epochs) 实验。

背景：
  文档 7.5 重测（exp_reward_sweep.py）显示，在 大批次(0.2) + LoRA rank + v_th 可训练 +
  固定小 LR=0.01 的配置下，log-likelihood(log_softmax) 反转为全场最优（val_acc=0.8545，
  超越 0.84 记录）。本脚本进一步改进目标：
    - rank 提升到 72（原 32）：增加可训练子空间容量，期望突破 0.8545。
    - LR 使用线性递减（optax.linear_schedule 0.01 -> 0.001）：长训练后期用小学习率微调，
      抑制 logits 尺度发散，稳定 log-likelihood 奖励。
    - 训练时长拉长到 10000 epochs：观察长程收敛与最终平台。

奖励：仅 log-likelihood（log_softmax(logits)[label]）。

用法：
    python exp_rank72_lrdecay.py
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
HIDDEN = [128, 128]      # 固定 2 层
NUM_CLASSES = 10
MNIST_DIR = os.environ.get("MNIST_DIR") or r"D:\Rust\snn_t1\mnist_data"
# WSL 下 Windows 盘符路径不生效，自动回退到 /mnt/d 挂载路径（原生 Windows 行为不变）
if not os.path.isdir(MNIST_DIR) and os.path.isdir("/mnt/d/Rust/snn_t1/mnist_data"):
    MNIST_DIR = "/mnt/d/Rust/snn_t1/mnist_data"

# ---- 改进配置 ----------------------------------------------------------------
# rank=72 相比 rank=32 的 LoRA 中间张量(...,rank)显著增大，0.2*60000=12000 的 batch 在
# RTX 4090(24G) 上会 OOM，故 batch 从 0.2 降至 0.1(=6000) 以适配高 rank 的显存需求。
BATCH_RATIO = 0.1
LR_START = 0.01          # 线性递减起点（与 exp_reward_sweep 的固定 0.01 对齐）
LR_END = 0.001           # 线性递减终点
RANK = 72                # 原 32 -> 提升到 72
MAX_EPOCHS = 10000       # 原 3000 -> 拉长到 10000
TRAIN_TAU = False
seed = 42                # 换一个 seed 以免与旧实验重复/对照
VAL = 1024
EVAL_EVERY = 100
# 其余超参沿用 7.5 重测基线
T = 8
SIGMA = 0.2
NOISE_REUSE = 0
GROUP_SIZE = 0

CSV_PATH = "results_rank72_lrdecay_loglik.csv"


def reward_loglik(logits, labels):
    """正确类 log_softmax（连续、无界，( -inf, 0 ]）——7.5 重测全场最优奖励。"""
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), labels]


# ---- 模型（与 exp_reward_sweep.py 同款可训练 v_th 2 层 SNN） -------------------
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


def run():
    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    # 线性递减学习率：0.01 -> 0.001，跨度 MAX_EPOCHS
    lr_sched = optax.linear_schedule(LR_START, LR_END, transition_steps=MAX_EPOCHS)
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
        raw = reward_loglik(logits, labels)
        fits = NOISER.convert_fitnesses(frozen_noiser, state["noiser_params"], raw)
        state["noiser_params"], state["params"] = jit_update(
            state["noiser_params"], state["params"], fits, it)
        return float(accuracy_from_logits(logits, labels))

    def log_lr_at(epoch):
        """打印当前步对应的线性衰减学习率（仅供日志参考）。"""
        frac = min(1.0, epoch / MAX_EPOCHS)
        return LR_START + (LR_END - LR_START) * frac

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
            print(f"  epoch {epoch:5d} | lr {log_lr_at(epoch):.5f} | "
                  f"val {last_val:.4f} | best_train {best:.4f}", flush=True)
    if last_val == 0.0 or MAX_EPOCHS % EVAL_EVERY != 0:
        last_val = eval_acc()
    return last_val, best


def main():
    print(f"配置: batch={num_envs}, lr 线性 {LR_START}->{LR_END}, rank={RANK}, "
          f"epochs={MAX_EPOCHS}, 2 层 [128,128], v_th 可训练, T={T}, sigma={SIGMA}, "
          f"group_size={GROUP_SIZE}, reward=loglik", flush=True)

    val, best = run()
    print(f"\n===== 结果 =====")
    print(f"  reward=loglik  rank={RANK}  lr 线性 {LR_START}->{LR_END}  "
          f"{MAX_EPOCHS} epochs")
    print(f"  val_acc={val:.4f}  best_train={best:.4f}")

    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["rank", "lr_start", "lr_end", "max_epochs", "seed",
                        "val_acc", "best_train"])
        w.writerow([RANK, LR_START, LR_END, MAX_EPOCHS, seed,
                    round(val, 5), round(best, 5)])
    print(f"results appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
