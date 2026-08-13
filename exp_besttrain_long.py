"""超长训练实验：验证 best_train 是否随训练单调爬升，尝试逼近 0.9。

背景：
  besttrain_sweep 4 个配置（T16/Wide256/Rank128/Base-warmcos）best_train 均落 0.79~0.82，
  未突破 rank72 线性衰减的 0.8332。趋势显示 best_train 随训练轮数**单调缓慢上升**
  （Base-warmcos 在 5800~6000 仍从 0.817 → 0.823，未平台）。本脚本用当前最优结构
  （[128,128] + rank=72 + loglik + warmup+cosine + v_th 可训练）跑**超长训练**，
  检验 best_train 是否继续爬升、能否逼近 0.9（0.8332 → 0.9 需 +6.7pp，纯靠延长训练）。

配置与 besttrain_sweep 的 Base-warmcos 完全一致，仅 MAX_EPOCHS 大幅拉长 + EVAL_EVERY 加大。

用法：
    python exp_besttrain_long.py
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

# ---- 配置（= besttrain_sweep 的 Base-warmcos，仅拉长训练） ----------------------
BATCH_RATIO = 0.1          # 6000（适配 rank72 显存）
PEAK_LR = 0.01             # warmup+cosine 峰值学习率
MAX_EPOCHS = 30000         # 超长训练（是 base 6000 的 5 倍）
VARIANTS = [0.2]           # sigma 变体（0.2 已知最优）；可扩展为 [0.2] 或 [0.15, 0.2]
RANK = 72
T = 8
seed = 42                  # 与 besttrain_sweep / rank72 实验同 seed，便于横向对比
VAL = 1024
EVAL_EVERY = 250           # 每 250 epochs 评估一次测试集（长训练下降低频率）
NOISE_REUSE = 0
GROUP_SIZE = 0

CSV_PATH = "results_besttrain_long.csv"


def reward_loglik(logits, labels):
    """正确类 log_softmax（连续、无界，( -inf, 0 ]）——7.5 重测全场最优奖励。"""
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), labels]


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


def run(sigma):
    print(f"\n===== [long] sigma={sigma} rank={RANK} {MAX_EPOCHS} epochs =====", flush=True)
    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    # warmup+cosine：peak 0.01（文档 7.6 最优调度）
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR, warmup_steps=max(10, MAX_EPOCHS // 10),
        decay_steps=MAX_EPOCHS, end_value=PEAK_LR * 0.05,
    )
    frozen_noiser, n0 = NOISER.init_noiser(
        p0, sigma, lr_sched, solver=optax.adamw,
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
            print(f"  [sigma={sigma}] epoch {epoch:6d} | val {last_val:.4f} | "
                  f"best_train {best:.4f}", flush=True)
    if last_val == 0.0 or MAX_EPOCHS % EVAL_EVERY != 0:
        last_val = eval_acc()
    return best, last_val


def main():
    print(f"配置: batch={num_envs}, peak_lr={PEAK_LR}, warmup+cosine, "
          f"epochs={MAX_EPOCHS}, reward=loglik, rank={RANK}, [128,128], "
          f"v_th 可训练, seed={seed}, sigma 变体={VARIANTS}", flush=True)

    results = []
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["sigma", "rank", "max_epochs", "val_acc", "best_train"])
        f.flush()

    for sigma in VARIANTS:
        best, val = run(sigma)
        results.append((sigma, val, best))
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([sigma, RANK, MAX_EPOCHS, round(val, 5), round(best, 5)])
            f.flush()
        print(f"  [sigma={sigma}] val_acc={val:.4f} best_train={best:.4f}", flush=True)

    print("\n===== 超长训练 best_train 汇总（loglik + warmup+cosine + rank72） =====")
    for sigma, val, best in results:
        print(f"  sigma={sigma:<4} val={val:.4f} best_train={best:.4f}")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
