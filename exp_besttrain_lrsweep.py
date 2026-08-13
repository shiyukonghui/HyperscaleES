"""best_train 决定性 LR 扫描：loglik + rank72 + 不同学习率。

背景校准：
  文档 7.16 的 0.840 实为 **val_acc**（表头 | group_size | val_acc |），并非 best_train。
  实测大 LR(0.03)+硬0/1 的 best_train 仅 ~0.82，低于 rank72+loglik 的 0.8332。
  全项目 best_train 纪录 = rank72 + loglik + 线性衰减(0.01->0.001) 的 **0.8332**。
  LR 是影响 best_train 的最强单一因素，但 loglik+rank72 下尚未系统扫不同 LR。

本脚本：固定 batch=6000（rank72 显存安全），loglik 奖励，rank=72，v_th 可训练，
2 层 [128,128]，T=8，sigma=0.2，扫描固定 LR 多档 + 线性衰减对照，各跑 MAX_EPOCHS，
对比 best_train，寻找可能突破 0.8332 的学习率。

用法：
    python exp_besttrain_lrsweep.py
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

# ---- 配置 ----------------------------------------------------------------
BATCH_RATIO = 0.1          # 6000（rank72 显存安全）
RANK = 72
T = 8
SIGMA = 0.2
NOISE_REUSE = 0
GROUP_SIZE = 0
MAX_EPOCHS = 8000          # 每个 LR 的训练更新次数
seed = 42
VAL = 1024
EVAL_EVERY = 100

CSV_PATH = "results_besttrain_lrsweep.csv"

# LR 候选：名称 -> (kind, lr_start, lr_end)
# kind: "fixed" 固定 lr_start；"linear" 线性 lr_start->lr_end
CONFIGS = [
    ("fixed_003",  "fixed",  0.003, None),
    ("fixed_005",  "fixed",  0.005, None),
    ("fixed_01",   "fixed",  0.01,  None),
    ("fixed_02",   "fixed",  0.02,  None),
    ("linear_01",  "linear", 0.01,  0.001),
]


def reward_loglik(logits, labels):
    """正确类 log_softmax（连续、无界，( -inf, 0 ]）——best_train 纪录用的奖励。"""
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


def run(kind, lr_start, lr_end):
    label = f"{kind}-{lr_start}" + (f"->{lr_end}" if kind == "linear" else "")
    print(f"\n===== [LR {label}] =====", flush=True)
    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    if kind == "fixed":
        lr_sched = lr_start
    else:  # linear
        lr_sched = optax.linear_schedule(lr_start, lr_end, transition_steps=MAX_EPOCHS)
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
            print(f"  [LR {label}] epoch {epoch:5d} | val {last_val:.4f} | "
                  f"best_train {best:.4f}", flush=True)
    if last_val == 0.0 or MAX_EPOCHS % EVAL_EVERY != 0:
        last_val = eval_acc()
    return best, last_val


def main():
    print(f"配置: batch={num_envs}, rank={RANK}, loglik, T={T}, sigma={SIGMA}, "
          f"epochs={MAX_EPOCHS}, v_th 可训练, [128,128], seed={seed}", flush=True)
    print(f"LR 候选: {[n for n, *_ in CONFIGS]}", flush=True)

    results = []
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["name", "kind", "lr_start", "lr_end", "max_epochs",
                        "val_acc", "best_train"])
        f.flush()

    for name, kind, lr_start, lr_end in CONFIGS:
        best, val = run(kind, lr_start, lr_end)
        results.append((name, val, best))
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([name, kind, lr_start, lr_end or "", MAX_EPOCHS,
                                    round(val, 5), round(best, 5)])
            f.flush()
        print(f"  [{name}] val_acc={val:.4f} best_train={best:.4f}", flush=True)

    print("\n===== best_train LR 扫描汇总（loglik + rank72） =====")
    for name, val, best in sorted(results, key=lambda x: -x[2]):
        print(f"  {name:<14} val={val:.4f} best_train={best:.4f}")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
