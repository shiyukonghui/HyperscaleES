"""best_train 提升扫描：多配置对比，目标将训练集准确率(best_train) 推到 0.9+。

背景：
  rank72 + lr 线性递减 + loglik 长程实验（exp_rank72_lrdecay.py）best_train=0.8332。
  本脚本尝试多种高潜力的容量/调度配置，用 loglik 奖励（已证最优）+ warmup+cosine 调度
  （文档 7.6 证明最优收敛调度）+ v_th 可训练的 2 层可变宽度 SNN，对比 best_train 上升。

候选配置（覆盖不同方向）：
  A. T16        ：T=16（时间分辨率×2，SNN 表示能力更强），[128,128], rank=72
  B. Wide256    ：[256,256]（隐藏层宽度×2，容量↑），T=8, rank=72
  C. Rank128    ：rank=128（可训练子空间↑），[128,128], T=8
  D. Base-warmcos：基准 [128,128], T=8, rank=72（对照组，验证 warmup+cosine 优于线性衰减）

共同：reward=loglik, 调度=warmup+cosine(lr 峰值 0.01), batch=6000(0.1)（适配高 rank 显存），
     MAX_EPOCHS 可配（默认 6000），每配置独立同 seed 初始化。

用法：
    python exp_besttrain_sweep.py
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
NUM_CLASSES = 10
MNIST_DIR = os.environ.get("MNIST_DIR") or r"D:\Rust\snn_t1\mnist_data"
# WSL 下 Windows 盘符路径不生效，自动回退到 /mnt/d 挂载路径（原生 Windows 行为不变）
if not os.path.isdir(MNIST_DIR) and os.path.isdir("/mnt/d/Rust/snn_t1/mnist_data"):
    MNIST_DIR = "/mnt/d/Rust/snn_t1/mnist_data"

# ---- 公共配置 ----------------------------------------------------------------
BATCH_RATIO = 0.1          # 6000（适配高 rank 显存，rank72 在 12000 会 OOM）
PEAK_LR = 0.01             # warmup+cosine 峰值学习率（与 rank72 实验起点 0.01 对齐）
MAX_EPOCHS = 6000         # 每个配置的训练更新次数
TRAIN_TAU = False
seed = 42                  # 与 rank72 实验同 seed，便于横向对比
VAL = 1024
EVAL_EVERY = 50

# 固定超参（沿用 7.5 重测基线）
SIGMA = 0.2
NOISE_REUSE = 0
GROUP_SIZE = 0

CSV_PATH = "results_besttrain_sweep.csv"

# 候选配置：名称 -> (T, HIDDEN, RANK)
CONFIGS = [
    ("T16",        16,   [128, 128], 72),
    ("Wide256",     8,   [256, 256], 72),
    ("Rank128",     8,   [128, 128], 128),
    ("Base-warmcos", 8,  [128, 128], 72),
]


def reward_loglik(logits, labels):
    """正确类 log_softmax（连续、无界，( -inf, 0 ]）——7.5 重测全场最优奖励。"""
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), labels]


class TrainableVthSNN2L(Model):
    """2 层 SNN，v_th 可训练（softplus 参数化），隐藏宽度可配。"""

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


def run_config(name, T, hidden, rank):
    """跑一个配置，返回 (best_train, last_val)。"""
    # 动态覆盖模型类的隐藏宽度（forward 中引用 HIDDEN）
    global HIDDEN
    HIDDEN = hidden

    print(f"\n===== [{name}] T={T} hidden={hidden} rank={rank} =====", flush=True)
    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=hidden,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    # warmup+cosine：peak 0.01，warmup 后余弦衰减（文档 7.6 最优调度）
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR, warmup_steps=max(10, MAX_EPOCHS // 10),
        decay_steps=MAX_EPOCHS, end_value=PEAK_LR * 0.05,
    )
    frozen_noiser, n0 = NOISER.init_noiser(
        p0, SIGMA, lr_sched, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=rank,
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
            print(f"  [{name}] epoch {epoch:5d} | val {last_val:.4f} | "
                  f"best_train {best:.4f}", flush=True)
    if last_val == 0.0 or MAX_EPOCHS % EVAL_EVERY != 0:
        last_val = eval_acc()
    return best, last_val


def main():
    print(f"配置: batch={num_envs}, peak_lr={PEAK_LR}, warmup+cosine, "
          f"epochs={MAX_EPOCHS}, reward=loglik, v_th 可训练, seed={seed}", flush=True)

    results = []
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["name", "T", "hidden", "rank", "val_acc", "best_train"])
        f.flush()

    for name, T, hidden, rank in CONFIGS:
        best, val = run_config(name, T, hidden, rank)
        results.append((name, T, hidden, rank, val, best))
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([name, T, ",".join(map(str, hidden)), rank,
                                    round(val, 5), round(best, 5)])
            f.flush()
        print(f"  [{name}] val_acc={val:.4f} best_train={best:.4f}", flush=True)

    print("\n===== best_train 提升扫描汇总（loglik + warmup+cosine） =====")
    for name, T, hidden, rank, val, best in results:
        print(f"  {name:<14} T={T:<3} hidden={hidden} rank={rank:<4} "
              f"val={val:.4f} best_train={best:.4f}")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
