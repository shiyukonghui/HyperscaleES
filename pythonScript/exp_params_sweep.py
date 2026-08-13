"""2 层 SNN 参数扫描：T（时间步）/ sigma（扰动幅度）/ noise_reuse / group_size 控制变量法。

固定：batch=12000, LR=0.03, epochs=3000, rank=32, v_th 可训练, 2 层 [128,128], 硬 0/1 奖励。
控制变量法：每个维度只改一个参数，其余取基准值（T=8, sigma=0.2, noise_reuse=0, group_size=0）。

设计：
  - 每个配置（参数名-值）独立跑一遍，与 exp_vth_trainable.py 相同的 2 层可训练 v_th 模型。
  - 每维度扫描后输出小计，最后汇总对比，确定各参数收益边际。
  - 结果追加写盘（results_params_sweep.csv）。

用法：
    python exp_params_sweep.py
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
LR = 0.1           # warmup+cosine 的 peak 学习率
RANK = 32
MAX_EPOCHS = 3000
TRAIN_TAU = False
seed = 0
VAL = 1024
EVAL_EVERY = 50

# 固定非扫描维度（本实验：batch=0.2, LR 用 warmup+cosine(0.1), T=8, sigma=0.2）
BASE = {"T": 8, "sigma": 0.2, "noise_reuse": 0, "group_size": 0}
# 仅扫描 group_size 0~50。注意：convert_fitnesses 要求 batch(12000) 能被 group_size 整除，
# 因此在 (0,50] 内只有 12000 的约数 5/10/15/20/24/30/40/48 是合法取值，加上 gs=0（无分组基线）。
SWEEP = {
    "group_size": [0, 5, 10, 15, 20, 30, 40, 48],
}

CSV_PATH = "results_params_sweep_gs0_50_warmcos.csv"


class TrainableVthSNN2L(Model):
    """2 层 SNN，v_th 可训练（softplus 参数化）。与 exp_vth_trainable.py 同款但固定 2 层。"""

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

# ---- 模型 / noiser 初始化（T 只影响 forward，noiser 与参数有关，每个配置全建）
def build(T, sigma, noise_reuse, group_size):
    """按给定参数构建模型 + noiser + JIT，返回 run()（跑满 MAX_EPOCHS）。"""
    key = jax.random.key(seed)
    model_key, es_key, dkey = jax.random.split(key, 3)
    frozen_p, p0, scan_map, es_map = TrainableVthSNN2L.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(p0, es_key, scan_map)
    # warmup+cosine 调度：peak LR(0.1)，warmup 后余弦衰减（与 exp_lr_schedule.py 的 warmcos 同口径）
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR, warmup_steps=max(10, MAX_EPOCHS // 10),
        decay_steps=MAX_EPOCHS, end_value=LR * 0.05,
    )
    frozen_noiser, n0 = NOISER.init_noiser(
        p0, sigma, lr_sched, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=RANK,
        noise_reuse=noise_reuse, group_size=group_size,
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

    # 用可变容器保存步间状态，避免依赖模块级 global（每个配置独立）
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
        raw = fitness_from_logits(logits, labels)
        fits = NOISER.convert_fitnesses(frozen_noiser, state["noiser_params"], raw)
        state["noiser_params"], state["params"] = jit_update(
            state["noiser_params"], state["params"], fits, it)
        return float(accuracy_from_logits(logits, labels))

    def run():
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

    return run


def main():
    print(f"固定: batch={num_envs}, lr={LR}, rank={RANK}, epochs={MAX_EPOCHS}, "
          f"2 层 [128,128], v_th 可训练, 硬 0/1", flush=True)
    print(f"基准值: {BASE}", flush=True)

    results = []
    # 初始化 CSV
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["param", "value", "val_acc", "best_train"])
        f.flush()

    # 控制变量法：每个维度只动一个
    for param, values in SWEEP.items():
        print(f"\n===== 扫描 {param} : {values} =====", flush=True)
        for v in values:
            cfg = dict(BASE)
            cfg[param] = v
            print(f"[{param}={v}] ...", flush=True)
            run = build(**cfg)
            val, best = run()
            results.append((param, v, val, best))
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([param, v, round(val, 5), round(best, 5)])
                f.flush()
            print(f"  [{param}={v}] val_acc={val:.3f} best_train={best:.3f}", flush=True)

    print("\n===== 参数扫描汇总（控制变量法） =====")
    for param in SWEEP:
        print(f"--- {param} ---")
        for p, v, val, best in results:
            if p == param:
                print(f"  {param}={v:>6}  val_acc={val:.4f}  best_train={best:.4f}")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
