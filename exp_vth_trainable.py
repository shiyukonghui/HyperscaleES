"""把 LIF 超参（v_th / tau_m）作为可训练参数 + 可变网络深度的实验脚本。

方案：
  - 原 SNNModel 中 v_th、tau_m 属于 frozen_params（不参与进化更新）。
  - 本脚本内嵌 TrainableVthSNN：把 v_th 改为 PARAM 类型放进可训练 params 树，
    参与 ES 更新（全参扰动 + AdamW）。为恒正且平滑，用 softplus 参数化：
      实际阈值 v_th = softplus(raw_vth)，初始 raw_vth = log(exp(0.3) - 1)（即初始 v_th = 0.3）。
  - VTH_PER_LAYER = True 时每层拥有独立可训练阈值（v_th1..v_thN），解决深层网络
    中"单一全局阈值无法匹配各层电流量级"导致的脉冲链衰减/失效问题。
  - TRAIN_TAU = True 时 tau_m 同样改为可训练参数（softplus 参数化，初始 = 20.0）。
  - 网络深度由 HIDDEN 决定（本次实验 [128,128,128]，即 3 层 LIF 隐藏层）。
  - 其余与 rank 扫描实验完全一致：T=8, sigma=0.2, 硬 0/1 奖励。

用法：
    python exp_vth_trainable.py
"""

import csv
import os
import time

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
HIDDEN = [128, 128, 128]   # 本次实验：比基线多一层 LIF 隐藏层（[128,128] -> [128,128,128]）
NUM_CLASSES = 10
MNIST_DIR = os.environ.get("MNIST_DIR") or r"D:\Rust\snn_t1\mnist_data"
# WSL 下 Windows 盘符路径不生效，自动回退到 /mnt/d 挂载路径（原生 Windows 行为不变）
if not os.path.isdir(MNIST_DIR) and os.path.isdir("/mnt/d/Rust/snn_t1/mnist_data"):
    MNIST_DIR = "/mnt/d/Rust/snn_t1/mnist_data"

# ---- 固定配置（单点测试） ---------------------------------------------------
BATCH_RATIO = 0.2    # 批次比例（0.2 × 60000 = 12000）
LR = 0.03            # 固定 LR
RANK = 32            # LoRA rank（上一实验的性价比平衡点）
MAX_EPOCHS = 1000    # 固定更新次数（含 10 次标定）
TRAIN_TAU = False    # tau 冻结（7.13 负面结论）；仅 v_th 可训练
VTH_PER_LAYER = True # 逐层独立 v_th：每层一个可训练阈值（解决深层网络脉冲链衰减）
T = 8
sigma = 0.2
seed = 0
VAL = 1024
EVAL_EVERY = 50

# 结果按网络结构与 v_th 模式分文件，避免混淆
CSV_PATH = (f"results_vth_trainable_h{'_'.join(map(str, HIDDEN))}"
            f"{'_perlayer' if VTH_PER_LAYER else ''}.csv")


class TrainableVthSNN(Model):
    """SNN 变体：v_th（及可选 tau_m）作为可训练参数（softplus 参数化，恒正）。

    与 SNNModel 的区别：v_th 不在 frozen_params，而是作为 PARAM 放进 params 树参与
    ES 更新；TRAIN_TAU=True 时 tau_m 同样进 params。frozen_params 仅保留剩余冻结超参。
    网络深度由全局 HIDDEN 决定：循环构建 fc1..fc{N+1}（N 个 LIF 隐藏层 + 读出层）。
    """

    @classmethod
    def rand_init(cls, key, in_dim, hidden_dims, num_classes, tau_m=20.0,
                  v_th=0.3, dtype=jnp.float32):
        # 每个权重层分配一个 key：fc1..fcN（隐藏）+ fc{N+1}（读出）+ out_gain
        keys = jax.random.split(key, len(hidden_dims) + 2)
        layers_kwargs = {}
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            layers_kwargs[f"fc{i+1}"] = MM.rand_init(keys[i], prev, h, dtype)
            prev = h
        layers_kwargs[f"fc{len(hidden_dims)+1}"] = MM.rand_init(
            keys[len(hidden_dims)], prev, num_classes, dtype)
        layers_kwargs["out_gain"] = Parameter.rand_init(
            keys[-1], None, None, jnp.ones((1,)), dtype=dtype)
        # v_th：逐层独立（每层初始 0.3）或全局共享（softplus 参数化）
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        if VTH_PER_LAYER:
            for i in range(len(hidden_dims)):
                layers_kwargs[f"v_th{i+1}"] = Parameter.rand_init(
                    None, None, None, raw_vth0, dtype=dtype)
        else:
            layers_kwargs["v_th"] = Parameter.rand_init(None, None, None, raw_vth0, dtype=dtype)
        frozen_params = {}
        if TRAIN_TAU:
            # tau_m 也可训练：初始 raw = log(expm1(20))，softplus(raw) = 20
            layers_kwargs["tau_m"] = Parameter.rand_init(
                None, None, None,
                jnp.log(jnp.expm1(jnp.asarray(tau_m, dtype=dtype))), dtype=dtype)
        else:
            frozen_params["tau_m"] = jnp.asarray(tau_m, dtype=dtype)
        layers = merge_inits(**layers_kwargs)
        return CommonInit(frozen_params or None, layers.params,
                          layers.scan_map, layers.es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        x = x.astype(common_params.params["fc1"].dtype)
        n_layers = len(HIDDEN)
        # 各层阈值：逐层独立时每层一个（softplus 恒正）；全局模式时各层共享同一值
        if VTH_PER_LAYER:
            vths = [jax.nn.softplus(call_submodule(Parameter, f"v_th{i+1}", common_params))
                    for i in range(n_layers)]
        else:
            vth = jax.nn.softplus(call_submodule(Parameter, "v_th", common_params))
            vths = [vth] * n_layers
        if TRAIN_TAU:
            # tau_m 可训练：softplus(raw_tau) 恒正，初始 = 20.0
            raw_tau = call_submodule(Parameter, "tau_m", common_params)
            tau = jax.nn.softplus(raw_tau)
        else:
            tau = common_params.frozen_params["tau_m"]

        # 逐层 LIF：每层用自己的 v_th 做线性投影 + LIF 脉冲，输出作为下一层输入
        cur = x                                    # (T, in_dim)
        for i in range(n_layers):
            def proj(x_t):
                return call_submodule(MM, f"fc{i+1}", common_params, x_t)
            currents = jax.vmap(proj)(cur)         # (T, h_i)
            v0 = jnp.zeros((HIDDEN[i],), dtype=currents.dtype)
            lif_params = {"tau_m": tau, "v_th": vths[i]}
            cur = run_lif(lif_params, currents, v0)  # (T, h_i)

        rate = jnp.mean(cur, axis=0)               # (h_N,)
        logits = call_submodule(MM, f"fc{n_layers+1}", common_params, rate)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


# ---- data -----------------------------------------------------------------
x_tr, y_tr = get_mnist_arrays("train", data_dir=MNIST_DIR)
x_te, y_te = get_mnist_arrays("test", data_dir=MNIST_DIR)
n_train = x_tr.shape[0]
num_envs = int(round(BATCH_RATIO * n_train))

# ---- 模型初始化 -------------------------------------------------------------
key = jax.random.key(seed)
model_key, es_key, data_key = jax.random.split(key, 3)
frozen_params, params, scan_map, es_map = TrainableVthSNN.rand_init(
    model_key, in_dim=IN_DIM, hidden_dims=HIDDEN,
    num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
)
total_params = sum(p.size for p in jax.tree.leaves(params))
print(f"total_params = {total_params} (含可训练 v_th)", flush=True)

# ---- noiser + JIT（固定 rank） ----------------------------------------------
es_tree_key = simple_es_tree_key(params, es_key, scan_map)
frozen_noiser, noiser_params = NOISER.init_noiser(
    params, sigma, LR, solver=optax.adamw,
    solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=RANK,
)
jit_forward = jax.jit(jax.vmap(
    lambda n, p, i, x: TrainableVthSNN.forward(NOISER, frozen_noiser, n, frozen_params,
                                               p, es_tree_key, i, x),
    in_axes=(None, None, 0, 0)))
jit_forward_eval = jax.jit(jax.vmap(
    lambda n, p, x: TrainableVthSNN.forward(NOISER, frozen_noiser, n, frozen_params,
                                            p, es_tree_key, None, x),
    in_axes=(None, None, 0)))
jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
    frozen_noiser, n, p, es_tree_key, f, i, es_map))


def current_vth():
    """读取当前实际阈值列表（逐层独立时每层一个，否则单值）。"""
    if VTH_PER_LAYER:
        return [float(jax.nn.softplus(params[f"v_th{i+1}"])) for i in range(len(HIDDEN))]
    return [float(jax.nn.softplus(params["v_th"]))]


def fmt_vth(vals):
    """把阈值列表格式化为 "0.300/0.400/..." 字符串。"""
    return "/".join(f"{v:.3f}" for v in vals)


def current_tau():
    """读取当前实际 tau_m（可训练时 softplus(raw_tau)，否则返回初始 20.0）。"""
    if TRAIN_TAU:
        return float(jax.nn.softplus(params["tau_m"]))
    return float(frozen_params["tau_m"])


def eval_acc():
    global noiser_params, params
    idx = jax.random.permutation(jax.random.key(1), x_te.shape[0])[:VAL]
    imgs = jnp.asarray(x_te[idx], dtype=DTYPE)
    labels = jnp.asarray(y_te[idx], dtype=jnp.int32)
    spikes = poisson_encode(imgs, T, jax.random.key(1)).transpose(1, 0, 2)
    logits = jit_forward_eval(noiser_params, params, spikes)
    return float(accuracy_from_logits(logits, labels))


def make_step(data_key):
    """返回 (step(epoch) -> epoch_idx, acc_train) 闭包，每次采样新批次。"""
    def step(epoch):
        global data_key, noiser_params, params
        data_key2, enc, per = jax.random.split(data_key, 3)
        data_key = data_key2
        idx = jax.random.permutation(per, n_train)[:num_envs]
        imgs = jnp.asarray(x_tr[idx], dtype=DTYPE)
        labels = jnp.asarray(y_tr[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, T, enc).transpose(1, 0, 2)
        it = (jnp.full(num_envs, epoch, dtype=jnp.int32),
              jnp.arange(num_envs, dtype=jnp.int32))
        logits = jit_forward(noiser_params, params, it, spikes)
        raw = fitness_from_logits(logits, labels)
        fits = NOISER.convert_fitnesses(frozen_noiser, noiser_params, raw)
        noiser_params, params = jit_update(noiser_params, params, fits, it)
        return epoch, float(accuracy_from_logits(logits, labels))
    return step


def main():
    print(f"hidden = {HIDDEN}, batch = {num_envs} ({BATCH_RATIO}x train), lr = {LR}, "
          f"sigma = {sigma}, rank = {RANK}, max_epochs = {MAX_EPOCHS}", flush=True)
    print(f"初始 v_th = [{fmt_vth(current_vth())}], 初始 tau_m = {current_tau():.3f} "
          f"(VTH_PER_LAYER={VTH_PER_LAYER}, TRAIN_TAU={TRAIN_TAU})", flush=True)

    step = make_step(data_key)

    # 标定 s/epoch（10 次真实更新，含 JIT）
    cal = 10
    start = time.time()
    best = 0.0
    for e in range(cal):
        _, acc = step(e)
        best = max(best, acc)
    cal_s = (time.time() - start) / cal
    print(f"  calib: {cal_s:.3f}s/epoch", flush=True)

    t0 = time.time()
    epoch = cal
    last_val = 0.0
    while epoch < MAX_EPOCHS:
        _, acc = step(epoch)
        best = max(best, acc)
        epoch += 1
        if epoch % EVAL_EVERY == 0:
            last_val = eval_acc()
            print(f"  epoch {epoch:5d} | val {last_val:.3f} | best_train {best:.3f} "
                  f"| v_th [{fmt_vth(current_vth())}] | tau {current_tau():.3f} "
                  f"| {(time.time()-t0):.0f}s", flush=True)

    last_val = eval_acc() if last_val == 0.0 and epoch > cal else last_val
    elapsed = time.time() - t0
    vth_final = current_vth()
    tau_final = current_tau()

    # 结果写盘（v_th 多值时以 "/" 连接）
    hidden_str = "-".join(map(str, HIDDEN))
    vth_str = "/".join(f"{v:.5f}" for v in vth_final)
    fresh = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["hidden", "batch", "lr", "rank", "epochs", "val_acc",
                        "best_train", "vth_final", "tau_final", "s_per_epoch", "time_s"])
        w.writerow([hidden_str, num_envs, LR, RANK, epoch, last_val, best,
                    vth_str, round(tau_final, 5),
                    round(cal_s, 5), round(elapsed)])
        f.flush()

    print("\n===== 可训练 LIF 超参实验汇总 =====")
    print(f"epochs={epoch} val_acc={last_val:.3f} best_train={best:.3f} "
          f"v_th(0.300 -> [{fmt_vth(vth_final)}]) tau(20.0 -> {tau_final:.3f}) "
          f"({elapsed:.0f}s)")
    print(f"\nresults appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
