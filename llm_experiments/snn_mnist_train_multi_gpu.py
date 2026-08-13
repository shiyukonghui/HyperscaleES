"""8×GPU 多卡放大训练：SNN(LIF) + HyperscaleES 演化策略（无反向传播）。

设计（参照 general_do_evolution_multi_gpu.py 的 shard_map 模式）：
  - mesh = ('data', num_gpus)，单进程多卡（本机 8×4090）
  - 泊松编码 + 前向：shard_map 内完成，batch 按 P('data') 分片（每卡只编码/前向自己分片）
  - 全局唯一 thread_id（jnp.arange(total_batch) 按 P('data') 分片）→ 跨卡噪声扰动不碰撞
  - fitness 分片计算 → 主机汇总全 batch
  - do_updates：复制式（全 batch fitness，P() 规格，各卡计算同一梯度，与 LLM 多卡脚本一致）
  - v_th 可训练（softplus 恒正，逐层独立），reward 默认 loglik（实验 7.5 重测最优）

用法（服务器）：
  .venv/bin/python -m llm_experiments.snn_mnist_train_multi_gpu \
      --batch 60000 --rank 64 --num-epochs 3000 \
      --mnist-dir ~/mnist_data --csv-out records/results_multigpu.csv
"""

import argparse
import csv
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import numpy as np
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
    get_mnist_arrays, poisson_encode,
)

NOISER = hs.noiser.eggroll.EggRoll
DTYPE = jnp.float32
IN_DIM = 28 * 28
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# 模型：v_th 可训练（softplus 恒正）的 SNN 变体，逐层独立阈值
# ---------------------------------------------------------------------------
class TrainableVthSNN(Model):
    """与 SNNModel 的区别：v_th 不在 frozen_params，而是 PARAM 进 params 树参与
    ES 更新（softplus 参数化恒正）。tau_m 冻结。隐藏层数由 v_th 参数个数推导。"""

    @classmethod
    def rand_init(cls, key, in_dim, hidden_dims, num_classes, tau_m=20.0,
                  v_th=0.3, dtype=jnp.float32):
        keys = jax.random.split(key, len(hidden_dims) + 2)
        layers_kwargs = {}
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            layers_kwargs[f"fc{i + 1}"] = MM.rand_init(keys[i], prev, h, dtype)
            prev = h
        layers_kwargs[f"fc{len(hidden_dims) + 1}"] = MM.rand_init(
            keys[len(hidden_dims)], prev, num_classes, dtype)
        layers_kwargs["out_gain"] = Parameter.rand_init(
            keys[-1], None, None, jnp.ones((1,)), dtype=dtype)
        # 逐层独立 v_th：raw 初始 = log(exp(v_th)-1)，实际阈值 = softplus(raw) = v_th
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        for i in range(len(hidden_dims)):
            layers_kwargs[f"v_th{i + 1}"] = Parameter.rand_init(
                None, None, None, raw_vth0, dtype=dtype)
        frozen_params = {"tau_m": jnp.asarray(tau_m, dtype=dtype)}
        layers = merge_inits(**layers_kwargs)
        return CommonInit(frozen_params, layers.params,
                          layers.scan_map, layers.es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        x = x.astype(common_params.params["fc1"].dtype)
        vth_keys = sorted(
            (k for k in common_params.params if k.startswith("v_th")),
            key=lambda k: int(k.replace("v_th", "")),
        )
        n_layers = len(vth_keys)
        vths = [jax.nn.softplus(call_submodule(Parameter, k, common_params))
                for k in vth_keys]
        cur = x  # (T, in_dim) 单样本，batch 由外层 vmap 并行
        for i in range(n_layers):
            cur = jax.vmap(
                lambda xt: call_submodule(MM, f"fc{i + 1}", common_params, xt)
            )(cur)
            v0 = jnp.zeros((cur.shape[-1],), dtype=cur.dtype)
            lif_p = {"tau_m": common_params.frozen_params["tau_m"],
                     "v_th": vths[i]}
            cur = run_lif(lif_p, cur, v0)  # (T, h) 脉冲
        rate = jnp.mean(cur, axis=0)  # (h_last,)
        logits = call_submodule(MM, f"fc{n_layers + 1}", common_params, rate)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


def fitness_from_logits(logits, labels, reward="loglik"):
    """每样本原始奖励：loglik（默认，7.5 重测最优）或硬 0/1。"""
    if reward == "binary":
        pred = jnp.argmax(logits, axis=-1)
        return (pred == labels).astype(jnp.float32)
    if reward == "loglik":
        ll = jax.nn.log_softmax(logits, axis=-1)
        return ll[jnp.arange(labels.shape[0]), labels]
    raise ValueError(f"unknown reward: {reward}")


def parse_args():
    p = argparse.ArgumentParser(description="SNN + ES 8×GPU 多卡放大训练")
    p.add_argument("--batch", type=int, default=60000, help="总 batch（<=60000，训练集全量）")
    p.add_argument("--rank", type=int, default=64, help="LoRA rank")
    p.add_argument("--T", type=int, default=8, help="泊松编码/SNN 时间步")
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-schedule", choices=["fixed", "linear", "cosine"], default="fixed")
    p.add_argument("--reward", choices=["loglik", "binary"], default="loglik")
    p.add_argument("--group-size", type=int, default=0, help="组内归一化（0=全局 z-score）")
    p.add_argument("--num-epochs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=str, default="128,128")
    p.add_argument("--mnist-dir", default=None, help="MNIST IDX 目录（服务器 ~/mnist_data）")
    p.add_argument("--validate-every", type=int, default=50)
    p.add_argument("--val-batch", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--csv-out", default=None, help="结果 CSV 路径")
    return p.parse_args()


def main():
    args = parse_args()
    hidden_dims = [int(h) for h in args.hidden.split(",")]
    assert args.batch <= 60000, "batch 不能超过训练集大小 60000"

    devices = jax.devices()
    num_gpus = len(devices)
    assert args.batch % num_gpus == 0, f"batch {args.batch} 需能被 {num_gpus} 整除"
    per_gpu_batch = args.batch // num_gpus
    print(f"[env] devices={num_gpus} ({[d.platform for d in devices]}), "
          f"batch={args.batch} (per_gpu={per_gpu_batch}), rank={args.rank}, "
          f"T={args.T}, reward={args.reward}")

    # 把 (batch, ...) 数组切分为 (num_gpus, per_gpu, ...)，交给 pmap 按设备分发
    def split_batch(x):
        x = jnp.asarray(x)
        return x.reshape((num_gpus, per_gpu_batch) + tuple(x.shape[1:]))

    # --- 模型 / noiser -----------------------------------------------------
    master_key = jax.random.key(args.seed)
    model_key, es_key, enc_base = jax.random.split(master_key, 3)

    MODEL = TrainableVthSNN
    frozen_params, params, scan_map, es_map = MODEL.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=hidden_dims,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    if args.lr_schedule == "fixed":
        lr_schedule = args.lr
    elif args.lr_schedule == "linear":
        lr_schedule = optax.linear_schedule(args.lr, args.lr * 0.1, args.num_epochs)
    else:  # cosine
        lr_schedule = optax.cosine_decay_schedule(args.lr, args.num_epochs)

    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params, args.sigma, lr_schedule,
        solver=optax.adamw, solver_kwargs={"b1": 0.9, "b2": 0.999},
        rank=args.rank, group_size=args.group_size, freeze_nonlora=False,
    )

    # --- 数据 --------------------------------------------------------------
    mnist_dir = args.mnist_dir or os.environ.get("MNIST_DIR")
    x_train, y_train = get_mnist_arrays("train", data_dir=mnist_dir)
    x_test, y_test = get_mnist_arrays("test", data_dir=mnist_dir)
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    print(f"[data] train={n_train}, test={n_test}, dir={mnist_dir}")

    # --- 前向：pmap 把 batch 按设备分发（每设备编码 + 前向 + fitness）------
    # 不用 shard_map/自动 SPMD：LIF 内部 jax.lax.scan 在手动轴/随机噪声下与
    # vmap 冲突（JAX 0.11），pmap 每设备是普通局部计算，无此问题。
    # 噪声按 (epoch, 全局 thread_id) 派生，thread_ids 分片后跨设备仍全局唯一。
    def _forward_pmap(noiser_params, params, epoch, thread_ids, images, labels):
        enc_key = jax.random.fold_in(
            jax.random.fold_in(enc_base, epoch), jax.lax.axis_index("data"))
        spikes = poisson_encode(images, args.T, enc_key).transpose(1, 0, 2)
        iterinfo = (jnp.full(thread_ids.shape, epoch, dtype=jnp.int32), thread_ids)
        logits = jax.vmap(
            lambda n, p, i, x: MODEL.forward(
                NOISER, frozen_noiser_params, n, frozen_params, p,
                es_tree_key, i, x),
            in_axes=(None, None, 0, 0),
        )(noiser_params, params, iterinfo, spikes)
        fitness = fitness_from_logits(logits, labels, args.reward)  # (per_gpu,)
        acc = jax.lax.pmean(
            jnp.mean((jnp.argmax(logits, -1) == labels).astype(jnp.float32)), "data")
        return fitness, acc

    forward_pmap = jax.pmap(
        _forward_pmap, axis_name="data",
        in_axes=(None, None, None, 0, 0, 0), out_axes=(0, None))

    # --- 验证（iterinfo=None，无扰动）--------------------------------------
    def _eval_pmap(noiser_params, params, images, labels):
        eval_key = jax.random.fold_in(
            jax.random.fold_in(enc_base, 1_000_000), jax.lax.axis_index("data"))
        spikes = poisson_encode(images, args.T, eval_key).transpose(1, 0, 2)
        logits = jax.vmap(
            lambda n, p, x: MODEL.forward(
                NOISER, frozen_noiser_params, n, frozen_params, p,
                es_tree_key, None, x),
            in_axes=(None, None, 0),
        )(noiser_params, params, spikes)
        return jax.lax.pmean(
            jnp.mean((jnp.argmax(logits, -1) == labels).astype(jnp.float32)), "data")

    eval_pmap = jax.pmap(
        _eval_pmap, axis_name="data",
        in_axes=(None, None, 0, 0), out_axes=None)

    # --- 复制式更新（全 batch fitness，各卡计算同一梯度）---------------------
    def _do_update(noiser_params, params, fitnesses, epoch, thread_ids):
        iterinfos = (jnp.full(fitnesses.size, epoch, dtype=jnp.int32), thread_ids)
        conv = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, fitnesses)
        noiser_params, new_params = NOISER.do_updates(
            frozen_noiser_params, noiser_params, params, es_tree_key,
            conv, iterinfos, es_map)
        return noiser_params, new_params

    do_update = jax.jit(_do_update)

    # --- 工具 --------------------------------------------------------------
    def evaluate(noiser_params, params):
        idx = jax.random.permutation(jax.random.key(args.seed), n_test)[:args.val_batch]
        imgs = jnp.asarray(x_test[idx], dtype=jnp.float32)
        labels = jnp.asarray(y_test[idx], dtype=jnp.int32)
        val_per_gpu = args.val_batch // num_gpus
        assert args.val_batch % num_gpus == 0
        acc = eval_pmap(noiser_params, params,
                        imgs.reshape(num_gpus, val_per_gpu, IN_DIM),
                        labels.reshape(num_gpus, val_per_gpu))
        return float(jax.device_get(acc))

    # --- 编译热身 ----------------------------------------------------------
    print("Compiling...")
    t0 = time.time()
    dummy_imgs = split_batch(jnp.zeros((args.batch, IN_DIM), dtype=jnp.float32))
    dummy_labels = split_batch(jnp.zeros(args.batch, dtype=jnp.int32))
    dummy_th = split_batch(jnp.arange(args.batch, dtype=jnp.int32))
    _ = forward_pmap(noiser_params, params, jnp.asarray(0, jnp.int32),
                     dummy_th, dummy_imgs, dummy_labels)
    val_per_gpu = args.val_batch // num_gpus
    _ = eval_pmap(noiser_params, params,
                  jnp.zeros((num_gpus, val_per_gpu, IN_DIM), dtype=jnp.float32),
                  jnp.zeros((num_gpus, val_per_gpu), dtype=jnp.int32))
    _ = do_update(noiser_params, params, jnp.zeros(args.batch),
                  jnp.asarray(0, jnp.int32),
                  jnp.arange(args.batch, dtype=jnp.int32))
    print(f"Warm-up done in {time.time() - t0:.1f}s")

    # --- 训练循环 ----------------------------------------------------------
    csv_path = args.csv_out
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_acc", "val_acc", "best_val", "best_train",
                 "epoch_time", "cum_time"])

    best_val, best_train = 0.0, 0.0
    data_key = jax.random.fold_in(master_key, 7)
    cum_t = 0.0

    for epoch in range(args.num_epochs):
        t_ep = time.time()
        data_key, sub = jax.random.split(data_key)
        idx = jax.random.permutation(sub, n_train)[:args.batch]
        imgs = jnp.asarray(x_train[idx], dtype=jnp.float32)
        labels = jnp.asarray(y_train[idx], dtype=jnp.int32)
        thread_ids = jnp.arange(args.batch, dtype=jnp.int32)

        fitness_dev, train_acc = forward_pmap(
            noiser_params, params, jnp.asarray(epoch, jnp.int32),
            split_batch(thread_ids), split_batch(imgs), split_batch(labels))
        train_acc = float(jax.device_get(train_acc))

        fitness_full = jnp.asarray(fitness_dev).reshape(args.batch)  # pmap 输出已按设备堆叠
        noiser_params, params = do_update(
            noiser_params, params, fitness_full,
            jnp.asarray(epoch, jnp.int32), thread_ids)

        best_train = max(best_train, train_acc)
        msg = (f"epoch {epoch:5d} | train_acc {train_acc:.4f} | "
               f"best_train {best_train:.4f}")
        val_acc = None
        if epoch % args.validate_every == 0:
            val_acc = evaluate(noiser_params, params)
            best_val = max(best_val, val_acc)
            msg += f" | val_acc {val_acc:.4f} | best_val {best_val:.4f}"
        t_ep = time.time() - t_ep
        cum_t += t_ep
        msg += f" | {t_ep:.2f}s | cum {cum_t:.1f}s"
        if epoch % args.log_every == 0 or epoch == args.num_epochs - 1:
            print(msg, flush=True)
        if csv_path:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_acc:.6f}", f"{val_acc if val_acc is not None else ''}",
                     f"{best_val:.6f}", f"{best_train:.6f}",
                     f"{t_ep:.3f}", f"{cum_t:.1f}"])

    # --- 终值报告 ----------------------------------------------------------
    vth_vals = []
    for k in sorted((k for k in params if k.startswith("v_th")),
                    key=lambda k: int(k.replace("v_th", ""))):
        raw = jax.device_get(params[k])
        vth_vals.append(float(jax.nn.softplus(raw)))
    print(f"final v_th = {[round(v, 4) for v in vth_vals]}")
    print(f"best_val = {best_val:.4f} | best_train = {best_train:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
