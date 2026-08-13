"""小批次梯度累积 ⇔ 大批次训练：证明 + 复现脚本。

数学基础（docs/es_batch_equivalence_math.md 定理 2/4）：
  ES 前向是 jax.vmap（逐样本独立），fitness → convert_fitnesses 全局 z-score → do_updates
  的 einsum('nir,njr->ij') 对样本求和。因此：

    参数冻结 + K 段 chunk（全新噪声，thread_id 全局唯一）+ 拼接 raw fitness
    + 一次全局 z-score + 一次 optimizer 更新  ==  单个 batch = K*N_s 的训练（代数精确）。

与"naive 小批次多次"的本质区别（审查报告关键结论）：
    正确：冻结参数 + 全局 z-score + 一次更新  ⇒ 精确等价大批次（定理 2）；
    错误：每 chunk 局部 z-score + 每 chunk 单独更新 ⇒ 不等价（局部归一化破坏线性性）。

用法（仓库根目录，GPU 环境）：
  # 1) 只证明等价性（不训练）：累积 vs 单大批次 逐参数精确相等，naive 局部归一化不相等
  python -m llm_experiments.snn_mnist_train_accumulate --verify --batch 60000 --accumulate 5

  # 2) 复现 0.9（单卡小批次累积 = 大批次，显存不足可降 --rank 或调大 --accumulate）
  python -m llm_experiments.snn_mnist_train_accumulate \
      --batch 60000 --accumulate 5 --rank 64 --num-epochs 3000 \
      --mnist-dir /mnt/d/Rust/snn_t1/mnist_data
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
from hyperscalees.environments.snn_mnist import get_mnist_arrays, poisson_encode

NOISER = hs.noiser.eggroll.EggRoll
DTYPE = jnp.float32
IN_DIM = 28 * 28
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# 模型：v_th 可训练（softplus 恒正）的 SNN 变体（与 snn_mnist_train_multi_gpu.py 一致）
# ---------------------------------------------------------------------------
class TrainableVthSNN(Model):
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
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        for i in range(len(hidden_dims)):
            layers_kwargs[f"v_th{i + 1}"] = Parameter.rand_init(
                None, None, None, raw_vth0, dtype=dtype)
        frozen_params = {"tau_m": jnp.asarray(tau_m, dtype=dtype)}
        layers = merge_inits(**layers_kwargs)
        return CommonInit(frozen_params, layers.params, layers.scan_map, layers.es_map)

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
        cur = x
        for i in range(n_layers):
            cur = jax.vmap(
                lambda xt: call_submodule(MM, f"fc{i + 1}", common_params, xt)
            )(cur)
            v0 = jnp.zeros((cur.shape[-1],), dtype=cur.dtype)
            lif_p = {"tau_m": common_params.frozen_params["tau_m"], "v_th": vths[i]}
            cur = run_lif(lif_p, cur, v0)
        rate = jnp.mean(cur, axis=0)
        logits = call_submodule(MM, f"fc{n_layers + 1}", common_params, rate)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


def fitness_from_logits(logits, labels, reward="loglik"):
    """每样本原始奖励：loglik（默认）或硬 0/1。"""
    if reward == "binary":
        pred = jnp.argmax(logits, axis=-1)
        return (pred == labels).astype(jnp.float32)
    if reward == "loglik":
        ll = jax.nn.log_softmax(logits, axis=-1)
        return ll[jnp.arange(labels.shape[0]), labels]
    raise ValueError(f"unknown reward: {reward}")


def parse_args():
    p = argparse.ArgumentParser(description="小批次梯度累积 == 大批次：证明 + 复现")
    p.add_argument("--batch", type=int, default=60000, help="总 batch N_L（=K*chunk）")
    p.add_argument("--accumulate", type=int, default=5, help="累积 chunk 数 K（每 chunk 显存 = batch/K）")
    p.add_argument("--rank", type=int, default=64, help="LoRA rank")
    p.add_argument("--T", type=int, default=8, help="泊松编码/SNN 时间步")
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--reward", choices=["loglik", "binary"], default="loglik")
    p.add_argument("--group-size", type=int, default=0, help="组内归一化（0=全局 z-score）")
    p.add_argument("--num-epochs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=str, default="128,128")
    p.add_argument("--mnist-dir", default=None, help="MNIST IDX 目录")
    p.add_argument("--verify", action="store_true",
                   help="仅验证「累积==单大批次」的精确等价，不训练")
    p.add_argument("--validate-every", type=int, default=50)
    p.add_argument("--val-batch", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--csv-out", default=None)
    return p.parse_args()


def tree_max_abs_diff(a, b):
    """两个参数 pytree 的逐叶最大绝对差。"""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    diffs = [jnp.max(jnp.abs(x - y)) for x, y in zip(leaves_a, leaves_b)]
    return float(max(jax.device_get(d) for d in diffs))


def main():
    args = parse_args()
    hidden_dims = [int(h) for h in args.hidden.split(",")]
    assert args.batch % args.accumulate == 0, "batch 必须能被 accumulate 整除"
    chunk = args.batch // args.accumulate
    print(f"[env] devices={len(jax.devices())} batch={args.batch} "
          f"accumulate={args.accumulate} chunk={chunk} rank={args.rank} T={args.T}")

    master_key = jax.random.key(args.seed)
    model_key, es_key, enc_base = jax.random.split(master_key, 3)

    frozen_params, params, scan_map, es_map = TrainableVthSNN.rand_init(
        model_key, in_dim=IN_DIM, hidden_dims=hidden_dims,
        num_classes=NUM_CLASSES, tau_m=20.0, v_th=0.3, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params, args.sigma, args.lr, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999},
        rank=args.rank, group_size=args.group_size, freeze_nonlora=False,
    )

    mnist_dir = args.mnist_dir or os.environ.get("MNIST_DIR")
    x_train, y_train = get_mnist_arrays("train", data_dir=mnist_dir)
    x_test, y_test = get_mnist_arrays("test", data_dir=mnist_dir)
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    print(f"[data] train={n_train} test={n_test} dir={mnist_dir}")

    # 单设备前向（vmap over batch），与 snn_mnist_train.py 一致
    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: TrainableVthSNN.forward(
            NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0),
    ))
    # 干净评估（iterinfo=None，无扰动）
    jit_eval = jax.jit(jax.vmap(
        lambda n, p, x: TrainableVthSNN.forward(
            NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, None, x),
        in_axes=(None, None, 0),
    ))
    # 一次更新（fitnesses 已经全局 z-score）—— 用于 verify 证明
    jit_update = jax.jit(
        lambda n, p, f, i: NOISER.do_updates(
            frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

    # 内存高效的梯度累积更新：把全局 z-score 后的 fitness 切成 K 段，
    # 每段独立算 einsum 梯度（einsum 对样本线性），scan 累加，最后一次 solver 更新。
    # 数学上 == 单大批次 do_updates（见定理 2：einsum 线性 + 全局 z-score）。
    # 好处：更新步显存从 batch×784×rank 降到 chunk×784×rank（24G 卡也能跑 batch=60000+rank=64）。
    def _accumulated_update(noiser_params, params, conv_full, thread_ids_full, epoch):
        conv_chunks = conv_full.reshape(args.accumulate, chunk)   # (K, chunk)
        tid_chunks = thread_ids_full.reshape(args.accumulate, chunk)

        def step(grad_acc, xs):
            conv_k, tid_k = xs
            iterinfo = (jnp.full(chunk, epoch, jnp.int32), tid_k)
            gk = jax.tree.map(
                lambda p, kk, m: NOISER._do_update(
                    p, kk, conv_k, iterinfo, m,
                    noiser_params["sigma"], frozen_noiser_params),
                params, es_tree_key, es_map)
            return jax.tree.map(lambda a, b: a + b, grad_acc, gk), None

        grad0 = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        grad_total, _ = jax.lax.scan(step, grad0, (conv_chunks, tid_chunks))
        # 每段 _do_update 已除以 sqrt(chunk)，K 段累加后再除以 sqrt(K) 恢复 sqrt(batch) 尺度
        grad_total = jax.tree.map(lambda g: g / jnp.sqrt(args.accumulate), grad_total)

        updates, new_opt = frozen_noiser_params["solver"].update(
            grad_total, noiser_params["opt_state"], params)
        noiser_params["opt_state"] = new_opt
        return noiser_params, optax.apply_updates(params, updates)

    accum_update = jax.jit(_accumulated_update)


    # ----------------------------------------------------------------------
    # 定理 2 证明：累积（全局 z-score + 一次更新）== 单大批次，逐参数精确相等；
    #              naive 局部 z-score + 每 chunk 更新 != 大批次（负对照）。
    # ----------------------------------------------------------------------
    if args.verify:
        print("\n[verify] 定理2 精确等价证明（K 段累积 == 单大批次）")
        vkey = jax.random.key(999)
        idx = jax.random.permutation(vkey, n_train)[:args.batch]
        imgs = jnp.asarray(x_train[idx], dtype=jnp.float32)
        labels = jnp.asarray(y_train[idx], dtype=jnp.int32)
        thread_ids = jnp.arange(args.batch, dtype=jnp.int32)
        # 同一编码：保证累积与单大批次用完全相同样本 + 完全相同噪声
        spikes = poisson_encode(imgs, args.T, enc_base).transpose(1, 0, 2)
        epoch = jnp.asarray(0, dtype=jnp.int32)

        def forward_batch(params_snap, spikes_b, tids):
            it = (jnp.full(spikes_b.shape[0], 0, dtype=jnp.int32), tids)
            return jit_forward(noiser_params, params_snap, it, spikes_b)

        # 路径 A：单大批次
        raw_A = fitness_from_logits(forward_batch(params, spikes, thread_ids), labels, args.reward)
        conv_A = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_A)
        _, params_A = jit_update(noiser_params, params, conv_A,
                                 (jnp.full(args.batch, 0, jnp.int32), thread_ids))

        # 路径 B：K 段累积（同一 spikes/thread_ids 切片，拼接后一次全局 z-score + 一次更新）
        raw_chunks = []
        for k in range(args.accumulate):
            sl = slice(k * chunk, (k + 1) * chunk)
            raw_chunks.append(fitness_from_logits(
                forward_batch(params, spikes[sl], thread_ids[sl]), labels[sl], args.reward))
        conv_B = NOISER.convert_fitnesses(
            frozen_noiser_params, noiser_params, jnp.concatenate(raw_chunks))
        _, params_B = jit_update(noiser_params, params, conv_B,
                                 (jnp.full(args.batch, 0, jnp.int32), thread_ids))

        # 路径 D：内存高效累积更新（chunked einsum，scan 累加）—— 用于训练的更新路径，也应等价
        _, params_D = accum_update(noiser_params, params, conv_A, thread_ids, epoch)

        # 路径 C（负对照）：每 chunk 局部 z-score + 每 chunk 单独更新（naive 小批次多次）
        params_C = params
        noiser_C = noiser_params
        for k in range(args.accumulate):
            sl = slice(k * chunk, (k + 1) * chunk)
            raw_k = fitness_from_logits(forward_batch(params_C, spikes[sl], thread_ids[sl]), labels[sl], args.reward)
            conv_k = NOISER.convert_fitnesses(frozen_noiser_params, noiser_C, raw_k)
            noiser_C, params_C = jit_update(noiser_C, params_C, conv_k,
                                            (jnp.full(chunk, 0, jnp.int32), thread_ids[sl]))

        d_AB = tree_max_abs_diff(params_A, params_B)   # 应≈0（精确等价，仅 float32 kernel 非确定性）
        d_AD = tree_max_abs_diff(params_A, params_D)   # 应≈0（chunked einsum 累积等价）
        d_AC = tree_max_abs_diff(params_A, params_C)   # 应>0（负对照不等价）
        print(f"  累积 vs 大批次 max|Δparam| = {d_AB:.3e}  (≈0，定理2 等价，残差为 float32 不同 batch 尺寸 kernel 的非确定性)")
        print(f"  chunked-einsum vs 大批次 = {d_AD:.3e}  (≈0，einsum 线性 → 分段累加等价)")
        print(f"  naive vs 大批次 max|Δparam| = {d_AC:.3e}  (>0，局部归一化破坏等价)")
        assert d_AB < 1e-3, f"定理2 失败：累积应≈大批次，实际 {d_AB:.3e}"
        assert d_AD < 1e-3, f"chunked-einsum 累积应≈大批次，实际 {d_AD:.3e}"
        assert d_AC > 1e-3, f"负对照失败：naive 局部归一化应与大批次有差异，实际 {d_AC:.3e}"
        print("[verify] PASS：累积==大批次（精确），chunked-einsum 等价，naive 局部归一化不相等\n")
        return

    # ----------------------------------------------------------------------
    # 训练循环：小批次累积 = 大批次，复现 0.9
    # ----------------------------------------------------------------------
    print("Compiling...")
    t0 = time.time()
    warm_spikes = jnp.zeros((chunk, args.T, IN_DIM), dtype=DTYPE)
    warm_it = (jnp.zeros(chunk, dtype=jnp.int32), jnp.arange(chunk, dtype=jnp.int32))
    _ = jit_forward(noiser_params, params, warm_it, warm_spikes)
    _ = jit_eval(noiser_params, params, jnp.zeros((args.val_batch, args.T, IN_DIM)))
    # 热身内存高效的累积更新（chunked einsum），而非全 batch 单次更新
    _ = accum_update(noiser_params, params, jnp.zeros(args.batch),
                     jnp.arange(args.batch, dtype=jnp.int32), jnp.asarray(0, jnp.int32))
    print(f"Warm-up done in {time.time() - t0:.1f}s")

    def evaluate():
        idx = jax.random.permutation(jax.random.key(1234), n_test)[:args.val_batch]
        imgs = jnp.asarray(x_test[idx], dtype=jnp.float32)
        labels = jnp.asarray(y_test[idx], dtype=jnp.int32)
        spikes = poisson_encode(imgs, args.T, jax.random.key(1_000_000)).transpose(1, 0, 2)
        logits = jit_eval(noiser_params, params, spikes)
        return float(jnp.mean((jnp.argmax(logits, -1) == labels).astype(jnp.float32)))

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

        # --- K 段累积（参数冻结，每段全新噪声 + 独立编码）---
        raw_chunks = []
        correct = 0
        for k in range(args.accumulate):
            sl = slice(k * chunk, (k + 1) * chunk)
            enc_key = jax.random.fold_in(jax.random.fold_in(enc_base, epoch), k)
            spikes_k = poisson_encode(imgs[sl], args.T, enc_key).transpose(1, 0, 2)
            it_k = (jnp.full(chunk, epoch, dtype=jnp.int32), thread_ids[sl])
            logits_k = jit_forward(noiser_params, params, it_k, spikes_k)
            raw_chunks.append(fitness_from_logits(logits_k, labels[sl], args.reward))
            correct += int(jnp.sum(jnp.argmax(logits_k, -1) == labels[sl]))

        # --- 拼接 → 一次全局 z-score → 一次内存高效的累积更新（== 单大批次）---
        raw_full = jnp.concatenate(raw_chunks)
        conv = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_full)
        noiser_params, params = accum_update(
            noiser_params, params, conv, thread_ids, epoch)

        train_acc = correct / args.batch
        best_train = max(best_train, train_acc)
        msg = f"epoch {epoch:5d} | train_acc {train_acc:.4f} | best {best_train:.4f}"
        val_acc = None
        if epoch % args.validate_every == 0:
            val_acc = evaluate()
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
                     f"{best_val:.6f}", f"{best_train:.6f}", f"{t_ep:.3f}", f"{cum_t:.1f}"])

    print(f"best_val = {best_val:.4f} | best_train = {best_train:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
