"""SNN 注意力（Hopfield / Mean-field）× 小批次等效大批次累积训练脚本。

融合两条已验证的工程/算法底座：
  1. 小批次等效大批次累积架构（docs/es_batch_accumulation_architecture.md）：
     参数冻结 K 段前向 + 拼接 raw fitness -> 一次全局 z-score -> chunked einsum 累积
     (+÷√K) -> 一次 optimizer 更新，严格等价单个大批次（定理 2：einsum 对样本线性）。
  2. patched-MNIST 注意力训练 + 参考 Softmax 等价性指标（llm_experiments/snn_attention_train.py）：
     --route hopfield / meanfield 两类 SNN 注意力，训练中周期测注意力权重误差 w_err
     （||p_snn - p_ref||）与输出余弦 cos_o。

本脚本把注意力模型放到大批次（默认 batch=60000 全量训练集）累积路径下验证：
  - 大批次累积下 Hopfield / Mean-field 对参考 Softmax 的注意力等价性演化（w_err / cos_o）；
  - 两路在同配置大批次下的性能对比（val_acc / best_val / best_train）。

用法（WSL venv 内，GPU）：
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /root/hyperscalees-venv/bin/python -m llm_experiments.snn_attention_train_accumulate \
      --route hopfield --batch 60000 --rank 64 --num-epochs 3000 \
      --mnist-dir /mnt/d/Rust/snn_t1/mnist_data --csv-out records/attention_accumulate/hopfield.csv

  --verify 模式：小规模断言累积（全局 z-score + chunked einsum）== 单大批次，
                 局部 z-score（naive）!= 大批次（负对照）。不训练。
"""

import argparse
import csv
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import jax.numpy as jnp
import optax

import hyperscalees as hs
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.snn_attention import (
    HopfieldAttnSNN,
    MeanFieldAttnSNN,
    model_rand_init,
    hopfield_attention,
    meanfield_attention,
    softmax_attention,
    _mk_qkv,
)
from hyperscalees.models.base_model import CommonParams
from hyperscalees.environments.snn_mnist import (
    get_mnist_arrays,
    poisson_encode,
    accuracy_from_logits,
)

NOISER = hs.noiser.eggroll.EggRoll
DTYPE = jnp.float32
NUM_CLASSES = 10
MNIST_DIR = os.environ.get("MNIST_DIR", r"D:\Rust\snn_t1\mnist_data")


# ---------------------------------------------------------------------------
# 奖励（与 snn_attention_train.py / snn_mnist_train_accumulate.py 一致的 loglik）
# ---------------------------------------------------------------------------
def reward_from_logits(logits, labels, reward="loglik"):
    """每样本 ES 奖励：loglik（默认，平滑稠密）或硬 0/1 binary。"""
    if reward == "binary":
        pred = jnp.argmax(logits, axis=-1)
        return (pred == labels).astype(jnp.float32)
    if reward == "loglik":
        ll = jax.nn.log_softmax(logits, axis=-1)
        return ll[jnp.arange(labels.shape[0]), labels]
    raise ValueError(f"unknown reward: {reward}")


# ---------------------------------------------------------------------------
# 28x28 图像切成 P x P patch token（每个 token = 展平的 patch 像素）
# ---------------------------------------------------------------------------
def patch_images(images, patch_px):
    """images: (batch, 28*28) float in [0,1] -> (batch, P*P, patch_px*patch_px)。"""
    assert 28 % patch_px == 0
    side = 28 // patch_px
    images = images.reshape(images.shape[0], side, patch_px, side, patch_px)
    return images.transpose(0, 1, 3, 2, 4).reshape(
        images.shape[0], side * side, patch_px * patch_px
    )


def parse_args():
    p = argparse.ArgumentParser(description="SNN 注意力 × 小批次等效大批次累积训练")
    p.add_argument("--route", choices=["hopfield", "meanfield"], default="hopfield")
    p.add_argument("--patch-px", type=int, default=7,
                   help="patch 边长（28//patch_px 个 token，7 -> 4x4=16 tokens）")
    p.add_argument("--d-head", type=int, default=16, help="Q/K/V 投影维度")
    p.add_argument("--batch", type=int, default=60000, help="总 batch N_L（=K*chunk）")
    p.add_argument("--accumulate", type=int, default=0,
                   help="累积段数 K（0=按显存公式自动取）")
    p.add_argument("--T", type=int, default=8, help="泊松编码/SNN 时间步")
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--rank", type=int, default=64, help="LoRA 噪声 rank")
    p.add_argument("--n-iter", type=int, default=8,
                   help="注意力迭代/population 步数")
    p.add_argument("--tau-m", type=float, default=20.0)
    p.add_argument("--proj-gain", type=float, default=2.0,
                   help="Q/K/V rate-encoder sigmoid 斜率")
    p.add_argument("--reward", choices=["loglik", "binary"], default="loglik")
    p.add_argument("--num-epochs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate-every", type=int, default=50)
    p.add_argument("--val-batch", type=int, default=2000)
    p.add_argument("--mnist-dir", default=None, help="MNIST IDX 目录")
    p.add_argument("--verify", action="store_true",
                   help="仅验证累积==单大批次（小规模），不训练")
    p.add_argument("--csv-out", default=None)
    return p.parse_args()


def tree_max_abs_diff(a, b):
    """两个 pytree 逐叶最大绝对差。"""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    diffs = [jnp.max(jnp.abs(x - y)) for x, y in zip(leaves_a, leaves_b)]
    return float(max(jax.device_get(d) for d in diffs))


def mem_safe_accumulate(batch, rank):
    """按显存公式自动选累积段数 K：chunk ≤ 0.765e6/rank（B 矩阵安全水位）。

    注意力模型最大 LoRA 矩阵为 q/k/v (49x16)，B=(chunk, 49, rank) 远小于 SNN 的
    (chunk, 784, rank)，故此处取相同保守公式即可，rank 越低 chunk 可越大。
    """
    max_chunk = max(1, int(0.765e6 / rank))
    max_chunk = min(max_chunk, batch)
    acc = batch // max_chunk
    if acc < 1:
        acc = 1
    while batch % acc != 0:
        acc += 1
    return acc


def compute_equivalence(frozen_params, params, es_tree_key, mean_rate, route,
                        frozen_noiser=None, noiser_params=None):
    """SNN 注意力权重 vs 参考 Softmax 的等价性指标（基于累积后的同一 params）。

    ``mean_rate``: (S, 1, num_tokens, d) 单时间步 rate 输入，使 iterinfo=None 的
    同一组 Q/K/V 投影同时喂给 SNN 核心与参考 Softmax。返回 (w_err, cos_o) 均值。
    """
    def attn_core(q, k, v, beta):
        if route == "hopfield":
            return hopfield_attention(q, k, v, g_inh=frozen_params["g_inh"],
                                      tau_a=frozen_params["tau_a"],
                                      beta=beta, n_iter=frozen_params["n_iter"])
        return meanfield_attention(q, k, v, gamma=frozen_params["gamma"],
                                   beta=beta, n_iter=frozen_params["n_iter"])

    def one(x):
        # iterinfo=None => 无 LoRA 噪声，SNN 核心与参考共用同一 Q/K/V
        common = CommonParams(NOISER, frozen_noiser, noiser_params,
                              frozen_params, params, es_tree_key, None)
        q, k, v = _mk_qkv(common, x)
        beta = jax.nn.softplus(params["beta"])
        p_snn, o_snn = attn_core(q, k, v, beta)
        p_ref, o_ref = softmax_attention(q, k, v, beta)
        w_err = jnp.mean(jnp.abs(p_snn - p_ref))
        cos_num = jnp.sum(o_snn * o_ref)
        cos_den = jnp.maximum(
            jnp.sqrt(jnp.sum(o_snn ** 2) * jnp.sum(o_ref ** 2)), 1e-6)
        return w_err, cos_num / cos_den

    w_errs, cos_os = jax.vmap(one)(mean_rate)
    return float(jnp.mean(w_errs)), float(jnp.mean(cos_os))


def main():
    args = parse_args()
    assert 28 % args.patch_px == 0, "patch_px 必须整除 28"
    num_tokens = (28 // args.patch_px) ** 2
    token_in_dim = args.patch_px ** 2

    if args.accumulate <= 0:
        args.accumulate = mem_safe_accumulate(args.batch, args.rank)
    assert args.batch % args.accumulate == 0, "batch 必须能被 accumulate 整除"
    chunk = args.batch // args.accumulate

    print(f"[env] devices={len(jax.devices())} route={args.route} batch={args.batch} "
          f"accumulate={args.accumulate} chunk={chunk} rank={args.rank} T={args.T}")

    master_key = jax.random.key(args.seed)
    model_key, es_key, enc_base = jax.random.split(master_key, 3)

    frozen_params, params, scan_map, es_map = model_rand_init(
        args.route, model_key, token_in_dim=token_in_dim, num_tokens=num_tokens,
        num_classes=NUM_CLASSES, d_head=args.d_head, tau_m=args.tau_m,
        proj_gain=args.proj_gain, n_iter=args.n_iter, dtype=DTYPE,
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=max(20, args.num_epochs // 10),
        decay_steps=args.num_epochs, end_value=args.lr * 0.1,
    )
    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params, args.sigma, lr_schedule, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=args.rank,
    )

    mnist_dir = args.mnist_dir or os.environ.get("MNIST_DIR") or MNIST_DIR
    x_train, y_train = get_mnist_arrays("train", data_dir=mnist_dir)
    x_test, y_test = get_mnist_arrays("test", data_dir=mnist_dir)
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    print(f"[data] train={n_train} test={n_test} patched-{num_tokens} tokens, "
          f"token_dim={token_in_dim} dir={mnist_dir}")

    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, x: (HopfieldAttnSNN if args.route == "hopfield"
                            else MeanFieldAttnSNN).forward(
            NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x),
        in_axes=(None, None, 0, 0),
    ))
    jit_eval = jax.jit(jax.vmap(
        lambda n, p, x: (HopfieldAttnSNN if args.route == "hopfield"
                         else MeanFieldAttnSNN).forward(
            NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, None, x),
        in_axes=(None, None, 0),
    ))
    jit_update = jax.jit(
        lambda n, p, f, i: NOISER.do_updates(
            frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

    # 内存高效累积更新：全局 z-score 后的 fitness 切成 K 段，每段独立 einsum
    # 梯度（einsum 对样本线性），scan 累加，最后一次 solver 更新。== 单大批次（定理 2）。
    def make_accumulated_update(accum, chk):
        def _accumulated_update(noiser_params, params, conv_full, thread_ids_full, epoch):
            conv_chunks = conv_full.reshape(accum, chk)
            tid_chunks = thread_ids_full.reshape(accum, chk)

            def step(grad_acc, xs):
                conv_k, tid_k = xs
                iterinfo = (jnp.full(chk, epoch, jnp.int32), tid_k)
                gk = jax.tree.map(
                    lambda p, kk, m: NOISER._do_update(
                        p, kk, conv_k, iterinfo, m,
                        noiser_params["sigma"], frozen_noiser_params),
                    params, es_tree_key, es_map)
                return jax.tree.map(lambda a, b: a + b, grad_acc, gk), None

            grad0 = jax.tree.map(lambda p: jnp.zeros_like(p), params)
            grad_total, _ = jax.lax.scan(step, grad0, (conv_chunks, tid_chunks))
            grad_total = jax.tree.map(lambda g: g / jnp.sqrt(accum), grad_total)

            updates, new_opt = frozen_noiser_params["solver"].update(
                grad_total, noiser_params["opt_state"], params)
            noiser_params["opt_state"] = new_opt
            return noiser_params, optax.apply_updates(params, updates)

        return jax.jit(_accumulated_update)

    accum_update = make_accumulated_update(args.accumulate, chunk)

    # -----------------------------------------------------------------------
    # 定理 2 证明：累积（全局 z-score + chunked einsum）== 单大批次；naive 局部 z-score 不等价
    # -----------------------------------------------------------------------
    if args.verify:
        vbatch = min(512, n_train)
        # 强制多段累积（>=2 chunk），使负对照（每 chunk 局部 z-score）有意义
        vacc = max(2, mem_safe_accumulate(vbatch // 2, args.rank))
        while vbatch % vacc != 0:
            vacc += 1
        vchunk = vbatch // vacc
        vkey = jax.random.key(999)
        idx = jax.random.permutation(vkey, n_train)[:vbatch]
        imgs = jnp.asarray(x_train[idx], dtype=jnp.float32)
        labels = jnp.asarray(y_train[idx], dtype=jnp.int32)
        thread_ids = jnp.arange(vbatch, dtype=jnp.int32)
        patch = patch_images(imgs, args.patch_px)
        vspikes = poisson_encode(patch, args.T, enc_base).transpose(1, 0, 2, 3)
        vepoch = jnp.asarray(0, jnp.int32)

        vfwd = jax.jit(jax.vmap(
            lambda n, p, i, x: (HopfieldAttnSNN if args.route == "hopfield"
                                else MeanFieldAttnSNN).forward(
                NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x),
            in_axes=(None, None, 0, 0),
        ))
        vupd = jax.jit(lambda n, p, f, i: NOISER.do_updates(
            frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

        def fwd_batch(psnap, sp, tids):
            it = (jnp.full(sp.shape[0], 0, jnp.int32), tids)
            return vfwd(noiser_params, psnap, it, sp)

        # 路径 A：单大批次
        raw_A = reward_from_logits(fwd_batch(params, vspikes, thread_ids), labels, args.reward)
        conv_A = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_A)
        _, params_A = vupd(noiser_params, params, conv_A,
                           (jnp.full(vbatch, 0, jnp.int32), thread_ids))

        # 路径 B：K 段累积 + 一次全局 z-score + 一次更新
        raw_chunks = []
        for k in range(vacc):
            sl = slice(k * vchunk, (k + 1) * vchunk)
            raw_chunks.append(reward_from_logits(
                fwd_batch(params, vspikes[sl], thread_ids[sl]), labels[sl], args.reward))
        conv_B = NOISER.convert_fitnesses(
            frozen_noiser_params, noiser_params, jnp.concatenate(raw_chunks))
        _, params_B = vupd(noiser_params, params, conv_B,
                           (jnp.full(vbatch, 0, jnp.int32), thread_ids))

        # 路径 D：chunked einsum 累积更新（训练实际路径，按 verify 规模构造）
        vaccum_update = make_accumulated_update(vacc, vchunk)
        _, params_D = vaccum_update(noiser_params, params, conv_A, thread_ids, vepoch)

        # 路径 C（负对照）：每 chunk 局部 z-score + 每 chunk 更新
        params_C = params
        noiser_C = noiser_params
        for k in range(vacc):
            sl = slice(k * vchunk, (k + 1) * vchunk)
            raw_k = reward_from_logits(fwd_batch(params_C, vspikes[sl], thread_ids[sl]),
                                       labels[sl], args.reward)
            conv_k = NOISER.convert_fitnesses(frozen_noiser_params, noiser_C, raw_k)
            noiser_C, params_C = vupd(noiser_C, params_C, conv_k,
                                      (jnp.full(vchunk, 0, jnp.int32), thread_ids[sl]))

        d_AB = tree_max_abs_diff(params_A, params_B)
        d_AD = tree_max_abs_diff(params_A, params_D)
        d_AC = tree_max_abs_diff(params_A, params_C)
        print(f"[verify] 累积 vs 大批次 max|Δparam| = {d_AB:.3e}  (≈0，定理2 等价)")
        print(f"[verify] chunked-einsum vs 大批次    = {d_AD:.3e}  (≈0，einsum 线性)")
        print(f"[verify] naive vs 大批次            = {d_AC:.3e}  (>0，负对照不等价)")
        assert d_AB < 1e-3, f"定理2 失败：累积应≈大批次，实际 {d_AB:.3e}"
        assert d_AD < 1e-3, f"chunked-einsum 应≈大批次，实际 {d_AD:.3e}"
        # 负对照：局部 z-score 应与大批次有可测差异（远超精确等价路径的 ~0 浮点底）
        assert d_AC > 1e-5, f"负对照失败：naive 局部归一化应有差异，实际 {d_AC:.3e}"
        print("[verify] PASS：累积==大批次（精确），chunked-einsum 等价，naive 局部归一化不相等")
        return

    # -----------------------------------------------------------------------
    # 训练循环：小批次累积 = 大批次（注意力等价性指标基于累积后的同一 params）
    # -----------------------------------------------------------------------
    print("Compiling...")
    t0 = time.time()
    warm_spikes = jnp.zeros((chunk, args.T, num_tokens, token_in_dim), dtype=DTYPE)
    warm_it = (jnp.zeros(chunk, dtype=jnp.int32), jnp.arange(chunk, dtype=jnp.int32))
    _ = jit_forward(noiser_params, params, warm_it, warm_spikes)
    _ = jit_eval(noiser_params, params,
                 jnp.zeros((args.val_batch, args.T, num_tokens, token_in_dim), dtype=DTYPE))
    _ = accum_update(noiser_params, params, jnp.zeros(args.batch),
                     jnp.arange(args.batch, dtype=jnp.int32), jnp.asarray(0, jnp.int32))
    print(f"Warm-up done in {time.time() - t0:.1f}s")

    def evaluate():
        idx = jax.random.permutation(jax.random.key(1234), n_test)[:args.val_batch]
        imgs = jnp.asarray(x_test[idx], dtype=jnp.float32)
        labels = jnp.asarray(y_test[idx], dtype=jnp.int32)
        patch = patch_images(imgs, args.patch_px)
        spikes = poisson_encode(patch, args.T, jax.random.key(1_000_000)).transpose(1, 0, 2, 3)
        logits = jit_eval(noiser_params, params, spikes)
        acc = float(accuracy_from_logits(logits, labels))

        # 等价性指标：均值 rate token，iterinfo=None 无噪声 Q/K/V，用累积更新后的 params
        sample_size = min(64, args.val_batch)
        sub = spikes[:sample_size]
        mean_rate = sub.mean(axis=1, keepdims=True)   # (S, 1, N, D)
        w_err, cos_o = compute_equivalence(
            frozen_params, params, es_tree_key, mean_rate, route=args.route,
            frozen_noiser=frozen_noiser_params, noiser_params=noiser_params)
        return acc, w_err, cos_o

    csv_path = args.csv_out
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_acc", "val_acc", "best_val", "best_train",
                 "w_err", "cos_o", "epoch_time", "cum_time"])

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
        patch = patch_images(imgs, args.patch_px)

        # --- K 段累积（参数冻结，每 chunk 全新噪声 + 独立编码）---
        raw_chunks = []
        correct = 0
        for k in range(args.accumulate):
            sl = slice(k * chunk, (k + 1) * chunk)
            enc_key = jax.random.fold_in(jax.random.fold_in(enc_base, epoch), k)
            spikes_k = poisson_encode(patch[sl], args.T, enc_key).transpose(1, 0, 2, 3)
            it_k = (jnp.full(chunk, epoch, jnp.int32), thread_ids[sl])
            logits_k = jit_forward(noiser_params, params, it_k, spikes_k)
            raw_chunks.append(reward_from_logits(logits_k, labels[sl], args.reward))
            correct += int(jnp.sum(jnp.argmax(logits_k, -1) == labels[sl]))

        # --- 拼接 -> 一次全局 z-score -> 一次累积更新（== 单大批次）---
        raw_full = jnp.concatenate(raw_chunks)
        conv = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_full)
        noiser_params, params = accum_update(
            noiser_params, params, conv, thread_ids, epoch)

        train_acc = correct / args.batch
        best_train = max(best_train, train_acc)
        msg = f"epoch {epoch:5d} | train_acc {train_acc:.4f} | best {best_train:.4f}"
        val_acc = w_err = cos_o = None
        if epoch % args.validate_every == 0 or epoch == args.num_epochs - 1:
            val_acc, w_err, cos_o = evaluate()
            best_val = max(best_val, val_acc)
            msg += (f" | val_acc {val_acc:.4f} | best_val {best_val:.4f}"
                    f" | w_err {w_err:.4f} | cos_o {cos_o:.4f}")
        t_ep = time.time() - t_ep
        cum_t += t_ep
        msg += f" | {t_ep:.2f}s | cum {cum_t:.1f}s"
        print(msg, flush=True)
        if csv_path:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_acc:.6f}",
                     f"{val_acc if val_acc is not None else ''}",
                     f"{best_val:.6f}", f"{best_train:.6f}",
                     f"{w_err if w_err is not None else ''}",
                     f"{cos_o if cos_o is not None else ''}",
                     f"{t_ep:.3f}", f"{cum_t:.1f}"])

    print(f"[{args.route}] best_val={best_val:.4f} best_train={best_train:.4f} "
          f"final w_err={w_err:.4f} cos_o={cos_o:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
