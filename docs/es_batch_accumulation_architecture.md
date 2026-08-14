# 小批次等效大批次：训练架构实现与参数配置逻辑

> 本文总结**已验证**的"小批次等效大批次"训练架构的**实现过程**与**参数配置逻辑**。该架构在单卡 RTX 4090（24GB）
> 上用梯度累积复现了 8×4090 大批次训练的精度（best_val 0.9149 vs 0.9152），等价性已由 `--verify` 模式逐参数证明。
> 理论根基见 [es_batch_equivalence_math.md](es_batch_equivalence_math.md)（定理 2 梯度累积代数精确 + einsum 线性）。

---

## 1. 架构总览：两层累积

核心思想是**把大批次训练拆成"前向累积"与"更新累积"两层**，每层都严格等价于单大批次：

```
单大批次 batch=N_L
   │
   ▼ 数学等价（定理 2）
前向累积：参数冻结，K 段 chunk（各 N_s=N_L/K）顺序前向，拼接 raw fitness
   │
   ▼ 一次全局 z-score（对所有 N_L 个 fitness，不能局部归一化）
   │
   ▼ 数学等价（einsum 对样本线性）
更新累积：K 段 chunked einsum（jax.lax.scan），累加梯度，÷√K 恢复尺度
   │
   ▼ 一次 optimizer 更新（adamw，状态只在累积后更新一次）
```

**两条等价性的本质**：
1. 前向是 `jax.vmap`（逐样本独立）→ 切块拼接 == 整批；
2. 更新 `einsum('nir,njr->ij')` 对样本索引 $n$ 是线性求和 → 分段求和 == 整批求和。

---

## 2. 实现过程

### 2.1 前向累积（参数冻结 + 全新噪声）

```python
raw_chunks = []
for k in range(K):                          # K 段顺序前向，参数冻结
    sl = slice(k * chunk, (k + 1) * chunk)
    enc_key = jax.random.fold_in(jax.random.fold_in(enc_base, epoch), k)   # 每 chunk 独立编码
    spikes_k = poisson_encode(imgs[sl], T, enc_key).transpose(1, 0, 2)
    it_k = (jnp.full(chunk, epoch, jnp.int32), thread_ids[sl])   # 全局唯一噪声方向
    logits_k = jit_forward(noiser_params, params, it_k, spikes_k)
    raw_chunks.append(fitness_from_logits(logits_k, labels[sl], reward))

raw_full = jnp.concatenate(raw_chunks)      # 拼接成 (N_L,)
```

**要点**：`thread_ids` 必须全局唯一（跨 chunk 不碰撞），保证每个样本的扰动方向独立；`params` 在整个 K 段循环内不变（冻结）。

### 2.2 一次全局 z-score

```python
conv = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_full)  # 全局 mean/std
```

**关键**：z-score 的 mean/std 必须对**全部 N_L 个 fitness** 一次计算。若每 chunk 单独归一化（局部 mean/std），
则破坏线性性，等价性不成立（`--verify` 负对照验证了这一点，误差 5.9e-02）。

### 2.3 更新累积（chunked einsum）

```python
def _accumulated_update(noiser_params, params, conv_full, thread_ids_full, epoch):
    conv_chunks = conv_full.reshape(K, chunk)
    tid_chunks = thread_ids_full.reshape(K, chunk)

    def step(grad_acc, xs):
        conv_k, tid_k = xs
        iterinfo = (jnp.full(chunk, epoch, jnp.int32), tid_k)
        gk = jax.tree.map(
            lambda p, kk, m: NOISER._do_update(p, kk, conv_k, iterinfo, m,
                                               noiser_params["sigma"], frozen_noiser_params),
            params, es_tree_key, es_map)
        return jax.tree.map(lambda a, b: a + b, grad_acc, gk), None   # 累加梯度

    grad0 = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    grad_total, _ = jax.lax.scan(step, grad0, (conv_chunks, tid_chunks))
    grad_total = jax.tree.map(lambda g: g / jnp.sqrt(K), grad_total)  # 尺度恢复
    updates, new_opt = frozen_noiser_params["solver"].update(grad_total, noiser_params["opt_state"], params)
    noiser_params["opt_state"] = new_opt
    return noiser_params, optax.apply_updates(params, updates)
```

**内存收益**：更新步的 einsum 中间张量从 `batch×784×rank`（rank=64 时 ~26GB）降到 `chunk×784×rank`（~2.4GB），
使 24GB 单卡可跑 batch=60000 + rank=64（甚至 rank=1024）。

### 2.4 尺度恢复（÷√K 的推导）

`_do_update` 返回 `-einsum/√N`。每 chunk 返回 `-einsum_k/√chunk`，K 段累加后除以 √K：

$$
\frac{1}{\sqrt K}\sum_{k=1}^K \frac{-einsum_k}{\sqrt{chunk}}
=\frac{-\sum_k einsum_k}{\sqrt{K\cdot chunk}}
=\frac{-einsum_{\text{total}}}{\sqrt{N_L}},
$$

与全 batch 单次更新**严格一致**。

---

## 3. 等价性证明（`--verify` 四路径）

| 路径 | 做法 | vs 大批次 max\|Δparam\| | 判定 |
|---|---|---:|---|
| A | 单大批次（基准） | — | — |
| B | 前向累积 + 全局 z-score + 一次更新 | 1.0e-04 | ✅ 等价 |
| D | chunked einsum 更新累积（训练实际路径） | 7.9e-06 | ✅ 等价（更紧） |
| C | naive：每 chunk 局部 z-score + 每 chunk 更新 | 5.9e-02 | ❌ 不等价（负对照） |

正对照比负对照小 ~5700 倍，证明**全局 z-score + 分段 einsum 累加 == 大批次**。

---

## 4. 参数配置逻辑

### 4.1 总 batch $N_L$：由目标精度决定

- batch 是 ES 学习能力的**主要杠杆**（方差 ∝ 1/N）。本任务 `N_L=60000`（全量训练集）→ 0.91+。
- 单卡 batch=12000 → 0.85；batch=60000 → 0.91。**优先把 batch 拉满**。

### 4.2 accumulate / chunk：由显存约束决定（核心公式）

前向与更新都受 LoRA 噪声张量 `B=(chunk, 784, rank)` 主导，其显存为

$$
M_B = chunk \times 784 \times rank \times 4\ \text{bytes}.
$$

实测安全水位 $M_B \le 2.4\ \text{GB}$（rank=64/chunk=12000 稳定；rank=96/chunk=12000 的 3.6GB OOM），故

$$
chunk \le \frac{2.4\times10^9}{784\times rank\times 4} \approx \frac{0.765\times10^6}{rank},
\qquad
accumulate = \frac{N_L}{chunk}\ (\text{取整且整除} N_L).
$$

扫描中实际采用的映射：

| rank | accumulate | chunk | $M_B$ |
|---:|---:|---:|---:|
| 64 | 5 | 12000 | 2.4 GB |
| 96 | 8 | 7500 | 2.2 GB |
| 128 | 10 | 6000 | 2.4 GB |
| 256 | 20 | 3000 | 2.4 GB |
| 512 | 40 | 1500 | 2.4 GB |
| 1024 | 80 | 750 | 2.4 GB |

### 4.3 rank：显存-精度权衡（实测平坦）

- 扫描 rank 64→1024，best_val 波动于 0.9109~0.9152（极差 0.0043，噪声量级内），**rank 对精度影响很弱**。
- rank=256 峰值 0.9152，rank=64 为 0.9137——两者差异在噪声范围内。
- rank 越大显存越大（需更小 chunk）、单 epoch 越慢（rank=1024 ~0.57s vs rank=64 ~0.19s），但精度增益≈0。
- **结论：batch 是主要杠杆，rank 是次要超参；rank=64 为性价比最优。**

### 4.4 固定超参（文档 §10 复现配置）

| 参数 | 值 |
|---|---|
| sigma / lr / reward | 0.2 / 0.01（固定）/ loglik |
| T / hidden | 8 / [128,128] |
| 优化器 | adamw（b1=0.9, b2=0.999） |
| v_th | 可训练（softplus 恒正，逐层独立） |
| epochs / seed | 3000 / 0 |

---

## 5. 关键代码文件

- 训练 + 证明脚本：[snn_mnist_train_accumulate.py](../llm_experiments/snn_mnist_train_accumulate.py)
- rank 扫描驱动：[exp_rank_sweep_accumulate.py](../pythonScript/exp_rank_sweep_accumulate.py)
- 数学推导：[es_batch_equivalence_math.md](es_batch_equivalence_math.md)
- 完整实验报告：[es_batch_equivalence_experiment.md](es_batch_equivalence_experiment.md)

---

## 6. 常见坑与规避

| 坑 | 后果 | 规避 |
|---|---|---|
| 每 chunk 局部 z-score | 等价性破坏（5.9e-02） | 全局 mean/std 一次计算 |
| 每 chunk 单独 optimizer 更新 | 变成"多 epoch 复用旧噪声"，不等价 | 累积后一次 solver 更新 |
| chunk 过大（高 rank） | 前向 OOM | 按 4.2 公式缩小 chunk |
| thread_id 跨 chunk 重复 | 噪声方向碰撞，探索不足 | thread_id 全局唯一（arange(N_L) 切片） |
| 同批均值中心化的 1/N 偏差 | 估计器偏置 $(1-\frac1N)\nabla F_\sigma$ | N≥60000 时偏差 1.7e-5，可忽略 |
| 忘记 ÷√K 尺度恢复 | 更新幅度偏大 √K 倍 | 累加后 `÷√K` |

---

## 7. 一句话总结

**小批次等效大批次 = 参数冻结的前向累积 + 全局 z-score + chunked einsum 更新累积 + 一次 optimizer 更新**；
其中 `chunk` 由显存公式 `chunk ≤ 0.765e6/rank` 唯一确定，`rank` 是次要超参（batch 才是主要杠杆）。
