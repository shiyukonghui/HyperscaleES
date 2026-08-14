# 大批次累积训练下的注意力等价性验证：Hopfield vs Mean-field

> 本文总结在 **小批次等效大批次累积架构**（[es_batch_accumulation_architecture.md](es_batch_accumulation_architecture.md)）
> 下，对两类 SNN 注意力模型 —— **Hopfield**（能量竞争软注意力）与 **Mean-field**（Wilson-Cowan 群体动力学）——
> 在 **batch=60000 全量、rank=64** 大批次累积训练下的**注意力等价性**（相对参考 Softmax 的逼近程度）与**性能**对比。
>
> 核心发现：**Hopfield 几乎精确复现 Softmax 注意力（w_err≈1e-8、cos_o≈1.0）且精度更高（best_val 0.580 vs 0.550）；
> Mean-field 显著偏离 Softmax（w_err 0.029、cos_o 0.90）且精度/收敛更慢** —— 注意力等价性与性能正相关。

---

## 1. 背景与目标

项目已确立两点基础：

1. **小批次等效大批次**（[es_batch_accumulation_architecture.md](es_batch_accumulation_architecture.md)）：
   参数冻结 K 段前向 + 一次全局 z-score + chunked einsum 累积（`÷√K`）+ 一次 optimizer 更新，
   严格等价单大批次（定理 2），使单卡 24GB 能复现 8×4090 大批次精度。
2. **两类 SNN 注意力模型**（`src/hyperscalees/models/snn_attention.py`）：
   - `HopfieldAttnSNN`：LIF 注意力神经元 + 全局抑制 → 能量竞争弛豫 → Softmax 权重（[注意力机制数学等价snn迁移.md](注意力机制数学等价snn迁移.md) 第 一 节）；
   - `MeanFieldAttnSNN`：Wilson-Cowan 群体动力学的值可用度 `r_j`，与指数相似度组合成注意力权重（第 二 节）。

原本两类模型只能用小批次（默认 256）训练。本工作把这两类注意力模型放到**大批次累积**路径下，回答：

- 大批次累积训练推进时，SNN 注意力对参考 Softmax 的**逼近度**（权重误差 `w_err = mean|p_snn - p_ref|`、输出余弦 `cos_o`）如何演化、收敛到多小；
- 两路在同配置大批次下的**性能**（val_acc / best_val / best_train）与收敛效率对比。

---

## 2. 方法与配置

- **任务**：patched-MNIST（28×28 切 4×4 patch，每 patch 为 token，16 tokens × 49 维）。
- **目标大批次**：`batch=60000`（全量训练集），`--accumulate 5`（chunk=12000，与累积文档 rank=64 映射一致）。
- **rank**：64；**T**：8；**sigma**：0.2；**lr**：0.03（warmup-cosine）；**n_iter**：8；**d_head**：16；**seed**：0。
- **优化器**：adamw（b1=0.9, b2=0.999）；**epochs**：3000；**val_batch**：2000（每 100 epoch 验证一次）。
- **运行环境**：WSL2 Ubuntu + RTX 4090（`/root/hyperscalees-venv`，jax 0.11.0）。
  - 环境变量：`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_FLAGS=--xla_gpu_autotune_level=1`。

**累积更新核心**（严格等价单大批次，见 [snn_attention_train_accumulate.py](../llm_experiments/snn_attention_train_accumulate.py)）：

```python
# K 段前向（参数冻结）-> 拼接 raw -> 一次全局 z-score
conv = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_full)
# chunked einsum 累积：每段 _do_update 已 ÷√chunk，K 段累加后再 ÷√K
grad_total = jax.tree.map(lambda g: g / jnp.sqrt(accumulate), grad_total)
updates, new_opt = solver.update(grad_total, opt_state, params)
```

**等价性指标**：基于同一批累积更新后的 `params`（iterinfo=None，无 LoRA 噪声的 Q/K/V），
计算 SNN 注意力权重 `p_snn` 与参考 Softmax `p_ref` 的权重误差 `w_err` 与读出余弦 `cos_o`。

---

## 3. 数学等价性验证（`--verify`）

用小规模（512 样本）断言累积架构对注意力模型也成立：

| 路径 | 做法 | vs 单大批次 max\|Δparam\| | 判定 |
|---|---|---:|---|
| A | 单大批次（基准） | — | — |
| B | 前向累积 + 全局 z-score + 一次更新 | 0.000e+00 | ✅ 精确等价 |
| D | chunked einsum 更新累积（训练实际路径） | 0.000e+00 | ✅ 精确等价 |
| C | naive 局部 z-score + 每 chunk 更新（负对照） | 1.001e-04 | ❌ 不等价 |

正对照为精确 0，负对照非零（~1e-4）⇒ 证明全局 z-score + 分段 einsum 累加 == 单大批次，累积架构对注意力模型完全成立。

---

## 4. 大批次训练结果

### 4.1 总览

| route | best_val | best_train | 终末 w_err | 终末 cos_o | 每 epoch | 墙钟(3000ep) |
|---|---:|---:|---:|---:|---:|---:|
| **hopfield** | **0.5800** | **0.5409** | **~1e-8** | **1.0000** | 0.138s | ~413s |
| **meanfield** | **0.5500** | **0.4930** | **0.0292** | **0.9019** | 0.138s | ~415s |

两路均从随机初始化（val≈0.15，10% 之上）经大批次累积训练收敛到 0.55~0.58，每 epoch 耗时相当（~0.14s），无 OOM、无噪声碰撞。

### 4.2 训练轨迹（每 100 epoch 采样）

**Hopfield**（`records/attention_accumulate/hopfield.csv`）：

| epoch | train_acc | val_acc | w_err | cos_o |
|---|---:|---:|---:|---:|
| 0 | 0.102 | 0.149 | 5.1e-09 | 1.0000 |
| 300 | 0.209 | 0.236 | 1.4e-08 | 1.0000 |
| 600 | 0.382 | 0.435 | 3.8e-08 | 1.0000 |
| 900 | 0.440 | 0.501 | 3.1e-08 | 1.0000 |
| 1500 | 0.501 | 0.556 | 2.5e-08 | 1.0000 |
| 2100 | 0.525 | 0.565 | 2.5e-08 | 1.0000 |
| 2900 | 0.537 | 0.579 | 2.3e-08 | 1.0000 |

Hopfield 的 `w_err` **全程维持在 ~1e-8**（即几乎精确等于 Softmax），`cos_o` 恒为 1.0000，同时 val_acc 稳定爬升到 0.58。

**Mean-field**（`records/attention_accumulate/meanfield.csv`）：

| epoch | train_acc | val_acc | w_err | cos_o |
|---|---:|---:|---:|---:|
| 0 | 0.102 | 0.146 | 0.0037 | 0.997 |
| 200 | 0.129 | 0.112 | 0.0079 | 0.989 |
| 400 | 0.291 | 0.333 | 0.0802 | 0.608 |
| 600 | 0.379 | 0.444 | 0.0774 | 0.634 |
| 900 | 0.422 | 0.483 | 0.0709 | 0.678 |
| 1500 | 0.454 | 0.528 | 0.0484 | 0.796 |
| 2100 | 0.477 | 0.528 | 0.0348 | 0.872 |
| 2900 | 0.493 | 0.546 | 0.0295 | 0.900 |

Mean-field 的 `w_err` 从初期的 ~0.004 **随学习放大到中期的 ~0.08**（`cos_o` 一度跌到 ~0.6），随后逐步回落到终末 0.029（`cos_o` 回升到 0.90）—— 说明其注意力权重在学到结构化模式时与 Softmax 的**结构性偏离**被放大，且始终未像 Hopfield 那样精确复现 Softmax。

---

## 5. 结论

1. **累积架构对注意力模型完全成立**：`--verify` 证明前向累积 + 全局 z-score + chunked einsum == 单大批次（精确 0），负对照非零——小批次等效大批次不仅适用于普通 SNN，也适用于两类注意力的 Q/K/V/读出全链路。

2. **大批次累积下，Hopfield 注意力几乎精确等于 Softmax**：全程 `w_err≈1e-8`、`cos_o≈1.0`，即使 val_acc 已爬到 0.58。
   这印证了 [注意力机制数学等价snn迁移.md](注意力机制数学等价snn迁移.md) 中"Hopfield 能量竞争 → Softmax 权重"的现代 Hopfield 理论：此超参（g_inh=0.5, tau_a=5, n_iter=8）下弛豫稳态确实收敛到 Softmax 不动点。

3. **Mean-field 注意力显著偏离 Softmax 且随学习放大**：w_err 峰值 ~0.08（cos_o 一度 ~0.6），终末 0.029。其 Wilson-Cowan 动力学的软阈值非线性未能完全还原指数 Softmax，结构上更"近似"而非"等价"。

4. **注意力等价性与性能正相关**：更贴近 Softmax 的 Hopfield（w_err≈1e-8）best_val 0.580，比偏离的 Mean-field（w_err 0.029）高 +3pp（best_train +4.8pp）；两路每 epoch 耗时相同。在"用 SNN 实现注意力"的语义下，**Hopfield 是更优的 Softmax 替代**。

5. **工程启示**：既然 Hopfield 几乎恒等于 Softmax 而 Mean-field 会偏离，用 Hopfield 路由可获得"Softmax 的确定性 + SNN 的生物可解释性"；若目标是显式研究 SNN 注意力的非线性偏离，Mean-field 才提供可观测的偏差信号。

---

## 6. 复现

```bash
# WSL venv，单路训练（batch=60000, rank=64, 3000ep）
XLA_PYTHON_CLIENT_PREALLOCATE=false /root/hyperscalees-venv/bin/python \
  -m llm_experiments.snn_attention_train_accumulate \
  --route hopfield --batch 60000 --rank 64 --num-epochs 3000 \
  --validate-every 100 --val-batch 2000 --lr 0.03 \
  --mnist-dir /mnt/d/Rust/snn_t1/mnist_data \
  --csv-out records/attention_accumulate/hopfield.csv

# 数学等价验证（小规模，不训练）
... -m llm_experiments.snn_attention_train_accumulate --route hopfield --verify

# 两路对比驱动（顺序跑，或 --skip-trained 仅聚合已有 CSV）
... pythonScript/exp_attention_accumulate_compare.py \
    --batch 60000 --rank 64 --num-epochs 3000 --mnist-dir /mnt/d/Rust/snn_t1/mnist_data
```

## 7. 关键文件

- 累积训练脚本：[snn_attention_train_accumulate.py](../llm_experiments/snn_attention_train_accumulate.py)
- 对比驱动：[exp_attention_accumulate_compare.py](../pythonScript/exp_attention_accumulate_compare.py)
- 模型：[snn_attention.py](../src/hyperscalees/models/snn_attention.py)
- 结果：`records/attention_accumulate/`（hopfield.csv / meanfield.csv / comparison.csv / 两路 .log）
- 数学基础：[es_batch_equivalence_math.md](es_batch_equivalence_math.md)、[es_batch_accumulation_architecture.md](es_batch_accumulation_architecture.md)
- 注意力理论：[注意力机制数学等价snn迁移.md](注意力机制数学等价snn迁移.md)
