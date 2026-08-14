# 逐 token 脉冲自注意力（selfattn_heads）训练失败根因分析

> 本文记录在 **小批次等效大批次累积架构**（[es_batch_accumulation_architecture.md](es_batch_accumulation_architecture.md)）下，
> 使用 `--model selfattn_heads`（多头 + 位置编码 + 多块残差的逐 token 脉冲自注意力，见
> `src/hyperscalees/models/snn_self_attention_heads.py`）训练 MNIST 分类，
> **3000 epochs 正确率始终停留在 ~10%（随机猜测水平），完全无学习**的实验事实与根因分析。
>
> 结论：**失败不是 OOM 或调参问题，而是模型架构（硬阈值 LIF + 行内归一化）与 HyperscaleES
> 演化策略（ES）训练范式不匹配**——硬阈值使「奖励对参数扰动的增益」近似为零，ES 梯度无从累积，
> 参数停在随机初始化附近。

---

## 1. 实验事实

运行命令（WSL / GPU，`--accumulate 10`，chunk=6000，OOM 已解决）：

```bash
python -m llm_experiments.snn_attention_train_accumulate \
  --model selfattn_heads --batch 60000 --accumulate 10 --rank 64 \
  --num-epochs 3000 --validate-every 1000 --val-batch 2000 \
  --csv-out results_sa_heads_a10_3000.csv
```

结果（`results_sa_heads_a10_3000.csv`）：

| 指标 | 初始 | 中段 | 最终 |
|------|------|------|------|
| train_acc | 0.0992 | 0.1026 | 0.0987 |
| best_train | 0.0992 | 0.1037 | 0.1041 |
| val_acc | 0.0865 | 0.1210 | 0.0860 |
| best_val | 0.0865 | 0.1210 | 0.1210 |

- 全 3000 epochs 中 train_acc 在 **0.098 ~ 0.104** 区间内原地波动，无任何上升趋势。
- MNIST 是 10 类分类，~0.10 恰为**随机猜测水平** → 模型完全没有学到任何判别信息。
- 日志无崩溃、无 OOM 中断（那条 `RESOURCE_EXHAUSTED: out of memory` 是 JAX allocator 的一次性重试警告，
  与 `--model attention` 验证一致，未中断训练）。

### 关于 w_err=0 / cos_o=1 的说明

CSV 中 `w_err` 恒为 0、`cos_o` 恒为 1 是 **`selfattn_heads` 模型的占位返回值**，并非训练成功信号。
见 `llm_experiments/snn_attention_train_accumulate.py` 的 `compute_equivalence()`：

```python
if model == "selfattn_heads":
    return (0.0, 1.0)   # 加深多头版注意力在块内，无法用单头 w_err 有意义表示，占位
```

因此这两个指标对这个模型**不可用**，判断训练成败只能用 train/val acc。

---

## 2. 训练机制回顾：HyperscaleES 演化策略（ES）

本仓库**不是反向传播**，而是**黑盒演化策略**（`src/hyperscalees/noiser/eggroll.py`）：

1. **前向**：对每个样本给可训练参数加 LoRA 噪声（`do_mm`：`base_ans + x @ B @ A.T`，
   `A, B` 由 `get_lora_update_params` 依 `(epoch, thread_id)` 随机生成）；
2. **奖励**：loglik / binary 奖励 → `convert_fitnesses` 做全局 z-score；
3. **更新**：`_simple_lora_update` 计算
   `grad = einsum('nir,njr->ij', scores * A, B) / N`，再交给 adamw 更新参数。

关键：`grad = mean(scores * perturbation)`。**这个算法要求「奖励 scores 随参数扰动平滑、连续地变化」**，
才能累积出非零的相关性梯度。若奖励对扰动不敏感（扰动前后输出几乎不变），则 `scores` 与扰动近似独立，
`mean(scores * perturbation) ≈ 0`，参数几乎不更新。

这是理解本失败的核心前提。

---

## 3. 根因：硬阈值 LIF + 行内归一化破坏 ES 梯度信号

`selfattn_heads` 模型在两处使用了**硬阈值（binary spike）+ 归一化**，恰好打断 ES 所需的相关性信号。

### 3.1 硬阈值 LIF 前端编码

`snn_self_attention_heads.py` 的 `_rate()`（`_forward` 中 `enc()` 调用）：

```python
def _rate(proj, vth, tau_m):
    spikes = run_lif({"tau_m": tau_m, "v_th": vth}, proj, v0)  # spike = (v >= v_th) 硬阈值
    return jnp.mean(spikes, axis=0)                            # 离散 spike-count rate
```

- 输出是 **0/1 发放计数**，是离散、阶梯化的；
- 参数微扰在绝大多数情况下**不改变发放模式**（膜电压差一点点不会跨阈值），
  因此奖励对扰动几乎不变 → 相关性 ≈ 0。

### 3.2 注意力核心：行内 max 归一化 + 硬阈值竞争

`snn_self_attention.py` 的 `row_softmax_lif()`（`selfattn_heads` 复用）：

```python
h_i = h_i / (jnp.max(jnp.abs(h_i)) + 1e-6)   # 行内 max 归一化：压平每行差异
u_new, s = lif_step({"tau_m": tau_m, "v_th": vth}, u, h_i - g_inh * G)  # 硬阈值发放
p = count_final / (jnp.sum(count_final) + 1e-6)  # 发放计数归一化
```

- **行内 max 归一化**把每个 token 的相似度压到相同幅度，进一步抹掉扰动带来的相对差异；
- **硬阈值 LIF 竞争**再离散化一次，输出对输入微扰完全不敏感。

### 3.3 proj_gain 参数「只存不读」

`frozen_params["proj_gain"]`（签名参数、存储均有）在 `snn_self_attention.py` 与
`snn_self_attention_heads.py` 的**前向/编码函数中从未被读取**（Grep 只在 `rand_init` 签名与
`frozen_params` 存储处命中）。它本应像连续速率版那样控制 Q/K/V 编码的动态范围，但没有接上。

---

## 4. 对照：能正常学习的连续速率版（snn_attention）

`snn_attention.py` 是已验证能学到 ~58% 的基线，其设计**刻意避开硬阈值**：

- 前端编码 `_rate_encode`：**连续 sigmoid**（`sigmoid(gain * mean_p / (1 + |mean_p|))`），无硬阈值；
- 注意力 `hopfield_attention` / `meanfield_attention`：**连续 Hopfield 松弛 + Boltzmann softmax**，无硬阈值。

其模块注释明确写道：

> "avoids the fragile binary-spike threshold **so the ES-trained model does not collapse**"

即**故意不用二进制硬阈值，以免 ES 训练失效**。而 `selfattn_heads` 恰恰用回了硬阈值 LIF，
回到了 ES 无法驾驭的路径。

文件 `docs/注意力机制数学等价snn迁移.md` 也已记录：**逐时刻严格等价 ANN→SNN 迁移不可行**，
本文再次印证：对 ES 训练而言，硬阈值脉冲层是「梯度」毒药。

---

## 5. 影响范围与修复方向

### 影响范围

- `src/hyperscalees/models/snn_self_attention.py`（单头逐 token 自注意力，`--model selfattn`）
  与 `snn_self_attention_heads.py`（多头，`--model selfattn_heads`）**均受影响**——
  两者共享同一 `row_softmax_lif` 硬阈值核心与 `_rate` 硬阈值编码，`proj_gain` 同样「只存不读」。
- 连续速率版 `snn_attention.py` / `snn_attention_lif.py` 不受影响（无硬阈值注意力核心）。

### 修复方向（供后续决策，本文不实施）

1. **改用连续/分级注意力**：注意力核心弃用硬阈值 LIF 竞争，改回连续 Boltzmann / Hopfield
   松弛（gated），使奖励对扰动平滑；
2. **前端编码去硬阈值**：`_rate` 改为连续 sigmoid 编码（复用 `snn_attention._rate_encode`），
   并真正接入 `proj_gain` 控制动态范围；
3. 若要保留脉冲性，可只在**输出层**做少量稀疏激励，避免在注意力核心注入硬阈值。

---

## 6. 复现与验证

复现命令见 §1。可用以下对照快速验证根因：

```bash
# 连续速率版（应能正常学习，best_val 高出一个数量级）
python -m llm_experiments.snn_attention_train_accumulate \
  --model attention --route hopfield --batch 60000 --accumulate 10 \
  --rank 64 --num-epochs 3000 --validate-every 1000 --val-batch 2000
```

---

*文档创建时间：2026-08-14。数据来源：`results_sa_heads_a10_3000.csv`、WSL 训练日志 `sa_heads_a10_3000.log`。*
