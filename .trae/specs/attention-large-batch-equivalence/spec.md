# 大批次累积训练下的注意力等价性验证（Hopfield vs Mean-field）Spec

## Why

项目已用 [小批次等效大批次累积架构](file:///f:/PythonProject/HyperscaleES/docs/es_batch_accumulation_architecture.md)
在单卡 24GB 上用梯度累积复现了 8×4090 大批次训练的精度（best_val 0.9149 vs 0.9152），
等价性已由 `--verify` 逐参数证明（全局 z-score + chunked einsum 对样本线性 → 分段累加 == 整批）。

同时代码库已实现两类 **SNN 注意力模型**（[snn_attention.py](file:///f:/PythonProject/HyperscaleES/src/hyperscalees/models/snn_attention.py)）：
`HopfieldAttnSNN`（能量竞争软注意力）与 `MeanFieldAttnSNN`（Wilson-Cowan 群体动力学），
但现有训练脚本 [snn_attention_train.py](file:///f:/PythonProject/HyperscaleES/llm_experiments/snn_attention_train.py)
仅支持小批次（默认 batch=256）训练。

本 spec 旨在：**复用已验证的累积架构，把注意力模型放到大批次（batch=60000）下训练**，
验证两个关注点：
1. **注意力等价性**：随着大批次累积训练推进，SNN 注意力权重对参考 Softmax 的逼近程度
   （权重误差 `||p_snn - p_ref||`、输出余弦 `cos_o`）如何演化、收敛到多小。
2. **Hopfield vs Mean-field 性能**：两种注意力路由在同配置大批次下的 val_acc / best_val / 训练效率对比。

## What Changes

1. 新增训练脚本 [snn_attention_train_accumulate.py](file:///f:/PythonProject/HyperscaleES/llm_experiments/snn_attention_train_accumulate.py)：
   - 融合 [snn_mnist_train_accumulate.py](file:///f:/PythonProject/HyperscaleES/llm_experiments/snn_mnist_train_accumulate.py)
     的**累积架构**（参数冻结 K 段前向 + 拼接 → 一次全局 z-score → chunked einsum 累积 + `÷√K` → 一次 optimizer 更新）
     与 [snn_attention_train.py](file:///f:/PythonProject/HyperscaleES/llm_experiments/snn_attention_train.py)
     的 **patched-MNIST 注意力训练 + 等价性指标**（`w_err`/`cos_o`）。
   - 批量在 WSL2 GPU（RTX 4090）上运行，目标 **batch=60000（全量训练集）、rank=64（用户指定）**。
   - 通过 `--route {hopfield, meanfield}` 复用同一个累积训练循环，支持两路对比。
   - 周期验证（iterinfo=None）时同时记录 `train_acc / val_acc / best_val / best_train / w_err / cos_o` 到 CSV。

2. 提供**累积模式下的等价性指标**：
   - 复用 `compute_equivalence` 逻辑（参数冻结、mean-rate token 输入、iterinfo=None 无噪声 Q/K/V）
     在训练过程中周期计算 SNN 注意力与参考 Softmax 的 `w_err` 与 `cos_o`。
   - **关键**：等价性指标必须在累积（大批次）路径下计算才算验证了"大批次下的注意力等价性"，
     即用累积更新后的同一 `params` 评估。

3. 新增驱动脚本 [exp_attention_accumulate_compare.py](file:///f:/PythonProject/HyperscaleES/pythonScript/exp_attention_accumulate_compare.py)：
   - 顺序运行 hopfield / meanfield 两路累积训练（batch=60000, rank=64, 同种子同配置），
     从各自 CSV 汇总 **best_val / best_train / 终末 w_err / cos_o / 每 epoch 耗时**。
   - 输出对比 CSV（`records/attention_accumulate/`）与对比总结。

4. **不改动**任何现有文件：复用 `SNNAttentionModel`/`HopfieldAttnSNN`/`MeanFieldAttnSNN`、
   `poisson_encode`/`get_mnist_arrays`、`EggRoll`、`NOISER._do_update`、`_accumulated_update` 模式。

## Impact

- Affected specs：本 spec 建立在已完成的两个 spec 之上 —— `add-snn-mnist-training`
  （SNN+演化训练框架）与 `scale-snn-mnist-8gpu`（多卡放大）、以及累积架构文档
  `es_batch_accumulation_architecture.md`（工程底座）。
- Affected code：
  - 新增 `llm_experiments/snn_attention_train_accumulate.py`（核心交付物）。
  - 新增 `pythonScript/exp_attention_accumulate_compare.py`（两路对比驱动）。
  - **不改动** `src/hyperscalees/models/snn_attention.py` / `environments/snn_mnist.py` /
    `noiser/eggroll.py` / 既有训练脚本。
- Affected records：新增 `records/attention_accumulate/` 结果目录（hopfield.csv / meanfield.csv / comparison.csv）。
- 内存影响：注意力模型最大 LoRA 矩阵为 `q/k/v`（49×16），`B=(chunk, 49, rank)`，
  远小于 SNN 的 `(chunk, 784, rank)`，故 batch=60000/rank=64 下 chunk 可放宽，显存无压力。
  但仍沿用 `chunk ≤ 0.765e6/rank` 的保守公式与 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 环境。

## ADDED Requirements

### Requirement: 累积训练脚本 `snn_attention_train_accumulate.py`

系统 SHALL 提供把 SNN 注意力模型（Hopfield / Mean-field）在**大批次累积架构**下训练的脚本。

#### Scenario: 累积前向（参数冻结 + 拼接）
- **WHEN** 对某 epoch 运行，目标 batch=60000
- **THEN** 将 batch 切成 K 段 chunk 顺序前向（`params` 全程冻结、thread_id 全局唯一、每 chunk 独立编码），
  拼接 `raw` fitness 为 `(batch,)`。

#### Scenario: 一次全局 z-score + chunked einsum 累积更新
- **WHEN** 拼接完成
- **THEN** 对全部 batch fitness 做**一次全局** `convert_fitnesses`（不按 chunk 局部归一化），
  再以 chunk 分段的 `_do_update` 经 `jax.lax.scan` 累加梯度、`÷√K` 恢复尺度、一次 solver 更新
  —— 严格等于单大批次（等价性由累积架构保证）。

#### Scenario: 周期验证 + 等价性指标
- **WHEN** 达到 `validate_every` epoch
- **THEN** 用 `iterinfo=None` 在验证集上测 `val_acc`，并基于**同一累积后的 `params`**
  计算参考 Softmax 等价性指标 `w_err = mean|p_snn - p_ref|` 与 `cos_o`（输出余弦），写入 CSV。

#### Scenario: 参数化与记录
- **WHEN** 启动脚本
- **THEN** 支持 `--route {hopfield,meanfield}`、`--batch`、`--accumulate`、`--rank`、`--T`、
  `--sigma`、`--lr`、`--n-iter`、`--patch-px`、`--d-head`、`--num-epochs`、`--seed`、`--csv-out`；
  CSV 列含 `epoch,train_acc,val_acc,best_val,best_train,w_err,cos_o,epoch_time,cum_time`。

### Requirement: 两路对比驱动 `exp_attention_accumulate_compare.py`

系统 SHALL 提供脚本顺序运行 hopfield / meanfield 两路大批次累积训练并汇总对比。

#### Scenario: 运行并汇总
- **WHEN** 运行驱动脚本
- **THEN** 依次跑 `--route hopfield` 与 `--route meanfield`（同 batch=60000/rank=64/种子/配置），
  从各自 CSV 汇总 best_val / best_train / 终末 w_err / cos_o / 每 epoch 耗时，
  写入 `records/attention_accumulate/comparison.csv`，并在终端打印两路对比表。

### Requirement: 大批次下的注意力等价性验证

系统 SHALL 在**大批次（batch=60000）累积训练**路径下量化 Hopfield 与 Mean-field
对参考 Softmax 注意力的逼近程度，并对比两路大批次性能。

#### Scenario: 等价性与性能对比成立
- **WHEN** 两路训练完成
- **THEN** 报告两路的终末 `w_err` / `cos_o`（注意力等价性）与 `best_val` / `best_train`（性能），
  判定大批次累积训练下 Hopfield 与 Mean-field 谁更逼近 Softmax、谁精度更高、各自收敛效率。

## MODIFIED Requirements

无（不改动既有脚本与 noiser 核心，纯新增）。

## REMOVED Requirements

无。
