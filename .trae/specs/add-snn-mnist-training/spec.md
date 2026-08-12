# SNN 模型 + 演化算法训练 MNIST Spec

## Why
HyperscaleES 提供了一套"无反向传播"的演化策略训练框架（Noiser + Model 抽象），
对 SNN（脉冲神经网络）天然友好——因为 SNN 的阶跃/脉冲函数不可微，演化训练正好绕开代理梯度等难题。
本代码库目前**没有任何 SNN 实现**，本 spec 旨在：
1. 新增一个符合现有 `Model` 接口的 SNN 模型（LIF 神经元、泊松编码）；
2. 复用现有 `Noiser` 演化算法，在经典 MNIST 任务上完成训练与评估，验证该算法对 SNN 的适配性。

## What Changes
- 新增 SNN 神经元模块与模型的 JAX 实现（符合 `hyperscalees.models.common.Model` 接口）。
- 新增 SNN 的 MNIST 数据加载模块（含泊松编码）。
- 新增单卡训练脚本，沿用 `tests/end_to_end_test.py` 的 `jax.jit + jax.vmap + Noiser.do_updates` 循环。
- 新增验证/测试脚本，验证 SNN 前向正确性与训练收敛到可接受的 MNIST 精度。
- **不修改** 现有 `Noiser` 接口与实现（`do_updates`/`convert_fitnesses` 等完全复用）。

## Design Decisions（已与用户确认）
- **神经元模型**：LIF（leaky integrate-and-fire），含膜电位泄漏项。
- **输入编码**：泊松编码（按像素强度概率随机发放脉冲序列）。
- **运行环境**：单卡 JAX（`jax.jit` + `jax.vmap`），CPU/GPU 均可运行，不做分布式 `shard_map` 版本。

## Impact
- Affected specs: 无既有 spec 被修改（新增 change-id）。
- Affected code:
  - `src/hyperscalees/models/snn.py`（新增）—— SNN 模型实现。
  - `src/hyperscalees/envs/snn_mnist.py`（新增）—— MNIST 数据、泊松编码与 fitness 打分。
  - `llm_experiments/` 或 `scripts/` 下训练脚本（新增）—— SNN+MNIST 训练入口。
  - `tests/` 下测试（新增）—— SNN 前向与训练冒烟测试。

## ADDED Requirements

### Requirement: SNN LIF 神经元与模型
系统 SHALL 提供符合 `hyperscalees.models.base_model.Model` 接口（`rand_init` / `_forward`）的 SNN 模型。

#### Scenario: 初始化 SNN 模型
- **WHEN** 调用 `SNNModel.rand_init(key, ...)`
- **THEN** 返回 `(frozen_params, params, scan_map, es_map)`，其中权重矩阵标记为 `MM_PARAM`、可训练标量（如阈值/常数）标记为 `PARAM`、时间常数等冻结项放入 `frozen_params`。

#### Scenario: SNN 前向（含时间展开）
- **WHEN** 调用 `SNNModel.forward(NOISER, ..., iterinfo, x)`，其中 `x` 为 `(T, batch, in_dim)` 的脉冲输入
- **THEN** 在模块内部循环 `T` 个时间步运行 LIF 神经元动力学（膜电位积分、泄漏、阈值发放、reset），
  全时间步复用同一个 `iterinfo` 对应的噪声扰动（不随时间步改变扰动），返回读出层在时间轴上聚合后的 logits。

#### Scenario: 无扰动推理 / 带噪声生成
- **WHEN** `iterinfo=None` 时调用
- **THEN** 走不带噪声的正常前向（对应 Noiser 中 `do_mm` 等 `iterinfo is None` 分支），用于验证集评估；
  当 `iterinfo=(epoch, thread_id)` 时走带扰动前向，用于训练样本生成。

### Requirement: MNIST 数据加载与泊松编码
系统 SHALL 提供 MNIST 数据集加载，并将 28×28 灰度图像编码为泊松脉冲序列。

#### Scenario: 数据加载
- **WHEN** 训练/验证脚本请求 batch 数据
- **THEN** 从标准 MNIST 源（`jax`/`tensorflow_datasets`/`keras.datasets` 中库内已具备的可用来源）加载训练/测试图像与标签，并归一化到 `[0,1]`。

#### Scenario: 泊松编码
- **WHEN** 一张 28×28 图像需要编码为输入
- **THEN** 对每个 `(pixel_value in [0,1])`，在 `T` 个时间步内以概率 `pixel_value` 独立采样伯努利发放，得到 `(T, 28*28)` 的 0/1 脉冲张量；编码使用与训练循环分离的随机 key。

### Requirement: 单卡训练循环
系统 SHALL 提供单卡 JAX 训练脚本，复用现有 `Noiser` 演化更新完成 MNIST 分类训练。

#### Scenario: 演化训练
- **WHEN** 运行训练脚本
- **THEN** 每轮（epoch）循环执行：
  - 取 MNIST batch → 泊松编码 → `jax.vmap(forward, in_axes=(..., 0, 0))` 生成带噪声候选输出；
  - 按分类正确性计算每个样本的原始 fitness（如正确=1 / 错误=0，或平滑奖励）；
  - `NOISER.convert_fitnesses` 归一化；
  - `NOISER.do_updates` 更新 `noiser_params` 与 `params`；
  - 周期性在测试集上以 `iterinfo=None` 评估，打印/记录准确率。

#### Scenario: 训练脚本参数化
- **WHEN** 启动训练脚本
- **THEN** 支持命令行/常量配置：`num_epochs`、`batch_size`（=并行环境数 `num_envs`）、`T` 时间步、`sigma`、`lr`、噪声 `rank`、神经元层规模、`seed` 等，参数默认值写于脚本顶部便于调参。

### Requirement: 测试与验证
系统 SHALL 提供测试，验证 SNN 前向正确性与训练可收敛。

#### Scenario: 前向正确性测试
- **WHEN** 运行 SNN 测试
- **THEN** 断言：模型 `rand_init` 返回结构正确；`forward` 在 `iterinfo=None` 与带扰动两种情况下均能输出形状正确的 logits `(batch, num_classes)`；无扰动前向结果可复现（同 seed 稳定）。

#### Scenario: 训练冒烟测试
- **WHEN** 运行训练冒烟测试（少量 epoch + 少量样本）
- **THEN** 训练循环完整跑通、无报错，且验证准确率相对随机猜测（10 类约 10%）有明显提升，证明演化算法确实驱动了学习。
