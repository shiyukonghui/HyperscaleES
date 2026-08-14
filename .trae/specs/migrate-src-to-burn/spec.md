# 将 src/hyperscalees 迁移到 burn（Rust）Spec

## Why

`src/hyperscalees` 是一套以 JAX 实现的"无反向传播"演化策略（ES）训练框架：
`Model` 抽象 + `Noiser`（演化算法）+ SNN/SNN-Attention/RL/LLM 模型 + 环境。
当前整套算法绑定 JAX/cuDNN/GPU，无法脱离 Python 栈在原生环境运行。

本 spec 旨在把 `src/hyperscalees` **全部迁移**到 Rust 深度学习框架 **burn**，
做到：
- **功能逻辑等价**：算法结构、维度、更新公式、收敛行为与 Python/JAX 参考实现等价；
- **测试等价**：Rust 侧用 `#[test]`（容差 / 分布 / property 断言）对照 Python 参考实现；
- **测试驱动开发**：每个模块先写失败测试，再实现，最后验证。

## 已确认决策（用户指定）

- **迁移范围**：**全部 src** —— `noiser`、`models`（common / snn / snn_attention / rl / llm）、`environments`（snn_mnist / llm_bandits）。
- **等价标准**：**算法逻辑等价** —— 噪声分布、维度、更新公式、收敛行为等价；RNG 用 burn 原生实现，不逐位复刻 JAX threefry2x32；测试用容差/分布/property 断言。
- **项目形态**：在仓库根**新增 Rust workspace**（`burn_impl/`），用 cargo + `burn[ndarray]`（CPU，默认可测）+ `burn[tch]`（GPU 可选），TDD 测试内嵌在 `#[cfg(test)]`。

## What Changes

1. 新增 Rust workspace `burn_impl/`（Cargo workspace）。
2. 将 `noiser` 家族迁移为 Rust `Noiser` trait + 各实现（Base/EggRoll/OpenES/Sparse/AltEggRoll/EggRollBS）。
3. 将 `models.common` 迁移为 burn `Module` 组件（Parameter/MM/TMM/Embedding/Linear/MLP + layer_norm/ACTIVATIONS + es_tree_key）。
4. 迁移 `models.snn`（LIF + SNNModel）。
5. 迁移 `models.snn_attention`（Hopfield / MeanField / Softmax 参考 + SNNAttentionModel）。
6. 迁移 `models.rl` 的**结构逻辑**（Input/OutputProcessor/ActorCriticMLP 的张量与层组织；gymnax/distrax 依赖以 Rust 侧等价结构替代）。
7. 迁移 `models.llm`（LLM base + RWKV7 组件 + qrwkv6 组件 + 分词器**结构/纯算法**部分；依赖大权重的预训练管线不做端到端复刻）。
8. 迁移 `environments.snn_mnist`（IDX 解析、poisson_encode、fitness/accuracy）。
9. 迁移 `environments.llm_bandits` 的**纯奖励评分逻辑**（strip_thoughts / extract_* / 评分函数）；数据与外部库（reasoning_gym/math_verify/datasets）保持 Python 侧接线。
10. 提供 Python↔Rust 等价性测试基架（同一组确定性输入/种子，Python 导出参考输出，Rust 断言匹配）。

## Impact

- Affected specs：无既有 spec 被修改（新增 change-id `migrate-src-to-burn`）。
- Affected code：
  - 新增 `burn_impl/`（Workspace + 各 crate + 测试）。
  - **不修改** `src/hyperscalees/**`（Python 参考实现原样保留，作为等价性对照基准）。
  - **不删除** Python 源码；Rust 迁移为并行镜像，二者共存。
- 外部依赖：`pyproject.toml` 不变；burn 依赖在 `burn_impl/` 内新声明。

## 等价性测试策略（贯穿全部 Requirement）

- **参考基架**：新增 `burn_impl/ref_py/` 或其等价脚本，导出**确定性参考向量/标量**（fixed seed + fixed input），供 Rust 测试对照。
- **容差约定**：浮点断言使用相对/绝对容差（如 `abs_tol=1e-4, rel_tol=1e-3`），因为 burn（ndarray/tch）与 JAX 的归约顺序/数值内核不同。
- **分布断言**：噪声/脉冲等随机量断言其**统计性质**（均值/方差/维度/支持集），不断言逐位值。
- **property 断言**：维度、dtype、形状、守恒（如 softmax 和=1、fitness 归一化前后形状一致）等。

---

## 迁移分阶段（每阶段独立可交付、可验证）

- **阶段 A — 核心算法（基线）**：noiser 全家族 + models/common + models/snn + environments/snn_mnist。自包含、无外部数据依赖（MNIST 用 IDX 或合成数据），是本 spec 的核心交付物。
- **阶段 B — SNN 注意力**：models/snn_attention（Hopfield/MeanField/Softmax）。
- **阶段 C — RL 结构**：models/rl 的层与张量组织。
- **阶段 D — LLM 结构**：models/llm 的组件结构与纯算法（RWKV7 / qrwkv6 前向结构、归一化、tokenizer 纯算法）。
- **阶段 E — Bandit 环境纯逻辑**：llm_bandits 的纯评分函数。

---

## ADDED Requirements

### Requirement: Rust workspace 与 burn 脚手架

系统 SHALL 提供 `burn_impl/` Cargo workspace，所有 crate 依赖 `burn`（ndarray 后端默认，tch 可选），并能 `cargo test` 全绿。

#### Scenario: 脚手架可编译可测
- **WHEN** 在 `burn_impl/` 执行 `cargo test`
- **THEN** 各 crate 编译通过，测试套件运行；具备 burn ndarray 后端 `Backend` 实例可做张量运算（`Tensor<B, 2>` 基本算子可用）。

### Requirement: 阶段 A（核心算法）迁移 — 功能/测试等价

系统 SHALL 将 `noiser` 全家族、`models.common`、`models.snn`、`environments.snn_mnist` 迁移为 Rust，且与 Python 参考在算法逻辑上等价。

#### Scenario: Noiser trait 与实现等价
- **WHEN** 对每个 Noiser（BaseNoiser/EggRoll/OpenES/Sparse/AltEggRoll/EggRollBS）构造相同确定性配置与相同输入
- **THEN** Rust 实现具备与 Python 相同的 `init_noiser` / `do_mm` / `do_Tmm` / `get_noisy_standard` / `convert_fitnesses` / `_do_update` / `do_updates` 语义：
  - `do_mm`（iterinfo=None）做 `x @ Wᵀ`；带 iterinfo 时叠加对应噪声更新（LoRA `A@Bᵀ` / 非 LoRA 满秩 / sparse 索引累加）。
  - `convert_fitnesses`：`group_size=0` 时全局 z-score `(s-mean)/sqrt(var+1e-5)`；否则按组归一化。
  - `_do_update`：按 `es_map` 分类（FULL=0/LORA=1/NOOP=2/NOOP=3）选更新函数，最终 `-grad * sqrt(num_envs)`；`do_updates` 经优化器（SGD/Adam/AdamW）更新参数。
  - 噪声分布为均值 0、方差 `sigma²` 的高斯（property 断言：统计均/方差在容差内）；LoRA 更新为 `einsum('nir,njr->ij')` 语义。

#### Scenario: common 组件等价
- **WHEN** 对 Parameter/MM/TMM/Embedding/Linear/MLP 构造确定性配置
- **THEN** `rand_init` 生成形状/分级正确（MM_PARAM=1/PARAM=0/EMB_PARAM=2 映射）；前向与 Python 等价：
  - `MM.do_mm` 输出 `x @ Wᵀ`（含噪声路径）；`TMM`/`Embedding` 同理。
  - `Linear` = weight 投影 + bias（可选）；`MLP` = 逐层 Linear + 激活（relu/silu/pqn），末层无激活。
  - `layer_norm`、`simple_es_tree_key`/`recursive_scan_split` 的**形状与 key 分裂结构**等价（结构断言，非逐位）。

#### Scenario: SNN（LIF + SNNModel）等价
- **WHEN** 对相同 `rand_init` 配置与相同输入 spike
- **THEN** `lif_step` 的 `v += dt/tau*(-v+cur)`、`spike=(v>=v_th)`、硬重置 `v*(1-spike)` 与 Python 等价；
  `SNNModel._forward` 在 T 时间步展开两层 LIF + 读出（平均发放率 → fc3 → out_gain），无扰动（iterinfo=None）输出确定性可复现；带扰动路径形状一致。

#### Scenario: snn_mnist 环境等价
- **WHEN** 给定同一批图像（`[0,1]`）
- **THEN** `poisson_encode(images, T, key)` 产生 `(T, batch, in_dim)` 的 0/1 张量，各时间步统计发放率≈`pixel`（分布断言）；`fitness_from_logits`/`accuracy_from_logits` 与 Python 一致。

#### Scenario: 收敛冒烟（可选，阶段 A 收尾）
- **WHEN** 运行极小规模训练循环（少量 envs + 少量 epoch，合成/MNIST 数据）
- **THEN** 循环端到端跑通、`do_updates` 确实改变参数，准确率不低于随机猜测（10 类 ≥ ~10%），与 Python `tests/snn_test.py::test_training_smoke` 语义一致。

### Requirement: 阶段 B — SNN 注意力迁移

系统 SHALL 将 `models.snn_attention` 迁移为 Rust，功能/测试等价。

#### Scenario: 注意力路由等价
- **WHEN** 对相同 Q/K/V 率向量与相同超参（g_inh/tau_a/gamma/n_iter/beta）
- **THEN** `hopfield_attention` 的松弛迭代 `u+=1/tau*(-u+h-g_inh*mean(u))`、Boltzmann 读出 `softmax(u)`、`o=p*v` 与 Python 等价；
  `meanfield_attention` 的 Wilson-Cowan 迭代与 `A_j∝exp(βqᵀk_j)r_j` 读出等价；
  `softmax_attention` 参考输出等价（`o=p*v`、`p=softmax(β q̄ᵀk)`）。
- **WHEN** 构造 HopfieldAttnSNN / MeanFieldAttnSNN 模型
- **THEN** `_ratio_encode` 的 `sigmoid(gain*mean/(1+|mean|))`、训参与冻结参数的边界（可训练 beta=softplus，非训练 beta=1/sqrt(d)）与 Python 等价。

### Requirement: 阶段 C — RL 结构迁移

系统 SHALL 将 `models.rl` 的**层/张量组织**迁移为 Rust（InputProcessor/OutputProcessor/ActorCriticMLP）。

#### Scenario: RL 结构等价
- **WHEN** 构造离散/连续观测-动作空间的 ActorCriticMLP
- **THEN** `rand_init` 结构（obs_embed→MLP→act_head 与可选 critic_head）与 Python 等价；前向张量形状一致；
  离散动作头输出 logits、连续头输出 mean+log_std 的**结构**一致（gymnax space 解析与 distrax 采样属于 Python 生态，Rust 侧以等价的 Rust 枚举/结构替代，不做分布库 1:1 复刻）。

### Requirement: 阶段 D — LLM 结构迁移

系统 SHALL 将 `models.llm` 的**组件结构与纯算法**迁移为 Rust（LLM base、RWKV7、qrwkv6、分词器纯算法）。

#### Scenario: RWKV7 / qrwkv6 前向结构等价
- **WHEN** 构造 RWKV7 与 qrwkv6 的 LayerNorm/GroupNorm/ChannelMixing/TimeMixing/Qwen2 等组件，给定确定性输入
- **THEN** 归一化（layer_norm/group_norm 带 weight/bias/eps）、前向张量形状与 Python 等价；
  时间步循环/状态递推（RWKV state, wkv6 matrix attention）在**结构层**（形状、状态规模、递推式）等价（数值上以单步/小 T 对照 Python）。
- **WHEN** 给定 token 序列与词表
- **THEN** 分词器（tokenizer）的**纯算法**（BPE 解码/编码表、特殊 token 处理、`_decode`/`_encode` 语义，不依赖 20B 大词表文件时可合成小词表测试）与 Python 等价。
- 说明：预训练权重加载、大词表 tokenizer 文件、完整 LLM 训练/推理管线**不在**本 spec 的端到端等价范围内（依赖大型外部资源），仅保证**组件结构**等价。

### Requirement: 阶段 E — Bandit 环境纯逻辑迁移

系统 SHALL 将 `environments.llm_bandits` 的**纯评分逻辑**迁移为 Rust。

#### Scenario: 纯评分函数等价
- **WHEN** 输入相同的生成文本
- **THEN** `strip_thoughts`/`extract_predicted_answer`/`extract_ground_truth`/`check_accuracy`/`single_fitness` 等纯字符串/正则逻辑与 Python 等价。
- 说明：数据集加载（`datasets`/`reasoning_gym`/`math_verify`）与 tokenizer 接线保留 Python 侧，Rust 只镜像可单测的纯函数。

---

## MODIFIED Requirements

无（不改动 Python 侧任何文件）。

## REMOVED Requirements

无（src/hyperscalees 保留原样）。

## 验收 / 完成定义

- `burn_impl/` 中 `cargo test` 全绿，覆盖阶段 A–E 的等价性断言。
- 每个模块遵循 TDD：先写失败测试 → 实现 → 测试转绿。
- Python 参考实现未被修改；等价性对照报告（可选记录 `records/` 或测试内断言）说明容差与统计口径。
