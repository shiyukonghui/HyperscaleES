# Checklist — 迁移 src/hyperscalees 到 burn

## 基础脚手架
- [x] `burn_impl/` Cargo workspace 存在，crate 依赖 burn（flex 后端），`cargo test` 冒烟通过。

## 阶段 A — 核心算法
- [x] `models/common` 组件已迁移：Parameter/MM/TMM/Embedding/Linear/MLP + layer_norm + ACTIVATIONS；TDD 测试断言 `x @ Wᵀ`、激活边界、es_map 分级、`simple_es_tree_key` 结构。
- [x] `noiser` Noiser trait 与 BaseNoiser/EggRoll 已迁移；TDD 测试断言无扰动 `x @ Wᵀ`、带扰动叠加、z-score fitness（全局/分组）、噪声统计、`do_updates` 改变参数。
- [x] OpenES 与 Sparse noiser 已迁移；TDD 测试断言稀疏索引/维度、非零个数、OpenES 无扰动分支。
- [x] AltEggRoll 与 EggRollBS 已迁移；TDD 测试断言两变体 fitness/更新语义对齐 Python（含 trust-region clip）。
- [x] `models/snn`（LIF + SNNModel）已迁移；TDD 测试断言 LIF 发放/不发放、无扰动可复现、有扰动形状、rand_init 结构。
- [x] `environments/snn_mnist` 已迁移；TDD 测试断言 IDX 解析与 Python 对照、poisson 发放率≈pixel、fitness/accuracy 一致。
- [x] 阶段 A 收敛冒烟：循环跑通、参数确实更新、准确率 ≥ 随机猜测。

## 阶段 B — SNN 注意力
- [x] `models/snn_attention` 已迁移（rate_encode/mk_qkv/hopfield/meanfield/softmax/SNNAttentionModel）；TDD 测试断言三种注意力输出与 Python 容差内一致、`p` 和为 1、模型形状正确。

## 阶段 C — RL 结构
- [x] `models/rl` 结构已迁移（Input/OutputProcessor、ActorCriticMLP，Space/Output 枚举）；TDD 测试断言 rand_init 结构与张量形状一致、（连续）mean+log_std 结构。

## 阶段 D — LLM 结构
- [x] `models/llm` 组件结构与纯算法已迁移（LLM base、RWKV7 LayerNorm/GroupNorm/ChannelMixing/TimeMixing、qrwkv6 RMSNorm/MLP/Attention、分词器纯算法）；TDD 测试断言归一化/单步前向与 Python 对照、inner_loop 递推、tokenizer 往返一致、状态形状正确。

## 阶段 E — Bandit 环境纯逻辑
- [x] `environments/llm_bandits` 纯评分逻辑已迁移（strip_thoughts/extract_* /check_accuracy/single_fitness/get_padded_prompt 等）；TDD 测试断言与 Python 同批样例评分等价。

## 收尾
- [x] `cargo test --workspace` 全绿（136 个测试全部通过）。
- [x] Python 参考实现未被修改（git status 仅 `burn_impl/` 与 `specs/`）；等价性口径（容差/统计）已记录于测试。
