# Tasks — 迁移 src/hyperscalees 到 burn（TDD）

> 通用约定：每个任务遵循 TDD（先写失败测试 → 实现 → 转绿）。测试放各自 crate 的 `#[cfg(test)]`。
> 浮点断言用绝对/相对容差；随机量做统计/分布断言；结构量做形状/维度断言。
> 等价性对照：阶段 A/B 以 Python 侧固定种子固定输入导出参考值/标量供 Rust 断言。

## 基础脚手架

- [x] Task 0: 建立 burn workspace 脚手架。
  - [x] 子任务 0.1: 创建 `burn_impl/` Cargo workspace（members 含 core/noiser/models/envs/facade），依赖 `burn`（0.21，flex 后端）。
  - [x] 子任务 0.2: 核心 crate `hyperscalees-core`，提供 `type B = burn::backend::Flex;` 与冒烟 `#[test]`。
  - [x] 子任务 0.3: `cargo test` 全绿（冒烟通过）。

## 阶段 A — 核心算法（noiser + common + snn + snn_mnist）

- [x] Task 1: 迁移 `models/common` 组件。
  - [x] 子任务 1.1: `Parameter`/`MM`/`TMM`/`Embedding`/`Linear`/`MLP` + `layer_norm` + ACTIVATIONS（relu/silu/pqn）。
  - [x] 子任务 1.2: `es_tree_key`/`recursive_scan_split` 的形状与 key 分裂结构。
  - [x] 子任务 1.3: TDD 测试：无噪声 `do_mm`=`x @ Wᵀ`；Linear/MLP 激活边界；es_map 分级；`simple_es_tree_key` 结构断言。
- [x] Task 2: 迁移 `noiser` trait 与 BaseNoiser/EggRoll。
  - [x] 子任务 2.1: `Noiser` trait 与 BaseNoiser（恒等/无更新）。
  - [x] 子任务 2.2: EggRoll：LoRA 噪声（`A@Bᵀ`）、group z-score fitness、`_simple_full_update`/`_simple_lora_update`/`_noop_update` 分派、SGD/Adam/AdamW。
  - [x] 子任务 2.3: TDD 测试：无扰动 `x @ Wᵀ`；带扰动叠加；convert_fitnesses 全局 vs 分组 z-score；噪声统计；do_updates 改变参数。
- [x] Task 3: 迁移 OpenES 与 Sparse noiser。
  - [x] 子任务 3.1: OpenES：满秩非 LoRA 噪声（`param + nonlora_update`）。
  - [x] 子任务 3.2: Sparse：`get_sparse_update_params`（idxjoint//b、%b、`q=k/(a*b)`），`do_mm` 稀疏 scatter-add；`_simple_sparse_update`。
  - [x] 子任务 3.3: TDD 测试：sparse 索引/维度/形状；稀疏更新非零个数符合 `k`；OpenES 无扰动分支等价。
- [x] Task 4: 迁移 AltEggRoll 与 EggRollBS。
  - [x] 子任务 4.1: AltEggRoll：`_simple_lora_update` 用 `sign(A)`/`sign(B)`。
  - [x] 子任务 4.2: EggRollBS：group_size 基线减法 `Z=(S-b)/(std+eps)`、首两列置 0、`per_dir_fitness=mean(Z,axis=0)`；`trust_region_norm` clip_by_global_norm。
  - [x] 子任务 4.3: TDD 测试：两变体 fitness/更新 shape 与关键数值语义对齐 Python。
- [x] Task 5: 迁移 `models/snn`（LIF + SNNModel）。
  - [x] 子任务 5.1: `lif_step`/`run_lif` 动力学。
  - [x] 子任务 5.2: `SNNModel.rand_init`/`_forward`：两层 LIF + 读出（平均发放率→fc3→out_gain）。
  - [x] 子任务 5.3: TDD 测试：LIF 强电流必发放/弱电流不发放；无扰动前向确定性；带扰动形状；rand_init 结构。
- [x] Task 6: 迁移 `environments/snn_mnist`。
  - [x] 子任务 6.1: IDX 读取（magic 0x803/0x801）→ 归一化 `[0,1]`（含 gz）。
  - [x] 子任务 6.2: `poisson_encode`；`fitness_from_logits`/`accuracy_from_logits`。
  - [x] 子任务 6.3: TDD 测试：poisson 发放率≈pixel；IDX 解析；fitness/accuracy。
- [x] Task 7: 阶段 A 收敛冒烟测试。
  - [x] 子任务 7.1: 极小规模训练循环（合成数据）端到端跑通。
  - [x] 子任务 7.2: TDD 断言：`do_updates` 改变参数；准确率 ≥ 随机猜测（10 类 ≥~10%）。

## 阶段 B — SNN 注意力

- [x] Task 8: 迁移 `models/snn_attention`。
  - [x] 子任务 8.1: `_rate_encode` 与 `_mk_qkv`。
  - [x] 子任务 8.2: `hopfield_attention`。
  - [x] 子任务 8.3: `meanfield_attention` 与 `softmax_attention`。
  - [x] 子任务 8.4: `SNNAttentionModel`/`HopfieldAttnSNN`/`MeanFieldAttnSNN`（trainable/冻结 beta）。
  - [x] 子任务 8.5: TDD 测试：三种注意力输出与 Python 参考在容差内一致；`p` 和为 1；`o=p*v` 形状；模型前向形状。

## 阶段 C — RL 结构

- [x] Task 9: 迁移 `models/rl` 的结构逻辑。
  - [x] 子任务 9.1: InputProcessor / OutputProcessor / ActorCriticMLP（Space/Output 枚举 + distrax 结构等价）。
  - [x] 子任务 9.2: TDD 测试：rand_init 结构、前向张量形状与 Python 一致；（连续）mean+log_std 结构。

## 阶段 D — LLM 结构

- [x] Task 10: 迁移 `models/llm` 的组件结构与纯算法。
  - [x] 子任务 10.1: LLM base + RWKV7：LayerNorm/GroupNorm/ChannelMixing/TimeMixing + BaseRWKV inner_loop 状态递推。
  - [x] 子任务 10.2: qrwkv6：Qwen2RMSNorm/Qwen2MLP/RWKV6Attention + inner_loop。
  - [x] 子任务 10.3: 分词器纯算法（BPE 编解码表、特殊 token；用小合成词表测试）。
  - [x] 子任务 10.4: TDD 测试：归一化/单步前向、inner_loop 小 T 递推、tokenizer 往返、贪婪最长匹配。

## 阶段 E — Bandit 环境纯逻辑

- [x] Task 11: 迁移 `environments/llm_bandits` 的纯评分逻辑。
  - [x] 子任务 11.1: `strip_thoughts`/`extract_predicted_answer`/`extract_ground_truth`/`check_accuracy`/`single_fitness`/`get_padded_prompt`/`make_rg_prompt`。
  - [x] 子任务 11.2: TDD 测试：与 Python 同批样例文本的评分结果等价。

## 收尾

- [x] Task 12: 全量验证与文档化。
  - [x] 子任务 12.1: `cargo test --workspace` 全绿（136 tests）；对照 `checklist.md` 逐项核验。
  - [x] 子任务 12.2: Python 参考实现未被修改（git status 仅 burn_impl/ 与 specs/）；等价性口径（容差/统计）已记录于测试。

# Task Dependencies

- Task 1 依赖 Task 0。
- Task 2 依赖 Task 1。
- Task 3、4 依赖 Task 2（共享 Noiser trait 与更新分派）。
- Task 5 依赖 Task 1 与 Task 2（EggRoll 供扰动前向）。
- Task 6 依赖 Task 0（独立，与 1-5 并行）；Task 7 依赖 5、6。
- Task 8 依赖 Task 1、5。
- Task 9 依赖 Task 1。
- Task 10 依赖 Task 1 与 Task 5。
- Task 11 依赖 Task 0（纯函数，与阶段 A 并行）。
- Task 12 依赖全部。
