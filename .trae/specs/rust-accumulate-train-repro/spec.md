# Rust 复刻「小批次等效大批次」累积训练（TrainableVthSNN）Spec

## Why

`docs/es_batch_accumulation_architecture.md` 描述的「参数冻结 K 段前向 + 一次全局 z-score + chunked einsum 更新累积（÷√K）+ 一次 optimizer 更新」训练架构，已在 Python/JAX 的 `llm_experiments/snn_mnist_train_accumulate.py` 上复现 best_val ≈ 0.9149。
上一 change-id `migrate-src-to-burn` 已把 `noiser` / `models.snn` / `environments.snn_mnist` / `EggRoll` 迁移为 Rust（`burn_impl/`），但**没有**：可训练的 softplus `v_th` SNN 变体、chunked einsum 累积更新 API、`loglik` 奖励、或一个可运行的累积训练二进制。

本 spec 的目标是：**用已迁移的 Rust 算法，编写一个 Rust 累积训练脚本（二进制），在真实 MNIST 上完整复刻该训练架构，跑出与 Python 同量级的 best_val（≈0.91，≥0.90），从而证明迁移在功能上完整。**

**已确认决策（用户指定）**
- 目标模型：**普通 SNN —— `TrainableVthSNN`**（每隐层可训练 softplus `v_th` + `out_gain`），即文档正文引用脚本用的是它。
- 验收标准：**复现 ~0.91 量级 best_val**（真实 MNIST，因 Rust 泊松编码用 burn 非可播种 RNG、噪声确定性流与 JAX threefry 不同，不追求逐位，允许后端/噪声差异下 ≥0.90）。
- 运行形态：`burn_impl/hyperscalees/` 内新增可运行二进制（`cargo run --release -p hyperscalees --bin accumulate_train`），在 `hyperscalees` facade crate（唯一可同时依赖 models+noiser+envs 的 crate）里接线，与现有 `snn_mnist_train.rs` 冒烟驱动一致地注入 EggRoll 噪声闭包。

## What Changes

1. **`hyperscalees-envs`：`fitness_from_logits` 增加 `loglik` 奖励**（Python 默认 `loglik`，现 Rust 仅 binary）。`loglik = log_softmax(logits)[batch, label]`。
2. **`hyperscalees-models`：新增 `TrainableVthSnn` 变体**（`snn.rs`）——每隐层可训练 `v_th`（`softplus` 恒正，`es_map=PARAM`）+ `out_gain`；`params()`/`es_map()`/`forward()` 对齐 Python `TrainableVthSNN`。（不修改现有 `SnnModel`。）
3. **`hyperscalees-noiser`：新增 chunked einsum 累积更新 API**（`eggroll.rs`）——镜像 Python `_accumulated_update`：把全长 `conv`/`thread_ids` 切成 K 段，逐段调用 `_do_update`（每段含 `÷N` 与 `×√N`，返回 `-grad_k*√chunk`），累加梯度，`÷√K` 恢复 `√batch` 尺度，最后**一次** `solver.update`。
4. **`hyperscalees`：新增可运行二进制 `src/bin/accumulate_train.rs`**，CLI 对齐 Python 脚本（`--batch --accumulate --rank --T --sigma --lr --reward --group-size --num-epochs --seed --hidden --mnist-dir --validate-every --val-batch --log-every --csv-out --verify`）。
5. 训练循环复刻文档架构：K 段前向（参数冻结、`thread_id` 全局唯一 `arange(batch)` 切片、每段独立编码）→ 拼接 raw fitness → **一次全局 z-score** → **chunked einsum 累积更新** → 单 epoch；周期在测试集评估，记录 `best_val`/CSV。
6. **`--verify` 模式**：镜像 Python 四路径断言——A 单大批次 / B 前向累积+全局 z-score+一次更新 / D chunked einsum 累积 / C naive 每段局部 z-score+每段更新（负对照）；断言 `Δ(A,B)≈0`、`Δ(A,D)≈0`、`Δ(A,C)>0`，证明累积 == 单大批次。
7. TDD：每个新增 API 先写失败测试（`#[cfg(test)]`）再实现；测试覆盖 softplus v_th、loglik 奖励、chunked 累积 == 全批更新、verify 判据。

## Impact

- Affected specs：新增 `rust-accumulate-train-repro`；不改 `migrate-src-to-burn`（已完成的迁移交付物）。
- Affected code：
  - `burn_impl/hyperscalees-envs/src/snn_mnist.rs`（新增 loglik reward）
  - `burn_impl/hyperscalees-models/src/snn.rs`（新增 `TrainableVthSnn`）
  - `burn_impl/hyperscalees-noiser/src/eggroll.rs`（新增累积更新 API，复用既有私有 `_do_update`/`do_update_with`）
  - `burn_impl/hyperscalees/src/bin/accumulate_train.rs`（新增二进制）
  - **不修改** `src/hyperscalees/**`（Python 参考）
- 外部依赖：`burn_impl` 内不新增第三方 crate；CLI 参数解析用 `std::env`（不引入 clap）以保持依赖最小。

## ADDED Requirements

### Requirement: loglik 奖励
系统 SHALL 在 `hyperscalees-envs::snn_mnist` 提供与 Python 等价的 `loglik` 逐样本奖励。

#### Scenario: loglik 奖励正确
- **WHEN** 给定确定性 `(logits, labels)`
- **THEN** 返回 `fitness = log_softmax(logits)[n, label_n]`（形状 `(batch,)`）；`reward="binary"` 保持现有硬 0/1 语义不变。

### Requirement: TrainableVthSnn 模型变体
系统 SHALL 提供可训练 softplus `v_th` 的 SNN 分类模型，训练参量与 es_map 对齐 Python `TrainableVthSNN`。

#### Scenario: 训练参量与结构正确
- **WHEN** 用 `(in_dim=784, hidden=[128,128], num_classes=10, tau_m=20.0, v_th=0.3)` 构造
- **THEN** `params()` 顺序为 `[fc1, fc2, fc3, out_gain, v_th1, v_th2]`（out_gain/v_th 以 `(1,1)` 进入 rank-2 管线）；`es_map()` 为 `[MM_PARAM, MM_PARAM, MM_PARAM, PARAM, PARAM, PARAM]`；每层 LIF 用 `softplus(v_th_i)` 作阈值；冻结 `tau_m`。

#### Scenario: 前向确定性
- **WHEN** 相同输入（`T,batch,784` spikes）两次无扰动 `forward`
- **THEN** 输出逐位一致且形状 `(batch, 10)`（clean 路径可复现）。

### Requirement: chunked einsum 累积更新
系统 SHALL 提供与 Python `_accumulated_update` 等价的 Rust API，使「K 段累积 == 单大批次更新」。

#### Scenario: 累积 == 全批更新
- **WHEN** 对相同 `(params, base_keys, es_map, 全长 conv, thread_ids, epoch)` 分别执行「全批一次性 `do_updates`」与「K 段累积更新」
- **THEN** 两者产出的参数逐元素最大绝对差 ≈0（容差 `abs_tol≈1e-5`），证明 einsum 分段累加等价；且只有**一次** `solver.update`（Adam/AdamW step 只 +1）。

### Requirement: accumulate_train 二进制 + 训练循环
系统 SHALL 提供一个可运行二进制，在真实 MNIST 上按文档架构训练并输出 `best_val`/CSV。

#### Scenario: 复现 ~0.91
- **WHEN** 运行 `cargo run --release -p hyperscalees --bin accumulate_train -- --batch 60000 --accumulate 5 --rank 64 --num-epochs 3000 --mnist-dir <dir> --csv-out <f>`
- **THEN** 训练循环端到端跑通；训练使用「K 段前向 + 全局 z-score + chunked einsum 累积」；周期评估测试集；最终 `best_val` 达到 ~0.91 量级（≥0.90，受 Rust RNG/后端差异允许波动）。

#### Scenario: verify 判据
- **WHEN** 运行 `accumulate_train --verify ...`
- **THEN** 输出并断言：`Δ(A,B)≈0`、`Δ(A,D)≈0`、`Δ(A,C)>0`（负对照），证明累积路径与单大批次等价。

## MODIFIED Requirements

无（不修改既有迁移交付物的对外行为；仅在 `snn_mnist.rs` 为 `fitness_from_logits` 扩展奖励类型、`snn.rs` 新增变体）。

## REMOVED Requirements

无。

## 验收 / 完成定义

- `burn_impl/` 内 `cargo test` 全绿（新增 TDD 测试）。
- `accumulate_train --verify` 通过（累积 == 单大批次，naive 不等价）。
- `accumulate_train`（完整/足够 epoch）在真实 MNIST 复现 best_val ≈0.91 量级（≥0.90）。
- Python `src/hyperscalees/**` 未修改；`git status` 仅涉及 `burn_impl/` 与 `specs/`。
