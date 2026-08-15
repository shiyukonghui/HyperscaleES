# Tasks — Rust 复刻「小批次等效大批次」累积训练（TrainableVthSNN）

> 通用约定：TDD——先写失败测试（`#[cfg(test)]`）再实现，测试转绿。
> 浮点断言用相对/绝对容差；随机量做统计/形状断言（与既有 migrate-src-to-burn 口径一致）。
> 本 change 在 `burn_impl/` 内增量完成，复用既有 crate 结构与迁移组件。

## 阶段 0 — GPU 训练实装（burn-cuda / CubeCL）【用户追加，已完成】

- [x] Task 0: `hyperscalees-core` 增加 `gpu` feature（`burn/cuda`，CubeCL CUDA 后端）+ `default_device()`/`is_gpu()`；flex 保持默认。
  - [x] 子任务 0.1: 后端类型按 feature 门控：`gpu` → `burn::backend::Cuda`，否则 `Flex`。
  - [x] 子任务 0.2: CUDA 冒烟测试 `smoke_gpu_matmul`（`CudaDevice::new(0)`）通过；`backend=cuda` 已在训练日志确认。
  - [x] 子任务 0.3: `hyperscalees` facade 增加 `gpu` feature 转发；`cargo build --features gpu` 通过（本机 CUDA Toolkit 12.8 + RTX 4090 + 驱动 610.74）。
- [x] Task 0b: GPU 批量向量化性能优化（逐样本前向/更新在 GPU 上不可行，60k batch 下 185s/epoch）。
  - [x] 子任务 0b.1: `models::snn` 新增 `BatchedNoiseFn` + `TrainableVthSnn::forward_batched`（整块 `(T,n,784)` 前向，等价 `jax.vmap`）。
  - [x] 子任务 0b.2: `noiser::eggroll` 新增 `batched_lora_noise`（并行 CPU 生成 + 一次上传）、`accumulated_update` 批量 einsum（2D gemm 等价式 `A_flat^T@B_flat`）、dense(FULL) 更新批量化。
  - [x] 子任务 0b.3: binary `segment_logits_batched`/`evaluate` 改批量路径。
  - [x] 子任务 0b.4: 性能 185s/epoch → ~28s/epoch（cache 切片上传成为新瓶颈：每 epoch 40GB H2D）。
- [x] Task 0c: 深度优化——GPU 内联噪声 + 仿射 z-score（消除全部 CPU→GPU 上传）【用户追加，已完成】。
  - [x] 子任务 0c.1: 启用 `burn/fusion` + `burn/autotune`（元素级链条融合单内核）。
  - [x] 子任务 0c.2: 定位瓶颈=slice_upload（每 epoch ~40GB 上传，占 24s/28s）。
  - [x] 子任务 0c.3: `noiser::eggroll` 新增内联辅助：`lora_einsum_raw/ones`、`dense_einsum_raw/ones`、`combine_affine_grads`（全局 z-score 仿射等价：`Σeinsum((raw-mean)/std) = (Σeinsum(raw) - mean·Σeinsum(1))/std`）。
  - [x] 子任务 0c.4: binary 训练热路径改为 GPU `Tensor::random` 生成噪声（反对称配对方差缩减），前向与梯度共享同一份 (A,B)；逐 chunk raw 加权累积 + 最后一次仿射修正 + 一次 solver；**零 CPU 随机数、零上传**。
  - [x] 子任务 0c.5: 等价性单测 `inline_affine_matches_accumulated_two_phase`（内联 ≡ 两阶段，容差 1e-4）通过。
  - [x] 子任务 0c.6: 性能 **~28s/epoch → ~1.3s/epoch**（3000 epoch ≈ 1.2h），GPU 利用率显著提升。

## 阶段 1 — 前置数据/奖励扩展

- [x] Task 1: `hyperscalees-envs::snn_mnist` 扩展 `fitness_from_logits` 支持 `loglik` 奖励。
  - [x] 子任务 1.1: 实现 `loglik = log_softmax(logits)[n, label_n]`（保持 `binary` 分支不变）。
  - [x] 子任务 1.2: TDD 测试——已知 logits/labels 的 loglik 值逐元素匹配手算/属性（形状 `(batch,)`、值有限）。
  - [x] 子任务 1.3: 验证 `binary` 语义回归不变。

## 阶段 2 — 可训练 v_th SNN 变体

- [x] Task 2: `hyperscalees-models::snn` 新增 `TrainableVthSnn` 变体（对齐 Python `TrainableVthSNN`）。
  - [x] 子任务 2.1: 模型结构 `[fc1(h1,784), fc2(h2,h1), fc3(10,h2), out_gain(1,)@(1,1), v_th1, v_th2]`；每层 LIF 阈值 `softplus(v_th_i)`，冻结 `tau_m`。
  - [x] 子任务 2.2: `params()` 顺序与 `es_map()`（fc1/2/3=MM_PARAM，out_gain/v_th1/v_th2=PARAM）对齐。
  - [x] 子任务 2.3: `forward(T,batch,784)->(batch,10)` 支持 `NoiseFn` 闭包（同 `SnnModel` 机制）。
  - [x] 子任务 2.4: TDD 测试——结构/es_map 正确；clean 前向确定性；softplus(v_th) 恒正；带扰动形状。

## 阶段 3 — chunked einsum 累积更新 API

- [x] Task 3: `hyperscalees-noiser::eggroll` 新增 K 段累积更新 API（镜像 Python `_accumulated_update`）。
  - [x] 子任务 3.1: `accumulated_update(...)` ——切 K 段 conv/thread_ids；逐段调用既有 `_do_update`（含 `×√N`），累加梯度；`÷√K`；`solver.update` 一次。
  - [x] 子任务 3.2: 只更新一次 optimizer step（Adam/AdamW 全局 step +1）。
  - [x] 子任务 3.3: TDD 测试——「K 段累积」产出参数与「全批一次性 `do_updates`」逐元素 ≈0（容差 1e-5）；solver step 计数只 +1。

## 阶段 4 — 可运行二进制 + 训练循环

- [x] Task 4: `hyperscalees/src/bin/accumulate_train.rs` 可运行二进制。
  - [x] 子任务 4.1: CLI 参数解析（`std::env`，不引 clap）：`--batch --accumulate --rank --T --sigma --lr --reward --group-size --num-epochs --seed --hidden --mnist-dir --validate-every --val-batch --log-every --csv-out --verify`。
  - [x] 子任务 4.2: MNIST 加载（`load_mnist_from_dir`）+ 索引采样 + 每段独立编码 + 全局唯一 `thread_ids`。
  - [x] 子任务 4.3: K 段前向（参数冻结，每段 per-thread `NoiseFn` 闭包）→ 拼接 raw → 全局 z-score → chunked einsum 累积更新。
  - [x] 子任务 4.4: 评估循环（`evaluate`）+ `best_val` 追踪 + CSV 输出（表头/追加）。
  - [x] 子任务 4.5: `--verify` 四路径（A 单大批次 / B 前向累积 / D chunked 累积 / C naive 负对照），断言 `Δ(A,B)≈0`、`Δ(A,D)≈0`、`Δ(A,C)>0`。

## 阶段 5 — 验证与复现

- [ ] Task 5: 全量验证并复现 ~0.91。
  - [ ] 子任务 5.1: `cargo test --workspace` 全绿。
  - [ ] 子任务 5.2: `accumulate_train --verify` 通过（累积==单大批次，naive 不等价）。
  - [ ] 子任务 5.3: 真实 MNIST 完整训练跑出 best_val ≈0.91 量级（≥0.90）并记录到 CSV；日志/结果与 Python 参考同架构对照。

# Task Dependencies

- Task 1 依赖 migrate-src-to-burn（envs 已迁移）。
- Task 2 依赖 Task 1（训练需统一奖励）与既有 `snn.rs`。
- Task 3 依赖既有 `eggroll.rs`（复用 `_do_update`/`do_update_with`/`Solver`），与 Task 2 可并行。
- Task 4 依赖 Task 2、Task 3（用到了变体与累积 API）。
- Task 5 依赖 Task 4。
