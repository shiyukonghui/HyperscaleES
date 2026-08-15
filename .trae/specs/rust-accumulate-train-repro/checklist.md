# Checklist — Rust 复刻「小批次等效大批次」累积训练（TrainableVthSNN）

## 阶段 1
- [ ] `snn_mnist::fitness_from_logits` 支持 `loglik`（log_softmax 收集），`binary` 分支回归不变；TDD 测试通过。

## 阶段 2
- [ ] `snn::TrainableVthSnn` 变体存在，`params()` 顺序 `[fc1,fc2,fc3,out_gain,v_th1,v_th2]`、`es_map()` 对齐 Python；每层 LIF 用 `softplus(v_th)`，`tau_m` 冻结；clean 前向确定性；支持 `NoiseFn`。
- [ ] TDD 测试：结构/es_map、softplus 恒正、clean 可复现、带扰动形状均通过。

## 阶段 3
- [ ] `eggroll::accumulated_update` 存在，逐段 `_do_update` 累加 + `÷√K` + 一次 `solver.update`。
- [ ] TDD 测试：「K 段累积」==「全批一次性 `do_updates`」逐元素 ≈0（容差 1e-5）；solver step 只 +1。

## 阶段 4
- [ ] `accumulate_train` 二进制存在，CLI 对齐 Python 脚本；MNIST 加载、每段独立编码、全局唯一 thread_ids、K 段前向 + 全局 z-score + chunked 累积更新、评估/`best_val`/CSV 均实现。
- [ ] `--verify` 四路径存在，断言 `Δ(A,B)≈0`、`Δ(A,D)≈0`、`Δ(A,C)>0`。

## 阶段 5
- [ ] `cargo test --workspace` 全绿（含新增 TDD）。
- [ ] `accumulate_train --verify` 通过（累积==单大批次，naive 不等价）。
- [ ] 真实 MNIST 完整训练复现 best_val ≈0.91 量级（≥0.90），CSV/日志已记录。
- [ ] Python `src/hyperscalees/**` 未被修改（git status 仅 `burn_impl/` 与 `specs/`）。
