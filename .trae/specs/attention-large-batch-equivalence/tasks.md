# Tasks

- [x] Task 1: 实现累积训练脚本 `llm_experiments/snn_attention_train_accumulate.py`
  - [x] SubTask 1.1: 复用 `snn_attention.py` 的 `HopfieldAttnSNN`/`MeanFieldAttnSNN`/`model_rand_init` 与 `snn_mnist.py` 的 `get_mnist_arrays`/`poisson_encode`；patched-MNIST 处理（`patch_images`）
  - [x] SubTask 1.2: 集成累积更新核心（仿 `snn_mnist_train_accumulate.py` 的 `_accumulated_update`）：K 段 chunked einsum `_do_update` 累加梯度、`÷√K`、一次 solver 更新；前向 K 段冻结参数拼接
  - [x] SubTask 1.3: 周期验证（iterinfo=None）+ 等价性指标：基于累积更新的同一 `params` 计算 `w_err`（权重误差）与 `cos_o`（输出余弦）
  - [x] SubTask 1.4: 配置化 CLI：--route / --batch / --accumulate / --rank / --T / --sigma / --lr / --n-iter / --patch-px / --d-head / --num-epochs / --seed / --validate-every / --csv-out；CSV 列 `epoch,train_acc,val_acc,best_val,best_train,w_err,cos_o,epoch_time,cum_time`
  - [x] SubTask 1.5: 环境变量 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，默认 batch=60000、rank=64，WSL `/mnt/d` MNIST 回退

- [x] Task 2: 本地小规模冒烟验证脚本正确性
  - [x] SubTask 2.1: 本机 CPU 跑小 batch / 小 accumulate：前向、全局 z-score、累积更新、等价性指标 `w_err`/`cos_o` 全链路通过，形状/数值合理
  - [x] SubTask 2.2: 确认累积更新数学等价存在（累积 vs 单大批次逐参数差≈0，naive 局部 z-score 不等价）—— 可复用 `--verify` 思路做小规模断言

- [x] Task 3: WSL2 GPU 大批次累积训练（hopfield 与 meanfield）
  - [x] SubTask 3.1: 在 WSL2 `/root/hyperscalees-venv` 下跑 `--route hopfield`：batch=60000, rank=64, 足够 epochs，输出 CSV
  - [x] SubTask 3.2: 跑 `--route meanfield`（同配置同种子）：输出 CSV
  - [x] SubTask 3.3: 观察训练中 `w_err`/`cos_o` 演化与 val_acc，确认大批次下注意力等价性逐步建立

- [x] Task 4: 两路对比驱动与结果汇总
  - [x] SubTask 4.1: 实现 `pythonScript/exp_attention_accumulate_compare.py`：顺序跑两路（同配置），汇总 `records/attention_accumulate/comparison.csv`
  - [x] SubTask 4.2: 结果落 `records/attention_accumulate/`（hopfield.csv / meanfield.csv / comparison.csv）

# Task Dependencies

- Task 2 依赖 Task 1（脚本就绪）
- Task 3 依赖 Task 2（脚本正确性验证通过）与 WSL2 环境
- Task 4 依赖 Task 3（两路训练完成）
- Task 1 与 Task 4 可与既有 spec 无关独立开发；Task 3 依赖环境可用（GPU 空闲）
