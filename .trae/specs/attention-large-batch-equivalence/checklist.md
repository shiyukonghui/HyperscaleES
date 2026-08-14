# Checklist

- [x] 累积训练脚本 `snn_attention_train_accumulate.py` 存在，支持 `--route {hopfield,meanfield}`，复用 `model_rand_init`/`poisson_encode`/`get_mnist_arrays`/`EggRoll`
- [x] 累积更新为"全局 z-score + chunked einsum 累加 + ÷√K + 一次 solver 更新"（严格等价单大批次，未按 chunk 局部归一化）
- [x] 前向 K 段参数冻结、thread_id 全局唯一、每 chunk 独立编码，拼接 raw fitness 为 (batch,)
- [x] 周期验证用 iterinfo=None 测 val_acc，且等价性指标（w_err 权重误差 / cos_o 输出余弦）基于同一累积后的 params 计算
- [x] CLI 覆盖 --route/--batch/--accumulate/--rank/--T/--sigma/--lr/--n-iter/--patch-px/--d-head/--num-epochs/--seed/--csv-out；CSV 列 `epoch,train_acc,val_acc,best_val,best_train,w_err,cos_o,epoch_time,cum_time`
- [x] 本机小规模冒烟通过：前向/累积更新/等价性指标全链路正确，数学等价小规模断言成立（--verify：累积 vs 大批次 0.000，chunked-einsum 0.000，naive 局部 z-score 1e-4 不等价）
- [x] WSL2 GPU 上 hopfield 与 meanfield 两路（batch=60000, rank=64, 同种子同配置）均跑通，无 OOM
- [x] 训练中记录 w_err/cos_o/val_acc 演化，能看出大批次下注意力等价性逐步建立（hopfield w_err≈1e-8 全程，meanfield w_err 0.03~0.05/col_o 0.8~0.9）
- [x] 对比驱动 `exp_attention_accumulate_compare.py` 汇总两路 best_val / best_train / 终末 w_err / cos_o / 每 epoch 耗时到 `records/attention_accumulate/comparison.csv`
- [x] 报告 Hopfield 与 Mean-field 在大批量下的注意力等价性（谁更逼近 Softmax）与性能（谁精度更高、收敛效率）
