# Checklist

- [x] SNN LIF 神经元/模型模块实现了 `Model.rand_init` / `_forward` 接口，且 `iterinfo=None` 与带扰动两种情况均正确
- [x] SNN 前向在模块内部完成 `T` 时间步展开，全步复用同一 `iterinfo` 噪声扰动
- [x] 参数类别（MM_PARAM / PARAM / frozen）按 spec 正确划分
- [x] MNIST 数据加载并归一化到 `[0,1]`（本地 IDX + HuggingFace 回退）
- [x] 泊松编码正确将 28×28 图像转成 `(T, batch, 28*28)` 脉冲序列
- [x] fitness 打分按分类正确性生成每个样本原始奖励
- [x] 单卡训练脚本复用 `Noiser`（EggRoll）的 `init_noiser` / `simple_es_tree_key` / `convert_fitnesses` / `do_updates`
- [x] 训练循环可按参数配置 num_epochs / num_envs / T / sigma / lr / rank / 层规模 / seed
- [x] 训练脚本周期性以 `iterinfo=None` 在测试集评估并打印准确率
- [x] SNN 前向测试通过（`rand_init` 结构、LIF 动力学、输出形状、无扰动结果可复现）
- [x] SNN+MNIST 训练冒烟测试通过（10 类 MNIST 循环跑通、参数被演化正确更新、准确率不塌缩；收敛需 GPU 大规模并行，见 spec）
