# Checklist

- [x] SSH 自动化可无交互连接 `xidian@172.18.12.5` 并返回命令输出
- [x] vllm 服务已停止，`nvidia-smi` 显示 8×4090 显存全部空闲（无进程、0 MiB）
- [x] 服务器 uv venv 环境可用：`import hyperscalees` + `SNNModel` 通过，`jax.devices()` 返回 8 个 GPU
- [x] 代码已同步到服务器（排除 `.venv`/`.egg-venv`/`.pylibs`/`__pycache__`），MNIST IDX 数据在服务器可读
- [x] 多卡脚本用 `jax.pmap` 把 batch 按设备分发（shard_map/SPMD 与 LIF scan 冲突的替代方案），fitness 按设备堆叠 reshape 为全 batch，`do_updates` 复制式
- [x] 全局 thread_id 跨卡唯一，样本噪声无碰撞（pmap 按设备切片）
- [x] 泊松编码在每设备内完成（key 按 epoch + 设备 axis_index 派生）
- [x] 配置 CLI 覆盖 batch / rank / T / sigma / lr / lr_schedule / reward / v_th / group_size / epochs / seed / mnist_dir
- [x] v_th 可训练（softplus 恒正参数化）已集成且默认开启
- [x] 本地小规模冒烟通过（pmap 前向 / 更新 / 评估，1 设备 CPU）
- [x] 服务器默认放大配置（batch=60000, rank=64, loglik）成功启动，8 卡占用，无 OOM
- [x] rank 探索完成（64/96/128，batch=60000 固定）：best_val 0.9152 / 0.9149 / 0.9089，best_train 0.8883 / 0.8880 / 0.8840，GPU0 峰值 26/42.6/44.3GB（无 OOM），收益边际在 rank=64
- [x] 训练日志记录 val_acc / best_train / 耗时；与单卡基线（0.8545 / 0.8584 / 0.8332）对比结果已记录（+5.7pp val / +5.5pp best_train）
