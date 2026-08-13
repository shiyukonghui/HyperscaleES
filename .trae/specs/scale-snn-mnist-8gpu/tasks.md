# Tasks

- [x] Task 1: 建立 SSH 自动化通道（本机直连，不经 WSL）
  - [x] SubTask 1.1: 准备本机 SSH 工具：paramiko（pip 安装，Python 3.13）直连方案
  - [x] SubTask 1.2: 验证可无交互执行命令（`pythonScript/ssh_remote.py` 密码认证，host 172.18.12.5）
  - [x] SubTask 1.3: 采集服务器信息：vip/Ubuntu 24.04、8×RTX 4090 48GB、驱动 580/CUDA 13、python 3.12、无 uv、无仓库、vllm 为 systemd 服务
- [x] Task 2: 停止 vllm 服务并释放 8 卡
  - [x] SubTask 2.1: 探测到 vllm 运行方式 = systemd 服务 `vllm-server.service`（TP=8）
  - [x] SubTask 2.2: `sudo -S systemctl stop vllm-server.service` 停止成功
  - [x] SubTask 2.3: 验证 8×4090 全部空闲（仅 4 MiB 占用，48505 MiB 空闲）
- [x] Task 3: 服务器环境搭建（uv venv + 依赖 + 代码 + 数据）
  - [x] SubTask 3.1: 安装 uv（官方脚本）
  - [x] SubTask 3.2: `uv venv .venv --python 3.12`
  - [x] SubTask 3.3: 安装 jax[cuda13] + optax + numpy + 运行时依赖 + `-e . --no-deps`
  - [x] SubTask 3.4: 验证通过：`jax.devices()` 返回 8 个 CudaDevice，imports ok
  - [x] SubTask 3.5: 代码同步（90 文件 4.3MB，排除本地 venv 大目录）
  - [x] SubTask 3.6: MNIST IDX 数据同步（4 个 gz，11.6MB → ~/mnist_data）
- [x] Task 4: 实现多卡训练脚本 `llm_experiments/snn_mnist_train_multi_gpu.py`（本地开发）
  - [x] SubTask 4.1: **改用 `jax.pmap`** 把 batch 按设备分发（shard_map/自动 SPMD 与 LIF scan 在 JAX 0.11 冲突，见 6.1 类问题）
  - [x] SubTask 4.2: 每设备内完成泊松编码 + 前向 + fitness；`jax.lax.pmean` 聚合准确率；fitness 按设备堆叠输出 reshape 为全 batch
  - [x] SubTask 4.3: 全局唯一 thread_id（`jnp.arange(batch)` 按设备切片，跨卡噪声不碰撞）
  - [x] SubTask 4.4: 配置化 CLI：batch / rank / T / sigma / lr / lr_schedule / reward / group_size / epochs / seed / hidden / mnist_dir / val / csv-out
  - [x] SubTask 4.5: v_th 可训练模型变体（softplus 恒正参数化，逐层独立）
  - [x] SubTask 4.6: 周期验证（iterinfo=None）+ 结果 CSV（epoch/train_acc/val_acc/best_val/best_train/时间）
- [x] Task 5: 本地小规模验证脚本正确性
  - [x] SubTask 5.1: 本地 CPU（1 设备）冒烟：前向/pmap/eval/更新/v_th 报告全链路通过
  - [x] SubTask 5.2: 噪声唯一性、fitness 形状正确（修复 pmap 输出多一维问题）
- [x] Task 6: 服务器执行放大训练
  - [x] SubTask 6.1: **默认放大配置（batch=60000, rank=64, loglik, lr=0.01, v_th 可训练, 3000 ep）完成：best_val 0.9152 / best_train 0.8883，333s**
  - [x] SubTask 6.2: `nvidia-smi` 验证 8 卡占用（GPU0 复制式更新峰值 ~26GB，其余 4.7GB 前向分片）
  - [x] SubTask 6.3: 无 OOM、无噪声碰撞；单 epoch ~0.11s
  - [x] SubTask 6.4: **rank 探索完成（batch=60000 固定，3000ep）**：rank 64 → 0.9152 / rank 96 → 0.9149 / rank 128 → 0.9089（GPU0 峰值 26/42.6/44.3GB 均未 OOM）。**收益边际在 rank=64**，更高 rank 持平或略降
- [x] Task 7: 结果分析与记录
  - [x] SubTask 7.1: 对比单卡基线（val 0.8545 / 峰值 0.8584 / best_train 0.8332）：**best_val +5.7pp（0.9152）、best_train +5.5pp（0.8883），突破平台**
  - [x] SubTask 7.2: CSV 已回传 `records/`（3 档）；`docs/snn_es_mnist_experiment.md` 已追加「10. 8×4090 多卡放大训练」小节

# Task Dependencies

- Task 2 依赖 Task 1（需先建立 SSH 通道与状态确认）
- Task 3 依赖 Task 1（venv/依赖安装与 GPU 空闲无强依赖，可与 Task 2 并行；代码/数据同步独立）
- Task 4 独立于服务器任务，可与 Task 1-3 并行（本地开发）
- Task 5 依赖 Task 4
- Task 6 依赖 Task 2（显存空闲）、Task 3（环境）、Task 4 + Task 5（脚本就绪）
- Task 7 依赖 Task 6
