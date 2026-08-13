# SNN + 演化算法 8×4090 多卡放大训练 Spec

## Why

实验记录（`docs/snn_es_mnist_experiment.md`）已确立该库的核心结论：

- **批次放大效应**（7.7/7.10）：演化策略学习能力主要靠扩大并行批次放大（梯度误差 ∝ σ/√N）；
  单卡 4090 上 batch=30000 达 ~0.76，进入 0.75~0.8 平台。
- **rank 边际点**（7.11）：rank=32 增益最大，rank=64 收益递减，**rank=128 在 batch=12000 时单卡 24GB 直接 OOM**——显存是当前瓶颈。
- **奖励与配置**（7.5 重测 / 7.12）：大批次 + 小固定 LR + LoRA rank + v_th 可训练下，**log-likelihood 奖励最优（0.8545）**，v_th 可训练 +2.8pp。
- 单卡最优记录：val 0.8545（7.5 重测）/ 训练峰值 0.8584（7.5.1）/ best_train 0.8332。

服务器 172.18.12.5 拥有 **8×RTX 4090（48GB/卡，合计 384GB）**，当前被 vllm 模型服务占用。
显存充裕：**批次可放大到 60000（全量训练集）**，并可**探索更大 rank（64/96/128，4×单卡边际点 32）**。
按 hyperscale 哲学放大批次与 rank，有望突破单卡 0.85 的平台。

## What Changes

1. **远程服务器部署（172.18.12.5）**
   - 本机**直接连接**（不经过 WSL 中转）：使用 PuTTY `plink`（`-pw` 密码参数）或 OpenSSH + `SSH_ASKPASS` 建立**无交互密码 SSH** 自动化通道。
   - **停止 vllm 模型服务**（先探测 systemd / docker / 裸进程，按实际方式停止），释放全部 8 卡显存。
   - 用 **uv 创建 venv** 并安装依赖（jax[cuda 变体] + optax + numpy + 运行时依赖 + `-e . --no-deps`），验证 `jax.devices()` 返回 8 个 GPU。
   - 同步代码（**排除本地大目录** `.venv`/`.egg-venv`/`.pylibs`/`__pycache__`，代码本体仅 ~4.5MB）与 MNIST IDX 数据（`D:\Rust\snn_t1\mnist_data`，4 个 gz 共 ~11.6MB）。

2. **新增多卡训练脚本 `llm_experiments/snn_mnist_train_multi_gpu.py`**（单进程 8 卡，参照 `general_do_evolution_multi_gpu.py` 的 shard_map 模式）：
   - `mesh = jax.make_mesh((8,), ('data',))`；前向 `shard_map` 将 batch 分片（`P('data')`）。
   - fitness 分片计算 → `process_allgather` 汇总全 batch → **复制式** `do_updates`（`P()` 规格，与 LLM 多卡脚本一致，不修改 EggRoll 核心）。
   - **全局唯一 thread_id**（`jnp.arange(total_batch)` 分片），保证跨卡噪声扰动不碰撞（EggRoll 按 `(epoch, thread_id)` 派生噪声）。
   - 泊松编码在 **shard 内完成**，避免整批跨卡复制。
   - 配置化 CLI：`batch / rank / T / sigma / lr / lr_schedule / reward(loglik|binary) / v_th_trainable / group_size / epochs / seed / mnist_dir`。
   - v_th 可训练模型变体（softplus 恒正参数化，参照 `pythonScript/exp_vth_trainable.py`），默认开启。
   - 周期验证（`iterinfo=None`）+ 结果 CSV 记录。

3. **放大训练配置**（基于实验结论）：
   - **默认**：`batch=60000`（全量训练集，per-GPU 7500，5×单卡 12000）、`rank=64` 起步、`T=8`、`reward=loglik`、`lr=0.01` 固定、`sigma=0.2`、`v_th` 可训练、`group_size=0`、≥3000 epochs。
   - **rank 探索**：在 batch=60000 固定下扫描 `rank ∈ {64, 96, 128}`（单卡边际点 32 的 2~4 倍），记录各档 val_acc/best_train 与耗时，确定 48GB 显存下的收益边际。
   - 更新内存预算：复制式 `do_updates` 的 fc1 中间张量 ≈ `(N, 784, rank)`；N=60000/rank=128 时更新峰值 ~40GB/卡（48GB 可容纳，配合 `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `MEM_FRACTION=0.9`）；OOM 时下调 rank。

4. **运行观测与结果记录**：`nohup` 后台运行、`nvidia-smi` 验证 8 卡占用、日志记录 val_acc/best_train/每 epoch 耗时；结果与单卡基线（0.8545 / 0.8584 / 0.8332）对比，CSV 回传本地 `records/`。

## Impact

- Affected specs：`add-snn-mnist-training`（已完成，本 spec 是其多卡放大延伸）。
- Affected code：
  - 新增 `llm_experiments/snn_mnist_train_multi_gpu.py`（核心交付物）。
  - **不改动** `src/hyperscalees/models/snn.py` / `environments/snn_mnist.py` / `noiser/eggroll.py`（v_th 可训练类内嵌于新脚本，参照 `exp_vth_trainable.py` 既有做法）。
  - 复用：`simple_es_tree_key`、`SNNModel`/v_th 变体、`poisson_encode`/`fitness_from_logits`/`accuracy_from_logits`、`EggRoll`。
- 服务器影响：vllm 服务被停止（用户已授权），8 卡转用于本训练；venv 与代码部署于服务器用户目录。
- 风险：复制式更新的显存上限（batch×rank 组合受限）；服务器驱动/JAX CUDA 版本需匹配（按 `nvidia-smi` 选择 `jax[cuda12]` 或 `jax[cuda13]`）。

## ADDED Requirements

### Requirement: SSH 自动化连接与服务器状态确认
系统 SHALL 通过本机 WSL + `sshpass` 以密码无交互连接 `xidian@172.18.12.5` 并执行远端命令，采集 GPU/驱动/环境信息。

#### Scenario: 建立连接并确认状态
- **WHEN** 执行 SSH 自动化命令
- **THEN** 返回 `nvidia-smi`、Python/uv 可用性、磁盘空间与是否已有仓库/venv

### Requirement: 停止 vllm 服务并释放 8 卡
系统 SHALL 探测 vllm 运行方式（systemd / docker / 裸进程）并停止之，验证 8×4090 显存全部空闲。

#### Scenario: vllm 停止成功
- **WHEN** 探测到 vllm 并执行停止（必要时 sudo）
- **THEN** `nvidia-smi` 显示 8 卡无进程占用、显存 0 MiB

### Requirement: uv venv 环境搭建
系统 SHALL 在服务器用 uv 创建独立 venv，安装 SNN 训练所需依赖并验证导入与 GPU 可见性。

#### Scenario: 环境可用
- **WHEN** venv 创建并安装依赖后执行验证命令
- **THEN** `import hyperscalees; from hyperscalees.models.snn import SNNModel` 通过，`jax.devices()` 返回 8 个 GPU

### Requirement: 多卡训练脚本
系统 SHALL 提供 `snn_mnist_train_multi_gpu.py`：shard_map 分片前向、allgather 汇总 fitness、复制式更新、shard 内泊松编码、全局唯一 thread_id、周期验证与 CSV 记录。

#### Scenario: 脚本可运行
- **WHEN** 以放大配置（batch=60000, rank=64, loglik）启动脚本
- **THEN** 8 卡参与计算，每 epoch 打印 train_acc / val_acc，无噪声碰撞、无 OOM

### Requirement: 放大配置与基线对比
系统 SHALL 以放大 batch（60000）与 rank（64/96/128 探索）运行训练，并将结果与单卡基线对比，判断是否突破 0.85。

#### Scenario: 训练完成
- **WHEN** 训练达到设定 epochs 或合理停点
- **THEN** 记录各 rank 档最终 val_acc / 峰值 val_acc / best_train / 耗时，与 0.8545 / 0.8584 / 0.8332 对比并写入 CSV

## MODIFIED Requirements

无（不改动既有单卡脚本与 noiser 核心）。

## REMOVED Requirements

无。
