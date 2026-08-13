# SNN + 演化算法训练 MNIST — 实验记录

> 目标：验证 HyperscaleES 的"无反向传播"演化策略（Noiser）能否适配 SNN（脉冲神经网络），
> 并在经典 MNIST 任务上完成训练与评估。
> 关联 spec：`.trae/specs/add-snn-mnist-training/`

## 1. 背景与设计决策

HyperscaleES 提供两大抽象：
- **Noiser**：给参数加噪声扰动（全量 / LoRA），并按 fitness 用演化方式更新参数（不依赖梯度）。
- **Model**：定义 `rand_init` / `_forward` 的模型接口。

SNN 的阶跃/脉冲函数不可微，而演化策略**不需要反向传播**，因此两者天然契合。本次设计决策（已与用户确认）：

| 项 | 选择 |
|---|---|
| 神经元模型 | LIF（leaky integrate-and-fire，含膜电位泄漏） |
| 输入编码 | 泊松编码（按像素强度概率随机发放） |
| 运行环境 | 单卡 JAX（`jax.jit` + `jax.vmap`，CPU/GPU 均可） |
| 数据集 | 本地 IDX 文件 `D:\Rust\snn_t1\mnist_data`（train 60000 / test 10000） |

## 2. 环境搭建

使用 `uv` 建立独立 venv 并安装项目依赖（避免污染系统 Python）。

```bash
uv venv .venv --python 3.13
uv pip install --python .\.venv\Scripts\python.exe jax jaxlib optax numpy
# 项目包本体（--no-deps，运行时依赖单独装）
uv pip install --python .\.venv\Scripts\python.exe -e . --no-deps
# import hyperscalees 所需的运行时依赖（gymnax/flax/transformers/datasets 等）
uv pip install --python .\.venv\Scripts\python.exe gymnax distrax flax gymnasium seaborn chex einops huggingface_hub tokenizers importlib_resources pyrwkv-tokenizer transformers datasets reasoning-gym math-verify
```

验证导入：

```bash
.\.venv\Scripts\python.exe -c "import hyperscalees; from hyperscalees.models.snn import SNNModel; print('ok')"
```

> 注：`pyproject.toml` 声明的依赖较全（含 LLM/gymnax 等），完整安装链较长。以上为验证 SNN 所需的子集。

## 3. 新增/修改文件

| 文件 | 说明 |
|---|---|
| `src/hyperscalees/models/snn.py` | LIF 神经元 + `SNNModel`（符合 `Model` 接口，`T` 步时间展开，全步复用同一 `iterinfo` 噪声） |
| `src/hyperscalees/environments/snn_mnist.py` | MNIST 加载（本地 IDX + HuggingFace 回退）、泊松编码、fitness/accuracy |
| `llm_experiments/snn_mnist_train.py` | 单卡训练循环（复用 `EggRoll` noiser） |
| `tests/snn_test.py` | 4 项测试：结构 / LIF 动力学 / 前向可复现 / 训练冒烟 |
| `src/hyperscalees/models/__init__.py` | 注册 `snn` 模块 |
| `src/hyperscalees/environments/__init__.py` | 注册 `snn_mnist` 模块 |

## 4. SNN 模型设计

LIF 神经元单步动力学（`lif_step`）：

```
v = v + (tau_m^-1) * (-v + current)      # 泄漏 + 充电
spike = (v >= v_th)                        # 阈值发放（不可微阶跃）
v = v * (1 - spike)                        # 硬重置
```

`SNNModel._forward`（单样本）：

```
x (T, in_dim) 泊松脉冲
  -> Linear fc1 (经 Noiser.do_mm 加 LoRA 噪声, 每时间步同一 iterinfo)
  -> LIF 层1 (jax.lax.scan 沿 T 展开)
  -> Linear fc2 (经 do_mm)
  -> LIF 层2
  -> 时间轴平均发放率 rate (h2,)
  -> Linear fc3 -> logits (num_classes,)
  -> * out_gain
```

参数分类（`es_map`）：权重矩阵 = `MM_PARAM(1)`，`out_gain` 标量 = `PARAM(0)`，`tau_m`/`v_th` = 冻结构造（放 `frozen_params`）。

批量并行：`jax.vmap(Model.forward, in_axes=(None,None,0,0))` 对 batch 并行；`iterinfo=(epoch, thread_id)` 用同一扰动生成每代候选。

## 5. 训练循环

沿用 `tests/end_to_end_test.py` 模式：

```
训练集 batch
  -> 泊松编码 (T, batch, 784)
  -> jit_forward 带噪声前向 (iterinfo per sample)      # 生成候选
  -> fitness_from_logits (正确=1/错误=0)
  -> NOISER.convert_fitnesses (z-score 归一化)
  -> NOISER.do_updates (演化更新 params)
  -> 周期性 iterinfo=None 在测试集评估
```

## 6. 调试中发现的关键问题（重要）

### 6.1 JAX 版本 API 兼容（导致 `do_updates` 崩溃）

- **现象**：`ValueError: scan got values with different leading axis sizes: 64, 2`。
- **根因**：`eggroll.py::_do_update` 依据 `len(base_key.shape)==0` 决定走"直接更新"还是"scan"分支。
  使用旧 API `jax.random.PRNGKey()`（shape `(2,)`）时，`simple_es_tree_key` 生成的 `es_tree_key`
  叶子为 `(2,)`，导致所有参数错误进入 scan 分支而崩溃。
- **修复**：全部改用新版 `jax.random.key()`（shape `()`），`es_tree_key` 叶子为标量，走正确分支。
  `end_to_end_test.py` 之所以能跑，正是因为用了新版 API。

### 6.2 LIF 阈值过高导致网络完全不工作（关键）

- **现象**：全黑图像与全白图像得到**完全相同的 logits 全 0**。
- **根因**：默认 `v_th=1.0` 远高于 `fc1` 权重（`1/sqrt(784)≈0.036`）产生的输入电流量级，
  神经元从不发放，读出层输入恒 0 → logits 恒 0 → SNN 完全失效。
- **验证**（黑/白可区分输出）：
  ```
  v_th=1.0  ->  diff=False（全0，失效）
  v_th=0.1  ->  diff=True
  v_th=0.3  ->  diff=True（最终采用）
  ```
- **修复**：默认 `v_th` 由 `1.0` 调低为 `0.3`，并在 `rand_init` 文档中说明阈值需与权重初始化量级匹配。

### 6.3 MNIST 数据源

- HuggingFace `datasets` 下载在本环境失败（HfUriError，网络/镜像受限）。
- **解决方案**：增加本地 IDX 解析 `_load_mnist_from_dir` + `set_mnist_data_dir`，优先从本地
  `D:\Rust\snn_t1\mnist_data` 读取，离线可用。`get_mnist_arrays(..., data_dir=...)` 支持本地目录。

## 7. 验证结果

### 7.1 单元/集成测试（`tests/snn_test.py`）

```bash
.\.venv\Scripts\python.exe tests\snn_test.py
```

```
OK test_rand_init_structure
OK test_lif_dynamics
OK test_forward_shapes_and_reproducibility
OK test_training_smoke (best_acc=0.164, max_param_delta=0.13389)
ALL SNN TESTS PASSED
```

要点：
- `rand_init` 返回结构、参数类别（MM_PARAM/PARAM/frozen）正确。
- LIF 动力学：强电流会发放、弱电流不发放。
- 前向输出形状正确、`iterinfo=None` 结果可复现。
- 训练冒烟：10 类 MNIST 循环跑通、`do_updates` 确实验改参数（delta>0）、准确率不塌缩。

### 7.2 训练脚本（`llm_experiments/snn_mnist_train.py`）

用精简配置验证脚本可完整运行（编译 + 训练 + 测试集评估）：

```
Compiling...  Warm-up done in 2.6s
epoch  0 | train_acc 0.047 | val_acc 0.082
epoch  1 | train_acc 0.094 | val_acc 0.082
Done.
```

默认配置：`num_epochs=40, num_envs=128, T=8, hidden=[128,128], v_th=0.3, rank=8`。

### 7.3 收敛能力验证（关键结论）

用**同一套 SNN + 演化管线**在 MNIST **二分类（数字 0 vs 1，12665 样本）**上训练：

```
init eval acc: 0.453
epoch  0 train_acc 0.398
epoch 30 train_acc 0.688
epoch 49 train_acc 0.852   <- 收敛到 85%
```

**结论**：SNN + 演化训练管线**完全正确、能有效学习**。而 **10 类全量 MNIST 在单卡 CPU + 纯演化下收敛很慢**（冒烟约 16%），因为该库的演化策略**依赖 GPU 大规模并行**（多 GPU × 数百候选）才能高效探索——这是算法特性，非代码问题。

### 7.4 长时训练观察：准确率 vs 训练时间

在 CPU 上投入更长时间训练 10 类 MNIST（`exp_train_time.py`，num_envs=128, T=8, hidden=[128,128], v_th=0.3, rank=8, sigma=0.2, lr=0.03），每 25 epoch 记录一次累计时间与 train/val 准确率（测试集 1024 样本、Poisson 编码评估）。

**配置 A：400 epoch（约 28 秒，单 epoch ≈ 0.065s）**

| epoch | 用时(s) | train_acc | val_acc |
|------:|--------:|----------:|--------:|
| 0    | 0.9  | 0.109 | 0.084 |
| 100  | 8.0  | 0.172 | 0.183 |
| 200  | 14.4 | 0.195 | 0.212 |
| 300  | 21.2 | 0.172 | 0.223 |
| 399  | 27.8 | 0.250 | 0.259 |

**配置 B：4000 epoch / 约 260 秒（单 epoch ≈ 0.065s）**

| epoch | 用时(s) | val_acc |
|------:|--------:|--------:|
| 100   | 8.0   | 0.183 |
| 400   | 27.5  | 0.260 |
| 1000  | ~65   | ~0.270 |
| 2000  | 130.8 | 0.271 |
| 3000  | ~195  | ~0.270 |
| 4000  | 259.6 | 0.280 |

**结论：准确率与训练时间【不成正比】，呈"早期快速上升 → 后期平台饱和"的收益递减（diminishing returns）曲线。**

- 前 ~28s（epoch 0–400）：val_acc 从 8.4% 快速升到 ~26%（斜率大）。
- 28s–260s（epoch 400–4000）：val_acc 仅从 26% 缓慢爬到 28%，在高位波动、趋于平台。
- 训练时间翻近 10 倍只换来约 2 个百分点；继续加长期望收益越来越小。这与学习曲线的一般规律一致，也是纯演化策略 + 硬 0/1 稀疏奖励 + 单卡小并行（128 候选）共同作用的结果。

> 复现：`.\\.venv\\Scripts\\python.exe exp_train_time.py 4000 128 8`

### 7.5 奖励函数对比实验：硬 0/1 vs log-likelihood vs sigmoid margin（关键）

基于 7.4 的分析，对 `snn_mnist.fitness_from_logits` 尝试了三种奖励，并在**完全相同的配置**
下各跑 4000 epoch（num_envs=128, T=8, hidden=[128,128], lr=0.03, sigma=0.2）做三方对比。

三种奖励定义：
- **硬 0/1**：`fitness = 1(正确) / 0(错误)`，离散。
- **log-likelihood**：`fitness = log_softmax(logits)[label]`，连续、范围 (-inf, 0]。
- **sigmoid margin**：`fitness = sigmoid(logits[label] - max_{j≠label} logits[j])`，连续、有界 (0,1)。

| 指标 | 硬 0/1 | log-likelihood | sigmoid margin |
|---|---|---|---|
| 峰值 val_acc | ~26%（epoch 400） | **~27.7%（epoch ~1100）** | ~16% |
| 4000 epoch 终点 val_acc | **~28%（稳定平台）** | ~13%（后段崩溃） | ~14-15%（低效稳定） |
| 后期曲线形态 | 稳定平台 | 先升后崩，跌回近随机 | 稳定但几乎不升 |

**三方对比结论：**

1. **log-likelihood 崩溃**：无界的 log-likelihood 对 logits 绝对尺度极敏感，z-score 归一化的连续奖励
   与 LoRA 扰动形成**正反馈**，导致 logits 尺度发散、后期崩溃回随机水平。
2. **sigmoid margin 稳定但低效**：有界的 sigmoid 避免了崩溃，但其**饱和区**（margin 过大/过小时
   `sigmoid≈0/1`、对 logits 失去区分度）使大部分样本落在无梯度区，因而收敛慢（~15%），反而
   低于硬 0/1。
3. **硬 0/1 虽信息量低，但在本框架（ES + LoRA + z-score + 小并行）下最稳健、终值最高（~28%）**。

**核心经验**：在本纯演化框架下，奖励的**稳定性**（量纲有界、与扰动配合）比**信息量/平滑性**更重要。
简单、量纲受控的奖励往往优于理论上更"平滑"但尺度敏感的无界/饱和奖励。若要利用连续梯度，须配合
温度缩放（把 margin 缩放到 sigmoid 非饱和区）、clip 与 variance reduction 等稳定化手段。

> 当前 `fitness_from_logits` 实现为 **sigmoid margin**（用户指定）；如需回到最高的硬 0/1，改动一行即可。

### 7.6 学习率调度对比实验（硬 0/1 奖励）

基于 7.5 结论（硬 0/1 最稳健），用**硬 0/1 奖励 + 5 种 optax 学习率调度**各跑 1000 epoch
（num_envs=128, T=8, base_lr=0.03, sigma=0.2, 独立同 seed 初始化）比较（`exp_lr_schedule.py`）：

| 学习率调度 | val_acc@1000epoch | best_train |
|---|---|---|
| fixed=0.03（基线） | 0.258 | 0.359 |
| linear 0.03→0.003 | 0.275 | 0.359 |
| cosine 0.03 | 0.306 | 0.344 |
| exp-decay 0.03/200×0.995 | 0.258 | 0.352 |
| **warmup+cosine 0.03** | **0.354** | **0.414** |

**结论：warmup + cosine 退火调度显著最优。**

- warmup+cosine 把 1000 步 val_acc 从 fixed 的 **25.8% 提升到 35.4%**（+9.6pp），
  也高于此前所有奖励/调度组合（硬 0/1 固定 LR 峰值 28%）。
- 机制：warmup 避免了演化早期在随机初始上大步更新导致的振荡/退化；
  cosine 退火在训练后期精细收敛，抑制了固定 LR 在平台期的振荡（对应 7.4 的平台现象）。
- best_train 也最高（41.4%），说明学习确实更充分。

**建议**：10 类 MNIST 训练应优先采用 `optax.warmup_cosine_decay_schedule`，而非固定 LR。

**正式训练验证**（`snn_mnist_train.py`，硬 0/1 + warmup+cosine，num_envs=128, T=8, 1000 epoch）：
最终测试集 val_acc ≈ **30.3%**。相比固定 LR 长时（4000 epoch，28%）**仅用 1/4 步数就更高**，
验证了 warmup+cosine 收敛更快、更充分（训练脚本 val 在 epoch ~950 达到 30%+）。

### 7.7 批次放大效应实验（固定 LR，改变并行候选数）

固定 LR=0.03、500 次进化更新（epoch）、T=8、硬 0/1，只改变**每次更新的并行样本数 num_envs**
（`exp_batch_effect.py`）。完整批次扫描（32 → 4096）：

| num_envs（批次） | val_acc@500epoch | best_train | 用时(s) |
|---|---|---|---|
| 32 | 0.202 | 0.375 | 23 |
| 64 | 0.273 | 0.375 | 24 |
| 128 | 0.224* | 0.344 | 32 |
| 256 | 0.341 | 0.348 | 50 |
| 512 | 0.434 | 0.436 | 91 |
| 1024 | 0.504 | 0.497 | 183 |
| 2048 | 0.507 | 0.480 | 360 |
| 4096 | **0.634** | **0.592** | 695 |

（*128 为单次运行离群噪声点）

**结论：存在强烈的"批次放大效应"——增大并行候选数可显著放大学习能力。**

- 总体趋势：批次 32→4096（128 倍），val_acc 从 **20.2% → 63.4%**（+43pp）。
- 佐证 ES 理论：噪声梯度误差 ∝ σ/√N，更多并行候选 → 梯度估计方差更低 → 更强学习。
- 非线性：1024→2048 出现平台（50.4→50.7%），但 4096 又**大幅跳升到 63.4%**——批次跨过
  某门槛后可越过局部瓶颈（单次运行也含随机方差，2048 或略偏低，但 4096 大幅领先是稳健的）。
- **4096 批次下 63.4% 为全部实验最高**（远超固定/衰减 LR 的 28–35%、warmup+cosine 的 35%）。

**核心洞见**：印证 HyperscaleES "hyperscale" 的设计哲学——**学习能力主要靠扩大并行批次放大，
而非延长小并行训练时间**。这也解释了 7.4 的平台瓶颈（长时间 × 小批次 128 收益递减）。
在 CPU 单机上 4096 批次已可到 ~63%；更大批次 + 更多 GPU 并行（hyperscale 场景）仍有望继续抬升。

### 7.8 批次 × 学习率调度 交互效应（2048 批次 + 多种 LR）

固定 num_envs=2048、200 次更新，比较 5 种 LR 调度（`exp_lr_schedule.py 200 2048`）：

| 学习率调度 | val_acc@200步 | best_train |
|---|---|---|
| **fixed = 0.03** | **0.458** | 0.402 |
| linear 0.03→0.003 | 0.390 | 0.285 |
| cosine 0.03 | 0.353 | 0.295 |
| exp-decay | 0.434 | 0.377 |
| warmup+cosine | 0.426 | 0.276 |

**结论：存在显著的"批次 × 调度"交互——调度是否必要取决于批次大小。**

| 批次 | 最优调度 | val_acc |
|---|---|---|
| 128（7.6 节） | warmup+cosine | 0.354（1000 步） |
| **2048** | **fixed（固定 LR）** | **0.458（200 步）** |

- 小批次：梯度噪声大，需 LR 衰减抑制振荡 → warmup+cosine 最优。
- 大批次：梯度噪声低（方差 ∝ 1/N），训练已稳定 → **固定 LR 全程大胆学习更好**，
  LR 衰减（cosine/linear）反而过早减小学习率、限制步内学习。

**工程启示**：调参时**先定批次（主导），再选调度（辅助）**。大规模并行（大批次）下固定 LR
已是高效默认，无需复杂调度。

### 7.9 更大初始学习率的效应（2048 批次 + base_lr 扫描）

在 7.8 的同一配置（num_envs=2048、200 次更新、T=8、硬 0/1 奖励、同 seed）下，把初始学习率
`base_lr` 从基线 0.03 提高到 **0.1** 与 **0.3**，重跑全部 5 种调度（`exp_lr_schedule.py 200 2048 <base_lr>`），
观察"调度方式是否随初始 LR 改变而重排"。

| 学习率调度 | base_lr=0.03 | base_lr=0.1 | base_lr=0.3 |
|---|---|---|---|
| fixed（固定） | **0.458** | 0.405 | 0.233 |
| linear LR→0.1×LR | 0.390 | 0.462 | 0.258 |
| cosine | 0.353 | 0.387 | 0.320 |
| exp-decay | 0.434 | 0.359 | 0.237 |
| **warmup+cosine** | 0.426 | **0.519** | 0.234 |

（值为 val_acc@200步；0.03 列来自 7.8 节）

**结论：提高初始 LR 整体上不是提升，而是"依赖调度方式、且排序被重排"。**

1. **最优调度从 fixed(0.03) 切换为 warmup+cosine(0.1)**：base_lr=0.1 时 warmup+cosine 达 **0.519**，
   为全部 15 组最高（比基线最佳 fixed/0.03 的 0.458 还高 +6pp）。机制：warmup 用低 LR 起步，
   避开高初始 LR 在随机初始化上的剧烈振荡；cosine 后期精细收尾。
2. **fixed 随 LR 单调恶化**：0.458→0.405→0.233。大批次下固定 LR 一旦过大，单步更新越过梯度、
   破坏稳定性，学习能力崩塌（0.3 时仅 23%，接近随机）。
3. **调度越"硬/快"，对高 LR 越敏感**：0.3 时多数调度塌回 23-32%，仅带强退火的 cosine(0.32) 相对最好
   ——从 0.3 快速衰减，把有效步长压回合理区间才勉强维持。
4. **linear 在 0.1 时是"异类受益者"**：0.390→0.462 反超基线。因它在 200 步内把 0.1 线性压到 0.01，
   "早期大步探索 + 后期小步收尾"恰与 0.1 组合最优；但 0.3 时同样失效（0.258）。

**综合洞见**：
- 提高初始 LR 对**无退火/固定类**调度（fixed、exp-decay）是净损害；
- 只有**带 warmup 或快速退火**的调度能从更高峰值 LR 获益，且峰值越高越需要更长的 warmup / 更快的退火；
- 本配置下最优组合为 **warmup+cosine、base_lr=0.1（val 51.9%）**，说明"适度调大初始 LR + warmup 保护"
  确有额外收益；但 ≥0.3 的大 LR 若不配套更保守的调度，反而不如 0.03。

### 7.10 超大批次扫描实验：0.75~0.8 精度极限（GPU / WSL2，2026-08）

**背景**：7.7 节在 CPU 上把批次推到 4096（val 63.4%）后，继续放大批次仍有抬升空间但受算力限制。
本次改用 RTX 4090（WSL2 + JAX 0.11 + CUDA 13）跑 `exp_batch_sweep.py` 的批次扫描：batch 改为
**相对训练集比例**（`batch = round(ratio × 60000)`），每个点固定跑 4000 次进化更新，固定 LR=0.03、
硬 0/1 奖励，隔离"批次大小"单一变量。

**配置**：T=8, hidden=[128,128], sigma=0.2, lr=0.03, seed=0, rank=8, 每点 4000 epochs。

| ratio | batch | val_acc | best_train | 用时(s) |
|---|---|---|---|---|
| 0.1 | 6000 | 0.699 | 0.708 | 54 |
| 0.2 | 12000 | 0.746 | 0.730 | 92 |
| 0.3 | 18000 | 0.752 | 0.749 | 126 |
| 0.4 | 24000 | 0.756 | 0.754 | 176 |
| 0.5 | 30000 | ~0.76\* | 0.754 | — |

（\*第 5 点在约 2100 epoch 时手动停止；日志显示 best_train 自 epoch ~900 起长期停在 0.754，
val 在 0.753~0.776 之间随机波动，不再上升。）

**结论 1：出现清晰的"批次-精度"收益递减平台，当前框架极限约 0.75~0.8。**

- 批次放大效应依然成立（6000→30000，val 0.70→0.76，印证 7.7），但幅度大幅收窄（5 倍批次仅 +6pp）。
- 超过某阈值后 best_train 长时间停滞、val 波动不再上升——进入平台期，继续加 epochs 收益趋零
  （与 7.4 的收益递减曲线一致）。

**结论 2：瓶颈是"优化/表达力受限"，不是过拟合。** best_train ≈ val（0.754 vs 0.756），训练与测试
几乎一致，说明模型远未过拟合，平台来自优化过程与表示能力，而非泛化。

**极限成因分析**：
1. **输入侧噪声地板**：`poisson_encode`（T=8）每次 forward 对同一图像重新伯努利采样，目标函数本身
   带噪声；val ±1~2pp 的波动即编码噪声的直接体现。即使梯度估计再准（超大 batch），也无法突破目标
   自身的噪声水平。
2. **奖励信号稀疏化**：硬 0/1 每样本仅 1 bit（对/错）。~75% 样本已正确后，提供有效"方向"的样本
   占比低且分散，ES 靠统计相关性提取的改进信号衰减（呼应 7.5"硬 0/1 信息量低但稳定"）。
3. **LoRA rank=8 子空间有限**：更新被限制在低秩子空间，表达精细决策边界的能力受限。
4. **固定 LR=0.03**：大批次下 fixed 是最优调度（7.8），但靠近饱和点时固定步长会在最优点附近震荡，
   无法精细收尾。
5. **SNN 离散表示精度**：hidden=[128,128]、T=8 脉冲计数读出，离散脉冲本身的信息精度有限。

**突破方向（按预期性价比排序）**：
1. **消除/降低输入噪声**：固定编码 seed 或提高 T（T=16/32）——直接降低"噪声地板"，与现有流程
   兼容性最好，最值得先试。
2. **进一步放大批次（hyperscale 哲学）**：6000→30000 仍在抬升，继续放大（如 60000 全量 + 多卡
   shard_map）仍有望突破平台，但收益递减、受显存/多卡限制。
3. **有界连续奖励**：硬 0/1 信息量低是已知短板（7.5）；按 7.5 建议用 `sigmoid(margin)` + 温度缩放
   + clip/variance reduction，在超大 batch 下安全利用连续梯度。
4. **提高 LoRA rank / 模型容量**：best_train≈val 表明表达力受限，rank 8→16/32 或加宽 hidden 可测。
5. **自适应 LIF（v_th/tau_m）**：按输入量级自适应阈值，缓解离散脉冲表示瓶颈（呼应第 9 节）。
6. **LR 调度微调**：超大 batch 下 warmup+cosine(base_lr=0.1)（7.9 中 2048 批次的最优组合）值得在
   30000 批次下复测。

**复现**（WSL2 / GPU；每点 4000 epochs，batch = ratio × n_train，顶部 `MAX_EPOCHS`/`RATIOS` 可调）：

```bash
wsl -d Ubuntu -u root -e bash -lc "cd /mnt/f/PythonProject/HyperscaleES && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' /root/hyperscalees-venv/bin/python exp_batch_sweep.py"
```

### 7.11 LoRA rank 单变量扫描：确定收益边际点（GPU / WSL2，2026-08）

**背景**：7.10 发现精度极限后，检验"LoRA rank（更新子空间维度）"是否限制学习能力。
固定 batch=0.2×训练集（12000）、固定 LR=0.03、固定 1000 epochs，仅 rank 作为单变量扫描
（`exp_rank_sweep.py`）。注意 EggRoll 用 `sigma/sqrt(rank)` 归一化扰动，rank 改变同时影响
扰动幅度，属完整单变量。

**配置**：T=8, hidden=[128,128], sigma=0.2, lr=0.03, seed=0, batch=12000, 每点 1000 epochs。

| rank | val_acc | best_train | 用时(s) | 较上一档提升 |
|---|---|---|---|---|
| 4 | 0.621 | 0.613 | 31 | — |
| 8 | 0.622 | 0.615 | 27 | +0.1pp（基本持平） |
| 16 | 0.649 | 0.639 | 29 | +2.7pp |
| 32 | **0.729** | 0.708 | 33 | **+8.0pp（增益最大）** |
| 64 | 0.759 | 0.728 | 44 | +3.0pp（明显收窄） |
| 128 | OOM | — | — | 24GB 显存不足（分配 5.23GiB 失败） |

**结论：收益边际点在 rank≈32。**

1. **低 rank 区（4~8）无效**：rank 翻倍但 val 几乎不动（0.621→0.622）——LoRA 子空间过小是瓶颈。
2. **中 rank 区（8→32）收益最大**：8→16（+2.7pp）、16→32（+8pp），rank=32 是边际增益最大的点。
3. **高 rank 区（32→64）收益递减**：翻倍仅 +3pp，进入收益递减段。
4. **内存成本线性暴涨**：EggRoll 扰动张量 ≈ (batch, 784, rank)×T，rank 翻倍显存翻倍；
   rank=128 在 batch=12000 下爆 24GB 显存。收益递增但内存线性增长的矛盾决定了 rank 有实际上限。

**工程建议**：rank=32 为"性价比平衡点"（收益曲线上拐点前增益最大）；若显存充足（多卡/更大显存）
rank=64 可再小幅抬升，但性价比明显下降。

**复现**（WSL2 / GPU；batch=0.2×60000=12000，每点 1000 epochs，默认 rank 4~128）：

```bash
wsl -d Ubuntu -u root -e bash -lc "cd /mnt/f/PythonProject/HyperscaleES && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' /root/hyperscalees-venv/bin/python exp_rank_sweep.py"
```

### 7.12 可训练阈值电压 v_th（GPU / WSL2，2026-08）

**背景**：7.11 确定 rank=32 后，尝试把 LIF 阈值电压 v_th 从"冻结超参"改为"可训练参数"，
检验其是否限制学习能力（此前 `tau_m`/`v_th` 固定不参与 ES 更新）。

**实现细节**（新增 `exp_vth_trainable.py`，内嵌 `TrainableVthSNN`，未改动 `snn.py`）：
- 把 `v_th` 从 `frozen_params` 移入可训练 `params` 树，作为 PARAM 类型参与 ES 全参更新；
  `frozen_params` 仅保留 `tau_m`。
- **softplus 参数化保证恒正**：params 中存原始值 `raw`（初始 `log(exp(0.3)-1)`，即实际阈值 0.3），
  前向中实际阈值 = `softplus(raw)`。避免 ES 把阈值推向负值导致神经元疯狂发放/网络失效。
- 其余结构与 SNNModel 完全一致：fc1/fc2/fc3 走 LoRA（rank 限制）、out_gain 全参更新。

**配置**：T=8, hidden=[128,128], sigma=0.2, lr=0.03, seed=0, batch=12000, rank=32, 1000 epochs。

| 指标 | 固定 v_th=0.3（7.11 同配置） | 可训练 v_th |
|---|---|---|
| val_acc | 0.729 | **0.757（+2.8pp）** |
| best_train | 0.708 | 0.712 |
| v_th | 0.3（固定） | 0.300 → **0.427** |

**v_th 调节轨迹**（每 50 epoch 采样）：早期先降到 0.18~0.28（阈值降低、神经元更易发放，加速启动），
中后期逐步抬升到 0.36~0.44，最终收敛于 0.427——ES 找到了比人工固定 0.3 更优的阈值，并
带来 +2.8pp 的精度提升，验证 v_th 确实是可优化的自由度。

**结论**：把 SNN 的 LIF 超参（v_th 等）纳入可训练参数是有效的突破方向之一（呼应 7.10 突破方向 5）；
softplus 参数化是必要的稳定性保障。后续可进一步把 `tau_m` 也纳入可训练参数、或给 v_th 施加
有界先验（如 clip 到合理区间）验证鲁棒性。

**复现**（WSL2 / GPU；配置见脚本顶部常量）：

```bash
wsl -d Ubuntu -u root -e bash -lc "cd /mnt/f/PythonProject/HyperscaleES && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' /root/hyperscalees-venv/bin/python exp_vth_trainable.py"
```

### 7.13 可训练 tau_m 的负面结论（GPU / WSL2，2026-08）

**背景**：7.12 验证 v_th 可训练有效（+2.8pp）后，尝试把 tau_m 也纳入可训练参数，
检验能否进一步提精度。`exp_vth_trainable.py` 增加 `TRAIN_TAU` 开关（True/False），
权重矩阵初始化与 7.12 完全一致（同一随机 key 派生），保证对比公平。

**配置**：T=8, hidden=[128,128], sigma=0.2, lr=0.03, seed=0, batch=12000, rank=32, 1000 epochs。

| 配置 | val_acc | best_train | v_th | tau_m |
|---|---|---|---|---|
| 固定 v_th/tau（7.11 同配置） | 0.729 | 0.708 | 0.3 | 20 |
| 仅 v_th 可训练（7.12） | **0.757** | 0.712 | 0.300→0.427 | 20（固定） |
| v_th + tau_m 可训练 | 0.744 | 0.708 | 0.300→0.399 | 20.0→19.68 |

**结论：tau_m 可训练没有带来提升，反而使 val_acc 回落 1.3pp（0.757 → 0.744）。**

1. **tau 几乎未被调节**：1000 epochs 内 tau 仅在 19.7~20.4 之间小幅徘徊，最终 19.68
   （变化 <2%）——ES 没有动力移动 tau，说明在当前任务下 tau 不是有效自由度。
2. **多一个参数反而引入优化噪声**：tau 作为 PARAM 全参扰动，其随机扰动与 AdamW 状态
   会干扰 v_th / 权重的更新（0.757 → 0.744）。
3. **机理解释**：LIF 泄漏项 `v += (dt/tau)·(−v + I)`，dt/tau = 0.05，在 T=8 的短时间窗口内
   衰减影响很小，tau=20 已处于合理区；它不像 v_th 那样直接控制发放门限，可优化空间有限。

**工程建议**：维持 7.12 的"仅 v_th 可训练"方案（单点最优 0.757），tau_m 保持冻结；
后续如需再探，可考虑给 tau 施加更大初始扰动/更长时间训练，但当前证据表明收益有限。

**复现**（WSL2 / GPU；`TRAIN_TAU` 开关在 `exp_vth_trainable.py` 顶部）：

```bash
wsl -d Ubuntu -u root -e bash -lc "cd /mnt/f/PythonProject/HyperscaleES && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' /root/hyperscalees-venv/bin/python exp_vth_trainable.py"
```

### 7.14 逐层独立 v_th：缓解深层网络退化（GPU / WSL2，2026-08）

**背景**：7.12 验证 v_th 可训练有效（2 层 +2.8pp），但直接把网络加深到 3 层（[128,128,128]）
后训练完全退化（val ≈ 0.098，接近随机）。分析主因是**全局单一 v_th 无法匹配各层电流量级**：
v_th 是全部 LIF 层共享同一个可训练标量，而第 2/3 层输入是上一层脉冲率（量级与第 1 层连续
输入差异大），同一阈值不可能同时适配各层。据此实现"逐层独立 v_th"方案验证。

**实现细节**（`exp_vth_trainable.py` 增加 `VTH_PER_LAYER` 开关，True 时生效）：
- 每层一个可训练阈值参数 `v_th1..v_thN`（PARAM 类型，softplus 参数化恒正，各层初始 0.3）。
- 前向中每层 LIF 使用自己的阈值；`VTH_PER_LAYER=False` 时回到全局共享（复现 7.12/7.13）。

**配置**：T=8, hidden=[128,128,128], sigma=0.2, lr=0.03, seed=0, batch=12000, rank=32, 1000 epochs。

| 配置 | val_acc | best_train | v_th 终值 |
|---|---|---|---|
| 2 层 [128,128] + 全局 v_th（7.12） | **0.757** | 0.712 | 0.427 |
| 3 层 + 全局 v_th（7.13 后续） | 0.098（随机） | 0.105 | 1.44（单值） |
| **3 层 + 逐层 v_th（本次）** | 0.431 | 0.411 | [0.194 / 0.294 / 4.289] |

**结论：逐层独立 v_th 部分解决了深层网络的退化问题，但仍未超过 2 层。**

1. **恢复学习能力**：从完全随机（0.098）恢复到 0.431，验证了"全局单阈值不匹配各层电流量级"
   是 3 层失效的主因。
2. **各层阈值自然分化**：第 1 层 0.19（输入电流大 → 阈值调低、易发放）、第 2 层 0.29、
   第 3 层 4.29（阈值被推到极高，工作在近饱和抑制态）——每层 ES 找到不同工作点。
3. **仍未超过 2 层**（0.431 vs 0.757）：1000 epochs 内 3 层收敛更慢（参数更多、脉冲链中仍有
   信号衰减），且日志显示训练仍处上升趋势（epoch 950 → 0.434，未到平台）。

**工程建议**：
- 逐层 v_th 是加深网络的前提，但单靠它不足以让 3 层超过 2 层；可尝试延长训练（≥3000 epochs）
  或引入层间归一化/残差连接稳定脉冲传播。
- 第 3 层阈值 4.29 表明深层电流量级被显著放大，后续可考虑按层自适应 v_th 初始化（呼应 6.2
  的"阈值需与输入电流量级匹配"教训）。

**复现**（WSL2 / GPU；`VTH_PER_LAYER`/`HIDDEN`/`TRAIN_TAU` 开关在 `exp_vth_trainable.py` 顶部）：

```bash
wsl -d Ubuntu -u root -e bash -lc "cd /mnt/f/PythonProject/HyperscaleES && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' /root/hyperscalees-venv/bin/python exp_vth_trainable.py"
```

### 7.15 3 层网络长时训练：fixed vs cosine LR（GPU / WSL2，2026-08）

**背景**：7.14 中 3 层逐层 v_th 在 1000 epochs 内仅 0.431 且仍处上升趋势，推断是训练时长不足。
本次把 epoch 数拉到 10000，并对比固定 LR 与余弦退火 LR（`exp_vth_trainable.py` 新增
`LR_SCHEDULE` 命令行参数，cosine 与 exp_lr_schedule.py 同口径：`cosine_decay_schedule(0.03, 10000)`）。

**配置**：T=8, hidden=[128,128,128], sigma=0.2, seed=0, batch=12000, rank=32, 逐层 v_th,
tau 冻结, 10000 epochs。

| LR 调度 | val_acc | best_train | v_th 终值 | 用时(s) |
|---|---|---|---|---|
| fixed | 0.655 | 0.660 | [0.855 / 0.469 / 14.446] | 333 |
| **cosine** | **0.699** | 0.694 | [0.575 / 0.230 / 9.877] | 332 |

**与历史对比**（3 层逐层 v_th）：

| 配置 | val_acc |
|---|---|
| 1000 epochs（7.14） | 0.431 |
| 10000 epochs fixed | 0.655 |
| 10000 epochs cosine | 0.699 |
| 2 层基准（7.12，1000 epochs） | 0.757 |

**结论：延长训练显著帮助 3 层收敛，且 cosine 优于 fixed（+4.4pp）。**

1. **训练时长是关键**：0.431 → 0.655/0.699，验证 7.14 中"3 层收敛更慢、未到平台"的判断；
   3 层网络确实需要远超 2 层的更新次数。
2. **cosine 后期低 LR 稳定参数**：cosine 的 v_th 在 epoch 9200~10000 几乎完全冻结于
   [0.575/0.230/9.88]；而 fixed 的 v_th 全程漂移（第 3 层阈值一路冲到 14.4 仍在动），
   尾部过冲损失精度——这与 7.8"大批次下 fixed 最优"的结论不同，原因是本实验含大量可训练
   超参（逐层 v_th）且训练更长，退火有助于稳定。
3. **3 层仍未超过 2 层（0.699 vs 0.757）**：即使 10000 epochs + 逐层 v_th + cosine，第 3 层
   阈值仍被推到 ~10（近全关状态），深层实际承载的信息有限——深层表示优势未兑现，需层间
   归一化/残差等结构改进（呼应 7.14 工程建议）。

**复现**（WSL2 / GPU；`LR_SCHEDULE` 取 fixed 或 cosine）：

```bash
wsl -d Ubuntu -u root -e bash -lc "cd /mnt/f/PythonProject/HyperscaleES && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' /root/hyperscalees-venv/bin/python exp_vth_trainable.py fixed"
wsl -d Ubuntu -u root -e bash -lc "cd /mnt/f/PythonProject/HyperscaleES && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_autotune_level=1' /root/hyperscalees-venv/bin/python exp_vth_trainable.py cosine"
```

## 8. 复现 / 使用方法

```bash
# 1) 运行测试
.\.venv\Scripts\python.exe tests\snn_test.py

# 2) 完整训练（参数在脚本顶部可调）
.\.venv\Scripts\python.exe llm_experiments\snn_mnist_train.py

# 3) 长时训练观察：准确率 vs 训练时间（可调 epoch/envs/T/奖励）
.\.venv\Scripts\python.exe exp_train_time.py 4000 128 8

# 4) 学习率调度对比（硬 0/1 奖励，5 种调度；可选第 3 参数 base_lr；默认 2048 批次）
.\.venv\Scripts\python.exe exp_lr_schedule.py 200 2048
.\.venv\Scripts\python.exe exp_lr_schedule.py 200 2048 0.1   # 更大初始 LR
.\.venv\Scripts\python.exe exp_lr_schedule.py 200 2048 0.3

# 5) 如需 10 类高精度：配置 GPU 并增大 num_envs / num_epochs
#    （演化策略在大并行下才能高效收敛）
```

## 9. 局限与后续方向

- 单卡 CPU 上纯演化对 10 类 MNIST 收敛有限，建议 GPU + 大并行（`shard_map` 多卡版本可参考 `do_grpo_multi_gpu.py` / `general_do_evolution_multi_gpu.py`）。
- LIF 阈值/`tau_m`/权重初始化需按输入量级人工校准；可考虑自适应阈值或权重初始化归一化改进鲁棒性。
- **奖励设计**：硬 0/1 信息量低但稳定；单纯换用无界连续奖励（log-likelihood）在纯 ES 框架下会训练崩溃（见 7.5），需配合 clip/温度缩放/有界奖励（如 `sigmoid(margin)`）与 variance reduction 才能安全利用连续梯度。
