# Tasks

- [x] Task 1: 实现 SNN LIF 神经元与模型模块 `src/hyperscalees/models/snn.py`
  - [x] SubTask 1.1: 实现 LIF 神经元单元（膜电位累积、泄漏、阈值发放、reset），封装为可在 `T` 步内 `jax.lax.scan` 的递推
  - [x] SubTask 1.2: 实现 `SNNModel`，符合 `Model` 接口（`rand_init` 返回 `(frozen_params, params, scan_map, es_map)`；`_forward(common_params, x)` 支持 `(T, in_dim)` 输入、内部 `T` 步展开并全步复用同一 `iterinfo` 扰动）
  - [x] SubTask 1.3: 参数分类：输入投影/权重标 `MM_PARAM`，可训练标量（out_gain）标 `PARAM`，冻结常数（tau_m/v_th）入 `frozen_params`
  - [x] SubTask 1.4: 读出层在时间轴聚合（平均发放率）后经分类头输出 logits

- [x] Task 2: 实现 MNIST 数据与泊松编码 + fitness 打分 `src/hyperscalees/environments/snn_mnist.py`
  - [x] SubTask 2.1: 加载 MNIST 训练/测试数据（支持本地 IDX 目录或 HuggingFace）并归一化到 `[0,1]`
  - [x] SubTask 2.2: 实现泊松编码：对每个像素在 `T` 步内以像素强度为概率采样伯努利，输出 `(T, batch, 28*28)` 脉冲
  - [x] SubTask 2.3: 实现 fitness 打分：将模型输出 logits 与标签比对，生成每个样本的原始奖励

- [x] Task 3: 单卡训练脚本 `llm_experiments/snn_mnist_train.py`（沿用 `end_to_end_test.py` 的 `jax.jit`+`jax.vmap`+`Noiser.do_updates` 循环）
  - [x] SubTask 3.1: 编写训练循环脚本，含参数（num_epochs / num_envs / T / sigma / lr / rank / 层规模 / seed）
  - [x] SubTask 3.2: 集成 `EggRoll` Noiser：`init_noiser`、`simple_es_tree_key`、`convert_fitnesses`、`do_updates`
  - [x] SubTask 3.3: 周期性（`iterinfo=None`）在测试集评估并打印准确率

- [x] Task 4: 测试 `tests/snn_test.py`
  - [x] SubTask 4.1: SNN 前向正确性测试（`rand_init` 结构、LIF 动力学、前向输出形状、无扰动结果可复现）
  - [x] SubTask 4.2: SNN+MNIST 训练冒烟测试（本地 IDX 数据，10 类 MNIST 循环跑通、参数被演化正确更新、准确率不塌缩）

# Task Dependencies
- Task 2 依赖 Task 1（SNN 模型是评分对象）
- Task 3 依赖 Task 1 与 Task 2
- Task 4 依赖 Task 1、2、3
