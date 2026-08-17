# Rust 可训练 SNN Transformer（snn_transformer）

> 在已迁移的 Rust ES 框架（`burn_impl/`）下，完成一个**可训练**的 SNN Transformer
> 架构并接入演化策略（ES）训练，针对 patched-MNIST 分类。

## 1. 动机与目标

仓库存在两类 SNN 注意力 Python 参考：

- `snn_attention.py`（连续速率版）：ES 能学到 ~58%，但它是「全局 query + 标量权重」
  的 **Attention-Pooling**，不是真 Transformer；
- `snn_self_attention_heads.py`（真逐 token 多头 + 位置编码 + 多块残差）：是真
  Transformer，但**在 ES 下完全不学**（训练 3000 epochs 一直 ~10% 随机水平），根因见
  `docs/es_selfattn_heads_train_failure_analysis.md`——硬阈值 LIF 竞争 + 行内归一化
  使「奖励对参数微扰的增益」≈0，ES 梯度无从累积。

本实现的目标：把 `snn_self_attention_heads.py` 的**架构形态**移植到 Rust，但采用该
分析文档 §5 的修复方向（**连续注意力核心 + 连续 Q/K/V 编码**），得到既真 Transformer、
又能在 ES 下学习的模型。

## 2. 设计

### 2.1 连续（可训练）注意力核心

- 注意力权重用 **Boltzmann / Hopfield 连续松弛**（与 `snn_attention.hopfield_attention`
  同源）：

  ```
  H = beta · (Q @ Kᵀ)            // (num_tokens, num_tokens) 逐 token 相似度
  u ← u + (1/τ)·(−u + H − g·mean(u))   // 迭代 n_iter 步，divisive 归一化
  A = softmax(u, 沿 key 轴)             // (num_tokens, num_tokens)，行和 ≈1
  O = A @ V                            // 序列输出，保留 token 结构
  ```

- Q/K/V 前端用**连续 sigmoid 速率编码**（`rate_encode`，接入 `proj_gain`），取代硬阈值
  `_rate`；FFN 用 swish 连续门控。全程无硬阈值，`score` 对扰动平滑，ES 可累积梯度。

### 2.2 架构（真 Transformer）

```
x: (T, in_dim)          单样本，in_dim = num_tokens·token_in_dim（patched 展平）
  -> reshape (T, num_tokens, token_in_dim)
  -> 逐 token Q/K/V 投影 + 连续速率编码   (num_tokens, d_model)
  -> + pos_emb (num_tokens, d_model)
  -> L 个块：多头连续自注意力 + 残差；swish 前馈 + 残差
  -> 池化（mean over tokens）-> out -> *out_gain -> logits (num_classes,)
```

### 2.3 噪声注入：按**参数索引**寻址（关键）

ES 需要「前向实际注入每个可训练参数的扰动」，且梯度 `mean(score · noise)` 才有效。
但多头 q/k/v/o/ff 权重**形状大量重复**，若按权重**形状**路由噪声会歧义。故 `TrainNoise`
按 `SnnTransformer::params()` 的**索引**寻址：

- `mm_indices[k]` ⇔ 第 k 个 LoRA（`MM_PARAM`）参数，`lora[k]` = 该参数的逐样本
  `(A (n,r,a), B (n,r,b))`；
- `pos_emb` / `out_gain` / `beta`（非 matmul 参数）用逐样本**稠密**加性噪声。

`nn(idx, x, w)` 闭包按 `idx` 查 `mm_indices` 应用 LoRA 噪声（`x@wᵀ + (x@B_sᵀ)@A_s`）。

> 注意：`batched_lora_noise` 返回 `(A(n,a,r), B(n,b,r))`，本模型与 `lora_einsum_pair`
> 期望 `(n,r,a)`/`(n,r,b)`，训练二进制里做了 `swap_dims(1,2)` 转换。

## 3. 文件

| 文件 | 说明 |
|------|------|
| `burn_impl/hyperscalees-models/src/snn_transformer.rs` | 模型 `SnnTransformer`（含 8 个单测：结构/es_map、注意力等价、确定性、噪声敏感、多块加深、位置编码、逐样本独立、参数索引布局） |
| `burn_impl/hyperscalees/src/bin/snn_transformer_train.rs` | 训练二进制（CPU flex / `--features gpu` CUDA 同一份代码） |

## 4. 训练

初始化时 `d_model` 必须能被 `num_heads` 整除。`params()`/`es_map()` 与既有
`TrainableVthSnn` 相同的扁平参数管线；`init_noiser` + 每 chunk `batched_lora_noise`
（MM，CPU 生成后上传）+ 稠密噪声（PARAM/EMB）→ `forward_batched` → loglik 奖励 →
`lora_einsum_pair` 与稠密加权梯度 → `combine_affine_grads`（全局 z-score 仿射修正）→
`solver.update` → `write_params` 回写。每 chunk 样本数需为偶数（反对称配对）。

**GPU 就绪**：`SnnTransformer` 与 ES 全路径只用泛型 burn 算子（matmul/softmax/
sigmoid/exp 等），`hyperscalees-core` 的 `gpu` feature 把 `B` 路由到 CUDA 后端，故
**同一训练二进制在 `--features gpu` 下直接跑 GPU**（`[env] backend=cuda` 行证实）。
`batched_lora_noise` 在 CPU 并行生成噪声后一次性上传，`lora_einsum_pair` 是 2D
GEMM（后端 matmul），均对 CUDA 兼容；无需 `cublas`/`oxide` 内核（那是
`accumulate_train` 那条优化热路径的事）。

用法（CPU flex 默认 / `--features gpu` 跑 CUDA）：

```bash
# 需去掉 sccache：Remove-Item Env:RUSTC_WRAPPER
# CPU 冒烟：
cargo run -p hyperscalees --bin snn_transformer_train -- \
    --batch 2048 --accumulate 4 --rank 16 --T 8 \
    --d-model 32 --num-heads 4 --num-blocks 2 \
    --num-epochs 300 --mnist-dir <dir> [--csv-out out.csv]

# GPU（RTX 4090 等）：加 --features gpu，开更大 d_model / 全批 60000 / 数千 epoch
cargo run --release -p hyperscalees --features gpu --bin snn_transformer_train -- \
    --batch 60000 --accumulate 8 --rank 16 --T 8 \
    --d-model 96 --num-heads 6 --num-blocks 3 \
    --num-epochs 2000 --mnist-dir <dir> [--csv-out out_gpu.csv]
```

- `--mnist-dir`：MNIST IDX 目录（默认 `D:\Rust\snn_t1\mnist_data`，或环境变量
  `MNIST_DIR`）。
- 小模型/短 epoch 冒烟测试已通过：`best_val 0.087→0.136`，验证全管线（噪声→前向→
  奖励→梯度→更新→回写→评估）端到端正确、参数确实更新。要看到显著精度（接近连续
  `snn_attention` 的 ~58% 量级），需要更大 d_model / 全批 60000 / 数千 epoch（与
  Python 参考一致），这正是 `--features gpu` 的目的。
- 跑 GPU 前需先 `cargo build -p hyperscalees --features gpu` 编译（首次会经 nvrtc
  JIT 预构建 burn-cuda 内核，耗时较长）；确保本机已装 NVIDIA CUDA Toolkit。

> 注：`snn_transformer_train` 为纯 CPU 可运行二进制，不依赖 `gpu` feature；而
> `accumulate_train` 等既有二进制引用 `oxide`/`cublas`（仅在 `--features gpu` 下存在），
> 故构建整个 `hyperscalees` 包仍需 gpu feature——这是仓库既有约定，非本新增内容引入。
