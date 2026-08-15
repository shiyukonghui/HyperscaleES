# SNN-ES 训练加速：混合后端设计方案与逐步优化记录

> 分支：`perf/cublas-einsum-pipeline`（相对 `main` 3305542）
> 目标：把 Rust/burn 实现的 ES 训练（MNIST SNN）压到 JAX 参考实现的水平。
> 基线：JAX 0.119–0.187 s/epoch（单卡 0.187s，多卡 0.119s）；本方案最终 **0.15 s/epoch**（已超单卡基线）。
> 硬件：RTX 4090，Windows WDDM；burn 0.21 + CubeCL CUDA 后端（禁用 fusion，原因见 §4.1）。

---

## 目录

1. [总体结论](#1-总体结论)
2. [当前架构：混合后端分工](#2-当前架构混合后端分工)
3. [逐步优化历程（含每步实测）](#3-逐步优化历程含每步实测)
4. [关键技术细节](#4-关键技术细节)
5. [失败尝试与结论（避免重复踩坑）](#5-失败尝试与结论避免重复踩坑)
6. [测量方法论（本项目的血泪教训）](#6-测量方法论本项目的血泪教训)
7. [如何复现与对照](#7-如何复现与对照)
8. [剩余优化空间](#8-剩余优化空间)

---

## 1. 总体结论

**不要问"C vs Rust 还是 burn 的差距"——真正的问题是内核/运行时质量差距（cuBLAS/XLA vs cubecl）。** 但结论不是"全换 cuBLAS"，而是**按形状分工**：

| 负载形状 | 最优实现 | 原因 |
|---|---|---|
| 瘦 M 长 K 的 2D GEMM（einsum：M=2a≤256，K=n/2·r=384000） | **cuBLAS（TF32）** | burn/cubecl 实测仅 ~1.6 TFLOP/s（并行度受 M×N 平铺限制）；cuBLAS 快约 2 倍，TF32 张量核再快一档 |
| 超多批小矩阵（前向噪声：12000 批 × (8,784)@(784,64)） | **burn（cubecl batched matmul）** | cuBLAS strided-batched 在此形状反而慢（fc1 噪声步 15ms vs cubecl 4.7ms） |
| 噪声生成（写 1.2GB+） | **vendored cubek-random 自定义内核** | 直接填充 burn 张量；配合"半噪声"方案生成量减半 |
| 泊松编码 / LIF / fitness / 参数更新 | **burn** | 元素级小算子，burn 足够快 |

**最大的一笔收益来自算法侧的重构（半噪声反对称方案），而非换后端**：噪声生成 85ms→19ms/epoch，且前向、einsum 均因"配对隐含"而零拷贝。换后端（cuBLAS einsum + TF32）再省约 60ms。

---

## 2. 当前架构：混合后端分工

### 2.1 每 chunk 的数据流（chunk=12000，accumulate=5，rank=64，T=8，隐藏层 128）

```
imgs (12000,784)
  │  poisson_encode（burn：一次 (T,n,784) Uniform + 一次比较）
  ▼
spikes_k (8,12000,784)
  │  gen_lora_noise_antipodal（cubek-random 的 random_normal，只生成前半！）
  ▼
A'_h (6000,64,128)  B'_h (6000,64,784)          ← 半噪声（配对隐含）
  │  forward_batched_lora_half（纯 burn：2 个半批 batched matmul + LIF）
  ▼
logits_k (12000,10)
  │  fitness / argmax / 正确率（burn）
  ▼
raw_k (12000,)
  │  配对合并 einsum：f_pair 加权 → cat → reshape
  │  → cublas::gemm_atb（cuBLAS，TF32，同一 stream，零同步）
  ▼
g_raw / g_ones（累加进 grad_acc / ones_acc）
```

### 2.2 各阶段归属与原因

| 阶段 | 实现 | 说明 |
|---|---|---|
| 泊松编码 | burn | 已是单张量向量化版（一次 Uniform (T,n,784) + lower 比较） |
| 噪声生成 | cubecl 内核（vendored `cubek-random`） | `random_normal` 直接写 burn 张量；**只生成 n/2 样本** |
| 前向（base + 噪声两步） | burn `forward_batched_lora_half` | cuBLAS 批量版实测更慢（§5.4） |
| einsum 梯度 | **cuBLAS `gemm_atb`** | 瘦 M 长 K 专用；TF32 与 XLA 默认一致（§4.4） |
| fitness/dense/更新 | burn | 小算子 |

### 2.3 半噪声方案的三个连锁收益（核心 trick）

反对称配对约定：`A'[n/2+i] = -A'[i]`、`B'[n/2+i] = -B'[i]`（方差缩减，ES 估计无偏）。

1. **einsum K 减半**：配对消去后半样本，raw/ones 合并为一次半 K GEMM：
   ```
   g_raw  = Σ_{i<half} (f_i + f_{i+half})·A'_i ⊗ B'_i
   g_ones = 2·Σ_{i<half} A'_i ⊗ B'_i
   ```
2. **生成量减半**：只生成前半张量 `(n/2, r, *)`，不生成取负副本（fc1 B' 2.4GB→1.2GB）。
3. **前向零拷贝**（最妙）：样本 `i+half` 的噪声注入是
   ```
   y[i+half] = x[i+half] @ (-B'_i)^T = -(x[i+half] @ B'_i^T)
   z[i+half] = y[i+half] @ (-A'_i)   = (x[i+half] @ B'_i^T) @ A'_i
   ```
   双重取负抵消 → 两半样本**共用同一份半噪声**各做一次 batched matmul 再 cat 拼接即可，
   不需要任何取负/拷贝，且与全量噪声版**逐位一致**（IEEE 中 `(-a)·(-b) = a·b` 精确成立；
   noise_bench `[0d]` maxdiff = 0.000e0）。

---

## 3. 逐步优化历程（含每步实测）

> 测量口径：同窗口背靠背 A/B（跨窗口对比不可靠，见 §6）；epoch 时间为稳态（非第 0 epoch）。

| 步骤 | 提交 | 改动 | 实测（同窗口） | 说明 |
|---|---|---|---|---|
| 0 | `3305542`(main) | 基线 | 1.31s → 0.28s | 更早的 accumulate_train 重构，非本分支 |
| 1 | `d753ad1` | einsum 首次换 cuBLAS；多流实验 | 0.28→0.25s | **事后证明是脏对照**（当时缺后续步骤，gen 成本掩盖了 einsum 收益）；多流失败（§5.1） |
| 2 | `8c2711a` | 反对称配对 prng 内核（一次内核生成完整配对张量） | 0.25→0.20s | 旧方案：后半线程重新生成同一序列并取负 |
| 3 | `98f3df9` | **半噪声方案**：只生成前半，配对由消费方施加（前向/ einsum 配套改造） | 0.23 vs 0.22s（D 基线） | gen 阶段 85→19ms；前向 2×半批 matmul 略增；净 +10ms 但 gen 大降，为后续铺路 |
| 4 | `8ed3d27` | **einsum 走 cuBLAS 同流 GEMM**（正确隔离后） | 0.22→0.16s | 关键一步：半噪声 + cuBLAS einsum 组合后收益才显形 |
| 5 | `72eadbe` | **einsum 默认 TF32 张量核**（XLA 默认一致） | 0.16→0.15s | 梯度噪声 O(1)，TF32 1e-3 相对误差无统计影响 |

### 各阶段 GPU 分解（ACC_PHASE=1，同步点门控，仅作结构参考）

| 阶段 | 步骤 2 之后 | 步骤 3 之后 | 步骤 4/5 之后 |
|---|---|---|---|
| poisson | ~10ms | ~10ms | ~9ms |
| gen | ~85ms | **~19ms** | ~19ms |
| fwd | ~75ms | ~80ms | ~75ms |
| einsum | ~120ms | ~130ms | ~68ms（cuBLAS） |
| fitness+dense | ~12ms | ~11ms | ~10ms |
| 稳态 epoch（无同步） | 0.21s | 0.22s | **0.15s** |

> 注意：同步门控的逐阶段数字对"多小内核"的 burn 路径有高估（时钟回落，见 §6.2），
> 无同步的 epoch 时间才是金标准。

---

## 4. 关键技术细节

### 4.1 为什么禁用 burn/fusion

burn 的 fusion（内核融合 tracer）在 CPU 侧做图追踪，实测引入 8–30× 的 CPU 开销
（epoch 0.74s → 0.55s，反而更慢）。本项目 CPU 入队本身不是瓶颈，融合省下的内核
启动被追踪开销吃掉。`hyperscalees-core/Cargo.toml` 的 `gpu` feature 明确不含
`burn/fusion`（有注释说明）。

### 4.2 cuBLAS 集成（cublas.rs）

- **依赖**：cudarc 0.19（cublas + driver + fallback-dynamic-loading），`cubecl 0.10`
  （cuda feature），`burn-cubecl 0.21`。
- **同流零同步**：cuBLAS handle 创建后 `cublasSetStream` 绑到 cubecl 的原始 stream
  （vendored cubecl-cuda 暴露 `CudaServer::raw_stream`）。于是 cuBLAS 调用与 burn
  算子**天然同流有序**，无需任何 cudaStreamSynchronize——逐 chunk 全量 sync 会打空
  GPU 流水线。输入指针经 `CudaServer::raw_device_ptr` 解析（走 cubecl 的 resolve
  机制，跨流依赖自动插入等待）。
- **列主序约定**（最容易错的部分）：
  - 行主序 `(k,m)` 矩阵 ≡ 列主序 `(m,k)` 矩阵，`lda = 行 stride`；
  - cuBLAS 输出 `C (m,n)` 是列主序 → 输出缓冲区按 `(n,m)` 行主序承载 `C^T`，
    返回前 `transpose()` 视图还原（下游 slice/add 对 strided 视图完全支持）；
  - burn 张量是 **pitched**（PitchedMemoryLayoutPolicy，16 字节行对齐），
    `lda/ldb/ldc` 一律取 `cube.meta.strides()` 的实际值，不能假设等于列数。
- **辅助函数**：`gemm_atb`（C=A^T·B，einsum 用）、`gemm_abt`（C=A·B^T，B 直接传
  权重避免方阵 transpose 不拷贝的坑）、`gemm`（C=A·B）、`batched_gemm_bt` /
  `batched_gemm`（批量噪声前向用，A/B 后证明训练形状上不敌 cubecl，保留作对照）。
- 数学模式：默认 `CUBLAS_TF32_TENSOR_OP_MATH`（§4.4）。

### 4.3 半噪声反对称方案的实现落点

| 文件 | 改动 |
|---|---|
| `vendor/cubek-random/src/normal_antipodal.rs` | 原"一次内核生成完整配对张量"的内核废弃（§5.3），文件留档说明 |
| `hyperscalees/src/cublas.rs` `gen_lora_noise_antipodal` | 只生成 `(n/2, r, a/b)` 前半（plain `random_normal`） |
| `hyperscalees-models/src/snn.rs` `forward_batched_lora_half` | 两半样本对同一半噪声做 batched matmul 后 cat；数学推导见 §2.3-3 |
| `hyperscalees-noiser/src/eggroll.rs` `lora_einsum_pair_half` | 直接消费前半张量（无切片）；f_pair 加权 + cat + 一次半 K GEMM |

配套校验（`noise_bench`）：
- `[0b]` 半 einsum vs 全量 einsum：maxdiff = 0.000e0（逐位一致）；
- `[0d]` 半前向 vs 全量前向：maxdiff = 0.000e0；cuBLAS 全量前向对照 1.2e-2（TF32 级）。

### 4.4 TF32（与 XLA/JAX 默认行为对齐）

XLA 的 fp32 matmul 默认允许 TF32（`allow_tf32` 默认开），因此 JAX 参考实现的
einsum 本就是 TF32 精度。本项目 cuBLAS handle 默认 `CUBLAS_TF32_TENSOR_OP_MATH`：
einsum 长 K（K=384000）归约实测 0.16→0.15s/epoch。梯度噪声本身 O(1) 量级，远大于
TF32 的 ~1e-3 相对误差，收敛轨迹与 fp32 同区间（20 epoch train_acc ≈0.088）。
`EINSUM_FP32=1` 可切回纯 fp32。

### 4.5 前向为什么保持 burn

前向噪声步是"12000 批 × 小矩阵"（每批 (8,784)@(784,64)）。专门实现的 cuBLAS
strided-batched 版本（`batched_gemm_bt`/`batched_gemm`/`forward_batched_lora_cublas`）
A/B 实测比 burn 慢 ~20ms/epoch（fc1 噪声步 15ms vs cubecl 4.7ms）。cuBLAS 的批量
内核在这种"海量小批"形状上不如 cubecl 的 batched matmul。**形状决定后端**。

---

## 5. 失败尝试与结论（避免重复踩坑）

| 尝试 | 结果 | 教训 |
|---|---|---|
| **5.1 多流流水线**（`StreamId(1).executes` 把编码/噪声/前向放到第二流与 einsum 重叠） | WDDM 下跨流事件同步 ~24× 慢（受控实验 1.81s → 43s），已回退 | Windows WDDM 上多流收益为负；单流 + 深队列才是对的 |
| **5.2 噪声跨 epoch 缓存**（fc1 噪声 14GB 缓存复用） | 缓存顶到 24GB 物理上限触发 WDDM 分页，5.8s/epoch，已回退 | 显存墙；每 chunk 现生成 + 半噪声方案才是对的 |
| **5.3 一次生成完整配对张量的内核**（后半线程重生成同一序列取负） | 计算量 2×，gen 85ms；后续"生成一次、双写两半"更差（双写流 0.28s/epoch） | **配对应由消费方隐含施加，而非生成侧物化**（半噪声方案） |
| **5.4 cuBLAS 前向/批量 GEMM** | 训练形状（12000 批×小矩阵）比 burn 慢 ~20ms/epoch | 批量小矩阵形状留给 cubecl |
| **5.5 burn cat 物化全量配对张量**（`cat([h, -h])`） | 0.30s/epoch（cat+neg 流量 7.2GB） | 不要为了"统一接口"物化后半 |
| **5.6 自定义 K 分片 einsum 内核**（块平铺 + 部分和二次归约） | fc1（m=256,n=784）的 A/B 重读流量 ~19GB/chunk → 37ms/chunk，整体不敌 cuBLAS，已删除 | 块平铺的跨块重读（A×n_tiles + B×m_tiles）在双大输出维上致命；小输出维（fc2/fc3）虽有改善但整体输 |
| **5.7 einsum 换向**（`g^T = B^T·A`，M=b 更大） | 无变化 | burn matmul 在 K=384000 上对任何朝向都差，换向不解决内核效率 |
| **5.8 einsum 连续化 LHS**（`transpose().clone()`） | 无变化 | 同上；该形状的瓶颈不是 LHS 布局 |
| **5.9 burn/fusion 融合** | CPU 追踪开销 8–30×，0.74s→0.55s | 本项目 CPU 入队不是瓶颈 |

---

## 6. 测量方法论（本项目的血泪教训）

### 6.1 跨窗口对比不可靠

同一二进制在不同时间点测量漂移达 ±75ms（GPU 时钟/热状态/WDDM 行为）。
**所有结论必须以同窗口背靠背 A/B 为准**（如步骤 4 的 0.22 vs 0.16 就是同一窗口连跑两次）。
早期"cuBLAS einsum 无收益"的负结论就是脏对照：对照组构建缺半噪声方案，
gen 成本（85ms）掩盖了 einsum 的收益。

### 6.2 同步门控的逐阶段计时会高估小内核

`ACC_PHASE=1` 在每个阶段末尾插 `into_scalar()` 同步点：GPU 在同步间隙空闲导致
时钟回落，多小内核的 burn 路径被高估（如 burn einsum 同步测 121ms，实际无同步
约 50–60ms）。**同步测得的阶段数只作结构参考；金标准是无同步 epoch 时间。**

### 6.3 bench 程序的内存污染

`noise_bench` 在长跑后（累计持有 ~10GB+ 张量）数字完全失真（同一操作 9ms 与 318ms
两个极端都出现过），疑似 WDDM 分页/时钟。**训练循环内测 + 同窗口 A/B 才是可信的。**

### 6.4 正确性校验的"线性区"技巧

真实阈值下 TF32 级差异会让 LIF 阈值翻转产生混沌放大（O(1) 级差异），无法作为等价
判据。等价性校验用 `v_th = -1e9` 强制全发放（线性区）：spike 图案确定性，误差只
来自 GEMM 数值（~1e-2），转置/布局错误仍给 O(1) 误差能抓住。半噪声 vs 全量前向
在非线性区也逐位一致（0.0），因为差异只来自 GEMM 数值顺序（IEEE 符号精确）。

---

## 7. 如何复现与对照

### 7.1 训练

```powershell
# 常规（默认：半噪声 + cuBLAS einsum TF32）
target\release\accumulate_train.exe --batch 60000 --accumulate 5 --rank 64 `
  --num-epochs N --mnist-dir D:\Rust\snn_t1\mnist_data

# 阶段级 GPU 计时（env 门控，默认零开销）
$env:ACC_PHASE='1'; $env:ACC_TIMING='1'

# 对照开关
$env:EINSUM='burn'      # einsum 切回 burn matmul
$env:EINSUM_FP32='1'    # cuBLAS 切回纯 fp32（默认 TF32）
```

### 7.2 正确性校验

```powershell
target\release\noise_bench.exe   # [0a]-[0d] 全部通过：
# [0a] cuBLAS 列主序小矩阵（精确）
# [0b] 半 einsum vs 全量 einsum（0.0，逐位一致）
# [0c]/[0c2] 半噪声分布
# [0e]-[0g] cuBLAS gemm/批量帮助函数
# [0d] 半前向 vs 全量前向（0.0）；cuBLAS 全量前向对照（TF32 级 1e-2）
```

### 7.3 单测

```powershell
cargo test --offline --release -p hyperscalees-noiser -p hyperscalees-models -p hyperscalees-envs
# 153 项全绿
```

### 7.4 构建注意（本机）

- 离线构建：`cargo build --offline`（本机网络受限）；
- sccache 不可用：构建前清 `RUSTC_WRAPPER`/`CARGO_BUILD_RUSTC_WRAPPER`，
  项目 `.cargo/config.toml` 的 rustc-wrapper 与本仓库 untracked 的 `rustc-shim.cmd`
  配套（构建 workaround，**不要提交**）；
- 勿用 PowerShell 直接改写含中文注释的源文件（编码损坏）；用编辑器/工具。

---

## 8. 剩余优化空间

| 方向 | 预期 | 备注 |
|---|---|---|
| 前向 base 2D GEMM（3× (96000,784)@(128,784)）换 cuBLAS `gemm_abt` | ~3–5ms | 需把前向的 base matmul 做成闭包注入（models crate 有 NoiseFn 先例） |
| LIF 融合内核（8 步 × 每步 ~7 个元素级算子） | ~5ms | 需要自定义 cube 内核或恢复轻量融合 |
| einsum 带宽优化（当前 ~4× 带宽下限） | ~10ms | cuBLAS 对 K=384000 的 split-K 行为已较好，收益递减 |
| 噪声生成内核再调优（当前 ~105GB/s 写） | ~5ms | plain 内核本身指令绑定（Box-Muller），或换 sum-of-uniforms 近似（改变分布语义，需谨慎） |
| 多卡（对标 JAX 多卡 0.119s） | 大规模 | 超出本分支范围 |

---

*文档维护：随分支演进更新；每步优化应附带同窗口 A/B 数字与本表的对照。*
