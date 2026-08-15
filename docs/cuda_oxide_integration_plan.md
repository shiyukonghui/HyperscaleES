# CUDA-Oxide 集成计划：用 NVIDIA 官方 Rust→CUDA 编译器替换/补充内核层

> 分支：`perf/cuda-oxide`（基于 main，main 已含全部加速：0.15s/epoch 混合后端方案）
> 状态：**调研完成，等待工具链就绪**（本机网络受限，见 §6）
> 相关文档：`docs/snn_es_burn_gpu_optimization.md`（现有混合后端方案）

---

## 1. CUDA-Oxide 是什么

[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)（NVIDIA Labs 实验项目，
v0.1.0 / v0.2.0 社区版已发布）是一个 **Rust-to-CUDA 编译器**：

- 本质是 **rustc 的 fork**（`rustc-codegen-cuda` 代码生成后端）；
- 用**标准、惯用的 Rust** 写 SIMT GPU 内核（`#[kernel]` 风格注解），直接编译到
  **PTX**——无 DSL、无外部语言绑定、无 C/CUDA C 代码；
- 官方 [book](https://nvlabs.github.io/cuda-oxide/index.html)：
  [Writing Your First Kernel](https://nvlabs.github.io/cuda-oxide/getting-started/hello-gpu.html)、
  [Building from Source](https://nvlabs.github.io/cuda-oxide/appendix/building-from-source.html)；
- 与 [cutile-rs](https://nvlabs.github.io/cutile-rs/main/guide/interoperability.html)
  （NVIDIA 的 CUDA driver API Rust 库）有官方互操作示例（`cutile_inter_kernel`）；
- 实测/评论（[Hands-On 教程](https://www.qwe.edu.pl/tutorial/cuda-oxide-nvidia-rust-cuda-compiler/#main-content)、
  [HN 讨论](https://hn.nuxt.dev/item/48096692)）：实验性质，内核语言特性仍在收敛
  （无 std、指针/原子操作受限等），但对"数值内核"（PRNG、逐元素、归约、GEMM 分块）
  已够用。

### 与本项目现有两条内核路径的对比

| 路径 | 现状 | cuda-oxide 的定位 |
|---|---|---|
| cubecl（burn 自带） | WGPU 风格 DSL 内核（vendored `cubek-random` 等），JIT 编译 | 可替换：标准 Rust 写内核，编译期出 PTX，加载即用（省 JIT） |
| cuBLAS（cudarc） | 稠密 GEMM 用官方库（einsum 已换 cuBLAS+TF32） | 不替换 cuBLAS；只补 cuBLAS 覆盖不到的定制内核 |
| 自定义 cube 内核 | 之前自定义 einsum 内核因平铺重读失败（文档 §5.6） | 用标准 Rust + 共享内存/寄存器分块重写，可彻底控制 tiling |

**定位**：不推翻现有混合后端（burn 算子 + cuBLAS einsum 已经 0.15s/epoch），而是
**把"自定义内核"这一类从 cubecl DSL 迁移到 cuda-oxide**，覆盖：
1. 噪声生成（gen，~19ms/epoch）：taus+lcg+Box-Muller，标准 Rust 写，控制 SFU/向量化；
2. einsum 合并 GEMM（~50ms/epoch）：瘦 M 长 K 形状的定制分块内核（含共享内存），
   目标超过 cuBLAS TF32（目前 ~15-20 TFLOP/s，理论可更高）；
3. LIF 融合（~8ms/epoch）：把 8 步 × 每步 ~7 个元素级算子融成 1-2 个内核；
4. 泊松编码（~9ms/epoch）：Uniform 生成 + 比较融合。

---

## 2. 集成架构

```
[内核 crate]  hyperscalees-kernels/（标准 Rust 源）
      │  用 cuda-oxide rustc fork 构建（cargo 子命令 / 直接 rustc）
      ▼
    PTX 文本（或 cubin）
      │  编译期 include_bytes! 嵌入
      ▼
[宿主 crate]  hyperscalees/
      │  cudarc driver API（已有依赖！）
      │    CudaContext::load_module(ptx) → Arc<CudaModule>
      │    module.load_function("kernel_name") → CudaFunction
      │    f.launch(grid, block, args...)   ← 同 stream，零同步
      ▼
    与 burn 张量互操作：复用 cublas.rs 的 raw_ptr() 机制
    （cubecl resolve → 原始设备指针 → 作为内核参数传入）
```

**为什么可行（已核实）**：`cudarc 0.19.9` 的 driver API 自带
`CudaContext::load_module` / `CudaModule::load_function` / `CudaFunction::launch`
（`cudarc/src/driver/mod.rs` 文档示例，以及 `result::module::load` raw API）。
与现有 cuBLAS 集成同模式：handle 绑到 cubecl 原始 stream → 天然同流有序、零同步。

### 2.1 内核与 burn 张量的数据契约

- burn 张量是 pitched（16 字节行对齐）——内核参数直接传**原始设备指针 + 实际
  strides**（与 `cublas.rs` 的 `raw_ptr`/`cube.meta.strides()` 相同做法）；
- 内核签名用裸指针 + 显式 stride 参数，不依赖 burn 内部布局；
- 所有内核在 cubecl 主 stream 上启动（`CudaContext::load_module` 与 cuBLAS 共用
  同一 context/stream）。

---

## 3. 落地步骤

### 阶段 A：工具链就绪（阻塞项）

1. 获取 cuda-oxide 编译器（Windows x86_64）：
   - GitHub Releases（v0.2.0 起有预构建二进制）或源码构建；
   - **本机 GitHub 不可达**（schannel `SEC_E_NO_CREDENTIALS`），需在可联网机器下载
     release zip 拷贝到本机，或修复网络（见 §6）；
2. 验证：用官方 hello-gpu 例子产出 PTX；
3. 确定集成到 cargo 的方式（cuda-oxide 的 `cargo-cuda` 子命令 / `rust-toolchain`
   切换 / 独立构建脚本产出 PTX 后 include_bytes!）。

### 阶段 B：最小闭环（PoC）

1. 新建 `hyperscalees-kernels` crate：第一个内核 = **噪声生成**（taus+lcg+
   Box-Muller，等价 `cubek-random::random_normal` 的序列构造）；
2. 宿主 `hyperscalees/src/oxide.rs`：module 加载封装 + 用 `raw_ptr` 把 CubeTensor
   指针传入内核；
3. noise_bench 新增校验：cuda-oxide 内核 vs `random_normal` 分布等价
   （mean/var 粗检 + 与现有 [0c] 同判据）；
4. A/B：gen 阶段耗时（当前 ~19ms/epoch）。**目标：≥ 当前内核吞吐（~100GB/s 写）**。

### 阶段 C：按优先级替换/新增内核

| 优先级 | 内核 | 当前成本 | 目标 |
|---|---|---|---|
| 1 | 噪声生成（半量） | ~19ms/epoch | 向量化写 + 控制 SFU 指令，≥ 现状 |
| 2 | einsum 合并 GEMM（瘦 M 长 K） | ~50ms/epoch（cuBLAS TF32） | 定制分块（共享内存 + 寄存器块，K 分片部分和），目标 > cuBLAS |
| 3 | LIF 融合 | ~8ms/epoch | 8 步 LIF 融合为 1-2 内核 |
| 4 | 泊松编码融合 | ~9ms/epoch | Uniform+比较 1 内核 |
| 5 | 半前向 batched matmul | ~45ms/epoch（burn） | 批量小 GEMM 定制内核（谨慎：burn 已优于 cuBLAS，需分块设计） |

每步都走同一流程：noise_bench 等价校验 → 训练循环 ACC_PHASE 阶段计时 →
**同窗口背靠背 A/B epoch 时间**（金标准，见主文档 §6）。

### 阶段 D：收尾

- 失败的尝试保留记录；保留 `EINSUM=burn` / `EINSUM_FP32=1` 等对照开关；
- 更新 `docs/snn_es_burn_gpu_optimization.md` 的架构表与优化历程。

---

## 4. 骨架（本分支已搭）

- `burn_impl/hyperscalees-kernels/`：候选内核 crate 骨架（标准 Rust 可编译，
  内核函数带详细注释说明 cuda-oxide 映射；toolchain 就绪后加 `#[kernel]` 注解）；
- `burn_impl/hyperscalees/src/oxide.rs`：宿主侧 module 加载封装骨架
  （`OxideModule::load(ptx_bytes)` + `launch`，用 cudarc driver API；
  当前仅编译通过 + 单元测试占位，PTX 未就位时不加载）；
- 本文件：集成计划。

---

## 5. 风险与备选

| 风险 | 缓解 |
|---|---|
| cuda-oxide 实验性，Windows 支持/内核语言特性不完整 | 阶段 B 先做最小闭环验证；失败则保留现状（已 0.15s，无损失） |
| 编译链与 cargo workspace 集成复杂（toolchain 切换） | 独立构建脚本产出 PTX + `include_bytes!`，宿主不依赖特殊 toolchain |
| 内核性能不如预期 | 每个内核都有 noise_bench 等价校验 + 同窗口 A/B；不达标即弃用 |
| 与 burn 张量生命周期（内存池复用）冲突 | 内核只读指针、不持有句柄；与 cuBLAS 集成同一套 resolve 机制 |

---

## 6. 本机网络障碍（当前阻塞项）

- 本机 HTTPS 全面受限：GitHub `git ls-remote`、crates.io、rsproxy.cn 均
  `SEC_E_NO_CREDENTIALS` / 连接被关闭（cargo 一直走 `--offline`）；
- 因此 **cuda-oxide 编译器二进制/源码无法在本机直接获取**；
- 需要：在可联网机器下载
  `https://github.com/NVlabs/cuda-oxide/releases`（v0.2.0 起的 Windows 预构建），
  拷贝到本机后按 §3-阶段 A 继续；或修复本机代理/证书。

---

*更新日志：2026-08 分支建立，调研完成；阶段 B 起每步更新。*
