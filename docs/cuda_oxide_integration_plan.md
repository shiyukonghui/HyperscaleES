# CUDA-Oxide 集成计划：用 NVIDIA 官方 Rust→CUDA 编译器替换/补充内核层

> 分支：`perf/cuda-oxide`（基于 main，main 已含全部加速：0.15s/epoch 混合后端方案）
> 状态：**阶段 A（工具链）+ 阶段 B（最小闭环 PoC）已完成**——PRNG 噪声内核
> 已用 cuda-oxide 编写、编译出 PTX 并集成进训练热路径（默认启用，`GEN_CUBEK=1`
> 可切回 cubek-random 对照）
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

## 2. 集成架构（已落地）

```
[内核 crate]  hyperscalees-kernels/（标准 Rust 源，cuda-oxide 例子里开发）
      │  用 cuda-oxide rustc fork 构建（`cargo oxide run <example>`）
      ▼
     PTX 文本（17.7KB，经 llvm-link + opt internalize/globaldce 裁剪）
      │  编译期 include_str! 嵌入（CONCAT! 保证 NUL 结尾）
      ▼
[宿主 crate]  hyperscalees/src/oxide.rs
      │  cudarc driver API（已有依赖！）
      │    cuModuleLoadData(ptx) → cuModuleGetFunction
      │    cuLaunchKernel(…, args 数组, extra=null)   ← 同 cubecl stream，零同步
      ▼
    与 burn 张量互操作：复用 cublas.rs 的 as_cube()/raw_ptr() 机制
    （cubecl resolve → 原始设备指针 → 作为内核参数传入）
```

**关键点（全部实测验证）**：

- **launch 顺序**：`cuLaunchKernel(f, gx,gy,gz, bx,by,bz, shared, hStream,
  kernelParams, extra)`——`kernelParams` 是参数值数组、`extra` 传 `null`；
  两者写反会报 `CUDA_ERROR_INVALID_VALUE`；
- **参数对齐**：`kernelParams` 指向的每个参数值需 8 字节对齐（`repr(C, align(8))`
  包装 u32/f32 标量参数），否则随机 `CUDA_ERROR_INVALID_VALUE`；
- **PTX 必须 NUL 结尾**（`cuModuleLoadData` 要求）：`include_str!` + `"\0"`；
- **PTX 必须裁剪**：链接全量 libdevice 后 977KB 的 PTX 会被驱动拒绝
  （`INVALID_PTX`）；`opt -passes="internalize,globaldce"` 裁掉未用符号 →
  17.7KB 可加载；
- **流绑定**：内核在 cubecl 主 stream（vendored `raw_stream`）上启动，与
  cuBLAS/训练热路径完全同流有序，零同步开销；
- **向量化写**：`#[repr(C, align(16))] struct F4([f32;4])` + `*mut F4` 写出 →
  PTX `st.global.v4.b32`；`opt -O3` 后 libdevice 数学内联为 `MUFU sqrt.approx`，
  无 `call.uni`。

### 2.1 内核与 burn 张量的数据契约

- burn 张量是 pitched（16 字节行对齐）——内核参数直接传**原始设备指针 + 实际
  strides**（与 `cublas.rs` 的 `raw_ptr`/`cube.meta.strides()` 相同做法）；
- 内核签名用裸指针 + 显式 stride 参数，不依赖 burn 内部布局；
- 所有内核在 cubecl 主 stream 上启动（与 cuBLAS 共用同一 context/stream）。
- **PRNG 内核特例**：输出是连续张量，直接按 `总元素数/128 线程` 平铺，
  每线程 128 元素（32 次 F4 向量写），无需 stride 参数。

---

## 3. 落地步骤（进度）

### 阶段 A：工具链就绪 —— ✅ 完成

1. 获取 cuda-oxide 编译器：用户提供 `cuda-oxide-0.2.1.zip`（源码包，含
   `rustc_codegen_cuda` 后端 + `cargo-oxide` 子命令 + 示例），解压到
   `cuda-oxide-0.2.1/`（**不入库**，外部源码树）；
2. 本机环境：`RUSTUP_TOOLCHAIN=nightly-2026-04-03`（rustc fork 的前端版本）、
   LLVM（`LIBCLANG_PATH`）、CUDA 12.8 Toolkit（`CUDA_HOME`）、
   `LIBNVVM_PATH=...\nvvm\bin\nvvm64_40_0.dll`（cuda-core 运行时校验用）；
3. Windows 移植补丁（本地改动，不入库）：
   - `cuda-bindings/build.rs`：`collect_lib_paths` 补 `{toolkit}/lib/x64`；
   - `cuda-core`：8 处 CUDA 12.8 bindgen 枚举 i32/u32 不匹配 → `as u32` 强转；
   - `oxide-artifacts`：宿主目标支持 `x86_64-pc-windows-msvc` → COFF 段
     （`IMAGE_SCN_CNT_INITIALIZED_DATA | MEM_READ | ALIGN_8BYTES`）；
   - 空 `ffi.lib` 占位（`llvm-ar` 生成，放 `target\debug\deps`，RUSTFLAGS
     `-L native=` 指过去）；
   - 后端 DLL 拷贝：`target\debug\deps\rustc_codegen_cuda.dll` →
     `target\debug\librustc_codegen_cuda.so`（cargo-oxide 按 `.so` 找路径，
     LoadLibrary 忽略扩展名）；
4. 验证：vecadd 示例产出 PTX 并在 driver API 下加载启动成功（3 流测试）。

### 阶段 B：最小闭环（PoC）—— ✅ 完成

1. `hyperscalees-kernels/` crate + `cuda-oxide-0.2.1/.../examples/snn_prng`：
   第一个内核 = **噪声生成**（taus+lcg+Box-Muller，等价
   `cubek-random::random_normal` 的序列构造），输出写半量反对称噪声张量
   `(n/2, r, a/b)`；
2. 宿主 `hyperscalees/src/oxide.rs`：`load_kernel`（cuModuleLoadData+
   cuModuleGetFunction）+ `launch`（cuLaunchKernel，kernelParams 数组）+ 
   `prng_normal_half(out, mean, std, device)`（种子取自 `next_seeds()`
   原子+时间混合，grid = 总线程数/256 块）；
3. `noise_bench` 新增 `[0c4] oxide_prng_check`：`Tensor::empty` +
   `oxide::prng_normal_half` → mean≈0 var≈1 断言（通过：mean=-0.001
   var=1.003）；`[0a]-[0d]` 全部原样通过；
4. 集成：`cublas.rs::gen_lora_noise_antipodal` 默认走 oxide 内核（
   `GEN_CUBEK=1` 回退 cubek-random）；训练热路径 epoch 0.17s（同步计时），
   gen 阶段 ~19-34ms，与 cubek-random 同窗口 A/B 持平（机器漂移 ±75ms
   内无显著差异）；
5. **正确性**：3 个 epoch 训练正常收敛（train_acc 0.0812→0.0817，与
   cubek 路径一致）。

### 阶段 C：按优先级替换/新增内核

| 优先级 | 内核 | 当前成本 | 状态 |
|---|---|---|---|
| 1 | 噪声生成（半量） | ~19ms/epoch | ✅ 完成（向量化写 + MUFU，≥ 现状） |
| 2 | einsum 合并 GEMM（瘦 M 长 K） | ~50ms/epoch（cuBLAS TF32） | ⬜ 待做：定制分块（共享内存 + 寄存器块，K 分片部分和），目标 > cuBLAS |
| 3 | LIF 融合 | ~8ms/epoch | ⬜ 待做：8 步 LIF 融合为 1-2 内核 |
| 4 | 泊松编码融合 | ~9ms/epoch | ⬜ 待做：Uniform+比较 1 内核 |
| 5 | 半前向 batched matmul | ~45ms/epoch（burn） | ⬜ 待做（谨慎：burn 已优于 cuBLAS，需分块设计） |

每步都走同一流程：noise_bench 等价校验 → 训练循环 ACC_PHASE 阶段计时 →
**同窗口背靠背 A/B epoch 时间**（金标准，见主文档 §6）。

### 阶段 D：收尾

- 失败的尝试保留记录；保留 `EINSUM=burn` / `EINSUM_FP32=1` / `GEN_CUBEK=1`
  等对照开关；
- 更新 `docs/snn_es_burn_gpu_optimization.md` 的架构表与优化历程。

---

## 4. 现状（本分支已落地的代码）

- `burn_impl/hyperscalees-kernels/`：内核 crate 骨架；
- `burn_impl/hyperscalees-kernels/ptx/prng_normal_half.ptx`：cuda-oxide 编译
  出的 PRNG 内核 PTX（17.7KB，`include_str!` 嵌入宿主）；
- `burn_impl/hyperscalees/src/oxide.rs`：`load_kernel` / `launch` /
  `launch_on_stream` / `kernel_function` / `prng_normal_half`（默认噪声路径）；
- `burn_impl/hyperscalees/src/cublas.rs`：`gen_lora_noise_antipodal` 默认
  走 oxide 内核（`GEN_CUBEK=1` 回退）；
- `burn_impl/hyperscalees/src/bin/noise_bench.rs`：`[0c4] oxide_prng_check`；
- `burn_impl/hyperscalees/src/bin/oxide_probe.rs`：调试用探测 bin
  （vecadd 3 流测试 / prng empty/zeros / manual vs oxide launch），保留备查；
- 内核源码：`cuda-oxide-0.2.1/.../crates/rustc-codegen-cuda/examples/snn_prng/`。

---

## 5. 风险与备选（实测更新）

| 风险 | 缓解 / 实测 |
|---|---|
| cuda-oxide 实验性，Windows 支持/内核语言特性不完整 | ✅ 已打通 Windows x86_64 全链路（见 §7 补丁清单）；语言特性对数值内核够用 |
| 编译链与 cargo workspace 集成复杂（toolchain 切换） | ✅ 已解决：独立构建脚本产出 PTX + `include_str!`，宿主不依赖特殊 toolchain |
| 内核性能不如预期 | ✅ PRNG 内核达标（向量化写 + MUFU，无 call.uni，~19ms 与 cubek 持平）；每个新内核仍走 noise_bench 等价校验 + 同窗口 A/B |
| 与 burn 张量生命周期（内存池复用）冲突 | ✅ 内核只读指针、不持有句柄；与 cuBLAS 集成同一套 resolve 机制 |
| 驱动/驱动 API 加载失败（INVALID_PTX 等） | ✅ 已踩平：NUL 结尾、internalize/globaldce 裁剪、参数 8 字节对齐、kernelParams/extra 顺序 |

---

## 6. 网络障碍（已解决）

- 本机 HTTPS 受限（GitHub/crates.io `SEC_E_NO_CREDENTIALS`），cargo 走
  `--offline`；crates.io 源码通过 **rsproxy 稀疏索引**拉取；
- cuda-oxide 编译器源码由用户提供 zip 拷贝到本机
  （`cuda-oxide-0.2.1/`，外部源码树，不入库）；
- git 走 `http.sslBackend=openssl`（schannel 不可用）。

---

## 7. Windows 构建步骤（复现手册）

### 7.1 环境变量（所有 cuda-oxide 构建）

```
RUSTUP_TOOLCHAIN=nightly-2026-04-03
LIBCLANG_PATH=C:\Users\wyl\scoop\apps\llvm\current\bin
CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
LIBNVVM_PATH=%CUDA_HOME%\nvvm\bin\nvvm64_40_0.dll
RUSTFLAGS=-L native=<cuda-oxide>\crates\rustc-codegen-cuda\target\debug\deps
RUSTC_WRAPPER=<repo>\burn_impl\rustc-shim.exe   （.cmd 版会被 cmd 8191 字符上限截断）
```

### 7.2 内核编译流水线（以 snn_prng 为例）

```
cd cuda-oxide-0.2.1\cuda-oxide-0.2.1
cargo oxide run snn_prng          # → crates\rustc-codegen-cuda\target\nvptx64-nvidia-cuda\debug\examples\snn_prng.ll
llvm-link -o snn_prng_full.ll snn_prng.ll %CUDA_HOME%\nvvm\libdevice\libdevice.10.bc
opt -passes="internalize,globaldce" -o snn_prng_trim.ll snn_prng_full.ll
opt -O3 -o snn_prng_opt.ll snn_prng_trim.ll
llc -O3 -march=nvptx64 -mcpu=sm_89 -o snn_prng_opt.ptx snn_prng_opt.ll
copy snn_prng_opt.ptx <repo>\burn_impl\hyperscalees-kernels\ptx\prng_normal_half.ptx
```

要点：`internalize,globaldce` 必须做（否则全量 libdevice PTX 977KB 被驱动拒载）；
`llc -mcpu=sm_89` 匹配 RTX 40 系（Ada）。

### 7.3 宿主重建

```
cd burn_impl
$env:RUSTC_WRAPPER=''; $env:CARGO_BUILD_RUSTC_WRAPPER=''
cargo build --offline --release -p hyperscalees --features gpu --bin noise_bench --bin accumulate_train
```

PTX 变更后重编宿主即生效（`include_str!` 嵌入）。校验：`noise_bench.exe`
的 `[0c4] oxide_prng_check`；训练收敛 sanity：
`accumulate_train.exe --batch 60000 --accumulate 5 --rank 64 --num-epochs 3
--mnist-dir D:\Rust\snn_t1\mnist_data`。

### 7.4 改了后端/运行时（cuda-core、oxide-artifacts 等）后

```
cd cuda-oxide-0.2.1\cuda-oxide-0.2.1\crates\rustc-codegen-cuda
cargo build            # 后端 DLL
copy target\debug\deps\rustc_codegen_cuda.dll target\debug\librustc_codegen_cuda.so
```

---

*更新日志：2026-08 分支建立，调研完成；阶段 A/B 完成（工具链打通 +
PRNG 内核集成默认启用）；后续每步更新。*
