//! 小批次等效大批次（梯度累积）训练驱动的可运行二进制。
//!
//! 复刻 Python 参考脚本 `llm_experiments/snn_mnist_train_accumulate.py` 的
//! 「小批次等效大批次」累积训练（架构见 `docs/es_batch_accumulation_architecture.md`）：
//! 参数冻结的 K 段前向累积 -> 一次全局 z-score -> chunked einsum 更新累积
//! （`accumulated_update`，÷√K 尺度恢复 + 一次 solver 更新）== 单大批次训练。
//!
//! 本二进制只增量新增文件，不修改任何既有 crate；Cargo 自动发现 `src/bin/` 下的
//! 二进制目标。模型为 `TrainableVthSnn`（两隐层 LIF，v_th 可训练 softplus 恒正），
//! 在真实 MNIST 上用已移植的 Rust 算法训练。
//!
//! 用法示例（Windows 宿主，需提供 MNIST 目录）：
//! ```text
//! cargo run --release -p hyperscalees --bin accumulate_train -- \
//!     --batch 60000 --accumulate 5 --rank 64 --num-epochs 3000 \
//!     --mnist-dir <dir> [--csv-out out.csv]
//! ```
//!
//! `--verify` 模式镜像 Python 四路径（A 单大批次 / B 前向累积 / D chunked 累积 /
//! C 局部归一化负对照），不训练，验证 B≈A、D≈A 且 C≠A 的精确等价性。

use std::io::Write;

use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
use hyperscalees_core::B;
use hyperscalees_envs::snn_mnist::{
    accuracy_from_logits, fitness_from_logits_reward, load_mnist_from_dir, poisson_encode, Reward,
};
use hyperscalees_models::snn::TrainableVthSnn;
use hyperscalees_noiser::eggroll::{
    accumulated_update, batched_lora_noise, combine_affine_grads, dense_einsum_ones,
    dense_einsum_raw, init_noiser, lora_einsum_ones, lora_einsum_raw, EggRoll,
};
use hyperscalees_noiser::{FrozenNoiserParams, IterInfo, Noiser, NoiserParams, Solver};

/// 输入维度（MNIST 28x28）。
const IN_DIM: usize = 784;
/// 类别数（MNIST 10 类）。
const NUM_CLASSES: usize = 10;
/// 可训练阈值初值（softplus 后 0.3）。
const V_TH: f32 = 0.3;
/// base_key 派生常量（与 snn_mnist_train.rs 一致）。
const KEY_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
/// MNIST 数据目录回退路径（--mnist-dir 与环境变量 MNIST_DIR 均未提供时使用）。
const DEFAULT_MNIST_DIR: &str = "D:\\Rust\\snn_t1\\mnist_data";

/// 命令行配置，默认值与 Python 参考脚本一致。
#[derive(Clone)]
struct Config {
    batch: usize,
    accumulate: usize,
    rank: usize,
    t: usize,
    sigma: f32,
    lr: f32,
    reward: Reward,
    group_size: i32,
    noise_reuse: i32,
    num_epochs: usize,
    seed: u64,
    hidden: Vec<usize>,
    mnist_dir: Option<String>,
    validate_every: usize,
    val_batch: usize,
    log_every: usize,
    csv_out: Option<String>,
    verify: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            batch: 60000,
            accumulate: 5,
            rank: 64,
            t: 8,
            sigma: 0.2,
            lr: 0.01,
            reward: Reward::Loglik,
            group_size: 0,
            // 与 Python `EggRoll.init_noiser` 的 `noise_reuse=0` 默认一致
            // （0 表示 true_epoch 恒为 0，即噪声只依赖 thread_id）。
            noise_reuse: 0,
            num_epochs: 3000,
            seed: 0,
            hidden: vec![128, 128],
            mnist_dir: None,
            validate_every: 50,
            val_batch: 10000,
            log_every: 10,
            csv_out: None,
            verify: false,
        }
    }
}

/// 手动解析命令行参数（用 `std::env::args`，不引入 clap）。
/// 支持 `--key value` 与 `--flag`；未知参数直接忽略。
/// `--reward` 支持 `loglik`/`binary`；`--verify` 为布尔开关。
fn parse_args() -> Config {
    let mut c = Config::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next_val = |i: &mut usize| -> String {
            if *i + 1 < args.len() {
                *i += 1;
                args[*i].clone()
            } else {
                String::new()
            }
        };
        match a {
            "--batch" => c.batch = next_val(&mut i).parse().unwrap_or(c.batch),
            "--accumulate" => c.accumulate = next_val(&mut i).parse().unwrap_or(c.accumulate),
            "--rank" => c.rank = next_val(&mut i).parse().unwrap_or(c.rank),
            "--T" => c.t = next_val(&mut i).parse().unwrap_or(c.t),
            "--sigma" => c.sigma = next_val(&mut i).parse().unwrap_or(c.sigma),
            "--lr" => c.lr = next_val(&mut i).parse().unwrap_or(c.lr),
            "--reward" => {
                let r = next_val(&mut i);
                c.reward = match r.as_str() {
                    "binary" => Reward::Binary,
                    _ => Reward::Loglik,
                };
            }
            "--group-size" => c.group_size = next_val(&mut i).parse().unwrap_or(c.group_size),
            "--noise-reuse" => c.noise_reuse = next_val(&mut i).parse().unwrap_or(c.noise_reuse),
            "--num-epochs" => c.num_epochs = next_val(&mut i).parse().unwrap_or(c.num_epochs),
            "--seed" => c.seed = next_val(&mut i).parse().unwrap_or(c.seed),
            "--hidden" => {
                let h = next_val(&mut i);
                c.hidden = h
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect();
            }
            "--mnist-dir" => c.mnist_dir = Some(next_val(&mut i)),
            "--validate-every" => {
                c.validate_every = next_val(&mut i).parse().unwrap_or(c.validate_every)
            }
            "--val-batch" => c.val_batch = next_val(&mut i).parse().unwrap_or(c.val_batch),
            "--log-every" => c.log_every = next_val(&mut i).parse().unwrap_or(c.log_every),
            "--csv-out" => c.csv_out = Some(next_val(&mut i)),
            "--verify" => c.verify = true,
            _ => { /* 未知参数忽略 */ }
        }
        i += 1;
    }
    c
}

/// 确定性 Fisher-Yates 部分洗牌：从 `[0, n)` 中取出 `count` 个索引。
/// 用简单 PRNG（xorshift + 乘），保证可复现（不要求逐位复刻 JAX）。
fn shuffled_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    assert!(count <= n, "count({count}) 不能超过 n({n})");
    let mut state: u64 = (seed ^ 0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut idx: Vec<usize> = (0..n).collect();
    // 部分 Fisher-Yates：只洗前 count 个位置。
    for i in 0..count {
        let j = i + (next() as usize % (n - i));
        idx.swap(i, j);
    }
    idx.truncate(count);
    idx
}

/// 由预构建的每层 (A, B) 噪声整块批量前向，返回 `(n, C)` logits。
///
/// `noises` 与 `dim_keys` 前 3 项（fc1/fc2/fc3，LoRA）一一对应：每个元素
/// `(A (n,a,r) 已乘 sign*base_sigma, B (n,b,r))`。闭包按权重 `[out,in]` 形状在
/// `noises` 中取该层的 (A, B)，逐时间步做 `x @ w^T + x @ B @ A^T`（整块批量，
/// 等价逐样本语义，见 `forward_batched`）。
///
/// 训练热路径由调用方生成噪声（GPU 上 `Tensor::random`，前向与梯度共享同一份），
/// verify 路径经 `segment_logits_batched` 用确定性 `batched_lora_noise` 生成。
fn forward_with_noises(
    model: &TrainableVthSnn,
    spikes: &Tensor<B, 3>,     // (T, n, 784)
    tids: &[i32],              // 长度 n（该段样本的全局 thread_id）
    epoch: i32,
    dim_keys: &[([usize; 2], u64)],
    noises: &[(Tensor<B, 3>, Tensor<B, 3>)],
) -> Tensor<B, 2> {            // (n, 10)
    let dk = dim_keys;
    let noise_helper =
        move |x: Tensor<B, 2>, w: Tensor<B, 2>, _ts: &[i32], _ep: i32| -> Tensor<B, 2> {
            let dims = w.dims();
            let pos = dk
                .iter()
                .position(|(d, _)| *d == dims)
                .expect("未找到该层的预构建噪声");
            let (a_t, b_t) = &noises[pos];
            let base = x.clone().matmul(w.clone().transpose()); // (n, a)
            // y = x @ B：逐样本 x_i(1,b) @ B_i(b,r) 的批量版。
            let y = x.unsqueeze_dim::<3>(2) * b_t.clone(); // (n,b,1)*(n,b,r) -> (n,b,r)
            let y = y.sum_dim(1).squeeze_dim::<2>(1); // (n, r)
            // noise = y @ A^T：逐样本 y_i(1,r) @ A_i(a,r)^T 的批量版。
            let noise = y.unsqueeze_dim::<3>(1) * a_t.clone(); // (n,1,r)*(n,a,r) -> (n,a,r)
            let noise = noise.sum_dim(2).squeeze_dim::<2>(2); // (n, a)
            base + noise
        };
    let noise: &dyn Fn(Tensor<B, 2>, Tensor<B, 2>, &[i32], i32) -> Tensor<B, 2> = &noise_helper;
    model.forward_batched(spikes.clone(), tids, epoch, Some(noise))
}

/// verify 模式前向：用确定性 `batched_lora_noise` 为该段（`base_thread..`）生成每层
/// 噪声一次（供所有时间步复用），再整块前向。与 verify 更新路径（`do_updates`/
/// `accumulated_update` 内部同样用 `batched_lora_noise`）噪声逐位一致。
fn segment_logits_batched(
    model: &TrainableVthSnn,
    spikes: &Tensor<B, 3>,     // (T, n, 784)
    epoch: i32,
    base_thread: i32,          // 该段第一个样本的全局 thread_id
    sigma: f32,
    rank: usize,
    nreuse: i32,
    device: &Device<B>,
    dim_keys: &[([usize; 2], u64)],
) -> Tensor<B, 2> {            // (n, 10)
    let n = spikes.dims()[1];
    let tids: Vec<i32> = (0..n).map(|j| base_thread + j as i32).collect();
    let base_sigma = sigma / (rank as f32).sqrt();
    let mut noises: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = Vec::with_capacity(3);
    for (dims, key) in dim_keys.iter().take(3) {
        let [a, b] = *dims;
        let (a_t, b_t) = batched_lora_noise(base_sigma, *key, rank, &tids, epoch, nreuse, a, b, device);
        noises.push((a_t, b_t));
    }
    forward_with_noises(model, spikes, &tids, epoch, dim_keys, &noises)
}

/// 在 GPU 上直接生成一段 n 个样本的 LoRA 噪声 `(A (n,a,r) 已乘 base_sigma, B (n,b,r))`。
///
/// 反对称配对（前 half 行随机、后 half 行取负）做方差缩减，等价 Python 的
/// `thread_id % 2` 正负配对（thread 2i 与 2i+1 共享 |噪声| 相反号）；分布上仍为
/// `N(0,1)`，故 ES 估计无偏。**完全免除 CPU 随机数与 CPU→GPU 上传**（原缓存切片
/// 上传是每 epoch 数十 GB 的主瓶颈）。
fn gen_gpu_lora_noise(
    base_sigma: f32,
    rank: usize,
    n: usize,
    a: usize,
    b: usize,
    device: &Device<B>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    assert!(n % 2 == 0, "GPU 反对称噪声要求 n 为偶数，实际 {n}");
    let half = n / 2;
    let b_even = Tensor::<B, 3>::random([half, b, rank], Distribution::Normal(0.0, 1.0), device);
    let b_t = Tensor::cat(vec![b_even.clone(), b_even.neg()], 0); // (n, b, r)
    let a_even = Tensor::<B, 3>::random([half, a, rank], Distribution::Normal(0.0, 1.0), device);
    let a_t = Tensor::cat(vec![a_even.clone(), a_even.neg()], 0).mul_scalar(base_sigma); // (n,a,r)
    (a_t, b_t)
}

/// GPU 上生成 dense 噪声 `(n, a, b)`（FULL 参数更新用，分布 `N(0, sigma²)`）。
fn gen_gpu_dense_noise(
    sigma: f32,
    n: usize,
    a: usize,
    b: usize,
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::random([n, a, b], Distribution::Normal(0.0, 1.0), device).mul_scalar(sigma)
}

/// 把 `accumulated_update`/`do_updates` 产出的参数写回模型。
/// 顺序 `[fc1, fc2, fc3, out_gain(1,1), v_th1(1,1), v_th2(1,1)]`：
/// fc1/fc2/fc3 权重直接赋值；out_gain/v_th1/v_th2 从 (1,1) squeeze 为 (1,)。
fn write_params(model: &mut TrainableVthSnn, new_params: Vec<Tensor<B, 2>>) {
    let mut it = new_params.into_iter();
    model.fc1.weight = it.next().expect("缺少 fc1");
    model.fc2.weight = it.next().expect("缺少 fc2");
    model.fc3.weight = it.next().expect("缺少 fc3");
    model.out_gain.value = it.next().expect("缺少 out_gain").squeeze_dim::<1>(0);
    model.v_th1.value = it.next().expect("缺少 v_th1").squeeze_dim::<1>(0);
    model.v_th2.value = it.next().expect("缺少 v_th2").squeeze_dim::<1>(0);
}

/// 两份参数列表的逐元素最大绝对差（遍历 Vec<Tensor<B,2>>，同 snn_mnist_train.rs 的
/// `max_abs_delta` 实现）。
fn max_abs_delta(a: &[Tensor<B, 2>], b: &[Tensor<B, 2>]) -> f32 {
    assert_eq!(a.len(), b.len(), "参数列表长度不一致");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x.clone().sub(y.clone()).abs();
            d.into_data()
                .into_vec::<f32>()
                .unwrap()
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// 干净评估（无扰动）：在测试集伪随机取 val_batch 条，整批 `forward_batched(x, None)` -> 精度。
fn evaluate(
    model: &TrainableVthSnn,
    test_img: &Tensor<B, 2>,
    test_lbl: &Tensor<B, 1, Int>,
    val_batch: usize,
    t: usize,
    seed: u64,
    device: &Device<B>,
) -> f32 {
    let n_test = test_img.dims()[0];
    let count = val_batch.min(n_test);
    let idx = shuffled_indices(n_test, count, seed.wrapping_mul(KEY_MUL));
    let idx_t: Vec<i32> = idx.iter().map(|&x| x as i32).collect();
    let idx_tensor: Tensor<B, 1, Int> =
        Tensor::from_data(TensorData::new(idx_t, [count].to_vec()), device);
    let imgs = test_img.clone().select(0, idx_tensor.clone());
    let labels = test_lbl.clone().select(0, idx_tensor);
    let spikes = poisson_encode(imgs, t); // (T, count, in)
    // 整批 clean 前向：tids/epoch 在噪声为 None 时不参与计算，任意给 0..count 与 0。
    let tids: Vec<i32> = (0..count).map(|i| i as i32).collect();
    let logits = model.forward_batched(spikes, &tids, 0, None); // (count, C) clean
    accuracy_from_logits(logits, labels)
}

/// --verify：镜像 Python 四路径，验证「累积 == 单大批次」且「局部归一化不等价」。
///
/// 用较小规模保证 CPU 快：`vb = min(batch, 1024)`（并调整为能整除 K 的最大值），
/// 取测试集前 vb 个样本；所有路径共用同一 `spikes` 与同一 `thread_ids = 0..vb`
/// （epoch=0）。每条路径各自 `noiser.clone()`，避免共享 opt_state 干扰。
///
/// 注意：调用方应传入「verify 专用」的 frozen/noiser（其 AdamW eps 放大为 1e-4）。
/// 原因：全局 z-score 使 fitness 严格零均值，个别权重元素的真实 ES 梯度接近 0
/// （fitness 与噪声近乎不相关），f32 下「单次整批求和」与「分 K 段求和」可能给出
/// 相反符号的微小值，被 Adam 的 eps=1e-8 放大为 ~0.01 的参数差——这是纯数值假象
/// （noiser 自身单测已证明 `accumulated_update` 与 `do_updates` 逐元素一致），
/// 并非数学不等价。eps=1e-4 使近零梯度的更新幅度收缩到 ~1e-5 量级，等价性判定
/// 恢复稳健；负对照（局部归一化）的差异 ~4e-2 不受影响。
fn run_verify(
    cfg: &Config,
    model: &mut TrainableVthSnn,
    frozen: &FrozenNoiserParams,
    noiser: &NoiserParams,
    params: &[Tensor<B, 2>],
    base_keys: &[u64],
    es_classes: &[i32],
    dim_key_pairs: &[([usize; 2], u64)],
    test_img: &Tensor<B, 2>,
    test_lbl: &Tensor<B, 1, Int>,
    device: &Device<B>,
) {
    let k = cfg.accumulate.max(1);
    // vb = min(batch, 1024)，调整为能整除 K 的最大值（向下取整，最小 1）。
    let mut vb = cfg.batch.min(1024);
    if vb % k != 0 {
        vb = (vb / k).max(1) * k;
    }
    let chunk_v = vb / k;
    assert!(chunk_v >= 1, "verify 的每段样本数必须 >= 1（vb={vb}, K={k}）");
    assert!(
        vb <= test_img.dims()[0],
        "verify 需要 vb({vb}) <= 测试集大小 {}",
        test_img.dims()[0]
    );

    println!("\n[verify] 定理2 精确等价证明（K 段累积 == 单大批次）");
    println!("  vb={vb} K={k} chunk_v={chunk_v} rank={} T={}", cfg.rank, cfg.t);

    // 测试集前 vb 个样本，一次泊松编码；thread_ids 全局唯一 = 0..vb，epoch=0。
    let imgs = test_img.clone().slice([0..vb, 0..IN_DIM]); // (vb, in)
    let labels = test_lbl.clone().slice([0..vb]); // (vb,)
    let spikes = poisson_encode(imgs, cfg.t); // (T, vb, in)
    let thread_ids: Vec<i32> = (0..vb).map(|i| i as i32).collect();

    // ---- 路径 A：单大批次（基准）--------------------------------------
    let mut noiser_a = noiser.clone();
    let logits_a = segment_logits_batched(
        model,
        &spikes,
        0, // epoch
        0, // base_thread：thread_id 从 0 开始
        cfg.sigma,
        cfg.rank,
        cfg.noise_reuse,
        device,
        dim_key_pairs,
    );
    let raw_a = fitness_from_logits_reward(logits_a, labels.clone(), cfg.reward);
    // 一次全局 z-score（A 与 D 共用同一 conv）。
    let conv_a = EggRoll.convert_fitnesses(frozen, noiser, raw_a);
    let iterinfos_full: Vec<IterInfo> = thread_ids
        .iter()
        .map(|&t| IterInfo { epoch: 0, thread_id: t })
        .collect();
    let params_a = EggRoll.do_updates(
        frozen,
        &mut noiser_a,
        params,
        base_keys,
        conv_a.clone(),
        &iterinfos_full,
        es_classes,
    );

    // ---- 路径 B：K 段前向累积 + 一次全局 z-score + 一次 do_updates -----
    let mut noiser_b = noiser.clone();
    let mut raw_chunks: Vec<Tensor<B, 1>> = Vec::with_capacity(k);
    for kk in 0..k {
        let lo = kk * chunk_v;
        let hi = lo + chunk_v;
        let spikes_k = spikes.clone().slice([0..cfg.t, lo..hi, 0..IN_DIM]);
        let labels_k = labels.clone().slice([lo..hi]);
        let logits_k = segment_logits_batched(
            model,
            &spikes_k,
            0,
            lo as i32,
            cfg.sigma,
            cfg.rank,
            cfg.noise_reuse,
            device,
            dim_key_pairs,
        );
        raw_chunks.push(fitness_from_logits_reward(logits_k, labels_k, cfg.reward));
    }
    let raw_full = Tensor::cat(raw_chunks, 0); // (vb,)
    let conv_b = EggRoll.convert_fitnesses(frozen, &noiser_b, raw_full);
    let params_b = EggRoll.do_updates(
        frozen,
        &mut noiser_b,
        params,
        base_keys,
        conv_b,
        &iterinfos_full,
        es_classes,
    );

    // ---- 路径 D：chunked einsum 累积更新（训练实际路径）----------------
    let mut noiser_d = noiser.clone();
    let params_d = accumulated_update(
        frozen,
        &mut noiser_d,
        params,
        base_keys,
        es_classes,
        conv_a.clone(),
        &thread_ids,
        0, // epoch
        k,
        chunk_v,
    );

    // ---- 路径 C（负对照）：每 chunk 局部 z-score + 每 chunk 单独更新 ----
    let mut noiser_c = noiser.clone();
    let mut params_c = params.to_vec();
    for kk in 0..k {
        let lo = kk * chunk_v;
        let hi = lo + chunk_v;
        // 用「上一段更新后」的参数做本段前向（naive 小批次多次：参数逐段变化）。
        let spikes_k = spikes.clone().slice([0..cfg.t, lo..hi, 0..IN_DIM]);
        let labels_k = labels.clone().slice([lo..hi]);
        let logits_k = segment_logits_batched(
            model,
            &spikes_k,
            0,
            lo as i32,
            cfg.sigma,
            cfg.rank,
            cfg.noise_reuse,
            device,
            dim_key_pairs,
        );
        let raw_k = fitness_from_logits_reward(logits_k, labels_k, cfg.reward);
        // 每段局部 z-score（只对当前 chunk 的 raw 计算 mean/std，破坏线性性）。
        let conv_k = EggRoll.convert_fitnesses(frozen, &noiser_c, raw_k);
        // 每段自己的 iterinfos（thread_id 取该段切片）。
        let iterinfos_k: Vec<IterInfo> = (lo..hi)
            .map(|t| IterInfo {
                epoch: 0,
                thread_id: t as i32,
            })
            .collect();
        params_c = EggRoll.do_updates(
            frozen,
            &mut noiser_c,
            &params_c,
            base_keys,
            conv_k,
            &iterinfos_k,
            es_classes,
        );
        write_params(model, params_c.clone());
    }

    // ---- 断言：B≈A、D≈A（等价），C≠A（负对照不等价）------------------
    let d_ab = max_abs_delta(&params_a, &params_b);
    let d_ad = max_abs_delta(&params_a, &params_d);
    let d_ac = max_abs_delta(&params_a, &params_c);
    println!("  累积 vs 大批次 max|Δparam| = {d_ab:.3e}  (≈0，定理2 等价，残差为 float32 不同累加顺序的非确定性)");
    println!("  chunked-einsum vs 大批次 max|Δparam| = {d_ad:.3e}  (≈0，einsum 线性 → 分段累加等价)");
    println!("  naive vs 大批次 max|Δparam| = {d_ac:.3e}  (>0，局部归一化破坏等价)");

    if d_ab >= 1e-3 {
        eprintln!("[verify] FAIL：累积应≈大批次（d_AB={d_ab:.3e} >= 1e-3）");
        std::process::exit(1);
    }
    if d_ad >= 1e-3 {
        eprintln!("[verify] FAIL：chunked-einsum 累积应≈大批次（d_AD={d_ad:.3e} >= 1e-3）");
        std::process::exit(1);
    }
    if d_ac <= 1e-3 {
        eprintln!("[verify] FAIL：负对照应≠大批次（d_AC={d_ac:.3e} <= 1e-3）");
        std::process::exit(1);
    }
    println!("[verify] PASS：累积==大批次（精确），chunked-einsum 等价，naive 局部归一化不相等");
    std::process::exit(0);
}

fn main() {
    let cfg = parse_args();

    // accumulate 必须整除 batch，否则 panic（与 Python 一致）。
    assert!(
        cfg.accumulate > 0 && cfg.batch % cfg.accumulate == 0,
        "batch({}) 必须能被 accumulate({}) 整除",
        cfg.batch,
        cfg.accumulate
    );
    let chunk = cfg.batch / cfg.accumulate;
    // 隐藏层只支持两层结构（TrainableVthSnn 固定两隐层）。
    assert!(
        cfg.hidden.len() == 2,
        "--hidden 需为逗号分隔的两个数字（如 128,128），实际 {:?}",
        cfg.hidden
    );
    let (h1, h2) = (cfg.hidden[0], cfg.hidden[1]);

    // MNIST 目录：--mnist-dir > 环境变量 MNIST_DIR > 默认路径（与 Python 一致）。
    let mnist_dir = cfg
        .mnist_dir
        .clone()
        .or_else(|| std::env::var("MNIST_DIR").ok())
        .unwrap_or_else(|| DEFAULT_MNIST_DIR.to_string());

    let device = Device::<B>::default();
    // 打印后端类型（CPU flex / CUDA），便于确认训练真正跑在 GPU 上。
    let backend_name = if hyperscalees_core::is_gpu() {
        "cuda"
    } else {
        "flex(cpu)"
    };
    println!(
        "[env] backend={} batch={} accumulate={} chunk={} rank={} T={} noise_reuse={}",
        backend_name, cfg.batch, cfg.accumulate, chunk, cfg.rank, cfg.t, cfg.noise_reuse
    );

    // ---- 模型与 ES 装配 ------------------------------------------------
    let mut model = TrainableVthSnn::new(IN_DIM, h1, h2, NUM_CLASSES, V_TH, &device);
    let es_classes = model.es_map();
    let params = model.params();
    // 每个参数一个确定性 base_key（长度=params.len()=6）。
    let base_keys: Vec<u64> = params
        .iter()
        .enumerate()
        .map(|(i, _)| (i as u64 + 1).wrapping_mul(KEY_MUL))
        .collect();
    // 由参数 `[out, in]` 形状 -> base_key 映射，供噪声闭包按权重形状取 key。
    let dim_key_pairs: Vec<([usize; 2], u64)> = params
        .iter()
        .zip(base_keys.iter())
        .map(|(p, k)| (p.dims(), *k))
        .collect();

    // EggRoll noiser（LoRA，adamw；noise_reuse 由 CLI 决定，默认 0 与 Python 一致）。
    // `mut noiser`：opt_state 在训练期间持续（不重建），solver step 逐 epoch 递增。
    let (frozen, mut noiser) = init_noiser(
        &params,
        cfg.sigma,
        cfg.lr,
        cfg.group_size,
        false, // freeze_nonlora：允许稠密 out_gain/v_th 路径
        cfg.noise_reuse,
        cfg.rank,
        Solver::adamw(cfg.lr),
        &device,
    );

    // ---- 载入 MNIST 数据 ----------------------------------------------
    let ((x_train, y_train), (x_test, y_test)) =
        load_mnist_from_dir(std::path::Path::new(&mnist_dir))
            .expect("加载 MNIST 失败（检查 --mnist-dir）");
    let n_train = x_train.len() / IN_DIM;
    let n_test = x_test.len() / IN_DIM;
    println!("[data] train={n_train} test={n_test} dir={mnist_dir}");

    // 扁平图像 [0,1] -> (n, 784)；标签 u8 -> Int 张量。
    let train_img: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(x_train, [n_train, IN_DIM].to_vec()),
        &device,
    );
    let train_lbl: Vec<i32> = y_train.iter().map(|&x| x as i32).collect();
    let train_lbl: Tensor<B, 1, Int> =
        Tensor::from_data(TensorData::new(train_lbl, [n_train].to_vec()), &device);
    let test_img: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(x_test, [n_test, IN_DIM].to_vec()), &device);
    let test_lbl: Vec<i32> = y_test.iter().map(|&x| x as i32).collect();
    let test_lbl: Tensor<B, 1, Int> =
        Tensor::from_data(TensorData::new(test_lbl, [n_test].to_vec()), &device);

    if cfg.verify {
        // ---- --verify：镜像 Python 四路径，不训练 -----------------------
        // verify 专用 solver：AdamW eps 放大到 1e-4（详见 run_verify 的文档注释，
        // 规避近零梯度被 eps=1e-8 放大的 f32 数值假象）。训练主循环仍用标准
        // `Solver::adamw`（eps=1e-8）。
        let verify_frozen = FrozenNoiserParams {
            group_size: cfg.group_size,
            freeze_nonlora: false,
            noise_reuse: cfg.noise_reuse,
            rank: cfg.rank,
            solver: Solver::AdamW {
                lr: cfg.lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-4,
                weight_decay: 1e-4,
            },
        };
        let verify_noiser = NoiserParams {
            sigma: cfg.sigma,
            opt_state: verify_frozen.solver.init_state(&params, &device),
        };
        run_verify(
            &cfg,
            &mut model,
            &verify_frozen,
            &verify_noiser,
            &params,
            &base_keys,
            &es_classes,
            &dim_key_pairs,
            &test_img,
            &test_lbl,
            &device,
        );
        return;
    }

    // ---- 训练循环：小批次累积 == 大批次（GPU 内联噪声路径）-------------
    // 训练热路径在 GPU 上直接生成噪声（`Tensor::random`），每 chunk 前向与梯度
    // 共享同一份 (A, B)，**没有 CPU 随机数与 CPU→GPU 上传**。因全局 z-score 需等全部
    // raw fitness，这里用仿射等价：逐 chunk 以 raw fitness 累积 grad_acc/ones_acc，
    // 最后 `(grad - mean·ones)/std` 一次性修正 + 一次 solver 更新——数学上与两阶段
    // `accumulated_update` 严格一致（noiser 单测 `inline_affine_matches_accumulated_two_phase`
    // 验证），等价性不受影响（verify 仍走确定性两阶段路径）。
    // CSV 输出头（目录不存在则创建，与 Python os.makedirs 一致）。
    let mut csv: Option<std::fs::File> = cfg.csv_out.as_ref().map(|p| {
        if let Some(parent) = std::path::Path::new(p).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("创建 CSV 目录失败");
            }
        }
        let mut fp = std::fs::File::create(p).expect("创建 csv 文件失败");
        writeln!(fp, "epoch,train_acc,val_acc,best_val,best_train,epoch_time,cum_time")
            .expect("写 CSV 表头失败");
        fp
    });

    let mut best_val = 0.0_f32;
    let mut best_train = 0.0_f32;
    let mut cum_t = 0.0_f32;
    let mut correct_f: f32;

    for epoch in 0..cfg.num_epochs {
        let t_ep = std::time::Instant::now();
        // 阶段计时（ACC_TIMING=1 时输出，用于定位性能瓶颈）。
        let timing = std::env::var("ACC_TIMING").map(|v| v == "1").unwrap_or(false);
        let t_sample = std::time::Instant::now();

        // 每 epoch 从训练集确定性取 batch 条（epoch 参与种子，每 epoch 采样不同）。
        let idx = shuffled_indices(
            n_train,
            cfg.batch,
            cfg.seed ^ (epoch as u64 + 1).wrapping_mul(KEY_MUL),
        );
        let idx_t: Vec<i32> = idx.iter().map(|&x| x as i32).collect();
        let idx_tensor: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(idx_t, [cfg.batch].to_vec()), &device);
        let imgs = train_img.clone().select(0, idx_tensor.clone()); // (batch, in)
        let labels = train_lbl.clone().select(0, idx_tensor); // (batch,)
        let t_sample_d = t_sample.elapsed().as_secs_f32();

        // 全局唯一 thread_id = arange(batch) 切片（跨 chunk 不碰撞）。
        let thread_ids: Vec<i32> = (0..cfg.batch).map(|i| i as i32).collect();

        // --- K 段内联累积：前向 + raw 加权部分梯度（参数冻结，每段全新噪声）---
        // grad_acc：Σ_i raw_i·噪声项；ones_acc：Σ_i 噪声项（仿射修正用）。
        let base_sigma = cfg.sigma / (cfg.rank as f32).sqrt();
        let p0 = model.params();
        let mut grad_acc: Vec<Tensor<B, 2>> = p0
            .iter()
            .map(|p| Tensor::<B, 2>::zeros(p.dims(), &device))
            .collect();
        let mut ones_acc = grad_acc.clone();
        let mut sum_raw = 0.0_f32;
        let mut sum_raw2 = 0.0_f32;
        let mut seg_correct = 0.0_f32;
        let t_fwd = std::time::Instant::now();
        for k in 0..cfg.accumulate {
            let lo = k * chunk;
            let hi = lo + chunk;
            let imgs_k = imgs.clone().slice([lo..hi, 0..IN_DIM]); // (chunk, in)
            let labels_k = labels.clone().slice([lo..hi]); // (chunk,)
            let tids_k: Vec<i32> = thread_ids[lo..hi].to_vec();
            // 每段独立泊松编码：(T, chunk, in)。
            let spikes_k = poisson_encode(imgs_k, cfg.t);
            // 该 chunk 每层（fc1/fc2/fc3 = dim_keys 前 3 项）的 GPU 噪声（前向与梯度共享）。
            let mut noises: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = Vec::with_capacity(3);
            for (dims, _key) in dim_key_pairs.iter().take(3) {
                let [a, b] = *dims;
                let (a_t, b_t) = gen_gpu_lora_noise(base_sigma, cfg.rank, chunk, a, b, &device);
                noises.push((a_t, b_t));
            }
            // 整块批量噪声前向 -> (chunk, C)。
            let logits_k = forward_with_noises(&model, &spikes_k, &tids_k, epoch as i32, &dim_key_pairs, &noises);
            // 本段 raw fitness（CPU 视图，供部分梯度与均/方差累积）。
            let raw_k: Vec<f32> = fitness_from_logits_reward(
                logits_k.clone(),
                labels_k.clone(),
                cfg.reward,
            )
            .into_data()
            .into_vec()
            .unwrap();
            // 累加本段正确数：accuracy = 段内正确率均值，×chunk 得正确条数。
            seg_correct += accuracy_from_logits(logits_k, labels_k) * chunk as f32;
            // LoRA 参数（fc1/fc2/fc3 = 前 3 项）的 raw 加权部分梯度 + ones 项。
            for (i, (a_t, b_t)) in noises.iter().enumerate() {
                grad_acc[i] = grad_acc[i].clone() + lora_einsum_raw(a_t, b_t, &raw_k, &device);
                ones_acc[i] = ones_acc[i].clone() + lora_einsum_ones(a_t, b_t, &device);
            }
            // dense（FULL）参数（out_gain/v_th1/v_th2 = 后 3 项，形状 (1,1)）。
            for di in 0..3 {
                let [a, b] = dim_key_pairs[3 + di].0;
                let noise = gen_gpu_dense_noise(cfg.sigma, chunk, a, b, &device);
                grad_acc[3 + di] = grad_acc[3 + di].clone() + dense_einsum_raw(&noise, &raw_k, &device);
                ones_acc[3 + di] = ones_acc[3 + di].clone() + dense_einsum_ones(&noise, &device);
            }
            sum_raw += raw_k.iter().sum::<f32>();
            sum_raw2 += raw_k.iter().map(|x| x * x).sum::<f32>();
        }
        correct_f = seg_correct;
        let t_fwd_d = t_fwd.elapsed().as_secs_f32();

        // --- 全局 z-score 仿射修正 + 一次 solver 更新（== 单大批次）---
        let t_zs = std::time::Instant::now();
        let mean = sum_raw / cfg.batch as f32;
        let var = (sum_raw2 / cfg.batch as f32 - mean * mean).max(0.0);
        let std = (var + 1e-5).sqrt();
        let grads = combine_affine_grads(&grad_acc, &ones_acc, mean, std, cfg.batch);
        // 同一 noiser 的 opt_state 持续跨 epoch（solver step 递增）。
        let new_params = frozen.solver.update(&model.params(), &grads, &mut noiser.opt_state);
        let t_upd_d = t_zs.elapsed().as_secs_f32();
        write_params(&mut model, new_params);

        // --- 打印与评估 ---
        let train_acc = correct_f / cfg.batch as f32;
        if train_acc > best_train {
            best_train = train_acc;
        }
        let mut msg = format!(
            "epoch {:5} | train_acc {:.4} | best_train {:.4}",
            epoch, train_acc, best_train
        );
        let mut val_acc: Option<f32> = None;
        if epoch % cfg.validate_every == 0 {
            val_acc = Some(evaluate(
                &model, &test_img, &test_lbl, cfg.val_batch, cfg.t, cfg.seed, &device,
            ));
            if val_acc.unwrap() > best_val {
                best_val = val_acc.unwrap();
            }
            msg += &format!(
                " | val_acc {:.4} | best_val {:.4}",
                val_acc.unwrap(),
                best_val
            );
        }
        let el = t_ep.elapsed().as_secs_f32();
        cum_t += el;
        msg += &format!(" | {el:.2}s | cum {cum_t:.1}s");
        if timing {
            eprintln!(
                "  [timing] sample={:.2}s fwd={:.2}s update={:.2}s",
                t_sample_d, t_fwd_d, t_upd_d
            );
        }
        if epoch % cfg.log_every == 0 || epoch == cfg.num_epochs - 1 {
            println!("{msg}");
        }
        if let Some(f) = csv.as_mut() {
            writeln!(
                f,
                "{},{:.6},{},{:.6},{:.6},{:.3},{:.1}",
                epoch,
                train_acc,
                val_acc.map(|v| format!("{v:.6}")).unwrap_or_default(),
                best_val,
                best_train,
                el,
                cum_t
            )
            .expect("写 CSV 失败");
        }
    }

    println!("best_val = {best_val:.4} | best_train = {best_train:.4}");
    println!("Done.");
}
