//! 可训练的 SNN Transformer 训练二进制（HyperscaleES 演化策略）。
//!
//! 把 [`SnnTransformer`]（多头 + 位置编码 + 多块残差的连续注意力版逐 token 脉冲
//! Transformer，见 `hyperscalees-models/src/snn_transformer.rs`）接入 Rust 已迁移的
//! ES 训练框架，在 patched-MNIST 上训练：
//!
//! ```text
//!   x: (n, 784)  像素
//!     -> patched 成 16 个 7x7 token（patch_px=7，num_tokens=16，token_in=49）
//!     -> poisson 编码 (T, n, 784)（784 = num_tokens*token_in）
//!     -> 演化策略：对每个可训练参数注入 LoRA（MM 权重）/ 稠密（PARAM/EMB_PARAM）
//!        逐样本噪声，前向求 logits，loglik 奖励，收益加权梯度 + 仿射修正
//!        （= 全局 z-score + solver 更新，等价单大批次）
//! ```
//!
//! 与 `accumulate_train.rs` 同构（小批次等效大批次累积），但模型为 Transformer，
//! 噪声按**参数索引**（而非权重形状）寻址，处理多头 q/k/v/o/ff 权重形状重复的多义性。
//!
//! 用法（同一二进制，CPU flex 默认 / `--features gpu` 跑 CUDA）：
//! ```text
//! # CPU 冒烟：
//! cargo run --release -p hyperscalees --bin snn_transformer_train -- \
//!     --batch 2048 --accumulate 4 --rank 16 --d-model 32 --num-heads 4 \
//!     --num-blocks 2 --num-epochs 200 --mnist-dir <dir> [--csv-out out.csv]
//! # GPU（RTX 4090 等）：加 --features gpu，开更大 d_model / 全批 60000 / 数千 epoch
//! #   可看到显著精度（量级对齐 Python 连续 snn_attention 的 ~58%）：
//! cargo run --release -p hyperscalees --features gpu --bin snn_transformer_train -- \
//!     --batch 60000 --accumulate 8 --rank 16 --T 8 --d-model 96 --num-heads 6 \
//!     --num-blocks 3 --num-epochs 2000 --mnist-dir <dir> [--csv-out out_gpu.csv]
//! ```
//! 模型（`SnnTransformer`）与 ES 全路径只用泛型 burn 算子（matmul/softmax/sigmoid/exp
//! 等），故同一份代码在 flex(CPU) 与 cuda(GPU) 后端下行为一致；`[env] backend=`
//! 行会如实打印当前后端（cpu/cuda），确认训练是否真正跑在 GPU 上。

use std::io::Write;

use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
use hyperscalees_core::B;
use hyperscalees_envs::snn_mnist::{
    accuracy_from_logits, fitness_from_logits_reward, load_mnist_from_dir, poisson_encode, Reward,
};
use hyperscalees_models::common::MM_PARAM;
use hyperscalees_models::snn_transformer::{SnnTransformer, TrainNoise};
use hyperscalees_noiser::eggroll::{batched_lora_noise, combine_affine_grads, init_noiser, lora_einsum_pair};
use hyperscalees_noiser::Solver;

/// 输入维度（MNIST 28x28）。
const IN_DIM: usize = 784;
/// patch 边长（28 // patch_px 个 token）。
const PATCH_PX: usize = 7;
/// token 数（4x4 = 16）。
const NUM_TOKENS: usize = (28 / PATCH_PX) * (28 / PATCH_PX);
/// 每 token 维度（7x7 = 49）。
const TOKEN_DIM: usize = PATCH_PX * PATCH_PX;
/// 类别数（MNIST 10 类）。
const NUM_CLASSES: usize = 10;
/// 可训练 beta 初始（softplus 后 = 1/sqrt(head_dim)）。
const KEY_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
/// MNIST 数据目录回退路径。
const DEFAULT_MNIST_DIR: &str = "D:\\Rust\\snn_t1\\mnist_data";

/// 命令行配置。
#[derive(Clone)]
struct Config {
    batch: usize,
    accumulate: usize,
    rank: usize,
    t: usize,
    sigma: f32,
    lr: f32,
    reward: Reward,
    num_epochs: usize,
    seed: u64,
    d_model: usize,
    num_heads: usize,
    num_blocks: usize,
    mnist_dir: Option<String>,
    validate_every: usize,
    val_batch: usize,
    log_every: usize,
    csv_out: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            batch: 2048,
            accumulate: 4,
            rank: 16,
            t: 8,
            sigma: 0.2,
            lr: 0.03,
            reward: Reward::Loglik,
            num_epochs: 300,
            seed: 0,
            d_model: 32,
            num_heads: 4,
            num_blocks: 2,
            mnist_dir: None,
            validate_every: 25,
            val_batch: 1000,
            log_every: 5,
            csv_out: None,
        }
    }
}

/// 手动解析命令行参数（与 `accumulate_train.rs` 一致的风格）。
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
            "--num-epochs" => c.num_epochs = next_val(&mut i).parse().unwrap_or(c.num_epochs),
            "--seed" => c.seed = next_val(&mut i).parse().unwrap_or(c.seed),
            "--d-model" => c.d_model = next_val(&mut i).parse().unwrap_or(c.d_model),
            "--num-heads" => c.num_heads = next_val(&mut i).parse().unwrap_or(c.num_heads),
            "--num-blocks" => c.num_blocks = next_val(&mut i).parse().unwrap_or(c.num_blocks),
            "--mnist-dir" => c.mnist_dir = Some(next_val(&mut i)),
            "--validate-every" => c.validate_every = next_val(&mut i).parse().unwrap_or(c.validate_every),
            "--val-batch" => c.val_batch = next_val(&mut i).parse().unwrap_or(c.val_batch),
            "--log-every" => c.log_every = next_val(&mut i).parse().unwrap_or(c.log_every),
            "--csv-out" => c.csv_out = Some(next_val(&mut i)),
            _ => { /* 未知参数忽略 */ }
        }
        i += 1;
    }
    c
}

/// 确定性 Fisher-Yates 部分洗牌：从 `[0, n)` 取 `count` 个索引。
fn shuffled_indices(n: usize, count: usize, seed: u64) -> Vec<usize> {
    assert!(count <= n, "count({count}) 不能超过 n({n})");
    let mut state: u64 = (seed ^ KEY_MUL) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..count {
        let j = i + (next() as usize % (n - i));
        idx.swap(i, j);
    }
    idx.truncate(count);
    idx
}

/// 28x28 图像切成 P x P patch token，再展平回 `(batch, 784)`。
///
/// `(batch, 784)` -> `(batch, 4, 7, 4, 7)` -> 重排 -> `(batch, 16, 49)` -> `(batch, 784)`。
/// 与 Python `patch_images` 布局一致：token 序号 = row_block*4 + col_block，token 内
/// 像素 = px*7 + py（行主序的 7x7 patch）。
fn patch_images(imgs: Tensor<B, 2>) -> Tensor<B, 2> {
    let [b, _] = imgs.dims();
    let side = 28 / PATCH_PX; // 4
    // (b, 4, 7, 4, 7)
    let t = imgs.reshape([b, side, PATCH_PX, side, PATCH_PX]);
    // 交换 px(轴2) 与 col_block(轴3) -> (b, 4, 4, 7, 7)
    let t = t.swap_dims(2, 3);
    // (b, 16, 49)
    let t = t.reshape([b, side * side, PATCH_PX * PATCH_PX]);
    // (b, 16*49)
    t.reshape([b, NUM_TOKENS * TOKEN_DIM])
}

/// 从平面参数列表写回模型（参数顺序见 `SnnTransformer::params`）。
fn write_params(model: &mut SnnTransformer, new_params: Vec<Tensor<B, 2>>) {
    let mut it = new_params.into_iter();
    model.in_q.weight = it.next().expect("缺 in_q");
    model.in_k.weight = it.next().expect("缺 in_k");
    model.in_v.weight = it.next().expect("缺 in_v");
    model.pos_emb = it.next().expect("缺 pos_emb");
    for blk in &mut model.blocks {
        for m in &mut blk.q {
            m.weight = it.next().expect("缺 block q");
        }
        for m in &mut blk.k {
            m.weight = it.next().expect("缺 block k");
        }
        for m in &mut blk.v {
            m.weight = it.next().expect("缺 block v");
        }
        blk.o.weight = it.next().expect("缺 block o");
        blk.ff1.weight = it.next().expect("缺 block ff1");
        blk.ff2.weight = it.next().expect("缺 block ff2");
    }
    model.out.weight = it.next().expect("缺 out");
    model.out_gain.value = it.next().expect("缺 out_gain").squeeze_dim::<1>(0);
    model.beta.value = it.next().expect("缺 beta").squeeze_dim::<1>(0);
}

/// 干净评估（无扰动）：测试集伪随机取 val_batch 条，整批 `forward_batched(None)` -> 精度。
fn evaluate(
    model: &SnnTransformer,
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
    let patched = patch_images(imgs); // (count, 784)
    let spikes = poisson_encode(patched, t); // (T, count, 784)
    let logits = model.forward_batched(spikes, None); // (count, C)
    accuracy_from_logits(logits, labels)
}

fn main() {
    let cfg = parse_args();

    assert!(
        cfg.accumulate > 0 && cfg.batch % cfg.accumulate == 0,
        "batch({}) 必须能被 accumulate({}) 整除",
        cfg.batch,
        cfg.accumulate
    );
    let chunk = cfg.batch / cfg.accumulate;
    assert!(
        chunk % 2 == 0,
        "每段样本数必须为偶数（反对称配对 LoRA 要求），实际 {chunk}"
    );
    assert!(
        28 % PATCH_PX == 0,
        "patch_px 必须整除 28"
    );

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
        "[env] backend={backend_name} batch={} accumulate={} chunk={} rank={} T={} d_model={} heads={} blocks={}",
        cfg.batch, cfg.accumulate, chunk, cfg.rank, cfg.t, cfg.d_model, cfg.num_heads, cfg.num_blocks
    );

    // ---- 模型与 ES 装配 ------------------------------------------------
    let mut model = SnnTransformer::new(
        TOKEN_DIM,
        NUM_TOKENS,
        NUM_CLASSES,
        cfg.d_model,
        cfg.num_heads,
        cfg.num_blocks,
        &device,
    );
    let es_map = model.es_map();
    let params = model.params();
    let n_params = params.len();
    // 每个参数一个确定性 base_key。
    let base_keys: Vec<u64> = params
        .iter()
        .enumerate()
        .map(|(i, _)| (i as u64 + 1).wrapping_mul(KEY_MUL))
        .collect();
    // MM 参数索引（LoRA）与稠密参数索引（PARAM/EMB_PARAM）。
    let mm_indices: Vec<usize> = (0..n_params).filter(|&i| es_map[i] == MM_PARAM).collect();
    // 稠密索引 = pos_emb(3) + out_gain/beta(末尾两个)，仅供统计。
    let _dense_indices: Vec<usize> = (0..n_params)
        .filter(|&i| es_map[i] != MM_PARAM)
        .collect();
    assert!(mm_indices.len() > 0 && _dense_indices.len() > 0);
    println!(
        "[model] params={n_params} mm(lora)={} dense={}",
        mm_indices.len(),
        _dense_indices.len()
    );

    // EggRoll noiser（LoRA + adamw；其余稠密参数走 FULL 随机搜索同流）。
    let (frozen, mut noiser) = init_noiser(
        &params,
        cfg.sigma,
        cfg.lr,
        0, // group_size
        false, // freeze_nonlora：允许稠密 pos_emb/out_gain/beta 路径
        0, // noise_reuse（噪声只依赖 thread_id）
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

    // CSV 输出。
    let mut csv: Option<std::fs::File> = cfg.csv_out.as_ref().map(|p| {
        if let Some(parent) = std::path::Path::new(p).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("创建 CSV 目录失败");
            }
        }
        let fp = std::fs::File::create(p).expect("创建 csv 文件失败");
        let mut w = std::io::BufWriter::new(fp);
        writeln!(w, "epoch,train_acc,val_acc,best_val,best_train,epoch_time,cum_time")
            .expect("写 CSV 表头失败");
        w.into_inner().map_err(|_| ()).unwrap()
    });

    let mut best_val = 0.0_f32;
    let mut best_train = 0.0_f32;
    let mut cum_t = 0.0_f32;

    for epoch in 0..cfg.num_epochs {
        let t_ep = std::time::Instant::now();

        // 每 epoch 从训练集确定性取 batch 条。
        let idx = shuffled_indices(
            n_train,
            cfg.batch,
            cfg.seed ^ (epoch as u64 + 1).wrapping_mul(KEY_MUL),
        );
        let idx_t: Vec<i32> = idx.iter().map(|&x| x as i32).collect();
        let idx_tensor: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(idx_t, [cfg.batch].to_vec()), &device);
        let imgs = train_img.clone().select(0, idx_tensor.clone()); // (batch, 784)
        let labels = train_lbl.clone().select(0, idx_tensor); // (batch,)
        let patch = patch_images(imgs); // (batch, 784) 已重排为 16x49 展平

        // ----- K 段累积：前向 + 收益加权部分梯度（仿射修正版）-----
        let base_sigma = cfg.sigma / (cfg.rank as f32).sqrt();
        let p0 = model.params();
        let mut grad_acc: Vec<Tensor<B, 2>> = p0
            .iter()
            .map(|p| Tensor::<B, 2>::zeros(p.dims(), &device))
            .collect();
        let mut ones_acc: Vec<Tensor<B, 2>> = grad_acc.clone();
        let mut sum_raw_t = Tensor::<B, 1>::zeros([1], &device);
        let mut sum_raw2_t = Tensor::<B, 1>::zeros([1], &device);
        let mut correct_t = Tensor::<B, 1>::zeros([1], &device);

        for k in 0..cfg.accumulate {
            let lo = k * chunk;
            let hi = lo + chunk;
            // 每 chunk 泊松编码：(T, chunk, 784)。
            let spikes_k = poisson_encode(
                patch.clone().slice([lo..hi, 0..IN_DIM]),
                cfg.t,
            );
            let tids: Vec<i32> = (lo..hi).map(|j| j as i32).collect();
            let labels_k = labels.clone().slice([lo..hi]);

            // 本 chunk 每个 MM 参数的 LoRA 噪声（按 mm_indices 顺序）。
            // `batched_lora_noise` 返回 `(A(n,a,r), B(n,b,r))`；`lora_einsum_pair` 与
            // 模型 `nn` 均期望 `(n,r,a)`/`(n,r,b)`，故做一次 dim1<->dim2 转置。
            let mut lora: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = Vec::with_capacity(mm_indices.len());
            for &idx in &mm_indices {
                let [a, b] = params[idx].dims();
                let (a_t, b_t) = batched_lora_noise(
                    base_sigma,
                    base_keys[idx],
                    cfg.rank,
                    &tids,
                    epoch as i32,
                    0,
                    a,
                    b,
                    &device,
                );
                lora.push((a_t.swap_dims(1, 2), b_t.swap_dims(1, 2)));
            }

            // 稠密参数的逐样本噪声。
            let pos_dim = model.pos_emb.dims(); // (num_tokens, d_model)
            let pos_noise: Tensor<B, 3> = Tensor::<B, 3>::random(
                [chunk, pos_dim[0], pos_dim[1]],
                Distribution::Normal(0.0, cfg.sigma as f64),
                &device,
            );
            let gain_noise: Tensor<B, 2> =
                Tensor::random([chunk, 1], Distribution::Normal(0.0, cfg.sigma as f64), &device);
            let beta_noise: Tensor<B, 2> =
                Tensor::random([chunk, 1], Distribution::Normal(0.0, cfg.sigma as f64), &device);

            // 噪声张量既给前向又给梯度，构造 TrainNoise 时传克隆。
            let tn = TrainNoise {
                lora: &lora,
                mm_indices: &mm_indices,
                pos_emb: Some(pos_noise.clone()),
                out_gain: Some(gain_noise.clone()),
                beta: Some(beta_noise.clone()),
            };
            let logits_k = model.forward_batched(spikes_k, Some(&tn)); // (chunk, C)
            let raw_k: Tensor<B, 1> =
                fitness_from_logits_reward(logits_k.clone(), labels_k.clone(), cfg.reward);

            sum_raw_t = sum_raw_t.clone() + raw_k.clone().sum();
            sum_raw2_t = sum_raw2_t.clone() + raw_k.clone().powf_scalar(2.0).sum();
            let pred = logits_k.argmax(1).reshape([chunk]);
            correct_t = correct_t.clone() + pred.equal(labels_k).float().sum();

            // LoRA（MM）梯度：g_raw = Σ raw·A⊗B；g_ones = Σ A⊗B。
            for (kidx, &mi) in mm_indices.iter().enumerate() {
                let (a_t, b_t) = &lora[kidx];
                let (g_raw, g_ones) = lora_einsum_pair(a_t, b_t, &raw_k, &device);
                grad_acc[mi] = grad_acc[mi].clone() + g_raw;
                ones_acc[mi] = ones_acc[mi].clone() + g_ones;
            }
            // 稠密（PARAM/EMB_PARAM）梯度：grad = Σ raw·noise；ones = Σ noise。
            // pos_emb（idx=3）：(chunk, nt, d)
            {
                let weighted = pos_noise.clone() * raw_k.clone().reshape([chunk, 1, 1]); // (chunk, nt, d)
                grad_acc[3] = grad_acc[3].clone()
                    + weighted.sum_dim(0).squeeze_dim::<2>(0); // (nt, d)
                ones_acc[3] = ones_acc[3].clone()
                    + pos_noise.sum_dim(0).squeeze_dim::<2>(0); // (nt, d)
            }
            // out_gain（idx = n_params-2）：(chunk, 1)
            let n_p = n_params;
            {
                let weighted = gain_noise.clone() * raw_k.clone().reshape([chunk, 1]); // (chunk, 1)
                grad_acc[n_p - 2] = grad_acc[n_p - 2].clone() + weighted.sum_dim(0);
                ones_acc[n_p - 2] = ones_acc[n_p - 2].clone() + gain_noise.sum_dim(0);
            }
            // beta（idx = n_params-1）：(chunk, 1)
            {
                let weighted = beta_noise.clone() * raw_k.reshape([chunk, 1]); // (chunk, 1)
                grad_acc[n_p - 1] = grad_acc[n_p - 1].clone() + weighted.sum_dim(0);
                ones_acc[n_p - 1] = ones_acc[n_p - 1].clone() + beta_noise.sum_dim(0);
            }
        }

        // 全局 z-score 仿射修正 + 一次 solver 更新。
        let mean = sum_raw_t.clone().into_scalar() / cfg.batch as f32;
        let var = (sum_raw2_t.clone().into_scalar() / cfg.batch as f32 - mean * mean).max(0.0);
        let std = (var + 1e-5).sqrt();

        let grads = combine_affine_grads(&grad_acc, &ones_acc, mean, std, cfg.batch);
        let new_params = frozen.solver.update(&model.params(), &grads, &mut noiser.opt_state);
        write_params(&mut model, new_params);

        // ---- 打印与评估 ----
        let train_acc = correct_t.clone().into_scalar() / cfg.batch as f32;
        if train_acc > best_train {
            best_train = train_acc;
        }
        let mut msg = format!(
            "epoch {:5} | train_acc {:.4} | best_train {:.4}",
            epoch, train_acc, best_train
        );
        let mut val_acc: Option<f32> = None;
        if epoch % cfg.validate_every == 0 {
            val_acc = Some(evaluate(&model, &test_img, &test_lbl, cfg.val_batch, cfg.t, cfg.seed, &device));
            if val_acc.unwrap() > best_val {
                best_val = val_acc.unwrap();
            }
            msg += &format!(" | val_acc {:.4} | best_val {:.4}", val_acc.unwrap(), best_val);
        }
        let el = t_ep.elapsed().as_secs_f32();
        cum_t += el;
        msg += &format!(" | {el:.2}s | cum {cum_t:.1}s");
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
