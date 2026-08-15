//! 噪声注入路径微基准：对比「广播乘法 + sum」与「3D batched matmul」等候选实现。
//!
//! 用法：cargo run --release -p hyperscalees --features gpu --bin noise_bench
//! 参数：--n 12000 --b 784 --a 128 --r 64 --T 8 --iters 10

use std::time::Instant;

use burn::tensor::{Distribution, Tensor};
use hyperscalees_core::B;

fn parse_args() -> (usize, usize, usize, usize, usize, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut n = 12000;
    let mut b = 784;
    let mut a = 128;
    let mut r = 64;
    let mut t = 8;
    let mut iters = 10;
    let mut i = 0;
    while i < args.len() {
        let next = |i: &mut usize| -> String {
            if *i + 1 < args.len() {
                *i += 1;
                args[*i].clone()
            } else {
                String::new()
            }
        };
        match args[i].as_str() {
            "--n" => n = next(&mut i).parse().unwrap_or(n),
            "--b" => b = next(&mut i).parse().unwrap_or(b),
            "--a" => a = next(&mut i).parse().unwrap_or(a),
            "--r" => r = next(&mut i).parse().unwrap_or(r),
            "--T" => t = next(&mut i).parse().unwrap_or(t),
            "--iters" => iters = next(&mut i).parse().unwrap_or(iters),
            _ => {}
        }
        i += 1;
    }
    (n, b, a, r, t, iters)
}

fn time_ms<F: FnMut() -> Tensor<B, 2>>(warmup: bool, iters: usize, mut f: F) -> f64 {
    if warmup {
        let _ = f().into_scalar(); // 触发 JIT 编译 + autotune
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = f();
        out.into_scalar(); // 强制同步，计入完整执行时间
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.iter().sum::<f64>() / times.len() as f64
}

fn main() {
    let (n, b, a, r, t, iters) = parse_args();
    let device = Device::<B>::default();
    println!("n={n} b={b} a={a} r={r} T={t} iters={iters} backend=cuda");

    let x: Tensor<B, 3> = Tensor::random([t, n, b], Distribution::Bernoulli(0.3), &device);
    let w: Tensor<B, 2> = Tensor::random([a, b], Distribution::Normal(0.0, 0.1), &device);
    let x2: Tensor<B, 2> = Tensor::random([n, b], Distribution::Bernoulli(0.3), &device);

    // ---- 1. 广播乘法 + sum（当前实现，逐时间步）----
    let b_t: Tensor<B, 3> = Tensor::random([n, b, r], Distribution::Normal(0.0, 1.0), &device);
    let a_t: Tensor<B, 3> = Tensor::random([n, a, r], Distribution::Normal(0.0, 1.0), &device);
    let cur = time_ms(true, iters, || {
        let mut parts: Vec<Tensor<B, 2>> = Vec::with_capacity(t);
        for i in 0..t {
            let x_t = x
                .clone()
                .slice([i..i + 1, 0..n, 0..b])
                .squeeze_dim::<2>(0);
            let base = x_t.clone().matmul(w.clone().transpose());
            let y = x_t.clone().unsqueeze_dim::<3>(2) * b_t.clone();
            let y = y.sum_dim(1).squeeze_dim::<2>(1);
            let noise = y.unsqueeze_dim::<3>(1) * a_t.clone();
            let noise = noise.sum_dim(2).squeeze_dim::<2>(2);
            parts.push(base + noise);
        }
        Tensor::cat(parts, 0).mean_dim(0).squeeze_dim::<2>(0)
    });
    println!("[1] broadcast_mul_sum        : {cur:8.2} ms/chunk (当前实现)");

    // ---- 2. 3D batched matmul, m=1（逐时间步）----
    let a_tt: Tensor<B, 3> = a_t.clone().swap_dims(1, 2); // (n, r, a)
    let cur = time_ms(true, iters, || {
        let mut parts: Vec<Tensor<B, 2>> = Vec::with_capacity(t);
        for i in 0..t {
            let x_t = x
                .clone()
                .slice([i..i + 1, 0..n, 0..b])
                .squeeze_dim::<2>(0);
            let base = x_t.clone().matmul(w.clone().transpose());
            let y = x_t
                .clone()
                .unsqueeze_dim::<3>(1)
                .matmul(b_t.clone())
                .squeeze_dim::<2>(1); // (n,1,b)@(n,b,r) -> (n,r)
            let noise = y
                .clone()
                .unsqueeze_dim::<3>(1)
                .matmul(a_tt.clone())
                .squeeze_dim::<2>(1); // (n,1,r)@(n,r,a) -> (n,a)
            parts.push(base + noise);
        }
        Tensor::cat(parts, 0).mean_dim(0).squeeze_dim::<2>(0)
    });
    println!("[2] batched_m1_per_t         : {cur:8.2} ms/chunk");

    // ---- 3. 3D batched matmul, m=T 合并（每层一次）----
    let xp = x.clone().swap_dims(0, 1); // (n, T, b)
    let cur = time_ms(true, iters, || {
        let base = xp
            .clone()
            .reshape([n * t, b])
            .matmul(w.clone().transpose())
            .reshape([n, t, a]); // (n,T,a)
        let y = xp.clone().matmul(b_t.clone()); // (n,T,b)@(n,b,r) -> (n,T,r)
        let noise = y.matmul(a_tt.clone()); // (n,T,r)@(n,r,a) -> (n,T,a)
        (base + noise).mean_dim(1).squeeze_dim::<2>(1)
    });
    println!("[3] batched_mT_merged        : {cur:8.2} ms/chunk");

    // ---- 4. einsum 梯度 GEMM（当前实现）----
    let scores: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let scores_t: Tensor<B, 1> = Tensor::from_data(&scores[..], &device);
    let cur = time_ms(true, iters, || {
        let a_w = a_t.clone() * scores_t.clone().reshape([n, 1, 1]);
        let a_flat = a_w.swap_dims(1, 2).reshape([n * r, a]);
        let b_flat = b_t.clone().swap_dims(1, 2).reshape([n * r, b]);
        a_flat.transpose().matmul(b_flat)
    });
    println!("[4] einsum_gemm_2d           : {cur:8.2} ms/chunk");

    // ---- 4b. einsum 变体：不 transpose 的 GEMM (n*r,a)@(n*r,b)^T ----
    let cur = time_ms(true, iters, || {
        let a_w = a_t.clone() * scores_t.clone().reshape([n, 1, 1]);
        let a_flat = a_w.swap_dims(1, 2).reshape([n * r, a]);
        let b_flat = b_t.clone().swap_dims(1, 2).reshape([n * r, b]);
        a_flat.matmul(b_flat.transpose()).transpose()
    });
    println!("[4b] einsum_gemm_2d_alt      : {cur:8.2} ms/chunk");

    // ---- 5. poisson：循环 8 次 vs 单次 (T,n,b) ----
    let cur = time_ms(true, iters, || {
        let mut spikes: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        for _ in 0..t {
            let u: Tensor<B, 2> = Tensor::random([n, b], Distribution::Uniform(0.0, 1.0), &device);
            spikes.push(u.lower(x2.clone()).float().unsqueeze_dim(0));
        }
        Tensor::cat(spikes, 0).sum_dim(0).squeeze_dim::<2>(0).mean()
    });
    println!("[5] poisson_loop             : {cur:8.2} ms/chunk");
    let cur = time_ms(true, iters, || {
        let u: Tensor<B, 3> = Tensor::random([t, n, b], Distribution::Uniform(0.0, 1.0), &device);
        let x3 = x2.clone().unsqueeze_dim(0); // (1,n,b)
        u.lower(x3).float().sum_dim(0).squeeze_dim::<2>(0).mean()
    });
    println!("[5b] poisson_single           : {cur:8.2} ms/chunk");

    // ---- 6. 随机数生成吞吐 ----
    let cur = time_ms(true, iters, || {
        Tensor::<B, 3>::random([n, b, r], Distribution::Normal(0.0, 1.0), &device)
            .sum_dim(2)
            .squeeze_dim::<2>(2)
            .mean()
    });
    println!("[6] rand_normal(n,b,r)       : {cur:8.2} ms/chunk ({(n * b * r) as f64 * 4.0 / cur / 1e6} GB/s)");

    // ---- 7. matmul 基础吞吐 (n,b)@(b,a) ----
    let cur = time_ms(true, iters, || {
        let x2 = x2.clone();
        x2.matmul(w.clone().transpose()).mean_dim(1).squeeze_dim::<1>(1).mean()
    });
    println!("[7] matmul(n,b)x(b,a)        : {cur:8.2} ms/chunk");

    // ---- 8. LIF 步进（逐时间步, 2 层）----
    let cur = time_ms(true, iters, || {
        let mut v: Tensor<B, 2> = Tensor::zeros([n, 128], &device);
        for _ in 0..t {
            let cur_t = Tensor::<B, 2>::random([n, 128], Distribution::Normal(0.0, 1.0), &device);
            let charged = v.clone() + (v.neg() + cur_t).mul_scalar(1.0 / 20.0);
            let spike = charged.clone().greater_equal_elem(0.3).float();
            v = charged.clone() * spike.clone().neg().add_scalar(1.0);
        }
        v.mean()
    });
    println!("[8] lif_loop                 : {cur:8.2} ms/chunk");
}
