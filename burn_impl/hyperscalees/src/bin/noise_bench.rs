//! 噪声注入路径微基准：对比「广播乘法 + sum」与「3D batched matmul」等候选实现。
//!
//! 用法：cargo run --release -p hyperscalees --features gpu --bin noise_bench
//! 参数：--n 12000 --b 784 --a 128 --r 64 --T 8 --iters 10

use std::time::Instant;

use burn::tensor::{Device, Distribution, Tensor};
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

fn time_ms<F: FnMut() -> Tensor<B, 1>>(warmup: bool, iters: usize, mut f: F) -> f64 {
    if warmup {
        let _ = f().into_scalar(); // 触发 JIT 编译 + autotune
    }
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = f();
        out.into_scalar(); // 强制同步，计入完整执行时间
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let n = times.len() as f64;
    if n == 0.0 {
        return f64::NAN;
    }
    let sum: f64 = times.iter().sum();
    if sum.is_nan() {
        eprintln!("  [dbg] raw times = {times:?}");
        return f64::NAN;
    }
    sum / n
}

fn main() {
    let (n, b, a, r, t, iters) = parse_args();
    let device = Device::<B>::default();
    println!("n={n} b={b} a={a} r={r} T={t} iters={iters} backend=cuda");

    // ---- 0. cuBLAS 集成正确性校验（列主序映射容易出错，先验证数学等价）----
    #[cfg(feature = "gpu")]
    {
        // 确定性小矩阵：A (k=2, m=3), B (k=2, n=2)。
        let am: Tensor<B, 2> = Tensor::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &device,
        ); // (2, 3)
        let bm: Tensor<B, 2> =
            Tensor::from_data([[7.0_f32, 8.0], [9.0, 10.0]], &device); // (2, 2)
        let c_ref = am.clone().transpose().matmul(bm.clone());
        let c_cu = hyperscalees::cublas::gemm_atb(&am, &bm, &device);
        let va = c_ref.into_data().into_vec::<f32>().unwrap();
        let vb = c_cu.into_data().into_vec::<f32>().unwrap();
        let maxd = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!("[0a] cublas_gemm_check        : ref={va:?} cu={vb:?} maxdiff={maxd:.3e}");
        assert!(maxd < 1e-4, "cuBLAS gemm 数值错误");

        // lora_einsum_pair_cublas vs 通用 lora_einsum_pair（相同反对称输入）。
        let n_s = 8usize;
        let r_s = 3usize;
        let a_s = 2usize;
        let b_s = 3usize;
        let a_half: Tensor<B, 3> =
            Tensor::random([n_s / 2, r_s, a_s], Distribution::Normal(0.0, 1.0), &device);
        let b_half: Tensor<B, 3> =
            Tensor::random([n_s / 2, r_s, b_s], Distribution::Normal(0.0, 1.0), &device);
        let a_t = Tensor::cat(vec![a_half.clone(), a_half.neg()], 0);
        let b_t = Tensor::cat(vec![b_half.clone(), b_half.neg()], 0);
        let scores: Tensor<B, 1> = Tensor::random([n_s], Distribution::Normal(0.0, 1.0), &device);
        let (g1, o1) = hyperscalees_noiser::eggroll::lora_einsum_pair(&a_t, &b_t, &scores, &device);
        let (g2, o2) = hyperscalees::cublas::lora_einsum_pair_cublas(&a_t, &b_t, &scores, &device);
        let maxd = |x: Tensor<B, 2>, y: Tensor<B, 2>| {
            let vx = x.into_data().into_vec::<f32>().unwrap();
            let vy = y.into_data().into_vec::<f32>().unwrap();
            vx.iter()
                .zip(vy.iter())
                .map(|(u, v)| (u - v).abs())
                .fold(0.0_f32, f32::max)
        };
        let d_raw = maxd(g1, g2);
        let d_ones = maxd(o1, o2);
        println!("[0b] cublas_pair_einsum_check : raw={d_raw:.3e} ones={d_ones:.3e} (应 <1e-4)");
        assert!(d_raw < 1e-4 && d_ones < 1e-4, "cuBLAS 配对 einsum 数值错误");
    }

    let x: Tensor<B, 3> = Tensor::random([t, n, b], Distribution::Bernoulli(0.3), &device);
    let w: Tensor<B, 2> = Tensor::random([a, b], Distribution::Normal(0.0, 0.1), &device);
    let x2: Tensor<B, 2> = Tensor::random([n, b], Distribution::Bernoulli(0.3), &device);

    // ---- 1. 广播乘法 + sum（当前实现，逐时间步）----
    let b_t: Tensor<B, 3> = Tensor::random([n, b, r], Distribution::Normal(0.0, 1.0), &device);
    let a_t: Tensor<B, 3> = Tensor::random([n, a, r], Distribution::Normal(0.0, 1.0), &device);
    let cur = time_ms(true, iters, || {
        let mut parts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
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
            parts.push((base + noise).unsqueeze_dim::<3>(0));
        }
        Tensor::cat(parts, 0).mean_dim(0).squeeze_dim::<2>(0).mean()
    });
    println!("[1] broadcast_mul_sum        : {cur:8.2} ms/chunk (当前实现)");

    // ---- 2. 3D batched matmul, m=1（逐时间步）----
    let a_tt: Tensor<B, 3> = a_t.clone().swap_dims(1, 2); // (n, r, a)
    let cur = time_ms(true, iters, || {
        let mut parts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
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
            parts.push((base + noise).unsqueeze_dim::<3>(0));
        }
        Tensor::cat(parts, 0).mean_dim(0).squeeze_dim::<2>(0).mean()
    });
    println!("[2] batched_m1_per_t         : {cur:8.2} ms/chunk");

    // ---- 3. 3D batched matmul, m=T 合并（每层一次）----
    let xp = x.clone().swap_dims(0, 1).reshape([n, t, b]); // (n, T, b) 连续
    let cur = time_ms(true, iters, || {
        let base = xp
            .clone()
            .reshape([n * t, b])
            .matmul(w.clone().transpose())
            .reshape([n, t, a]); // (n,T,a)
        let y = xp.clone().matmul(b_t.clone()); // (n,T,b)@(n,b,r) -> (n,T,r)
        let noise = y.matmul(a_tt.clone()); // (n,T,r)@(n,r,a) -> (n,T,a)
        (base + noise).mean_dim(1).squeeze_dim::<2>(1).mean()
    });
    println!("[3] batched_mT_merged        : {cur:8.2} ms/chunk");

    // ---- 4. einsum 梯度 GEMM（当前实现）----
    let scores: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let scores_t: Tensor<B, 1> = Tensor::from_data(&scores[..], &device);
    let cur = time_ms(true, iters, || {
        let a_w = a_t.clone() * scores_t.clone().reshape([n, 1, 1]);
        let a_flat = a_w.swap_dims(1, 2).reshape([n * r, a]);
        let b_flat = b_t.clone().swap_dims(1, 2).reshape([n * r, b]);
        a_flat.transpose().matmul(b_flat).mean()
    });
    println!("[4] einsum_gemm_2d           : {cur:8.2} ms/chunk");

    // ---- 4b. einsum 变体：A 已按 (n,r,a) 布局（无 swap/reshape 拷贝）----
    let a_ra: Tensor<B, 3> = a_t.clone().swap_dims(1, 2).reshape([n, r, a]); // 连续 (n,r,a)
    let cur = time_ms(true, iters, || {
        let a_w = a_ra.clone() * scores_t.clone().reshape([n, 1, 1]);
        let a_flat = a_w.reshape([n * r, a]); // 连续，无拷贝
        let b_flat = b_t.clone().swap_dims(1, 2).reshape([n * r, b]);
        a_flat.transpose().matmul(b_flat).mean()
    });
    println!("[4b] einsum_gemm_A_ra        : {cur:8.2} ms/chunk");

    // ---- 4c. einsum 变体：A 与 B 都按 (n,r,*) 布局（无任何拷贝）----
    let b_rb: Tensor<B, 3> = b_t.clone().swap_dims(1, 2).reshape([n, r, b]); // 连续 (n,r,b)
    let cur = time_ms(true, iters, || {
        let a_w = a_ra.clone() * scores_t.clone().reshape([n, 1, 1]);
        let a_flat = a_w.reshape([n * r, a]); // 连续，无拷贝
        let b_flat = b_rb.clone().reshape([n * r, b]); // 连续，无拷贝
        a_flat.transpose().matmul(b_flat).mean()
    });
    println!("[4c] einsum_gemm_no_copy     : {cur:8.2} ms/chunk");

    // ---- 4d. 反对称配对合并 einsum（训练热路径：raw+ones 一次半 K GEMM）----
    let b_rab: Tensor<B, 3> = b_t.clone().swap_dims(1, 2).reshape([n, r, b]); // (n,r,b)
    let a_ra: Tensor<B, 3> = a_t.clone().swap_dims(1, 2).reshape([n, r, a]); // (n,r,a)
    let cur = time_ms(true, iters, || {
        let half = n / 2;
        let a_half = a_ra.clone().slice([0..half, 0..r, 0..a]);
        let b_half = b_rab.clone().slice([0..half, 0..r, 0..b]);
        let f_pair = scores_t
            .clone()
            .slice([0..half])
            .add(scores_t.clone().slice([half..n]));
        let b_w = b_half.clone() * f_pair.reshape([half, 1, 1]);
        let b2 = b_half.mul_scalar(2.0);
        let a_stack = Tensor::cat(vec![a_half.clone(), a_half], 2); // (half, r, 2a)
        let b_stack = Tensor::cat(vec![b_w, b2], 2); // (half, r, 2b)
        let a_flat = a_stack.reshape([half * r, 2 * a]);
        let b_flat = b_stack.reshape([half * r, 2 * b]);
        a_flat.transpose().matmul(b_flat).mean()
    });
    println!("[4d] einsum_pair_halfk       : {cur:8.2} ms/chunk");

    // ---- 4e. einsum 变体：lhs 显式 contiguify（避免转置视图的非合并读）----
    let cur = time_ms(true, iters, || {
        let a_w = a_ra.clone() * scores_t.clone().reshape([n, 1, 1]);
        let a_flat = a_w.reshape([n * r, a]); // 连续
        let a_lhs = a_flat.transpose().reshape([a, n * r]); // 拷贝 -> 连续 (a, n*r)
        let b_flat = b_rab.clone().reshape([n * r, b]); // 连续
        a_lhs.matmul(b_flat).mean()
    });
    println!("[4e] einsum_contig_lhs       : {cur:8.2} ms/chunk");

    // ---- 4f. 配对合并（M 堆叠：raw/ones 共享 B_half，半 K 一次 GEMM）----
    let cur = time_ms(true, iters, || {
        let half = n / 2;
        let a_half = a_ra.clone().slice([0..half, 0..r, 0..a]);
        let b_half = b_rab.clone().slice([0..half, 0..r, 0..b]);
        let f_pair = scores_t
            .clone()
            .slice([0..half])
            .add(scores_t.clone().slice([half..n]));
        let a_w = a_half.clone() * f_pair.reshape([half, 1, 1]);
        let a_stack = Tensor::cat(vec![a_w, a_half], 2); // (half, r, 2a)
        let a_flat = a_stack.reshape([half * r, 2 * a]);
        let b_flat = b_half.reshape([half * r, b]);
        a_flat.transpose().matmul(b_flat).mean()
    });
    println!("[4f] einsum_pair_contig      : {cur:8.2} ms/chunk");

    // ---- 4k. cuBLAS 配对 einsum（训练热路径实际使用的实现）----
    #[cfg(feature = "gpu")]
    {
        let cur = time_ms(true, iters, || {
            let half = n / 2;
            let a_half = a_ra.clone().slice([0..half, 0..r, 0..a]);
            let b_half = b_rab.clone().slice([0..half, 0..r, 0..b]);
            let f_pair = scores_t
                .clone()
                .slice([0..half])
                .add(scores_t.clone().slice([half..n]));
            let a_w = a_half.clone() * f_pair.reshape([half, 1, 1]);
            let a_stack = Tensor::cat(vec![a_w, a_half], 2);
            let a_flat = a_stack.reshape([half * r, 2 * a]);
            let b_flat = b_half.reshape([half * r, b]);
            let (g1, g2) =
                hyperscalees::cublas::lora_einsum_pair_cublas(&a_ra, &b_rab, &scores_t, &device);
            (g1 + g2).mean()
        });
        println!("[4k] einsum_pair_cublas     : {cur:8.2} ms/chunk");

    // ---- 4l. 纯 cuBLAS gemm（无 prep）：(2a, half*r)@(half*r, b) ----
    #[cfg(feature = "gpu")]
    {
        let half = n / 2;
        let a_flat2: Tensor<B, 2> = a_ra
            .clone()
            .slice([0..half, 0..r, 0..a])
            .reshape([half * r, a])
            .transpose()
            .reshape([half * r, a])
            .transpose();
        let _ = a_flat2; // 占位避免未使用
        let cur = time_ms(true, iters, || {
            let a_m: Tensor<B, 2> = a_ra
                .clone()
                .slice([0..half, 0..r, 0..a])
                .reshape([half * r, a]);
            let b_m: Tensor<B, 2> = b_rab
                .clone()
                .slice([0..half, 0..r, 0..b])
                .reshape([half * r, b]);
            let c1 = hyperscalees::cublas::gemm_atb(&a_m, &b_m, &device);
            let c2 = hyperscalees::cublas::gemm_atb(&a_m, &b_m, &device);
            (c1 + c2).mean()
        });
        println!("[4l] gemm_atb_pure_x2      : {cur:8.2} ms/chunk (2 次 gemm)");
    }
    }

    // ---- 4h. raw_halfk 形状：(128, 384000)@(384000, 784)，转置 lhs 视图 ----
    let cur = time_ms(true, iters, || {
        let half = n / 2;
        let a_half = a_ra.clone().slice([0..half, 0..r, 0..a]);
        let b_half = b_rab.clone().slice([0..half, 0..r, 0..b]);
        let f_pair = scores_t
            .clone()
            .slice([0..half])
            .add(scores_t.clone().slice([half..n]));
        let a_w = a_half * f_pair.reshape([half, 1, 1]);
        let a_flat = a_w.reshape([half * r, a]);
        let b_flat = b_half.reshape([half * r, b]);
        a_flat.transpose().matmul(b_flat).mean()
    });
    println!("[4h] raw_halfk_strided_lhs   : {cur:8.2} ms/chunk");

    // ---- 4i. 同形状但 lhs 显式拷贝为连续 ----
    let cur = time_ms(true, iters, || {
        let half = n / 2;
        let a_half = a_ra.clone().slice([0..half, 0..r, 0..a]);
        let b_half = b_rab.clone().slice([0..half, 0..r, 0..b]);
        let f_pair = scores_t
            .clone()
            .slice([0..half])
            .add(scores_t.clone().slice([half..n]));
        let a_w = a_half * f_pair.reshape([half, 1, 1]);
        let a_flat = a_w.reshape([half * r, a]);
        let a_lhs = a_flat.transpose().reshape([a, half * r]); // 连续
        let b_flat = b_half.reshape([half * r, b]);
        a_lhs.matmul(b_flat).mean()
    });
    println!("[4i] raw_halfk_contig_lhs    : {cur:8.2} ms/chunk");

    // ---- 4j. ones_halfk 形状：(128, 384000)@(384000, 784) ----
    let cur = time_ms(true, iters, || {
        let half = n / 2;
        let a_half = a_ra.clone().slice([0..half, 0..r, 0..a]);
        let b_half = b_rab.clone().slice([0..half, 0..r, 0..b]);
        let a_flat = a_half.reshape([half * r, a]);
        let b_flat = b_half.reshape([half * r, b]);
        a_flat.transpose().matmul(b_flat).mul_scalar(2.0).mean()
    });
    println!("[4j] ones_halfk               : {cur:8.2} ms/chunk");

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
    let gbps = (n * b * r) as f64 * 4.0 / cur / 1e6;
    println!("[6] rand_normal(n,b,r)       : {cur:8.2} ms/chunk ({gbps:.0} GB/s)");

    // ---- 6b. 反对称生成（当前 gen_gpu_lora_noise 结构）----
    let cur = time_ms(true, iters, || {
        let half = n / 2;
        let b_even: Tensor<B, 3> =
            Tensor::random([half, b, r], Distribution::Normal(0.0, 1.0), &device);
        let b_full = Tensor::cat(vec![b_even.clone(), b_even.neg()], 0); // (n, b, r)
        let a_even: Tensor<B, 3> =
            Tensor::random([half, r, a], Distribution::Normal(0.0, 1.0), &device);
        let a_full =
            Tensor::cat(vec![a_even.clone(), a_even.neg()], 0).mul_scalar(0.025); // (n,r,a)
        b_full.sum_dim(1).squeeze_dim::<2>(1).sum() + a_full.sum_dim(1).squeeze_dim::<2>(1).sum()
    });
    println!("[6b] antipodal_gen           : {cur:8.2} ms/chunk");

    // ---- 7. matmul 基础吞吐 (n,b)@(b,a) 与 (T*n,b)@(b,a) ----
    let cur = time_ms(true, iters, || {
        x2.clone().matmul(w.clone().transpose()).mean_dim(1).squeeze_dim::<1>(1).mean()
    });
    println!("[7] matmul(n,b)x(b,a)        : {cur:8.2} ms/chunk");
    let cur = time_ms(true, iters, || {
        x.clone()
            .reshape([n * t, b])
            .matmul(w.clone().transpose())
            .mean_dim(1)
            .squeeze_dim::<1>(1)
            .mean()
    });
    println!("[7b] matmul(Tn,b)x(b,a)      : {cur:8.2} ms/chunk");

    // ---- 8. LIF 步进（逐时间步, 2 层）----
    let cur = time_ms(true, iters, || {
        let mut v: Tensor<B, 2> = Tensor::zeros([n, 128], &device);
        for _ in 0..t {
            let cur_t: Tensor<B, 2> =
                Tensor::random([n, 128], Distribution::Normal(0.0, 1.0), &device);
            let charged = v.clone() + (v.neg() + cur_t).mul_scalar(1.0 / 20.0);
            let spike = charged.clone().greater_equal_elem(0.3).float();
            v = charged.clone() * spike.clone().neg().add_scalar(1.0);
        }
        v.mean()
    });
    println!("[8] lif_loop                 : {cur:8.2} ms/chunk");

    // ---- 9. sum_dim 归约：广播中间量的总和（当前噪声路径的归约成本）----
    let cur = time_ms(true, iters, || {
        let y = x2.clone().unsqueeze_dim::<3>(2) * b_t.clone(); // (n,b,1)*(n,b,r)
        y.sum_dim(1).squeeze_dim::<2>(1).mean()
    });
    println!("[9] mul_then_sumdim         : {cur:8.2} ms/chunk");

    // ---- 10. 3D batched matmul 单独（无 base matmul 干扰）----
    let cur = time_ms(true, iters, || {
        xp.clone().matmul(b_t.clone()).mean_dim(1).squeeze_dim::<2>(1).mean()
    });
    println!("[10] batched_matmul_only     : {cur:8.2} ms/chunk");

    // ---- 10b. 同上，但 rhs 为列主序视图（B' 存 (n,r,b) 后 swap）----
    let b_rab: Tensor<B, 3> = b_t.clone().swap_dims(1, 2).reshape([n, r, b]); // 连续 (n,r,b)
    let b_br_view: Tensor<B, 3> = b_rab.clone().swap_dims(1, 2); // (n,b,r) 列主序视图
    let cur = time_ms(true, iters, || {
        xp.clone().matmul(b_br_view.clone()).mean_dim(1).squeeze_dim::<2>(1).mean()
    });
    println!("[10b] batched_matmul_colmajor_rhs: {cur:8.2} ms/chunk");

    // ---- 10c. 完整噪声路径（合并 T，rhs 列主序，A' 连续 (n,r,a)）----
    let cur = time_ms(true, iters, || {
        let base = xp
            .clone()
            .reshape([n * t, b])
            .matmul(w.clone().transpose())
            .reshape([n, t, a]);
        let y = xp.clone().matmul(b_br_view.clone()); // rhs 列主序
        let noise = y.matmul(a_ra.clone()); // rhs 连续 (n,r,a)
        (base + noise).mean_dim(1).squeeze_dim::<2>(1).mean()
    });
    println!("[10c] merged_full_colmajor   : {cur:8.2} ms/chunk");

    // ---- 11. CPU 入队吞吐（无同步）：小张量逐 op 的 CPU 开销 ----
    let small: Tensor<B, 2> = Tensor::random([n, 64], Distribution::Normal(0.0, 1.0), &device);
    let t0 = Instant::now();
    let mut acc = small.clone();
    for _ in 0..2000 {
        acc = acc.clone().mul_scalar(1.0000001);
    }
    acc.mean().into_scalar(); // 末尾一次同步
    let per_op_ms = t0.elapsed().as_secs_f64() * 1000.0 / 2000.0;
    println!("[11] enqueue_elemwise        : {per_op_ms:8.4} ms/op (CPU 入队)");

    // ---- 12. matmul 入队吞吐 ----
    let w64: Tensor<B, 2> = Tensor::random([64, 64], Distribution::Normal(0.0, 1.0), &device);
    let t0 = Instant::now();
    let mut acc2 = small.clone();
    for _ in 0..500 {
        acc2 = acc2.clone().matmul(w64.clone());
    }
    acc2.mean().into_scalar();
    let per_op_ms = t0.elapsed().as_secs_f64() * 1000.0 / 500.0;
    println!("[12] enqueue_matmul          : {per_op_ms:8.4} ms/op (CPU 入队)");

    // ---- 13. random 入队吞吐 ----
    let t0 = Instant::now();
    let mut acc3 = small.clone();
    for _ in 0..500 {
        acc3 = acc3.clone()
            + Tensor::<B, 2>::random([n, 64], Distribution::Normal(0.0, 1.0), &device);
    }
    acc3.mean().into_scalar();
    let per_op_ms = t0.elapsed().as_secs_f64() * 1000.0 / 500.0;
    println!("[13] enqueue_random          : {per_op_ms:8.4} ms/op (CPU 入队)");

    // ---- 14. 多流重叠测试：S0/S1 各 50 次 GEMM，单流 vs 双流 ----
    let t0 = Instant::now();
    let mut a1 = small.clone();
    for _ in 0..50 {
        a1 = a1.clone().matmul(w64.clone());
    }
    a1.mean().into_scalar();
    let single = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let s1 = cubecl::stream_id::StreamId { value: 1 };
    let f = s1.executes(|| {
        let mut a2 = small.clone();
        for _ in 0..50 {
            a2 = a2.clone().matmul(w64.clone());
        }
        a2
    });
    let mut a0 = small.clone();
    for _ in 0..50 {
        a0 = a0.clone().matmul(w64.clone());
    }
    let r14 = (a0 + f).mean().into_scalar();
    let dual = t0.elapsed().as_secs_f64();
    println!(
        "[14] stream_overlap          : single={single:.3}s dual={dual:.3}s (dual≈single 无重叠；≈single/2 有重叠)"
    );

    // ---- 15. 真实负载多流测试：每流做一次 3 层噪声生成（≈一次 prefetch 的 GPU 量）----
    let do_gen = |dev: &Device<B>| {
        let mut acc: Tensor<B, 1> = Tensor::zeros([1], dev);
        for (aa, bb) in [(a, b), (128usize, 128usize), (10usize, 128usize)] {
            let half = n / 2;
            let b_even: Tensor<B, 3> =
                Tensor::random([half, r, bb], Distribution::Normal(0.0, 1.0), dev);
            let b_t = Tensor::cat(vec![b_even.clone(), b_even.neg()], 0);
            let a_even: Tensor<B, 3> =
                Tensor::random([half, r, aa], Distribution::Normal(0.0, 1.0), dev);
            let a_t = Tensor::cat(vec![a_even.clone(), a_even.neg()], 0).mul_scalar(0.025);
            acc = acc.clone() + b_t.sum_dim(1).squeeze_dim::<2>(1).sum();
            acc = acc.clone() + a_t.sum_dim(1).squeeze_dim::<2>(1).sum();
        }
        acc
    };
    let t0 = Instant::now();
    let r1 = do_gen(&device);
    let r2 = do_gen(&device);
    (r1 + r2).into_scalar();
    let seq = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let f = s1.executes(|| do_gen(&device));
    let r0 = do_gen(&device);
    (r0 + f).into_scalar();
    let par = t0.elapsed().as_secs_f64();
    println!(
        "[15] gen_real_overlap        : seq={seq:.3}s par={par:.3}s (par≈seq 无重叠；≈seq/2 有重叠)"
    );
}
