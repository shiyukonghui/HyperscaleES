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
    if std::env::var("BENCH_RAW").map(|v| v == "1").unwrap_or(false) {
        eprintln!("    raw={times:?}");
    }
    sum / n
}

/// 同 time_ms，但闭包返回 rank-3 张量（用 into_data 同步）。
fn time_ms3<F: FnMut() -> Tensor<B, 3>>(warmup: bool, iters: usize, mut f: F) -> f64 {
    if warmup {
        let _ = f().into_data();
    }
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = f();
        out.into_data(); // 强制同步
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    if std::env::var("BENCH_RAW").map(|v| v == "1").unwrap_or(false) {
        eprintln!("    raw={times:?}");
    }
    times.iter().sum::<f64>() / times.len() as f64
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

        // [0b] 半噪声配对 einsum 校验：lora_einsum_pair_half（只存前半，配对隐含）
        // 与 lora_einsum_pair（完整配对张量）数学一致。
        let n_s = 8usize;
        let r_s = 3usize;
        let a_s = 2usize;
        let b_s = 3usize;
        let a_half: Tensor<B, 3> =
            Tensor::random([n_s / 2, r_s, a_s], Distribution::Normal(0.0, 1.0), &device);
        let b_half: Tensor<B, 3> =
            Tensor::random([n_s / 2, r_s, b_s], Distribution::Normal(0.0, 1.0), &device);
        let a_t = Tensor::cat(vec![a_half.clone(), a_half.clone().neg()], 0);
        let b_t = Tensor::cat(vec![b_half.clone(), b_half.clone().neg()], 0);
        let scores: Tensor<B, 1> = Tensor::random([n_s], Distribution::Normal(0.0, 1.0), &device);
        let (g1, o1) = hyperscalees_noiser::eggroll::lora_einsum_pair(&a_t, &b_t, &scores, &device);
        let (g2, o2) =
            hyperscalees_noiser::eggroll::lora_einsum_pair_half(&a_half, &b_half, &scores, &device);
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
        println!("[0b] half_pair_einsum_check   : raw={d_raw:.3e} ones={d_ones:.3e} (应 <1e-4)");
        assert!(d_raw < 1e-4 && d_ones < 1e-4, "半噪声配对 einsum 数值错误");

        // [0c] 半量噪声生成校验：返回 (n/2, r, *) 张量，分布 N(mean, std²)。
        let n_s = 16usize;
        let r_s = 64usize;
        let b_s = 64usize;
        let (a_g, b_g) = hyperscalees::cublas::gen_lora_noise_antipodal(
            n_s, r_s, 16, b_s, 0.25, &device,
        );
        assert_eq!(b_g.dims(), [n_s / 2, r_s, b_s], "B' 应为 (n/2, r, b)");
        assert_eq!(a_g.dims(), [n_s / 2, r_s, 16], "A' 应为 (n/2, r, a)");
        let bv = b_g.into_data().into_vec::<f32>().unwrap();
        let half = n_s / 2 * r_s * b_s;
        let mean = bv.iter().sum::<f32>() / half as f32;
        let var = bv.iter().map(|x| x * x).sum::<f32>() / half as f32 - mean * mean;
        println!("[0c] half_noise_check        : mean={mean:.3} var={var:.3} (应≈0, ≈1)");
        assert!(mean.abs() < 0.1 && (var - 1.0).abs() < 0.2, "B' 半噪声分布异常");
        let av = a_g.into_data().into_vec::<f32>().unwrap();
        let mean_a = av.iter().sum::<f32>() / av.len() as f32;
        let var_a = av.iter().map(|x| x * x).sum::<f32>() / av.len() as f32 - mean_a * mean_a;
        println!("[0c2] half_noise_A_check     : mean={mean_a:.3} var={var_a:.3} (应≈0, ≈0.0625)");
        assert!(
            mean_a.abs() < 0.1 && (var_a - 0.0625).abs() < 0.02,
            "A' 半噪声分布异常"
        );

        // [0c4] cuda-oxide PRNG 内核校验：半量正态填充（PTX 经 cudarc 加载），
        // 分布与 cubek-random 一致（不逐位——种子不同；统计同分布即可）。
        // 注：PRNG 内核按连续扁平写，必须传 1D 张量（见集成文档 §10 bug 4）。
        let b_ox: Tensor<B, 1> = Tensor::empty([n_s / 2 * r_s * b_s], &device);
        hyperscalees::oxide::prng_normal_half(&b_ox, 0.0, 1.0, &device).unwrap();
        let oxv = b_ox.into_data().into_vec::<f32>().unwrap();
        let n_ox = oxv.len() as f32;
        let mean_ox = oxv.iter().sum::<f32>() / n_ox;
        let var_ox = oxv.iter().map(|x| x * x).sum::<f32>() / n_ox - mean_ox * mean_ox;
        println!(
            "[0c4] oxide_prng_check        : mean={mean_ox:.3} var={var_ox:.3} (应≈0, ≈1)"
        );
        assert!(
            mean_ox.abs() < 0.1 && (var_ox - 1.0).abs() < 0.2,
            "cuda-oxide PRNG 分布异常"
        );

        // [0m] cuda-oxide einsum 内核校验：einsum_pair_fused vs burn fp32 参考
        //（lora_einsum_pair_half 同款计算序列：f_pair 加权 + cat 拼接 + matmul），
        // 容差相对 1e-3（fp32 累加顺序差异；布局/配对错误给出 O(1) 级误差）。
        let n_e = 2000usize; // K = 1000·16 = 16000
        let r_e = 16usize;
        let a_e = 32usize;
        let b_e = 48usize; // 48 有 pitch（burn 256B 对齐 → 行 stride 64）
        let half_e = n_e / 2;
        // 数据隔离实验：与示例完全一致的确定性均匀数据（从 CPU 上传）
        let mut seed = 0x1234_5678_9abc_def0u64;
        let fill = |len: usize, scale: f32, s: &mut u64| -> Vec<f32> {
            let mut v = vec![0.0f32; len];
            for x in v.iter_mut() {
                *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((*s >> 33) as u32 as f64) / (1u64 << 31) as f64 - 1.0;
                *x = (u as f32) * scale;
            }
            v
        };
        let a_h: Tensor<B, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(fill(half_e * r_e * a_e, 1.0, &mut seed), [half_e, r_e, a_e].to_vec()),
            &device,
        );
        let b_h: Tensor<B, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(fill(half_e * r_e * b_e, 1.0, &mut seed), [half_e, r_e, b_e].to_vec()),
            &device,
        );
        let scores_e: Tensor<B, 1> = Tensor::from_data(
            burn::tensor::TensorData::new(fill(n_e, 1.0, &mut seed), [n_e].to_vec()),
            &device,
        );
        let f_pair = scores_e
            .clone()
            .slice([0..half_e])
            .add(scores_e.clone().slice([half_e..n_e]));
        let a_w = a_h.clone() * f_pair.reshape([half_e, 1, 1]);
        let a_stack = Tensor::cat(vec![a_w, a_h.clone()], 2).reshape([half_e * r_e, 2 * a_e]);
        let b_flat = b_h.clone().reshape([half_e * r_e, b_e]);
        let g_ref = a_stack.transpose().matmul(b_flat); // (2a, b)
        let g_raw_ref = g_ref.clone().slice([0..a_e, 0..b_e]).reshape([a_e, b_e]);
        let g_ones_ref = g_ref.slice([a_e..2 * a_e, 0..b_e]).reshape([a_e, b_e]).mul_scalar(2.0);
        let (o_raw, o_ones) =
            hyperscalees::oxide::einsum_pair_fused(&a_h, &b_h, &scores_e, &device).unwrap();
        // 对照：cuBLAS 版（全量反对称构造 → 半量配对等价）
        let a_full = Tensor::cat(vec![a_h.clone(), a_h.clone().neg()], 0); // (n, r, a)
        let b_full = Tensor::cat(vec![b_h.clone(), b_h.clone().neg()], 0); // (n, r, b)
        let (c_raw, c_ones) =
            hyperscalees::cublas::lora_einsum_pair_cublas(&a_full, &b_full, &scores_e, &device);
        let maxd = |x: Tensor<B, 2>, y: Tensor<B, 2>| {
            x.into_data()
                .into_vec::<f32>()
                .unwrap()
                .iter()
                .zip(y.into_data().into_vec::<f32>().unwrap().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max)
        };
        let d_oc_raw = maxd(o_raw.clone(), c_raw.clone());
        let d_oc_ones = maxd(o_ones.clone(), c_ones.clone());
        let d_om_raw = maxd(o_raw.clone(), g_raw_ref.clone());
        let d_om_ones = maxd(o_ones.clone(), g_ones_ref.clone());
        let d_cm_raw = maxd(c_raw.clone(), g_raw_ref.clone());
        let d_cm_ones = maxd(c_ones.clone(), g_ones_ref.clone());
        let scale_om = maxd(g_raw_ref.clone(), Tensor::<B, 2>::zeros([a_e, b_e], &device)) + 1.0;
        println!(
            "[0m] oxide_einsum_check       : raw={d_om_raw:.3e} ones={d_om_ones:.3e} (应 <{:.1e})",
            scale_om * 1e-3
        );
        println!(
            "[0m]  oxide-vs-cublas         : raw={d_oc_raw:.3e} ones={d_oc_ones:.3e} | cublas-vs-ref: raw={d_cm_raw:.3e} ones={d_cm_ones:.3e}"
        );
        {
            let ov = o_raw.into_data().into_vec::<f32>().unwrap();
            let rv = g_raw_ref.into_data().into_vec::<f32>().unwrap();
            for i in 0..8 {
                println!("[0m]   elem[{i}]: oxide={:.3} ref={:.3}", ov[i], rv[i]);
            }
            // m=1 行前 4 个 + m=31 行前 4 个
            for m in [1usize, 31] {
                for n in 0..4 {
                    let i = m * b_e + n;
                    println!(
                        "[0m]   (m={m},n={n}): oxide={:.3} ref={:.3}",
                        ov[i], rv[i]
                    );
                }
            }
            // 错位假设检查：oxide[m=1] vs ref[其他 m]
            for (mm, label) in [(1usize, "m=1"), (17, "m=17"), (9, "m=9"), (25, "m=25")] {
                println!(
                    "[0m]   ref[{label},n=0] = {:.3}  (oxide[m=1,n=0] = {:.3})",
                    rv[mm * b_e],
                    ov[1 * b_e]
                );
            }
            let mut bad_m = vec![0usize; a_e];
            for (i, (a, b)) in ov.iter().zip(rv.iter()).enumerate() {
                if (a - b).abs() > 1.0 {
                    bad_m[i / b_e] += 1;
                }
            }
            println!("[0m]   bad-by-m: {bad_m:?}");
            // ones 输出前 4 个（m=0..3, n=0）
            {
                let ov2 = o_ones.into_data().into_vec::<f32>().unwrap();
                let rv2 = g_ones_ref.into_data().into_vec::<f32>().unwrap();
                for m in 0..4 {
                    println!(
                        "[0m]   ones(m={m},n=0): oxide={:.3} ref={:.3}",
                        ov2[m * b_e],
                        rv2[m * b_e]
                    );
                }
            }
            // 打印最大偏差的位置（m, n）与值
            let mut worst = (0usize, 0.0f32);
            for (i, (a, b)) in ov.iter().zip(rv.iter()).enumerate() {
                let d = (a - b).abs();
                if d > worst.1 {
                    worst = (i, d);
                }
            }
            println!(
                "[0m]   worst: idx={} (m={}, n={}) oxide={:.3} ref={:.3}",
                worst.0,
                worst.0 / b_e,
                worst.0 % b_e,
                ov[worst.0],
                rv[worst.0]
            );
            // 统计错元素分布：按 n 列计数
            let mut bad_n = vec![0usize; b_e];
            for (i, (a, b)) in ov.iter().zip(rv.iter()).enumerate() {
                if (a - b).abs() > 1.0 {
                    bad_n[i % b_e] += 1;
                }
            }
            println!("[0m]   bad-by-n: {bad_n:?}");
            let mut bad_m = vec![0usize; a_e];
            for (i, (a, b)) in ov.iter().zip(rv.iter()).enumerate() {
                if (a - b).abs() > 1.0 {
                    bad_m[i / b_e] += 1;
                }
            }
            println!("[0m]   bad-by-m: {bad_m:?}");
        }
        assert!(
            d_om_raw < scale_om * 1e-3 && d_om_ones < scale_om * 1e-3,
            "cuda-oxide einsum 数值错误"
        );

        // [0d] cuda-oxide 融合 LIF 扫描校验：lif_fused vs burn run_lif。
        // 内核与 run_lif 同为 f32 同序列（v + (cur - v)·leak → fma 后逐位一致），
        // 用随机 v0（非零初值路径）与训练同款 tau_m=20/v_th=0.3。
        {
            use hyperscalees_models::snn::{run_lif, LifParams};
            let cur_l: Tensor<B, 3> = Tensor::random(
                [5, 12000, 128],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let v0_l: Tensor<B, 2> =
                Tensor::random([12000, 128], Distribution::Normal(0.0, 1.0), &device);
            let sp_ox =
                hyperscalees::oxide::lif_fused(&cur_l, &v0_l, 20.0, 0.3, &device).unwrap();
            let sp_ref = run_lif(
                LifParams { tau_m: 20.0, v_th: 0.3 },
                cur_l,
                v0_l,
            );
            let vo = sp_ox.into_data().into_vec::<f32>().unwrap();
            let vr = sp_ref.into_data().into_vec::<f32>().unwrap();
            let bad = vo
                .iter()
                .zip(vr.iter())
                .filter(|(x, y)| (*x - *y).abs() > 1e-6)
                .count();
            let maxd = vo
                .iter()
                .zip(vr.iter())
                .map(|(x, y)| (*x - *y).abs())
                .fold(0.0_f32, f32::max);
            println!(
                "[0d] oxide_lif_check          : bad={bad}/{} maxdiff={maxd:.2e} (应全 0)",
                vo.len()
            );
            assert!(bad == 0, "cuda-oxide LIF 数值错误（bad={bad}）");

            // [0d] 变体 2：非 256B 对齐 h=100（行 pitch 128 ≠ 100）——回归保护
            //（阶段 C 矩阵测试曾暴露 LIF 扁平访问 bug，见集成文档 §10 bug 3）。
            let cur_l2: Tensor<B, 3> = Tensor::random(
                [5, 4000, 100],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let v0_l2: Tensor<B, 2> =
                Tensor::random([4000, 100], Distribution::Normal(0.0, 1.0), &device);
            let sp_ox2 = hyperscalees::oxide::lif_fused(&cur_l2, &v0_l2, 20.0, 0.3, &device).unwrap();
            let sp_ref2 = run_lif(LifParams { tau_m: 20.0, v_th: 0.3 }, cur_l2, v0_l2);
            let vo2 = sp_ox2.into_data().into_vec::<f32>().unwrap();
            let vr2 = sp_ref2.into_data().into_vec::<f32>().unwrap();
            let bad2 = vo2
                .iter()
                .zip(vr2.iter())
                .filter(|(x, y)| (*x - *y).abs() > 1e-6)
                .count();
            println!("[0d] oxide_lif_check(h=100) : bad={bad2}/{} (应全 0)", vo2.len());
            assert!(bad2 == 0, "cuda-oxide LIF h=100 数值错误（bad={bad2}）");
        }

        // [0p] cuda-oxide 融合泊松编码统计校验（RNG 与 burn 不同源，仅承诺统计等价）：
        // p 线性斜坡 [0,1] → 全体发放率 ≈ 0.5、低半 ≈ 0.25、高半 ≈ 0.75，输出全 0/1。
        {
            use hyperscalees_envs::snn_mnist::poisson_encode;
            let n_p = 12000usize;
            let h_p = 784usize;
            let t_p = 8usize;
            let total_p = n_p * h_p;
            let p_vals: Vec<f32> = (0..total_p)
                .map(|i| (i as f32 + 0.5) / total_p as f32)
                .collect();
            let p_imgs: Tensor<B, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(p_vals.clone(), [n_p, h_p].to_vec()),
                &device,
            );
            let sp_ox =
                hyperscalees::oxide::poisson_encode_fused(&p_imgs, t_p, &device).unwrap();
            let sp_bu = poisson_encode(p_imgs.clone(), t_p);
            let rate = |v: Vec<f32>| -> (f32, f32, f32, bool) {
                let is01 = v.iter().all(|x| *x == 0.0 || *x == 1.0);
                let mut sum = 0.0f32;
                let mut sum_lo = 0.0f32;
                let mut sum_hi = 0.0f32;
                for i in 0..total_p {
                    let mut r = 0.0f32;
                    for tt in 0..t_p {
                        r += v[tt * total_p + i];
                    }
                    r /= t_p as f32;
                    sum += r;
                    if p_vals[i] < 0.5 {
                        sum_lo += r;
                    } else {
                        sum_hi += r;
                    }
                }
                (
                    sum / total_p as f32,
                    sum_lo / (total_p as f32 * 0.5),
                    sum_hi / (total_p as f32 * 0.5),
                    is01,
                )
            };
            let (g_ox, lo_ox, hi_ox, is01_ox) = rate(sp_ox.into_data().into_vec::<f32>().unwrap());
            let (g_bu, lo_bu, hi_bu, is01_bu) = rate(sp_bu.into_data().into_vec::<f32>().unwrap());
            println!(
                "[0p] oxide_poisson_check   : oxide rate={g_ox:.4}/{lo_ox:.4}/{hi_ox:.4} 01={is01_ox} | burn {g_bu:.4}/{lo_bu:.4}/{hi_bu:.4} (应 ≈0.5/0.25/0.75)"
            );
            assert!(
                is01_ox && (g_ox - 0.5).abs() < 0.01 && (lo_ox - 0.25).abs() < 0.01
                    && (hi_ox - 0.75).abs() < 0.01,
                "cuda-oxide 泊松编码统计异常: {g_ox}/{lo_ox}/{hi_ox}"
            );
        }

        // [0e] 通用 gemm 帮助函数校验（容差 1e-1：burn 侧该形状启用 TF32，cuBLAS 为
        // 纯 fp32，差异 ~2.5e-4 相对值；转置/布局错误会给出 O(1) 误差仍能抓住）。
        let m0 = 12usize;
        let k0 = 784usize;
        let n0 = 16usize;
        let am: Tensor<B, 2> = Tensor::random([m0, k0], Distribution::Normal(0.0, 1.0), &device);
        let bm: Tensor<B, 2> = Tensor::random([k0, n0], Distribution::Normal(0.0, 1.0), &device);
        let c_ref = am.clone().matmul(bm.clone());
        let c_cu = hyperscalees::cublas::gemm(&am, &bm, &device);
        let va = c_ref.into_data().into_vec::<f32>().unwrap();
        let vb = c_cu.into_data().into_vec::<f32>().unwrap();
        let maxd = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!("[0e] cublas_gemm_check       : maxdiff={maxd:.3e} (应 <1e-1)");
        assert!(maxd < 1e-1, "cuBLAS gemm 错误");

        // [0f] batched gemm 帮助函数校验：batched_gemm_bt / batched_gemm。
        // 约定：batched_gemm_bt 消费批在中间维的 x (m, n, k)（前向 (T, n, *) 布局），
        // 输出批在第一维 (n, m, r)；batched_gemm 消费 (n, m, k) 批在第一维。
        let nb = 4usize;
        let mb = 3usize;
        let kb = 784usize;
        let rb = 64usize;
        let xp: Tensor<B, 3> =
            Tensor::random([mb, nb, kb], Distribution::Normal(0.0, 1.0), &device);
        let b3: Tensor<B, 3> =
            Tensor::random([nb, rb, kb], Distribution::Normal(0.0, 1.0), &device);
        let y_ref = xp.clone().swap_dims(0, 1).matmul(b3.clone().swap_dims(1, 2)); // (n, m, r)
        let y_cu = hyperscalees::cublas::batched_gemm_bt(&xp, &b3, &device);
        let va = y_ref.clone().into_data().into_vec::<f32>().unwrap();
        let vb = y_cu.clone().into_data().into_vec::<f32>().unwrap();
        let maxd = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!("[0f] cublas_batched_bt_check  : maxdiff={maxd:.3e} (应 <1e-1)");
        println!("      ref={:?} cu={:?}", &va[..8], &vb[..8]);
        assert!(maxd < 1e-1, "cuBLAS batched_gemm_bt 错误");
        let lb = 16usize;
        // a3 缩放到 ~1/30：y 是上一级 GEMM 结果（std≈28，burn 侧 TF32 误差 ~7e-2），
        // 两级误差传播后 burn 参考与 cuBLAS 的差 ~2e-2；真实布局错误给出 O(1) 级误差。
        let a3: Tensor<B, 3> =
            Tensor::random([nb, rb, lb], Distribution::Normal(0.0, 1.0 / 30.0), &device);
        let z_ref = y_ref.clone().matmul(a3.clone()); // (n, m, l)
        // 分支 1：y 为 batched_gemm_bt 的转置视图（每批列主序）。
        let z_cu = hyperscalees::cublas::batched_gemm(&y_cu, &a3, &device);
        // 分支 2：y 为连续 (n, m, k)（每批行主序）。
        let z_cu2 = hyperscalees::cublas::batched_gemm(&y_ref, &a3, &device);
        let va = z_ref.into_data().into_vec::<f32>().unwrap();
        let vb = z_cu.into_data().into_vec::<f32>().unwrap();
        let vc = z_cu2.into_data().into_vec::<f32>().unwrap();
        let maxd = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        let maxd2 = va
            .iter()
            .zip(vc.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!("[0g] cublas_batched_check     : maxdiff(view)={maxd:.3e} maxdiff(contig)={maxd2:.3e} (应 <1e-1)");
        assert!(maxd < 1e-1, "cuBLAS batched_gemm 错误（转置视图分支）");
        assert!(maxd2 < 1e-1, "cuBLAS batched_gemm 错误（连续分支）");

        // [0d] 半噪声前向等价性：forward_batched_lora_half（只存前半，配对隐含）与
        // forward_batched_lora（完整配对张量）一致。用 v_th = -1e9 强制全发放
        // （线性区）：spike 图案确定性，误差只来自 GEMM 数值（TF32 ~1e-2），避免
        // 阈值翻转的混沌放大（真实阈值下 TF32 差异也会导致 O(1) 级差异）。
        use hyperscalees_models::snn::TrainableVthSnn;
        let model = TrainableVthSnn::new(784, 16, 16, 10, 0.3, &device);
        let xf: Tensor<B, 3> = Tensor::random([3, 4, 784], Distribution::Bernoulli(0.5), &device);
        let rank_s = 64usize;
        let mut noises_h: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = Vec::new();
        let mut noises_f: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = Vec::new();
        for (aa, bb) in [(16usize, 784usize), (16, 16), (10, 16)] {
            let (a_h, b_h) =
                hyperscalees::cublas::gen_lora_noise_antipodal(4, rank_s, aa, bb, 0.25, &device);
            noises_f.push((
                Tensor::cat(vec![a_h.clone(), a_h.clone().neg()], 0),
                Tensor::cat(vec![b_h.clone(), b_h.clone().neg()], 0),
            ));
            noises_h.push((a_h, b_h));
        }
        let vth_q = -1e9_f32;
        let out_ref = model.forward_batched_lora(xf.clone(), vth_q, vth_q, &noises_f);
        let out_half = model.forward_batched_lora_half(xf.clone(), vth_q, vth_q, &noises_h);
        let out_cu = hyperscalees::cublas::forward_batched_lora_cublas(
            &model,
            xf,
            vth_q,
            vth_q,
            &noises_f,
            &device,
        );
        let va = out_ref.clone().into_data().into_vec::<f32>().unwrap();
        let vb = out_half.into_data().into_vec::<f32>().unwrap();
        let vc = out_cu.into_data().into_vec::<f32>().unwrap();
        let maxd = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        let maxd2 = va
            .iter()
            .zip(vc.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!("[0d] half_forward_check       : maxdiff(half)={maxd:.3e} maxdiff(cublas)={maxd2:.3e} (应 <1e-1)");
        assert!(maxd < 1e-1, "半噪声前向与 burn 前向不一致");
        assert!(maxd2 < 1e-1, "cuBLAS 前向与 burn 前向不一致");
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

        // ---- 4m. cuda-oxide 融合配对 einsum（训练热路径候选）----
        let (a_h4, b_h4) = hyperscalees::cublas::gen_lora_noise_antipodal(
            n, r, a, b, 0.025, &device,
        );
        let cur = time_ms(true, iters, || {
            let (g1, g2) =
                hyperscalees::oxide::einsum_pair_fused(&a_h4, &b_h4, &scores_t, &device).unwrap();
            (g1 + g2).mean()
        });
        println!("[4m] einsum_pair_oxide     : {cur:8.2} ms/chunk");

        // ---- 4d. cuda-oxide 融合 LIF 扫描（对照 [8] lif_loop burn 基线）----
        let cur_d: Tensor<B, 3> = Tensor::random(
            [t, n, 128],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v0_d: Tensor<B, 2> = Tensor::zeros([n, 128], &device);
        let cur = time_ms(true, iters, || {
            hyperscalees::oxide::lif_fused(&cur_d, &v0_d, 20.0, 0.3, &device)
                .unwrap()
                .mean()
        });
        println!("[4L] lif_fused_oxide       : {cur:8.2} ms/chunk");
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

    // ---- 4P. cuda-oxide 融合泊松编码（对照 [5b] burn 基线）----
    let cur = time_ms(true, iters, || {
        hyperscalees::oxide::poisson_encode_fused(&x2, t, &device)
            .unwrap()
            .mean()
    });
    println!("[4P] poisson_fused_oxide    : {cur:8.2} ms/chunk");

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
    let _ = (a0 + f).mean().into_scalar(); // sink：确保 dual 计时包含计算
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

    // ---- 16+. 训练真实形状（chunk=12000, T=8, r=64）的 cuBLAS 分片计时 ----
    let nch = 12000usize;
    let t8 = 8usize;
    let rk = 64usize;
    let xr: Tensor<B, 3> = Tensor::random([t8, nch, b], Distribution::Bernoulli(0.5), &device);
    let br: Tensor<B, 3> = Tensor::random([nch, rk, b], Distribution::Normal(0.0, 1.0), &device);
    let ar: Tensor<B, 3> = Tensor::random([nch, rk, a], Distribution::Normal(0.0, 1.0), &device);
    let wr: Tensor<B, 2> =
        Tensor::random([a, b], Distribution::Normal(0.0, 1.0 / 28.0), &device);
    let x2r = xr.clone().reshape([t8 * nch, b]);
    let br2 = br.clone().swap_dims(1, 2); // (n, b, r) 列主序视图（burn 参考用）
    let xp = xr.clone().swap_dims(0, 1).reshape([nch, t8, b]); // 连续拷贝（burn 参考用）
    // [16] base GEMM：x (96000,784) @ w^T (16,784) —— cuBLAS 与 burn。
    let cu = time_ms(true, 3, || {
        hyperscalees::cublas::gemm_abt(&x2r.clone(), &wr, &device).sum()
    });
    let t0 = Instant::now();
    for _ in 0..3 {
        let _ = x2r.clone().matmul(wr.clone().transpose());
    }
    Tensor::<B, 1>::zeros([1], &device).into_scalar();
    let bu = t0.elapsed().as_secs_f64() / 3.0 * 1000.0;
    println!("[16] gemm_abt_base_fc1        : cu={cu:7.2} burn={bu:7.2} ms/chunk");
    // [17] 噪声第一步：x (T,n,in)@B'^T —— cuBLAS 批在中间（不拷贝）vs burn（先拷贝）。
    let cu = time_ms3(true, 3, || hyperscalees::cublas::batched_gemm_bt(&xr, &br, &device));
    let _ = xp.clone().matmul(br2.clone()); // 预热
    let t0 = Instant::now();
    for _ in 0..3 {
        let _ = xp.clone().matmul(br2.clone());
    }
    Tensor::<B, 1>::zeros([1], &device).into_scalar();
    let bu = t0.elapsed().as_secs_f64() / 3.0 * 1000.0;
    println!("[17] batched_bt_fc1           : cu={cu:7.2} burn={bu:7.2} ms/chunk");
    // [17b] 同 [17] 但输入先 permute 为 (n,T,in) 连续（每批连续）。
    let cu = time_ms3(true, 3, || {
        hyperscalees::cublas::batched_gemm_bt_first(&xp, &br, &device)
    });
    println!("[17b] batched_bt_fc1_first    : cu={cu:7.2} ms/chunk");
    // [18] 噪声第二步：y (n,T,r)@A' —— 输入为 batched_gemm_bt 的转置视图。
    let yv = hyperscalees::cublas::batched_gemm_bt(&xr, &br, &device);
    let cu = time_ms3(true, 3, || hyperscalees::cublas::batched_gemm(&yv, &ar, &device));
    println!("[18] batched_gemm_fc1_view    : cu={cu:7.2} ms/chunk");
    // [19] einsum fc1：gemm_atb (384000,32)@(384000,784)。
    let a_half = ar.clone().slice([0..nch / 2, 0..rk, 0..a]);
    let f2: Tensor<B, 1> =
        Tensor::random([nch / 2], Distribution::Uniform(0.5, 1.5), &device);
    let a_w = a_half.clone() * f2.reshape([nch / 2, 1, 1]);
    let a_stack = Tensor::cat(vec![a_w, a_half], 2).reshape([nch / 2 * rk, 2 * a]);
    let b_stack = br.clone().slice([0..nch / 2, 0..rk, 0..b]).reshape([nch / 2 * rk, b]);
    let cu = time_ms(true, 3, || {
        hyperscalees::cublas::gemm_atb(&a_stack, &b_stack, &device).sum()
    });
    let t0 = Instant::now();
    for _ in 0..3 {
        let _ = a_stack
            .clone()
            .transpose()
            .matmul(b_stack.clone()); // (m,k)@(k,n)
    }
    Tensor::<B, 1>::zeros([1], &device).into_scalar();
    let bu = t0.elapsed().as_secs_f64() / 3.0 * 1000.0;
    println!("[19] einsum_gemm_fc1          : cu={cu:7.2} burn={bu:7.2} ms/chunk");
    // [20] 噪声生成 fc1（反对称内核）：B' (12000,64,784) + A' (12000,64,16)。
    let t0 = Instant::now();
    for _ in 0..3 {
        let _ = hyperscalees::cublas::gen_lora_noise_antipodal(nch, rk, a, b, 0.025, &device);
    }
    Tensor::<B, 1>::zeros([1], &device).into_scalar();
    let g1 = t0.elapsed().as_secs_f64() / 3.0 * 1000.0;
    println!("[20] antipodal_gen_fc1        : {g1:7.2} ms/chunk");
    // [21] 完整 fc1 噪声线性层（base+bt+gemm 全 cuBLAS）。
    let cu = time_ms3(true, 3, || {
        let base = hyperscalees::cublas::gemm_abt(&x2r.clone(), &wr, &device)
            .reshape([t8, nch, a]);
        let y = hyperscalees::cublas::batched_gemm_bt(&xr, &br, &device);
        let z = hyperscalees::cublas::batched_gemm(&y, &ar, &device);
        base + z.swap_dims(0, 1)
    });
    println!("[21] lora_linear_fc1_cublas   : cu={cu:7.2} ms/chunk");
}
