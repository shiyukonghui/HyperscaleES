//! 反对称配对的标准正态随机生成（本项目 patch）。
//!
//! 一次内核调用生成完整的 `(n, r, b)` 张量，其中**后半样本是前半的逐位取负**
//! （`out[n/2 + i] = -out[i]`）——即训练热路径 `gen_gpu_lora_noise` 的
//! `cat([e, -e])` 语义，但省去 neg 与 cat 两次全量拷贝（fc1 每 chunk 省 ~5ms）。
//!
//! 数值一致性：前半线程的种子/状态序列与 `random_normal` 完全相同（线程种子
//! `1000000007 * ABSOLUTE_POS`），因此前半输出与旧实现逐位一致；后半线程用
//! `ABSOLUTE_POS - half_threads` 的种子重新生成同一序列并取负写入——与
//! `cat([e, -e])` 的逐位结果一致。
//!
//! 前置条件：`out.size() / 2` 必须是 `N_VALUES_PER_THREAD` 的整数倍（本工作负载的
//! 所有形状均满足：n/2·r·b 均能被 128 整除）。

use core::f32::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::{
    View,
    layout::{
        Coords1d,
        linear::{LinearView, linear_view},
    },
};

use crate::{
    N_VALUES_PER_THREAD, get_seeds, lcg_step, prng_cube_count, taus_step_0, taus_step_1,
    taus_step_2, to_unit_interval_open,
};

/// 反对称配对的标准正态填充：`out[n/2 + i] = -out[i]`，`out[0..n/2] ~ N(mean, std²)`。
pub fn random_normal_antipodal<R: Runtime>(
    client: &ComputeClient<R>,
    mean: f32,
    std: f32,
    out: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let half_elems = out.size() / 2;
    assert!(
        out.size() % 2 == 0 && half_elems % N_VALUES_PER_THREAD == 0,
        "反对称配对要求 size 为偶数且 size/2 是 {N_VALUES_PER_THREAD} 的整数倍，实际 size={}",
        out.size()
    );
    let seeds = get_seeds();
    let n_threads_total = out.size().div_ceil(N_VALUES_PER_THREAD);
    let half_threads = (half_elems / N_VALUES_PER_THREAD) as u32;

    let cube_dim = CubeDim::new(client, n_threads_total);
    let cube_count = prng_cube_count(out.size(), cube_dim, N_VALUES_PER_THREAD);

    let output_vector_size = if out.size() % 4 == 0 { 4 } else { 1 };
    let address_type = out.required_address_type(dtype.size());
    let output = linear_view(out);

    prng_normal_antipodal_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        address_type,
        output_vector_size,
        output,
        mean,
        std,
        seeds[0],
        seeds[1],
        seeds[2],
        seeds[3],
        half_threads,
        N_VALUES_PER_THREAD,
        dtype,
    );

    Ok(())
}

#[cube(launch, address_type = "dynamic")]
fn prng_normal_antipodal_kernel<E: Numeric, N: Size>(
    output: &mut LinearView<Vector<E, N>, ReadWrite>,
    mean: f32,
    std: f32,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    half_threads: u32,
    #[comptime] n_values_per_thread: usize,
    #[define(E)] _dtype: StorageType,
) {
    // 与通用 prng_kernel 不同：这里每个线程写**连续**的 n_values_per_thread 个元素块
    // （float4 向量写，warp 内相邻线程写相邻 16B 块，依然完全合并），使「元素 ↔ 线程」
    // 映射为简单的块映射：元素 e 由线程 e/n_values_per_thread 写入。这样后半线程
    // （ABSOLUTE_POS >= half_threads）用前半对应线程的种子重新生成同一序列并取负，
    // 即可逐位得到 `out[n/2 + i] = -out[i]`（n/2 个元素 = half_threads 个线程块）。
    let vectors_per_thread = n_values_per_thread / N::value();
    let write_index_base = ABSOLUTE_POS as usize * vectors_per_thread;

    // 后半线程：用前半对应线程的种子重新生成同一序列，写时取负。
    let is_second_half = (ABSOLUTE_POS as u32) >= half_threads;
    let source_pos = if is_second_half {
        ABSOLUTE_POS - half_threads as usize
    } else {
        ABSOLUTE_POS
    };

    #[allow(arithmetic_overflow)]
    let thread_seed = 1000000007u32 * source_pos as u32;

    let mut state_0 = thread_seed + seed_0;
    let mut state_1 = thread_seed + seed_1;
    let mut state_2 = thread_seed + seed_2;
    let mut state_3 = thread_seed + seed_3;

    let mut output_vector_0 = Vector::empty();
    let mut output_vector_1 = Vector::empty();

    let num_iterations = n_values_per_thread / N::value() / 2;
    #[unroll(num_iterations <= 8)]
    for vector_index in 0..num_iterations {
        #[unroll]
        for i in 0..N::value() {
            state_0 = taus_step_0(state_0);
            state_1 = taus_step_1(state_1);
            state_2 = taus_step_2(state_2);
            state_3 = lcg_step(state_3);

            let int_random = state_0 ^ state_1 ^ state_2 ^ state_3;
            let unit_0 = to_unit_interval_open(int_random);

            state_0 = taus_step_0(state_0);
            state_1 = taus_step_1(state_1);
            state_2 = taus_step_2(state_2);
            state_3 = lcg_step(state_3);

            let int_random = state_0 ^ state_1 ^ state_2 ^ state_3;
            let unit_1 = to_unit_interval_open(int_random);

            let coeff = unit_0.ln() * -2.0;
            let coeff = coeff.sqrt() * std;
            let trigo_arg = 2.0 * PI * unit_1;

            let normal_0 = f32::cos(trigo_arg) * coeff + mean;
            let normal_1 = f32::sin(trigo_arg) * coeff + mean;

            output_vector_0[i] = E::cast_from(if is_second_half { -normal_0 } else { normal_0 });
            output_vector_1[i] = E::cast_from(if is_second_half { -normal_1 } else { normal_1 });
        }

        output[write_index_base + vector_index] = output_vector_0;
        output[write_index_base + num_iterations + vector_index] = output_vector_1;
    }
}
