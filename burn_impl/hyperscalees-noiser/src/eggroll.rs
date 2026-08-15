//! EggRoll: a LoRA-noised matmul noiser plus ES parameter updates, ported
//! from `src/hyperscalees/noiser/eggroll.py`.
//!
//! The math mirrors the Python exactly:
//!
//! * `get_lora_update_params` : seedable `N(0, 1)` noise of shape `(a+b, r)`,
//!   split into `B` (`b x r`) and `A` (`a x r`), with `A` scaled by
//!   `sign * base_sigma` (sign from `thread_id % 2`).
//! * `get_nonlora_update_params` : seedable `N(0, 1)` noise of `param.shape`
//!   scaled by `sign * base_sigma`.
//! * `convert_fitnesses` : global or per-group z-score.
//! * `do_updates` : `new_grad = -(update_fn * sqrt(N))`, then one optimizer
//!   step with the shared [`Solver`] plumbing.

use burn::tensor::{Device, Int, Tensor, TensorData};
use hyperscalees_core::B;

use crate::noiser::{
    noise_seed, DeterministicNoise, FrozenNoiserParams, IterInfo, Noiser, NoiserParams, Solver,
};

/// The EggRoll noiser. A zero-sized marker implementing [`Noiser`].
#[derive(Clone, Copy, Debug, Default)]
pub struct EggRoll;

/// Build the frozen + mutable noiser parameters, mirroring
/// `EggRoll.init_noiser`. `params` is used only to size the optimizer state.
pub fn init_noiser(
    params: &[Tensor<B, 2>],
    sigma: f32,
    // Kept for API parity with `EggRoll.init_noiser`; the learning rate already
    // lives inside `solver`.
    _lr: f32,
    group_size: i32,
    freeze_nonlora: bool,
    noise_reuse: i32,
    rank: usize,
    solver: Solver,
    device: &Device<B>,
) -> (FrozenNoiserParams, NoiserParams) {
    let frozen = FrozenNoiserParams {
        group_size,
        freeze_nonlora,
        noise_reuse,
        rank,
        solver,
    };
    let opt_state = frozen.solver.init_state(params, device);
    let noiser = NoiserParams { sigma, opt_state };
    (frozen, noiser)
}

// ---------------------------------------------------------------------------
// Per-parameter noise helpers
// ---------------------------------------------------------------------------

/// Derive the `(true_epoch, true_thread_idx, sign)` triple from an
/// [`IterInfo`] and the noise-reuse factor, as in
/// `get_lora_update_params` / `get_nonlora_update_params`.
pub(crate) fn epoch_thread_sign(info: &IterInfo, noise_reuse: i32) -> (i32, i32, f32) {
    let true_epoch = if noise_reuse == 0 { 0 } else { info.epoch / noise_reuse };
    let true_thread = info.thread_id / 2;
    let sign = if info.thread_id % 2 == 0 { 1.0 } else { -1.0 };
    (true_epoch, true_thread, sign)
}

/// Signature of a per-parameter `_simple_lora_update` variant. Used so that
/// [`eggroll::EggRoll`] and [`crate::alteggroll::AltEggRoll`] can share the
/// whole [`crate::noiser::Noiser`] plumbing while differing only in the sign of
/// the LoRA gradient (EggRoll: `A @ B.T`; AltEggRoll: `sign(A) @ sign(B).T`).
pub(crate) type LoraUpdateFn = fn(
    sigma: f32,
    key: u64,
    shape: [usize; 2],
    scores: &[f32],
    iterinfos: &[IterInfo],
    frozen: &FrozenNoiserParams,
    device: &Device<B>,
) -> Tensor<B, 2>;

/// Deterministic LoRA noise parameters: returns `(A, B)`.
///
/// `A` has shape `(a, r)` and `B` has shape `(b, r)`. `A` is scaled by
/// `sign * base_sigma` (the caller is expected to pass
/// `sigma / sqrt(rank)` as `base_sigma`).
pub fn get_lora_update_params(
    base_sigma: f32,
    key_seed: u64,
    rank: usize,
    info: &IterInfo,
    a: usize,
    b: usize,
    noise_reuse: i32,
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let (true_epoch, true_thread, sign) = epoch_thread_sign(info, noise_reuse);
    let mut rng = DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));
    // Shape (a + b, r) with A taking the lower rows and B the upper rows.
    let lora = rng.normal_tensor([a + b, rank], device);
    let b_t = lora.clone().slice([0..b, 0..rank]); // b x r
    let a_t = lora.slice([b..a + b, 0..rank]); // a x r
    (a_t.mul_scalar(sign * base_sigma), b_t)
}

/// 批量 LoRA 噪声：一次为 `tids` 中 n 个样本并行生成 `(A, B)` 对。
///
/// 返回 `(A, B)`：`A` 形状 `(n, a, r)` 且已乘 `sign * base_sigma`，`B` 形状
/// `(n, b, r)` 为原始标准正态。第 i 行与
/// `get_lora_update_params(base_sigma, key_seed, rank, &IterInfo{epoch, thread_id: tids[i]},
/// a, b, noise_reuse, device)` 逐位一致（见测试
/// `batched_lora_noise_matches_per_thread_get_lora_update_params`）。
///
/// 相比逐样本循环（60k 样本时在 GPU 上不可行），这里在 CPU 上按行分块并行填充
/// 一个 flat 向量（每线程独立 buffer，无数据竞争），一次性上传后交给调用方做
/// 批量 einsum（GPU 批量 matmul）。
pub fn batched_lora_noise(
    base_sigma: f32,
    key_seed: u64,
    rank: usize,
    tids: &[i32],
    epoch: i32,
    noise_reuse: i32,
    a: usize,
    b: usize,
    device: &Device<B>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let n = tids.len();
    let flat = generate_lora_flat(key_seed, rank, tids, epoch, noise_reuse, a, b);
    lora_flat_to_tensors(flat, n, a, b, rank, base_sigma, device)
}

/// 并行生成 LoRA 原始噪声 flat 向量（形状按行优先为 `(n, a+b, r)`）。
///
/// 每行 i 与 `get_lora_update_params` 使用完全相同的种子派生（`epoch_thread_sign` +
/// `noise_seed`）与采样序列（行优先 `(a+b)*rank` 个标准正态），并把 `sign` 乘到
/// 该行的 A 区（偏移 `b*rank` 起 `a*rank` 个元素）。**未乘 `base_sigma`**——调用方
/// 在 A 区统一缩放（`lora_flat_to_tensors` 或在缓存上传时）。
///
/// `noise_reuse == 0` 时噪声与 `epoch` 无关（`epoch_thread_sign` 恒取 true_epoch=0），
/// 因此可预生成一次跨 epoch 复用（见 [`LoraNoiseCache`]）。
fn generate_lora_flat(
    key_seed: u64,
    rank: usize,
    tids: &[i32],
    epoch: i32,
    noise_reuse: i32,
    a: usize,
    b: usize,
) -> Vec<f32> {
    let n = tids.len();
    // 每个样本（每行）的元素数：先 B 区 b*rank 个，后 A 区 a*rank 个。
    let row_len = (a + b) * rank;
    if n == 0 {
        return Vec::new();
    }
    // 并行度：取 CPU 核数；行数少于核数时按行数。
    let cores = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let n_threads = cores.min(n);
    let rows_per_thread = n.div_ceil(n_threads);
    // 每个线程独立生成自己的行块到独立 buffer（避免对同一 Vec 的共享可变借用），
    // 最后按行序合并；行内填充顺序与 `normal_tensor([a + b, rank], _)` 行优先完全一致。
    let chunks: Vec<Vec<f32>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for t in 0..n_threads {
            let start_row = t * rows_per_thread;
            let end_row = ((t + 1) * rows_per_thread).min(n);
            if start_row >= end_row {
                continue;
            }
            let tids_ref = &tids[start_row..end_row];
            handles.push(s.spawn(move || {
                let mut buf = vec![0.0_f32; (end_row - start_row) * row_len];
                for (local_i, &tid) in tids_ref.iter().enumerate() {
                    // 与 get_lora_update_params 完全相同的种子派生与采样序列。
                    let (true_epoch, true_thread, sign) =
                        epoch_thread_sign(&IterInfo { epoch, thread_id: tid }, noise_reuse);
                    let mut rng =
                        DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));
                    let row_start = local_i * row_len;
                    // 行内依次生成 (a+b)*rank 个标准正态（行优先）。
                    for k in 0..row_len {
                        buf[row_start + k] = rng.standard_normal();
                    }
                    // 把 sign 乘到 A 区（偏移 b*rank 起 a*rank 个元素）。
                    for k in 0..a * rank {
                        buf[row_start + b * rank + k] *= sign;
                    }
                }
                buf
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // 合并为单个 flat 向量（形状 (n, a+b, r) 行优先）。
    let mut flat = Vec::with_capacity(n * row_len);
    for c in chunks {
        flat.extend(c);
    }
    flat
}

/// 把 flat 噪声向量（行优先 `(n, a+b, r)`，A 区已乘 sign 未乘 base_sigma）上传为
/// `(A (n,a,r) 已乘 sign*base_sigma, B (n,b,r) 原始标准正态)`。
fn lora_flat_to_tensors(
    flat: Vec<f32>,
    n: usize,
    a: usize,
    b: usize,
    rank: usize,
    base_sigma: f32,
    device: &Device<B>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    if n == 0 {
        return (
            Tensor::<B, 3>::empty([0, a, rank], device),
            Tensor::<B, 3>::empty([0, b, rank], device),
        );
    }
    let lora = Tensor::<B, 3>::from_data(TensorData::new(flat, vec![n, a + b, rank]), device);
    let b_t = lora.clone().slice([0..n, 0..b, 0..rank]); // (n, b, r)
    let a_t = lora.slice([0..n, b..a + b, 0..rank]); // (n, a, r)
    // A 区内元素已乘 sign，这里再乘 base_sigma，等价于 get_lora_update_params 的
    // `sign * base_sigma`（IEEE 乘法符号规则保证逐位一致）。
    (a_t.mul_scalar(base_sigma), b_t)
}

/// 跨 epoch 复用的 LoRA 噪声缓存。
///
/// 因为 `noise_reuse == 0` 时噪声只依赖 `(base_key, thread_id)`（true_epoch 恒为 0），
/// 同一参数同一 thread 的噪声在所有 epoch 完全相同。因此可在训练启动前一次性并行
/// 生成全部噪声（CPU RAM，约 20GB @ batch=60000/rank=64），之后每个 epoch 每个 chunk
/// 只需从缓存切片 + 上传，**完全省去每 epoch 的 CPU 随机数生成**（原来约 10G 次/epoch）。
#[derive(Debug, Clone)]
pub struct LoraNoiseCache {
    /// 每个参数一个缓冲（行优先 `(batch, a+b, r)`，A 区已乘 sign 未乘 base_sigma）；
    /// `None` 表示该参数不是 LoRA（无缓存）。
    pub buffers: Vec<Option<Vec<f32>>>,
    /// 每个参数的形状 `(a, b)`。
    pub shapes: Vec<[usize; 2]>,
    /// 总样本数（= accumulate * chunk）。
    pub batch: usize,
    /// LoRA rank。
    pub rank: usize,
}

impl LoraNoiseCache {
    /// 从缓存取第 `param_idx` 个参数第 `[lo, hi)` 行（对应 `thread_ids[lo..hi]`）的
    /// `(A (n,a,r) 已乘 sign*base_sigma, B (n,b,r))`，一次性切片复制 + 上传。
    pub fn slice_upload(
        &self,
        param_idx: usize,
        lo: usize,
        hi: usize,
        base_sigma: f32,
        device: &Device<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let buf = self.buffers[param_idx].as_ref().expect("参数不是 LoRA（无缓存）");
        let [a, b] = self.shapes[param_idx];
        let rank = self.rank;
        let row_len = (a + b) * rank;
        let n = hi - lo;
        // from_data 需要 owned Vec，这里做一次切片复制；
        // base_sigma 缩放统一交给 `lora_flat_to_tensors`（对 A 区乘 base_sigma）。
        let slice = buf[lo * row_len..hi * row_len].to_vec();
        lora_flat_to_tensors(slice, n, a, b, rank, base_sigma, device)
    }
}

/// 为每个 LoRA 参数（`es_classes[i] == 1`）预生成全 batch 噪声缓存。
///
/// 仅支持 `noise_reuse == 0`（噪声与 epoch 无关才可跨 epoch 复用）；生成时 epoch 取 0
/// 即可（`epoch_thread_sign` 在 noise_reuse=0 下返回 true_epoch=0）。
pub fn build_lora_noise_cache(
    params: &[Tensor<B, 2>],
    base_keys: &[u64],
    es_classes: &[i32],
    batch: usize,
    rank: usize,
    noise_reuse: i32,
) -> LoraNoiseCache {
    assert!(
        noise_reuse == 0,
        "噪声缓存仅支持 noise_reuse=0（噪声与 epoch 无关才可跨 epoch 复用）"
    );
    let buffers = params
        .iter()
        .zip(base_keys.iter())
        .zip(es_classes.iter())
        .map(|((p, key), cls)| {
            if *cls != 1 {
                return None;
            }
            let [a, b] = p.dims();
            // tids = 0..batch（训练中 thread_ids 恒为 arange(batch)，缓存按位置索引）。
            let tids: Vec<i32> = (0..batch as i32).collect();
            let flat = generate_lora_flat(*key, rank, &tids, 0, 0, a, b);
            Some(flat)
        })
        .collect();
    let shapes = params.iter().map(|p| p.dims()).collect();
    LoraNoiseCache {
        buffers,
        shapes,
        batch,
        rank,
    }
}

/// Deterministic dense (`nonlora`) noise of the parameter's shape.
pub fn get_nonlora_update_params(
    base_sigma: f32,
    key_seed: u64,
    info: &IterInfo,
    shape: [usize; 2],
    noise_reuse: i32,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let (true_epoch, true_thread, sign) = epoch_thread_sign(info, noise_reuse);
    let mut rng = DeterministicNoise::new(noise_seed(key_seed, true_epoch, true_thread));
    let updates = rng.normal_tensor(shape, device);
    updates.mul_scalar(sign * base_sigma)
}

// ---------------------------------------------------------------------------
// Per-parameter update functions
// ---------------------------------------------------------------------------

/// `_simple_full_update`: `1/N * sum_i f_i * noise_i` or zeros when frozen.
pub(crate) fn simple_full_update(
    sigma: f32,
    key: u64,
    shape: [usize; 2],
    scores: &[f32],
    iterinfos: &[IterInfo],
    frozen: &FrozenNoiserParams,
    device: &Device<B>,
) -> Tensor<B, 2> {
    if frozen.freeze_nonlora {
        return Tensor::<B, 2>::zeros(shape, device);
    }
    let n = scores.len() as f32;
    let mut acc = Tensor::<B, 2>::zeros(shape, device);
    for (i, info) in iterinfos.iter().enumerate() {
        let up = get_nonlora_update_params(sigma, key, info, shape, frozen.noise_reuse, device);
        acc = acc + up.mul_scalar(scores[i]);
    }
    acc.mul_scalar(1.0 / n)
}

/// `_simple_lora_update`: `1/N * sum_i f_i * (A_i @ B_i^T)`.
pub(crate) fn simple_lora_update(
    sigma: f32,
    key: u64,
    shape: [usize; 2],
    scores: &[f32],
    iterinfos: &[IterInfo],
    frozen: &FrozenNoiserParams,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let n = scores.len() as f32;
    let rank = frozen.rank;
    let [a, b] = shape;
    let base_sigma = sigma / (rank as f32).sqrt();
    let mut acc = Tensor::<B, 2>::zeros([a, b], device);
    for (i, info) in iterinfos.iter().enumerate() {
        let (a_t, b_t) =
            get_lora_update_params(base_sigma, key, rank, info, a, b, frozen.noise_reuse, device);
        acc = acc + a_t.matmul(b_t.transpose()).mul_scalar(scores[i]);
    }
    acc.mul_scalar(1.0 / n)
}

/// `_noop_update`: zeros of the parameter's shape.
pub(crate) fn noop_update(shape: [usize; 2], device: &Device<B>) -> Tensor<B, 2> {
    Tensor::<B, 2>::zeros(shape, device)
}

/// `_do_update`: choose the update fn by `map_classification` and return the
/// *negated* gradient scaled by `sqrt(N)`. `lora_update` is the
/// `_simple_lora_update` variant (EggRoll uses `A @ B.T`; AltEggRoll uses
/// `sign(A) @ sign(B).T`).
pub(crate) fn do_update_with(
    lora_update: LoraUpdateFn,
    param: &Tensor<B, 2>,
    base_key: u64,
    fitnesses: &Tensor<B, 1>,
    iterinfos: &[IterInfo],
    map_class: i32,
    sigma: f32,
    frozen: &FrozenNoiserParams,
) -> Tensor<B, 2> {
    let device = param.device();
    let shape = param.dims();
    let scores = fitnesses.clone().into_data().into_vec::<f32>().unwrap();
    let g = match map_class {
        0 => simple_full_update(sigma, base_key, shape, &scores, iterinfos, frozen, &device),
        1 => lora_update(sigma, base_key, shape, &scores, iterinfos, frozen, &device),
        _ => noop_update(shape, &device),
    };
    let n = scores.len() as f32;
    g.mul_scalar(n.sqrt()).neg()
}

// ---------------------------------------------------------------------------
// Shared `Noiser` forward + update machinery
// ---------------------------------------------------------------------------
//
// These `pub(crate)` bodies let [`EggRoll`] and [`crate::alteggroll::AltEggRoll`]
// share the whole forward/update plumbing. The two noisers differ only in
// `_simple_lora_update` (EggRoll: `A @ B.T`; AltEggRoll: `sign(A) @ sign(B).T`),
// so `do_mm`/`do_Tmm`/`get_noisy_standard` are shared verbatim and only
// `do_updates` is parameterised by the `_simple_lora_update` variant.
// ([`crate::eggroll_bs::EggRollBS`] keeps its own module because it needs
// masked noise params and a baseline-subtraction `convert_fitnesses`.)

/// Shared `do_mm` (EggRoll/AltEggRoll): `base + x @ B @ A.T` under LoRA noise.
pub(crate) fn do_mm_impl(
    frozen: &FrozenNoiserParams,
    noiser: &NoiserParams,
    param: &Tensor<B, 2>,
    base_key: u64,
    iterinfo: Option<&IterInfo>,
    x: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let base = x.clone().matmul(param.clone().transpose());
    match iterinfo {
        None => base,
        Some(info) => {
            let [a, b] = param.dims();
            let (a_t, b_t) = get_lora_update_params(
                noiser.sigma / (frozen.rank as f32).sqrt(),
                base_key,
                frozen.rank,
                info,
                a,
                b,
                frozen.noise_reuse,
                &param.device(),
            );
            // base + x @ B @ A.T
            base + x.matmul(b_t).matmul(a_t.transpose())
        }
    }
}

/// Shared `do_Tmm` (EggRoll/AltEggRoll): `base + x @ A @ B.T` under LoRA noise.
#[allow(non_snake_case)]
pub(crate) fn do_Tmm_impl(
    frozen: &FrozenNoiserParams,
    noiser: &NoiserParams,
    param: &Tensor<B, 2>,
    base_key: u64,
    iterinfo: Option<&IterInfo>,
    x: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let base = x.clone().matmul(param.clone());
    match iterinfo {
        None => base,
        Some(info) => {
            let [a, b] = param.dims();
            let (a_t, b_t) = get_lora_update_params(
                noiser.sigma / (frozen.rank as f32).sqrt(),
                base_key,
                frozen.rank,
                info,
                a,
                b,
                frozen.noise_reuse,
                &param.device(),
            );
            // base + x @ A @ B.T
            base + x.matmul(a_t).matmul(b_t.transpose())
        }
    }
}

/// Shared `get_noisy_standard` (EggRoll/AltEggRoll): `param + nonlora` noise.
pub(crate) fn get_noisy_standard_impl(
    frozen: &FrozenNoiserParams,
    noiser: &NoiserParams,
    param: &Tensor<B, 2>,
    base_key: u64,
    iterinfo: Option<&IterInfo>,
) -> Tensor<B, 2> {
    match iterinfo {
        None => param.clone(),
        Some(_) if frozen.freeze_nonlora => param.clone(),
        Some(ref info) => {
            let shape = param.dims();
            let noise = get_nonlora_update_params(
                noiser.sigma,
                base_key,
                info,
                shape,
                frozen.noise_reuse,
                &param.device(),
            );
            param.clone() + noise
        }
    }
}

/// Shared `do_updates`: compute per-param grad via `_do_update` and apply the
/// solver. `lora_update` is the `_simple_lora_update` variant (differs between
/// EggRoll and AltEggRoll).
pub(crate) fn do_updates_impl(
    lora_update: LoraUpdateFn,
    frozen: &FrozenNoiserParams,
    noiser: &mut NoiserParams,
    params: &[Tensor<B, 2>],
    base_keys: &[u64],
    fitnesses: Tensor<B, 1>,
    iterinfos: &[IterInfo],
    es_classes: &[i32],
) -> Vec<Tensor<B, 2>> {
    if params.is_empty() {
        return Vec::new();
    }
    let grads: Vec<Tensor<B, 2>> = params
        .iter()
        .zip(base_keys.iter())
        .zip(es_classes.iter())
        .map(|((p, k), c)| do_update_with(lora_update, p, *k, &fitnesses, iterinfos, *c, noiser.sigma, frozen))
        .collect();
    frozen.solver.update(params, &grads, &mut noiser.opt_state)
}

/// K 段 chunked einsum 累积更新，镜像 `snn_mnist_train_accumulate.py` 的
/// `_accumulated_update`（小批次等效大批次架构，见
/// `docs/es_batch_accumulation_architecture.md`）。
///
/// `conv_full` 是全文全局 z-score 后的 fitness（形状 `(batch,)`），
/// `thread_ids` 为其逐样本的全局唯一线程 id，满足
/// `conv_full.len() == accumulate * chunk == thread_ids.len()`。
/// `accumulate` 即段数 K，`chunk` 为每段样本数。
///
/// 逐段 k：用 `epoch` 与第 k 段的 `thread_ids` 构造 `IterInfo` 列表，对每个
/// 参数调 `do_update_with(simple_lora_update, ...)`（返回 `-einsum_k/sqrt(chunk)`），
/// 跨全部 K 段累加梯度；累加后 `÷sqrt(K)` 恢复 `sqrt(batch)` 尺度；最后*一次*
/// `solver.update` 应用全部梯度。
///
/// 数学上 `sum_k -einsum_k/sqrt(chunk) / sqrt(K) = -einsum_total / sqrt(batch)`，
/// 与全批单次 `do_updates` 严格一致（仅 float32 累加顺序的非确定性差异），且
/// solver step 只 +1（而非逐段单独更新 +K）。
///
/// 无缓存版本（`noise_cache = None`）：LoRA 噪声逐段现场生成（CPU 并行 + 上传）；
/// 训练热路径请传 [`LoraNoiseCache`]，从缓存切片 + 上传，免去每 epoch 的 CPU 随机数。
pub fn accumulated_update(
    frozen: &FrozenNoiserParams,
    noiser: &mut NoiserParams,
    params: &[Tensor<B, 2>],
    base_keys: &[u64],
    es_classes: &[i32],
    conv_full: Tensor<B, 1>,
    thread_ids: &[i32],
    epoch: i32,
    accumulate: usize,
    chunk: usize,
) -> Vec<Tensor<B, 2>> {
    accumulated_update_cached(
        frozen,
        noiser,
        params,
        base_keys,
        es_classes,
        conv_full,
        thread_ids,
        epoch,
        accumulate,
        chunk,
        None,
    )
}

/// [`accumulated_update`] 的缓存版本：`noise_cache` 提供预生成的 LoRA 噪声，LoRA 参数
/// 从缓存切片上传（不再现场生成随机数）。缓存索引与 `params` 一一对应。
pub fn accumulated_update_cached(
    frozen: &FrozenNoiserParams,
    noiser: &mut NoiserParams,
    params: &[Tensor<B, 2>],
    base_keys: &[u64],
    es_classes: &[i32],
    conv_full: Tensor<B, 1>,
    thread_ids: &[i32],
    epoch: i32,
    accumulate: usize,
    chunk: usize,
    noise_cache: Option<&LoraNoiseCache>,
) -> Vec<Tensor<B, 2>> {
    if params.is_empty() {
        return Vec::new();
    }
    // 全文 fitness 的 CPU 视图（形状 (batch,)，batch = accumulate*chunk），各段直接
    // 切片复用，避免反复把 GPU 张量搬回 CPU。
    let scores_full: Vec<f32> = conv_full.clone().into_data().into_vec::<f32>().unwrap();
    // 每个参数的梯度累加器，初始化为 0（与 Python `jnp.zeros_like` 一致）。
    let mut grad_acc: Vec<Tensor<B, 2>> = params
        .iter()
        .map(|p| Tensor::<B, 2>::zeros(p.dims(), &p.device()))
        .collect();

    // 逐段 k：LoRA 参数走 GPU 批量 einsum 路径；dense（FULL）参数走批量噪声加权
    // 求和；NOOP 参数贡献为零。（60k batch 下逐样本循环在 GPU 上不可行，全部批量化。）
    let timing = std::env::var("ACC_TIMING").map(|v| v == "1").unwrap_or(false);
    let t_func = std::time::Instant::now();
    let mut t_slice = 0.0_f32;
    for k in 0..accumulate {
        let lo = k * chunk;
        let hi = lo + chunk;
        // 第 k 段的全局唯一 thread id 列表。
        let tids_k = &thread_ids[lo..hi];
        // 对每个参数计算该段的贡献并累加。
        for ((acc, p), (param_idx, (key, cls))) in grad_acc
            .iter_mut()
            .zip(params)
            .zip(base_keys.iter().zip(es_classes.iter()).enumerate())
        {
            if *cls == 1 {
                // LoRA 批量 einsum：一次为 chunk 个样本取 (A, B)，按 fitness 加权后
                // 批量 matmul，得到未缩放（不除以 chunk）的
                // `einsum_k = Σ_i f_i·(A_i @ B_i^T)`（A_i 已乘 sign*base_sigma）。
                let [a, b] = p.dims();
                let base_sigma = noiser.sigma / (frozen.rank as f32).sqrt();
                let t_s = std::time::Instant::now();
                let (a_t, b_t) = match noise_cache {
                    // 热路径：从缓存切片 + 上传（免 CPU 随机数）。
                    Some(cache) => cache.slice_upload(param_idx, lo, hi, base_sigma, &p.device()),
                    // 冷路径（verify/无缓存）：现场生成。
                    None => batched_lora_noise(
                        base_sigma,
                        *key,
                        frozen.rank,
                        tids_k,
                        epoch,
                        frozen.noise_reuse,
                        a,
                        b,
                        &p.device(),
                    ),
                };
                t_slice += t_s.elapsed().as_secs_f32();
                let scores_k = Tensor::<B, 1>::from_data(&scores_full[lo..hi], &p.device());
                let a_w = a_t * scores_k.reshape([chunk, 1, 1]); // (n, a, r) 按样本加权
                // einsum('nir,njr->ij') 的 2D gemm 等价式：把 A（(n,a,r)）与 B（(n,b,r)）
                // 各自 swap 成 (n,r,*) 后按行优先 reshape 为 (n*r, a) 与 (n*r, b)。
                // 两者按相同的 (n,r) 顺序展平，故 `A_flat^T @ B_flat = Σ_{n,r} A[n,:,r]⊗B[n,:,r]`
                // 正是 einsum 结果。相比 `(n,a,r)@(n,r,b)` 的批量 matmul（CubeCL 对 3D
                // batched gemm 优化差、且会物化 (n,a,b) 中间张量），2D gemm 走优化内核且
                // 不物化大中间量，是 GPU 热路径的关键加速。
                let a_flat = a_w.swap_dims(1, 2).reshape([chunk * frozen.rank, a]); // (n*r, a)
                let b_flat = b_t.swap_dims(1, 2).reshape([chunk * frozen.rank, b]); // (n*r, b)
                let g = a_flat.clone().transpose().matmul(b_flat); // (a, b)
                *acc = acc.clone() + g;
            } else if *cls == 0 {
                // FULL（dense）：批量噪声加权求和。逐样本语义与旧 `do_update_with` /
                // `simple_full_update` 一致：`einsum_k = Σ_i f_i·noise_i`，其中
                // `noise_i = sign_i·sigma·N(0,1)`（与 `get_nonlora_update_params` 完全
                // 相同的种子派生与采样序列）。批量版一次上传 (n,a,b) 噪声后加权求和，
                // 替代原逐样本循环（(1,1) 等小参数在 60k batch 下会产生 18 万次微小
                // GPU 操作，成为明显瓶颈）。
                let [a, b] = p.dims();
                let prod = a * b;
                let mut flat = vec![0.0_f32; chunk * prod];
                for (i, &tid) in tids_k.iter().enumerate() {
                    let (true_epoch, true_thread, sign) =
                        epoch_thread_sign(&IterInfo { epoch, thread_id: tid }, frozen.noise_reuse);
                    let mut rng =
                        DeterministicNoise::new(noise_seed(*key, true_epoch, true_thread));
                    let base = i * prod;
                    for j in 0..prod {
                        flat[base + j] = sign * noiser.sigma * rng.standard_normal();
                    }
                }
                let noise =
                    Tensor::<B, 3>::from_data(TensorData::new(flat, vec![chunk, a, b]), &p.device());
                let scores_k = Tensor::<B, 1>::from_data(&scores_full[lo..hi], &p.device());
                let weighted = noise * scores_k.reshape([chunk, 1, 1]); // (n,a,b)
                let einsum_k = weighted.sum_dim(0).squeeze_dim::<2>(0); // (a,b)
                *acc = acc.clone() + einsum_k;
            }
            // NOOP(2,3)：贡献为零，跳过。
        }
    }

    // 统一缩放：`-1/sqrt(chunk*accumulate) = -1/sqrt(batch)`，数学上等于
    // `-Σeinsum_k/sqrt(batch)`，与全批一次性 do_updates 严格一致（见模块注释，
    // 仅 float32 累加顺序的非确定性差异）。
    let scale = -1.0 / ((chunk * accumulate) as f32).sqrt();
    let grads: Vec<Tensor<B, 2>> = grad_acc.into_iter().map(|g| g.mul_scalar(scale)).collect();

    if timing {
        eprintln!(
            "  [accumulated_update] total={:.2}s slice_upload={:.2}s",
            t_func.elapsed().as_secs_f32(),
            t_slice
        );
    }
    // 最后一次性 solver 更新（step 只 +1，而非逐段 +K）。
    frozen.solver.update(params, &grads, &mut noiser.opt_state)
}

// ---------------------------------------------------------------------------
// 内联（GPU 噪声）路径辅助函数
// ---------------------------------------------------------------------------
//
// 训练热路径（见 `accumulate_train` 二进制）在 GPU 上直接生成噪声（`Tensor::random`），
// 前向与梯度共享受同一份 (A, B)，因此**完全没有 CPU 随机数与 CPU→GPU 上传**（原缓存
// 切片上传是主要瓶颈：batch=60000 时每 epoch 约 40GB 上传）。因「先全部前向、再全局
// z-score」需要先有全部 raw fitness 才能归一化，这里用**仿射等价**：全局 z-score 是
// `conv = (raw - mean)/std`，而 einsum 对 fitness 线性，故
//
//   Σ_k einsum(conv_k) = (Σ einsum(raw) - mean · Σ einsum(1)) / std
//
// 于是可以逐 chunk 用 raw fitness 累积 `grad_acc`（加权 einsum）与 `ones_acc`
// （einsum over 全 1），最后一次性仿射修正 + solver 更新——数学上与两阶段
// `accumulated_update` 严格一致（测试 `inline_affine_matches_accumulated_two_phase` 验证）。

/// `einsum('nri,nrj->ij')` 的 raw 加权版：`Σ_i f_i·(A_i @ B_i^T)`。
///
/// A' 与 B' 均按 `(n, r, *)` 连续布局存储（`A'[n,r,i] = A[n,i,r]` 且 A' 已乘
/// sign*base_sigma；`B'[n,r,j] = B[n,j,r]`），因此展平 `reshape([n*r, *])` 为零拷贝
/// 视图，2D GEMM `(a, n·r) @ (n·r, b)` 即 `Σ_{n,r} A[n,i,r]·B[n,j,r]`。
/// `scores` 为 GPU 张量（训练热路径不再把 fitness 搬回 CPU）。
pub fn lora_einsum_raw(
    a_t: &Tensor<B, 3>,     // (n, r, a)，A 已乘 sign*base_sigma
    b_t: &Tensor<B, 3>,     // (n, r, b)
    scores: &Tensor<B, 1>,  // (n,)
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [n, r, a] = a_t.dims();
    let _ = device;
    let scores_t = scores.clone().reshape([n, 1, 1]); // (n,1,1)
    let a_w = a_t.clone() * scores_t; // (n,r,a) 按样本加权
    let a_flat = a_w.reshape([n * r, a]); // 连续，零拷贝
    let b_flat = b_t.clone().reshape([n * r, b_t.dims()[2]]); // 连续，零拷贝
    a_flat.transpose().matmul(b_flat) // (a, b)
}

/// `einsum('nri,nrj->ij')` 的 unweighted 版：`Σ_i (A_i @ B_i^T)`（仿射修正的 ones 项）。
pub fn lora_einsum_ones(a_t: &Tensor<B, 3>, b_t: &Tensor<B, 3>, device: &Device<B>) -> Tensor<B, 2> {
    let [n, r, a] = a_t.dims();
    let _ = device;
    let a_flat = a_t.clone().reshape([n * r, a]); // 连续，零拷贝
    let b_flat = b_t.clone().reshape([n * r, b_t.dims()[2]]); // 连续，零拷贝
    a_flat.transpose().matmul(b_flat) // (a, b)
}

/// dense（FULL）raw 加权版：`Σ_i f_i·noise_i`（noise 形状 (n,a,b)；scores 为 GPU 张量）。
pub fn dense_einsum_raw(noise: &Tensor<B, 3>, scores: &Tensor<B, 1>, device: &Device<B>) -> Tensor<B, 2> {
    let n = noise.dims()[0];
    let _ = device;
    let scores_t = scores.clone().reshape([n, 1, 1]);
    let weighted = noise.clone() * scores_t; // (n,a,b)
    weighted.sum_dim(0).squeeze_dim::<2>(0) // (a,b)
}

/// dense（FULL）ones 版：`Σ_i noise_i`（仿射修正的 ones 项）。
pub fn dense_einsum_ones(noise: &Tensor<B, 3>, _device: &Device<B>) -> Tensor<B, 2> {
    noise.clone().sum_dim(0).squeeze_dim::<2>(0) // (a,b)
}

/// 反对称配对版 raw+ones 合并 einsum（训练热路径专用）。
///
/// 前提：`A'[half+i] = -A'[i]`、`B'[half+i] = -B'[i]`（见 `gen_gpu_lora_noise` 的
/// 反对称配对）。此时配对消去后半样本：
///
/// ```text
/// g_raw  = Σ_n f_n·(A'_n ⊗ B'_n) = Σ_{i<half} A''_i ⊗ B'_i，A''_i = (f_i + f_{half+i})·A'_i
/// g_ones = Σ_n (A'_n ⊗ B'_n)     = 2·Σ_{i<half} A'_i ⊗ B'_i
/// ```
///
/// 两者共享同一 `B_half`：把 `[A''; A_half]` 沿 a 轴拼接为 `(half, r, 2a)`，一次
/// 2D GEMM `(2a, half·r) @ (half·r, b)` 同时产出 `g_raw`（上半行）与 `g_ones'`
/// （下半行，末尾 ×2）。相比两个全 K GEMM（fc1 各 ~21ms），**FLOPs 恰好减半**
/// （半 K × 单次 M=2a 输出），且 B 只读一次。
pub fn lora_einsum_pair(
    a_t: &Tensor<B, 3>,     // (n, r, a)，A 已乘 sign*base_sigma，反对称配对
    b_t: &Tensor<B, 3>,     // (n, r, b)，反对称配对
    scores: &Tensor<B, 1>,  // (n,)
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [n, r, a] = a_t.dims();
    let b = b_t.dims()[2];
    let _ = device;
    assert!(n % 2 == 0, "配对 einsum 要求 n 为偶数，实际 {n}");
    let half = n / 2;
    // 前半（连续切片视图，零拷贝）。
    let a_half = a_t.clone().slice([0..half, 0..r, 0..a]); // (half, r, a)
    let b_half = b_t.clone().slice([0..half, 0..r, 0..b]); // (half, r, b)
    // g_raw 的加权 A：A''[i] = (f_i + f_{half+i})·A'_i。
    let f_pair = scores
        .clone()
        .slice([0..half])
        .add(scores.clone().slice([half..n])); // (half,)
    let a_w = a_half.clone() * f_pair.reshape([half, 1, 1]); // (half, r, a)
    // 拼接 + 展平（cat 输出连续；b_half 为连续切片，reshape 零拷贝）。
    let a_stack = Tensor::cat(vec![a_w, a_half], 2); // (half, r, 2a)
    let a_flat = a_stack.reshape([half * r, 2 * a]); // 连续
    let b_flat = b_half.reshape([half * r, b]); // 连续视图
    let g = a_flat.transpose().matmul(b_flat); // (2a, b)
    // 上半行 = g_raw；下半行 = g_ones'（×2 得 g_ones）。
    let g_raw = g.clone().slice([0..a, 0..b]).reshape([a, b]);
    let g_ones = g.slice([a..2 * a, 0..b]).reshape([a, b]).mul_scalar(2.0);
    (g_raw, g_ones)
}

/// 反对称配对版 raw+ones 合并 einsum（半噪声版，训练热路径专用）。
///
/// 与 [`lora_einsum_pair`] 数学完全一致，但噪声只存前半（配对隐含）：
/// `A'[half+i] = -A'[i]`、`B'[half+i] = -B'[i]` 由消费方约定，这里直接消费
/// `(half, r, a)` / `(half, r, b)` 张量，**无需切片**。噪声生成量减半
/// （fc1 B' 2.4GB → 1.2GB，gen 阶段 ~17ms → ~7ms/chunk）。
///
/// ```text
/// g_raw  = Σ_n f_n·(A'_n ⊗ B'_n) = Σ_{i<half} A''_i ⊗ B'_i，A''_i = (f_i + f_{half+i})·A'_i
/// g_ones = Σ_n (A'_n ⊗ B'_n)     = 2·Σ_{i<half} A'_i ⊗ B'_i
/// ```
///
/// 两者共享同一 `B_half`：把 `[A''; A_half]` 沿 a 轴拼接为 `(half, r, 2a)`，一次
/// 2D GEMM `(2a, half·r) @ (half·r, b)` 同时产出 `g_raw`（上半行）与 `g_ones'`
/// （下半行，末尾 ×2）。
pub fn lora_einsum_pair_half(
    a_half: &Tensor<B, 3>,   // (half, r, a)，A 已乘 sign*base_sigma（配对隐含）
    b_half: &Tensor<B, 3>,   // (half, r, b)（配对隐含）
    scores: &Tensor<B, 1>,   // (n,)
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [half, r, a] = a_half.dims();
    let b = b_half.dims()[2];
    let n = scores.dims()[0];
    let _ = device;
    assert_eq!(n, 2 * half, "半 einsum 要求 scores 长度 = 2×half，实际 {n} vs {half}");
    // g_raw 的加权 A：A''[i] = (f_i + f_{half+i})·A'_i。
    let f_pair = scores
        .clone()
        .slice([0..half])
        .add(scores.clone().slice([half..n])); // (half,)
    let a_w = a_half.clone() * f_pair.reshape([half, 1, 1]); // (half, r, a)
    // 拼接 + 展平（cat 输出连续；b_half 连续，reshape 零拷贝）。
    let a_stack = Tensor::cat(vec![a_w, a_half.clone()], 2); // (half, r, 2a)
    let a_flat = a_stack.reshape([half * r, 2 * a]); // 连续
    let b_flat = b_half.clone().reshape([half * r, b]); // 连续
    let g = a_flat.transpose().matmul(b_flat); // (2a, b)
    // 上半行 = g_raw；下半行 = g_ones'（×2 得 g_ones）。
    let g_raw = g.clone().slice([0..a, 0..b]).reshape([a, b]);
    let g_ones = g.slice([a..2 * a, 0..b]).reshape([a, b]).mul_scalar(2.0);
    (g_raw, g_ones)
}

/// 反对称配对版 raw-only 半 K einsum：`g_raw = Σ_i (f_i + f_{half+i})·A'_i ⊗ B'_i`。
///
/// 用于噪声已缓存（跨 epoch 固定）的层：ones 项同时被缓存后，每 epoch 只需这一个
/// `(a, half·r) @ (half·r, b)` GEMM（约 [pair 版] 的一半 FLOPs）。
pub fn lora_einsum_raw_halfk(
    a_t: &Tensor<B, 3>,     // (n, r, a)，A 已乘 sign*base_sigma，反对称配对
    b_t: &Tensor<B, 3>,     // (n, r, b)，反对称配对
    scores: &Tensor<B, 1>,  // (n,)
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [n, r, a] = a_t.dims();
    let b = b_t.dims()[2];
    let _ = device;
    assert!(n % 2 == 0, "配对 einsum 要求 n 为偶数，实际 {n}");
    let half = n / 2;
    let a_half = a_t.clone().slice([0..half, 0..r, 0..a]); // 连续视图
    let b_half = b_t.clone().slice([0..half, 0..r, 0..b]); // 连续视图
    let f_pair = scores
        .clone()
        .slice([0..half])
        .add(scores.clone().slice([half..n])); // (half,)
    let a_w = a_half * f_pair.reshape([half, 1, 1]); // (half, r, a)
    let a_flat = a_w.reshape([half * r, a]); // 连续
    let b_flat = b_half.reshape([half * r, b]); // 连续视图
    a_flat.transpose().matmul(b_flat) // (a, b)
}

/// 反对称配对版 ones-only 半 K einsum：`g_ones = 2·Σ_{i<half} A'_i ⊗ B'_i`。
///
/// 噪声跨 epoch 固定时可只算一次并缓存（每 epoch 省掉一半 einsum FLOPs）。
pub fn lora_ones_halfk(
    a_t: &Tensor<B, 3>,     // (n, r, a)，A 已乘 sign*base_sigma，反对称配对
    b_t: &Tensor<B, 3>,     // (n, r, b)，反对称配对
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [n, r, a] = a_t.dims();
    let b = b_t.dims()[2];
    let _ = device;
    assert!(n % 2 == 0, "配对 einsum 要求 n 为偶数，实际 {n}");
    let half = n / 2;
    let a_half = a_t.clone().slice([0..half, 0..r, 0..a]); // 连续视图
    let b_half = b_t.clone().slice([0..half, 0..r, 0..b]); // 连续视图
    let a_flat = a_half.reshape([half * r, a]); // 连续
    let b_flat = b_half.reshape([half * r, b]); // 连续视图
    a_flat.transpose().matmul(b_flat).mul_scalar(2.0) // (a, b)
}

/// 仿射修正 + 尺度恢复：最终 solver 输入梯度
/// `-(grad_acc - mean·ones_acc) / (std·sqrt(batch))`，与两阶段 `accumulated_update`
/// 的 `-Σeinsum(conv)/sqrt(batch)` 严格一致。
pub fn combine_affine_grads(
    grad_acc: &[Tensor<B, 2>],
    ones_acc: &[Tensor<B, 2>],
    mean: f32,
    std: f32,
    batch: usize,
) -> Vec<Tensor<B, 2>> {
    let scale = -1.0 / (std * (batch as f32).sqrt());
    grad_acc
        .iter()
        .zip(ones_acc.iter())
        .map(|(g, o)| (g.clone() - o.clone().mul_scalar(mean)).mul_scalar(scale))
        .collect()
}

impl Noiser for EggRoll {
    fn do_mm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        do_mm_impl(frozen, noiser, param, base_key, iterinfo, x)
    }

    #[allow(non_snake_case)]
    fn do_Tmm(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        do_Tmm_impl(frozen, noiser, param, base_key, iterinfo, x)
    }

    fn do_emb(
        &self,
        _frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        _param: &Tensor<B, 2>,
        _base_key: u64,
        _iterinfo: Option<&IterInfo>,
        _indices: Tensor<B, 1, Int>,
    ) -> Tensor<B, 2> {
        // Python raises NotImplementedError("Embedding is not implemented").
        unimplemented!("EggRoll embedding is not implemented")
    }

    fn get_noisy_standard(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &NoiserParams,
        param: &Tensor<B, 2>,
        base_key: u64,
        iterinfo: Option<&IterInfo>,
    ) -> Tensor<B, 2> {
        get_noisy_standard_impl(frozen, noiser, param, base_key, iterinfo)
    }

    fn convert_fitnesses(
        &self,
        frozen: &FrozenNoiserParams,
        _noiser: &NoiserParams,
        raw: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let group_size = frozen.group_size as usize;
        let n = raw.dims()[0];

        // Global mean / var of the full raw array (matches Python's use of
        // `jnp.var(raw_scores, keepdims=True)` in both branches).
        let mean = raw.clone().mean().into_scalar();
        let var = raw.clone().powf_scalar(2.0).mean().into_scalar() - mean * mean;
        let std = (var + 1e-5).sqrt();

        if group_size == 0 {
            // Global z-score.
            raw.add_scalar(-mean).mul_scalar(1.0 / std)
        } else {
            // Per-group mean (keepdims), global std.
            let n_groups = n / group_size;
            let groups = raw.reshape([n_groups, group_size]); // (groups, gs)
            let gmean = groups.clone().mean_dim(-1); // (groups, 1)
            (groups - gmean).mul_scalar(1.0 / std).reshape([n])
        }
    }

    fn do_updates(
        &self,
        frozen: &FrozenNoiserParams,
        noiser: &mut NoiserParams,
        params: &[Tensor<B, 2>],
        base_keys: &[u64],
        fitnesses: Tensor<B, 1>,
        iterinfos: &[IterInfo],
        es_classes: &[i32],
    ) -> Vec<Tensor<B, 2>> {
        do_updates_impl(simple_lora_update, frozen, noiser, params, base_keys, fitnesses, iterinfos, es_classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noiser::NoiserParams;

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    }

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn frozen_with(
        group_size: i32,
        freeze_nonlora: bool,
        noise_reuse: i32,
        rank: usize,
        solver: Solver,
    ) -> FrozenNoiserParams {
        FrozenNoiserParams {
            group_size,
            freeze_nonlora,
            noise_reuse,
            rank,
            solver,
        }
    }

    fn noiser_with(sigma: f32, frozen: &FrozenNoiserParams, params: &[Tensor<B, 2>]) -> NoiserParams {
        NoiserParams {
            sigma,
            opt_state: frozen.solver.init_state(params, &device()),
        }
    }

    // -- convert_fitnesses -------------------------------------------------

    #[test]
    fn convert_fitnesses_global_is_zero_mean_unit_std() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(1.0, &frozen, &[]);
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0, 4.0], &device());
        let out: Vec<f32> = to_vec(EggRoll.convert_fitnesses(&frozen, &noiser, raw));
        // mean=2.5, var=1.25, std=sqrt(1.25+1e-5)=1.118034.
        assert!(out.iter().map(|x| x * x).sum::<f32>() / 4.0 - 1.0 < 1e-3);
        assert!(out.iter().sum::<f32>() / 4.0 < 1e-3);
    }

    #[test]
    fn convert_fitnesses_group_matches_hand_computed() {
        let frozen = frozen_with(2, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(1.0, &frozen, &[]);
        let raw = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0, 4.0], &device());
        let out: Vec<f32> = to_vec(EggRoll.convert_fitnesses(&frozen, &noiser, raw));
        // groups [[1,2],[3,4]]; global std = sqrt(1.25+1e-5)=1.118034;
        // group means 1.5 and 3.5 -> each pair (dx/1.118034).
        let e = 0.5 / 1.11803399;
        assert!(out.iter().zip([-e, e, -e, e].iter()).all(|(a, b)| near(*a, *b, 1e-4)), "{out:?}");
    }

    // -- get_lora_update_params -------------------------------------------

    #[test]
    fn lora_update_params_shape_determinism_and_thread_var() {
        let a = 2usize;
        let b = 3usize;
        let r = 2usize;
        let key = 42u64;
        let info = IterInfo { epoch: 1, thread_id: 0 };

        let (a1, b1) = get_lora_update_params(0.5, key, r, &info, a, b, 0, &device());
        assert_eq!(a1.dims(), [a, r]);
        assert_eq!(b1.dims(), [b, r]);

        // Same inputs -> identical.
        let (a2, b2) = get_lora_update_params(0.5, key, r, &info, a, b, 0, &device());
        assert_eq!(to_vec(a1.clone()), to_vec(a2));
        assert_eq!(to_vec(b1.clone()), to_vec(b2));

        // Different true thread_idx (thread 0 vs thread 4 -> 0 vs 2) differs.
        let other = IterInfo { epoch: 1, thread_id: 4 };
        let (a3, _b3) = get_lora_update_params(0.5, key, r, &other, a, b, 0, &device());
        assert!(to_vec(a1).iter().zip(to_vec(a3).iter()).any(|(x, y)| x != y));
    }

    // -- batched_lora_noise ----------------------------------------------

    #[test]
    fn batched_lora_noise_matches_per_thread_get_lora_update_params() {
        // 小规模批量验证：batched_lora_noise 的每一行必须与按单个 tid 调
        // get_lora_update_params 的结果逐位一致（含 sign 与 base_sigma 处理）。
        let n = 3usize;
        let a = 2usize;
        let b = 3usize;
        let r = 2usize;
        let tids = [0_i32, 1, 2];
        let epoch = 5_i32;
        let noise_reuse = 1_i32;
        let base_sigma = 0.25_f32;
        let key = 1234u64;

        let (a_b, b_b) =
            batched_lora_noise(base_sigma, key, r, &tids, epoch, noise_reuse, a, b, &device());
        assert_eq!(a_b.dims(), [n, a, r]);
        assert_eq!(b_b.dims(), [n, b, r]);

        for (i, &tid) in tids.iter().enumerate() {
            let info = IterInfo { epoch, thread_id: tid };
            let (a_i, b_i) =
                get_lora_update_params(base_sigma, key, r, &info, a, b, noise_reuse, &device());
            // batched 的第 i 行与单样本结果逐位相等。
            let a_row = a_b.clone().slice([i..i + 1, 0..a, 0..r]).reshape([a, r]);
            let b_row = b_b.clone().slice([i..i + 1, 0..b, 0..r]).reshape([b, r]);
            assert_eq!(to_vec(a_row), to_vec(a_i), "A 第 {i} 行 (tid={tid}) 不一致");
            assert_eq!(to_vec(b_row), to_vec(b_i), "B 第 {i} 行 (tid={tid}) 不一致");
        }
    }

    #[test]
    fn batched_lora_noise_empty_tids_returns_empty_tensors() {
        // n == 0 时返回形状 [0, a, r] / [0, b, r] 的空张量。
        let (a_b, b_b) = batched_lora_noise(0.25, 1u64, 2, &[], 0, 0, 2, 3, &device());
        assert_eq!(a_b.dims(), [0, 2, 2]);
        assert_eq!(b_b.dims(), [0, 3, 2]);
        assert!(to_vec(a_b).is_empty());
        assert!(to_vec(b_b).is_empty());
    }

    // -- LoraNoiseCache ---------------------------------------------------

    #[test]
    fn lora_noise_cache_slice_upload_matches_batched_lora_noise() {
        // 缓存切片上传的结果必须与现场 batched_lora_noise 逐位一致（训练热路径
        // 依赖缓存 == 冷路径生成，保证前向与更新的噪声一致）。
        let a = 2usize;
        let b = 3usize;
        let r = 2usize;
        let batch = 5usize;
        let rank = r;
        let noise_reuse = 0_i32;
        let base_sigma = 0.25_f32;
        let key = 99u64;
        // 构造 params 中一个 LoRA 参数（形状 (a,b)），es_classes=[1]。
        let p = Tensor::<B, 2>::zeros([a, b], &device());
        let cache = build_lora_noise_cache(&[p], &[key], &[1], batch, rank, noise_reuse);
        assert_eq!(cache.buffers.len(), 1);
        assert!(cache.buffers[0].is_some());
        // 取 [2, 5) 段（对应 tids 2..5）。
        let (a_c, b_c) = cache.slice_upload(0, 2, 5, base_sigma, &device());
        assert_eq!(a_c.dims(), [3, a, r]);
        assert_eq!(b_c.dims(), [3, b, r]);
        let tids = [2_i32, 3, 4];
        let (a_b, b_b) = batched_lora_noise(base_sigma, key, r, &tids, 0, noise_reuse, a, b, &device());
        assert_eq!(to_vec(a_c), to_vec(a_b), "缓存 A 与现场生成不一致");
        assert_eq!(to_vec(b_c), to_vec(b_b), "缓存 B 与现场生成不一致");
    }

    // -- _simple_lora_update (via do_update path pieces) -------------------

    #[test]
    fn simple_lora_update_matches_hand_computed() {
        let a = 2usize;
        let b = 2usize;
        let r = 1usize;
        let key = 7u64;
        let sigma = 0.5_f32;
        let info = IterInfo { epoch: 0, thread_id: 0 }; // sign +1
        let frozen = frozen_with(0, false, 0, r, Solver::sgd(0.1));

        // Hand-compute the expected update with the same deterministic RNG.
        let (true_epoch, true_thread, sign) = epoch_thread_sign(&info, frozen.noise_reuse);
        let mut rng = DeterministicNoise::new(noise_seed(key, true_epoch, true_thread));
        let lora = rng.normal_tensor([a + b, r], &device());
        let lv = to_vec(lora); // length a+b = 4
        let b_raw = [lv[0], lv[1]]; // rows 0..2
        let a_raw = [lv[2], lv[3]]; // rows 2..4
        let base_sigma = sigma / (r as f32).sqrt();
        let sc = sign * base_sigma;
        // A = sc * a_raw (2x1); B = b_raw (2x1); expected = A @ B^T (2x2).
        let mut expected = [0.0_f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                expected[i * 2 + j] = (sc * a_raw[i]) * b_raw[j];
            }
        }

        let scores = [1.0_f32]; // N=1, fitness weight 1.
        let shape = [a, b];
        let got = simple_lora_update(sigma, key, shape, &scores, &[info], &frozen, &device());
        let gv = to_vec(got);
        assert!(gv.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "got {gv:?} exp {expected:?}");
    }

    // -- _do_update sign ---------------------------------------------------

    #[test]
    fn do_update_is_neg_grad_times_sqrt_n() {
        // N=1 dense update: g = noise; do_update = -g * sqrt(1) = -g.
        let sigma = 0.3_f32;
        let key = 5u64;
        let shape = [2, 1];
        let info = IterInfo { epoch: 0, thread_id: 0 };
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));

        // Hand-computed g.
        let (te, tt, sign) = epoch_thread_sign(&info, frozen.noise_reuse);
        let mut rng = DeterministicNoise::new(noise_seed(key, te, tt));
        let nv = to_vec(rng.normal_tensor(shape, &device()));
        let expected_g: Vec<f32> = nv.iter().map(|v| sign * sigma * v).collect();

        let param = Tensor::<B, 2>::zeros(shape, &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32], &device());
        let got = do_update_with(simple_lora_update, &param, key, &fitness, &[info], 0, sigma, &frozen);
        let gv = to_vec(got);
        let neg: Vec<f32> = expected_g.iter().map(|v| -v).collect();
        // got = -expected_g * sqrt(1)
        assert!(gv.iter().zip(expected_g.iter()).all(|(x, y)| near(*x, -y, 1e-5)), "got {gv:?} exp {neg:?}");
    }

    // -- do_updates SGD full pipeline -------------------------------------

    #[test]
    fn do_updates_sgd_full_pipeline_matches_formula() {
        // Two envs, dense update, shape [1, 2], lr=0.1, sigma=0.5.
        let lr = 0.1_f32;
        let sigma = 0.5_f32;
        let key = 99u64;
        let shape = [1, 2];
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(lr));
        let mut noiser = noiser_with(sigma, &frozen, &[]);

        let p = Tensor::<B, 2>::from_data([[1.0_f32, -1.0]], &device());
        let fitness = Tensor::<B, 1>::from_data([1.0_f32, 2.0], &device());
        let infos = [IterInfo { epoch: 0, thread_id: 0 }, IterInfo { epoch: 0, thread_id: 1 }];

        // Hand-compute g = 1/2 * (f0*u0 + f1*u1) with deterministic noise.
        let mut g = [0.0_f32; 2];
        for (i, info) in infos.iter().enumerate() {
            let (te, tt, sign) = epoch_thread_sign(info, 0);
            let mut rng = DeterministicNoise::new(noise_seed(key, te, tt));
            let nv = to_vec(rng.normal_tensor(shape, &device()));
            let f = fitness.clone().into_data().into_vec::<f32>().unwrap()[i];
            for (k, v) in nv.iter().enumerate() {
                g[k] += f * sign * sigma * v;
            }
        }
        for x in g.iter_mut() {
            *x /= 2.0; // /N
        }
        // param_new = param + lr * g * sqrt(N), N = 2.
        let n_root = (2.0_f32).sqrt();
        let expected: Vec<f32> = vec![
            1.0 + lr * g[0] * n_root,
            -1.0 + lr * g[1] * n_root,
        ];

        let updated = EggRoll.do_updates(&frozen, &mut noiser, &[p], &[key], fitness, &infos, &[0]);
        let uv = to_vec(updated[0].clone());
        assert!(uv.iter().zip(expected.iter()).all(|(x, y)| near(*x, *y, 1e-4)), "got {uv:?} exp {expected:?}");
    }

    // -- do_updates Adam runs without panic --------------------------------

    #[test]
    fn do_updates_adam_runs_without_panic() {
        let frozen = frozen_with(0, false, 0, 2, Solver::adam(0.01));
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let mut noiser = noiser_with(0.5, &frozen, &[p.clone()]);
        let fitness = Tensor::<B, 1>::from_data([1.0_f32, 2.0], &device());
        let infos = [IterInfo { epoch: 0, thread_id: 0 }, IterInfo { epoch: 0, thread_id: 2 }];
        // LoRA update path (map_class = 1).
        let updated = EggRoll.do_updates(&frozen, &mut noiser, &[p.clone()], &[3u64], fitness.clone(), &infos, &[1]);
        assert!(to_vec(updated[0].clone()).iter().all(|x| x.is_finite()));
        // Dense path as well.
        let updated2 = EggRoll.do_updates(&frozen, &mut noiser, &[p.clone()], &[4u64], fitness, &infos, &[0]);
        assert!(to_vec(updated2[0].clone()).iter().all(|x| x.is_finite()));
        // Running a second Adam step must not panic either.
        let fitness2 = Tensor::<B, 1>::from_data([0.5_f32, -1.0], &device());
        let updated3 = EggRoll.do_updates(&frozen, &mut noiser, &[p], &[5u64], fitness2, &infos, &[1]);
        assert!(to_vec(updated3[0].clone()).iter().all(|x| x.is_finite()));
    }

    // -- accumulated_update ----------------------------------------------

    #[test]
    fn accumulated_update_matches_do_updates_and_steps_once() {
        // 用 AdamW、小子尺寸（rank=2）、3 段 × 每段 4 样本，验证 K 段累积更新
        // （accumulated_update）与全批一次性 do_updates 产出参数逐元素一致，
        // 且 solver step 只 +1。
        let lr = 0.01_f32;
        let sigma = 0.5_f32;
        let rank = 2usize;
        let frozen = frozen_with(0, false, 0, rank, Solver::adamw(lr));
        let p = Tensor::<B, 2>::from_data(
            [
                [0.1_f32, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            &device(),
        );

        // 全长 conv（批长 = 3 段 × 4 = 12 样本）与全局唯一的 thread_ids。
        let conv_full = Tensor::<B, 1>::from_data(
            [
                1.0_f32, -0.5, 2.0, 0.5, // chunk 0
                -1.0, 1.5, 0.25, -0.75, // chunk 1
                1.0, -0.25, 0.75, -1.5, // chunk 2
            ],
            &device(),
        );
        let thread_ids: Vec<i32> = (0..12).collect();

        // 路径 A：全批一次性 do_updates（同一初态）。
        let mut noiser_full = noiser_with(sigma, &frozen, &[p.clone()]);
        let full_infos: Vec<IterInfo> = thread_ids
            .iter()
            .map(|&t| IterInfo { epoch: 3, thread_id: t })
            .collect();
        let updated_full = EggRoll.do_updates(
            &frozen,
            &mut noiser_full,
            &[p.clone()],
            &[42u64],
            conv_full.clone(),
            &full_infos,
            &[1],
        );

        // 路径 B：K 段累积更新（同一初态）。
        let mut noiser_acc = noiser_with(sigma, &frozen, &[p.clone()]);
        let updated_acc = accumulated_update(
            &frozen,
            &mut noiser_acc,
            &[p.clone()],
            &[42u64],
            &[1],
            conv_full.clone(),
            &thread_ids,
            3, // epoch
            3, // accumulate = K = 3 段
            4, // chunk = 每段 4 样本
        );

        // 逐元素最大绝对差 ≈ 0（float32 累加顺序差异，容差 1e-5）。
        let a = to_vec(updated_full[0].clone());
        let b = to_vec(updated_acc[0].clone());
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5, "max|Δparam| = {max_diff} (> 1e-5)");

        // solver step 只 +1（累积后一次 solver.update），而非逐段 +3。
        assert_eq!(noiser_acc.opt_state.step, 1);
        assert_eq!(noiser_full.opt_state.step, 1);
    }

    // -- 内联（GPU 噪声）辅助函数 -----------------------------------------

    #[test]
    fn lora_einsum_helpers_match_hand_computed() {
        // 小规模手算：A' (2, r=2, a=2)、B' (2, r=2, b=3)、scores [0.5, -1.0]。
        // 新布局 (n,r,*)：A'[n,r,i] = A[n,i,r]，B'[n,r,j] = B[n,j,r]。
        let a_t = Tensor::<B, 3>::from_data(
            [
                [[1.0_f32, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            &device(),
        ); // (2, 2, 2) = (n, r, a)
        let b_t = Tensor::<B, 3>::from_data(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            &device(),
        ); // (2, 2, 3) = (n, r, b)
        let scores = Tensor::<B, 1>::from_data([0.5_f32, -1.0], &device());

        // 手算 einsum('nir,njr->ij')：Σ_n Σ_r A[n,i,r]·B[n,j,r]；
        // 按 (n,r,*) 布局即 Σ_n Σ_r A'[n,r,i]·B'[n,r,j]。
        let raw = lora_einsum_raw(&a_t, &b_t, &scores, &device());
        let ones = lora_einsum_ones(&a_t, &b_t, &device());
        let mut exp_raw = vec![0.0_f32; 6];
        let mut exp_ones = vec![0.0_f32; 6];
        for n in 0..2 {
            for i in 0..2 {
                for j in 0..3 {
                    for r in 0..2 {
                        let v = a_t.clone().slice([n..n + 1, r..r + 1, i..i + 1]).into_scalar();
                        let w = b_t.clone().slice([n..n + 1, r..r + 1, j..j + 1]).into_scalar();
                        exp_ones[i * 3 + j] += v * w;
                        exp_raw[i * 3 + j] += scores.clone().slice([n..n + 1]).into_scalar() * v * w;
                    }
                }
            }
        }
        let got_raw = to_vec(raw);
        let got_ones = to_vec(ones);
        for k in 0..6 {
            assert!((got_raw[k] - exp_raw[k]).abs() < 1e-4, "raw[{k}]={} exp={}", got_raw[k], exp_raw[k]);
            assert!((got_ones[k] - exp_ones[k]).abs() < 1e-4, "ones[{k}]={} exp={}", got_ones[k], exp_ones[k]);
        }
    }

    #[test]
    fn dense_einsum_helpers_match_hand_computed() {
        // 小规模手算：noise (2, 1, 1)、scores [2.0, 3.0]。
        let noise = Tensor::<B, 3>::from_data([[[4.0_f32]], [[-1.0]]], &device()); // (2,1,1)
        let scores = Tensor::<B, 1>::from_data([2.0_f32, 3.0], &device());
        // Σ f_i·noise_i = 2*4 + 3*(-1) = 5；Σ noise_i = 3。
        let raw = dense_einsum_raw(&noise, &scores, &device());
        let ones = dense_einsum_ones(&noise, &device());
        assert!((to_vec(raw)[0] - 5.0).abs() < 1e-5);
        assert!((to_vec(ones)[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn lora_einsum_pair_matches_plain_raw_and_ones() {
        // 反对称配对版（合并单 GEMM）必须与全量 lora_einsum_raw/lora_einsum_ones
        // 逐位一致（容差 1e-4，仅浮点累加顺序差异）。
        let n = 4usize;
        let r = 3usize;
        let a = 2usize;
        let b = 3usize;
        // 构造反对称配对噪声：A'[half+i] = -A'[i]，B'[half+i] = -B'[i]。
        let a_half = Tensor::<B, 3>::from_data(
            [
                [[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ],
            &device(),
        ); // (2, 3, 2) = (half, r, a)
        let b_half = Tensor::<B, 3>::from_data(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ],
            &device(),
        ); // (2, 3, 3) = (half, r, b)
        let a_t = Tensor::cat(vec![a_half.clone(), a_half.neg()], 0); // (4, 3, 2)
        let b_t = Tensor::cat(vec![b_half.clone(), b_half.neg()], 0); // (4, 3, 3)
        let scores = Tensor::<B, 1>::from_data([0.5_f32, -1.0, 2.0, 0.25], &device());

        let (g_raw, g_ones) = lora_einsum_pair(&a_t, &b_t, &scores, &device());
        let exp_raw = lora_einsum_raw(&a_t, &b_t, &scores, &device());
        let exp_ones = lora_einsum_ones(&a_t, &b_t, &device());

        let gr = to_vec(g_raw);
        let er = to_vec(exp_raw);
        let go = to_vec(g_ones);
        let eo = to_vec(exp_ones);
        for k in 0..gr.len() {
            assert!(
                (gr[k] - er[k]).abs() < 1e-4,
                "pair raw[{k}]={} exp={}",
                gr[k],
                er[k]
            );
            assert!(
                (go[k] - eo[k]).abs() < 1e-4,
                "pair ones[{k}]={} exp={}",
                go[k],
                eo[k]
            );
        }

        // 拆分版（raw_halfk + ones_halfk）必须与合并版逐位一致。
        let g_raw2 = lora_einsum_raw_halfk(&a_t, &b_t, &scores, &device());
        let g_ones2 = lora_ones_halfk(&a_t, &b_t, &device());
        let gr2 = to_vec(g_raw2);
        let go2 = to_vec(g_ones2);
        for k in 0..gr2.len() {
            assert!(
                (gr2[k] - er[k]).abs() < 1e-4,
                "raw_halfk[{k}]={} exp={}",
                gr2[k],
                er[k]
            );
            assert!(
                (go2[k] - eo[k]).abs() < 1e-4,
                "ones_halfk[{k}]={} exp={}",
                go2[k],
                eo[k]
            );
        }
    }

    #[test]
    fn inline_affine_matches_accumulated_two_phase() {
        // 关键等价性：给定完全相同的噪声（用确定性缓存切片），内联路径
        // （raw 加权累积 + 仿射修正 + 一次 solver）必须与两阶段 accumulated_update
        // （全局 z-score + chunked einsum）产出参数一致。验证仿射恒等式
        // `Σeinsum((raw-mean)/std) = (Σeinsum(raw) - mean·Σeinsum(1))/std`。
        let lr = 0.01_f32;
        let sigma = 0.5_f32;
        let rank = 2usize;
        let frozen = frozen_with(0, false, 0, rank, Solver::adamw(lr));
        // 一个 LoRA 参数 (a=2, b=3)。
        let p = Tensor::<B, 2>::from_data([[0.1_f32, 0.2, 0.3], [0.4, 0.5, 0.6]], &device());
        // 批长 8 = K=2 × chunk=4；raw fitness（z-score 前的值）。
        let raw = vec![1.0_f32, -0.5, 2.0, 0.5, -1.0, 1.5, 0.25, -0.75];
        let thread_ids: Vec<i32> = (0..8).collect();
        // 确定性噪声缓存（与内联用相同切片）。
        let cache = build_lora_noise_cache(&[p.clone()], &[42u64], &[1], 8, rank, 0);

        // 两阶段路径：raw -> 全局 z-score -> accumulated_update（cache 提供相同噪声）。
        let raw_t = Tensor::<B, 1>::from_data(&raw[..], &device());
        let conv = EggRoll.convert_fitnesses(&frozen, &noiser_with(sigma, &frozen, &[p.clone()]), raw_t);
        let mut noiser_2p = noiser_with(sigma, &frozen, &[p.clone()]);
        let updated_2p = accumulated_update_cached(
            &frozen,
            &mut noiser_2p,
            &[p.clone()],
            &[42u64],
            &[1],
            conv,
            &thread_ids,
            0,
            2, // accumulate
            4, // chunk
            Some(&cache),
        );

        // 内联路径：逐 chunk 用 raw 累积 grad_acc/ones_acc，仿射修正 + 一次 solver。
        let base_sigma = sigma / (rank as f32).sqrt();
        let mut noiser_inl = noiser_with(sigma, &frozen, &[p.clone()]);
        let mut grad_acc = Tensor::<B, 2>::zeros(p.dims(), &device());
        let mut ones_acc = Tensor::<B, 2>::zeros(p.dims(), &device());
        let mut sum_raw = 0.0_f32;
        let mut sum_raw2 = 0.0_f32;
        for k in 0..2 {
            let lo = k * 4;
            let hi = lo + 4;
            let (a_t, b_t) = cache.slice_upload(0, lo, hi, base_sigma, &device());
            // 缓存为 (n,a,r)/(n,b,r) 布局；内联 einsum 新布局为 (n,r,*)，转视图即可。
            let a_ra = a_t.swap_dims(1, 2);
            let b_rb = b_t.swap_dims(1, 2);
            let scores_t = Tensor::<B, 1>::from_data(&raw[lo..hi], &device());
            grad_acc = grad_acc + lora_einsum_raw(&a_ra, &b_rb, &scores_t, &device());
            ones_acc = ones_acc + lora_einsum_ones(&a_ra, &b_rb, &device());
            sum_raw += raw[lo..hi].iter().sum::<f32>();
            sum_raw2 += raw[lo..hi].iter().map(|x| x * x).sum::<f32>();
        }
        let mean = sum_raw / 8.0;
        let var = sum_raw2 / 8.0 - mean * mean;
        let std = (var + 1e-5).sqrt();
        let grads = combine_affine_grads(&[grad_acc], &[ones_acc], mean, std, 8);
        let updated_inl = frozen.solver.update(&[p.clone()], &grads, &mut noiser_inl.opt_state);

        let a = to_vec(updated_2p[0].clone());
        let b = to_vec(updated_inl[0].clone());
        let max_diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-4, "内联 vs 两阶段 max|Δparam| = {max_diff} (> 1e-4)");
    }

    // -- init_noiser -------------------------------------------------------

    #[test]
    fn init_noiser_builds_frozen_and_noiser_params() {
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let (frozen, noiser) = init_noiser(&[p], 0.4, 0.01, 0, false, 0, 4, Solver::adamw(0.01), &device());
        assert_eq!(frozen.rank, 4);
        assert_eq!(frozen.group_size, 0);
        assert_eq!(noiser.sigma, 0.4);
        assert_eq!(noiser.opt_state.moments.len(), 1);
    }

    // -- get_noisy_standard & do_mm ---------------------------------------

    #[test]
    fn get_noisy_standard_adds_dense_noise_with_iterinfo() {
        let frozen = frozen_with(0, false, 0, 1, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        // Without iterinfo -> identity.
        let id = EggRoll.get_noisy_standard(&frozen, &noiser, &p, 1, None);
        assert_eq!(to_vec(id), vec![1.0, 2.0]);
        // With freeze_nonlora -> identity.
        let frz = frozen_with(0, true, 0, 1, Solver::sgd(0.1));
        let noiser2 = noiser_with(0.5, &frz, &[]);
        let id2 = EggRoll.get_noisy_standard(&frz, &noiser2, &p, 1, Some(&IterInfo { epoch: 0, thread_id: 1 }));
        assert_eq!(to_vec(id2), vec![1.0, 2.0]);
    }

    #[test]
    fn do_mm_adds_lora_noise_when_iterinfo_present() {
        let frozen = frozen_with(0, false, 0, 2, Solver::sgd(0.1));
        let noiser = noiser_with(0.5, &frozen, &[]);
        // param (out=1, in=2) -> a=1, b=2.
        let p = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0]], &device());
        let base = EggRoll.do_mm(&frozen, &noiser, &p, 8, None, x.clone());
        // x @ p.T = [[5]]
        assert_eq!(to_vec(base), vec![5.0]);
        let noisy = EggRoll.do_mm(&frozen, &noiser, &p, 8, Some(&IterInfo { epoch: 0, thread_id: 0 }), x);
        // x @ p.T + x @ B @ A.T ; must differ from base in general.
        assert!(to_vec(noisy)[0] != 5.0);
    }
}
