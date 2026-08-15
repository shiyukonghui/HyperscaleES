//! MNIST data loading, Poisson spike encoding, and fitness scoring for SNN training.
//!
//! Port of the Python `environments/snn_mnist` module to Rust using burn 0.21 (flex
//! backend, f32/i32). The training loop follows the same
//! "generation -> fitness -> noiser update" pattern; this module only supplies the data
//! + reward, and is agnostic to the underlying SNN model.

use std::io::{Error, ErrorKind, Read};

use burn::tensor::{Distribution, Tensor};
use hyperscalees_core::B;

/// Two leading bytes of a gzip file (the gzip "magic").
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// IDX magic number for an images file: magic(4) n(4) rows(4) cols(4) + payload.
const IDX_MAGIC_IMAGES: u32 = 0x803;
/// IDX magic number for a labels file: magic(4) n(4) + payload.
const IDX_MAGIC_LABELS: u32 = 0x801;

/// Standard absolute-size limits to guard against absurd headers (4 GiB would be silly here).
const MAX_ELEMS: usize = 1 << 30;

/// Filenames of the four standard MNIST IDX files, keyed by split name.
pub const IDX_IMAGES: [(&str, &str); 2] = [
    ("train", "train-images-idx3-ubyte.gz"),
    ("test", "t10k-images-idx3-ubyte.gz"),
];
pub const IDX_LABELS: [(&str, &str); 2] = [
    ("train", "train-labels-idx1-ubyte.gz"),
    ("test", "t10k-labels-idx1-ubyte.gz"),
];

fn invalid_data(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// If `data` is gzipped (leading `0x1f 0x8b`), decompress it; otherwise pass through.
fn gunzip_if_needed(data: &[u8]) -> std::io::Result<Vec<u8>> {
    if data.starts_with(&GZIP_MAGIC) {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(data).read_to_end(&mut out)?;
        Ok(out)
    } else {
        Ok(data.to_vec())
    }
}

/// Read a big-endian `u32` from `bytes` starting at `offset`.
fn be_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Parse an IDX images byte buffer (plain or gzipped) into normalized float pixels.
///
/// Returns `(flat data normalized to [0, 1], n, rows * cols)`.
pub fn read_idx_images(data: &[u8]) -> std::io::Result<(Vec<f32>, usize, usize)> {
    let raw = gunzip_if_needed(data)?;
    if raw.len() < 16 {
        return Err(invalid_data("images IDX buffer too short for header".into()));
    }
    let magic = be_u32(&raw, 0);
    if magic != IDX_MAGIC_IMAGES {
        return Err(invalid_data(format!(
            "expected images magic 0x{IDX_MAGIC_IMAGES:x}, got 0x{magic:x}"
        )));
    }
    let n = be_u32(&raw, 4) as usize;
    let rows = be_u32(&raw, 8) as usize;
    let cols = be_u32(&raw, 12) as usize;
    let per = rows.checked_mul(cols).ok_or_else(|| invalid_data("rows * cols overflow".into()))?;
    let total = n
        .checked_mul(per)
        .filter(|&t| t <= MAX_ELEMS)
        .ok_or_else(|| invalid_data("images payload size out of range".into()))?;

    let payload_end = 16usize
        .checked_add(total)
        .filter(|&e| e <= raw.len())
        .ok_or_else(|| invalid_data("images payload underflow".into()))?;

    let mut out = Vec::with_capacity(total);
    for &byte in &raw[16..payload_end] {
        out.push(byte as f32 / 255.0);
    }
    Ok((out, n, per))
}

/// Parse an IDX labels byte buffer (plain or gzipped) into a `Vec<u8>` of labels.
pub fn read_idx_labels(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let raw = gunzip_if_needed(data)?;
    if raw.len() < 8 {
        return Err(invalid_data("labels IDX buffer too short for header".into()));
    }
    let magic = be_u32(&raw, 0);
    if magic != IDX_MAGIC_LABELS {
        return Err(invalid_data(format!(
            "expected labels magic 0x{IDX_MAGIC_LABELS:x}, got 0x{magic:x}"
        )));
    }
    let n = be_u32(&raw, 4) as usize;
    let payload_end = 8usize
        .checked_add(n)
        .filter(|&e| e <= raw.len() && n <= MAX_ELEMS)
        .ok_or_else(|| invalid_data("labels payload underflow".into()))?;
    Ok(raw[8..payload_end].to_vec())
}

/// Load the four standard MNIST IDX files from `dir` (train + test).
///
/// Returns `((train_images, train_labels), (test_images, test_labels))`.
///
/// Mirrors Python `_load_mnist_from_dir`: images normalized to `[0, 1]`, labels kept raw.
pub fn load_mnist_from_dir(
    dir: &std::path::Path,
) -> std::io::Result<((Vec<f32>, Vec<u8>), (Vec<f32>, Vec<u8>))> {
    let read = |name: &str| -> std::io::Result<Vec<u8>> {
        std::fs::read(dir.join(name))
    };

    let train_imgs = read_idx_images(&read(IDX_IMAGES[0].1)?)?;
    let train_lbls = read_idx_labels(&read(IDX_LABELS[0].1)?)?;
    let test_imgs = read_idx_images(&read(IDX_IMAGES[1].1)?)?;
    let test_lbls = read_idx_labels(&read(IDX_LABELS[1].1)?)?;

    Ok((
        (train_imgs.0, train_lbls),
        (test_imgs.0, test_lbls),
    ))
}

/// Convert a batch of probabilities in `[0, 1]` (shape `(batch, in_dim)`) to Poisson
/// spike trains of `t` timesteps via independent Bernoulli sampling per element.
///
/// Roughly mirrors Python `poisson_encode`: for each timestep a fresh sample is drawn.
/// burn manages its own RNG (no JAX key), so the *statistical* property (per-element
/// firing rate ≈ pixel value) is what holds, not any bitwise equality.
///
/// 实现：一次 `(t, batch, in_dim)` 的 Uniform 随机 + 一次比较（替代逐时间步 t 次随机 +
/// 拼接），统计语义完全一致（每元素每时间步独立 Bernoulli），GPU 上减少 t-1 次随机
/// 内核与 1 次 cat。
///
/// Returns a `(t, batch, in_dim)` 0/1 `f32` tensor.
pub fn poisson_encode(images: Tensor<B, 2>, t: usize) -> Tensor<B, 3> {
    let device = images.device();
    let [batch, in_dim] = images.shape().dims();
    // Draw u ~ Uniform(0, 1) elementwise over (t, batch, in_dim)，spike = 1 if u < p else 0。
    let uniform = Tensor::<B, 3>::random(
        [t, batch, in_dim],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let probs = images.unsqueeze_dim(0); // (1, batch, in_dim) 广播
    uniform.lower(probs).float()
}

/// 奖励类型，对应 Python `fitness_from_logits(logits, labels, reward)` 的 `reward` 参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reward {
    /// 对数似然奖励：`log_softmax(logits, -1)[n, label_n]`。
    Loglik,
    /// 硬 0/1 奖励：`argmax(logits, -1) == label` 时为 1.0，否则为 0.0。
    Binary,
}

/// Per-sample hard reward: `1.0` where `argmax(logits, -1) == label`, else `0.0`.
///
/// Mirrors Python `fitness_from_logits` with `reward="binary"`; returned shape is `(batch,)`.
///
/// 保留该二元分支为默认行为，用于兼容既有调用（如 `accuracy_from_logits`）。
pub fn fitness_from_logits(logits: Tensor<B, 2>, labels: Tensor<B, 1, burn::tensor::Int>) -> Tensor<B, 1> {
    fitness_from_logits_reward(logits, labels, Reward::Binary)
}

/// 按指定的 [`Reward`] 类型计算批内每个样本的奖励，返回形状 `(batch,)`。
///
/// 语义与 Python `snn_mnist_train_accumulate.py` 的 `fitness_from_logits` 一致：
/// - [`Reward::Loglik`]：`log_softmax(logits)[n, label_n]`；
/// - [`Reward::Binary`]：`argmax(logits, -1) == label` 的硬 0/1。
pub fn fitness_from_logits_reward(
    logits: Tensor<B, 2>,
    labels: Tensor<B, 1, burn::tensor::Int>,
    reward: Reward,
) -> Tensor<B, 1> {
    match reward {
        Reward::Loglik => {
            // log_softmax 沿类别维（维度 1）-> 形状 (batch, n_classes)
            let log_logits = burn::tensor::activation::log_softmax(logits, 1);
            let batch = log_logits.shape().dims::<2>()[0];
            // 用 gather 沿类别维取出每个样本对应 label 的对数似然 -> (batch, 1)，再展平为 (batch,)
            let labels_2d = labels.reshape([batch, 1]);
            log_logits.gather(1, labels_2d).reshape([batch])
        }
        Reward::Binary => {
            let batch = logits.shape().dims::<2>()[0];
            // argmax over the class dimension -> (batch, 1) Int.
            let pred = logits.argmax(1).reshape([batch]);
            pred.equal(labels).float()
        }
    }
}

/// Mean over the batch of `argmax(logits) == label`, as `f32`.
///
/// Mirrors Python `accuracy_from_logits`.
pub fn accuracy_from_logits(logits: Tensor<B, 2>, labels: Tensor<B, 1, burn::tensor::Int>) -> f32 {
    fitness_from_logits(logits, labels).mean().into_scalar()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    /// Build a raw (uncompressed) IDX images byte buffer.
    fn idx_images_bytes(n: u32, rows: u32, cols: u32, payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&IDX_MAGIC_IMAGES.to_be_bytes());
        buf.extend_from_slice(&n.to_be_bytes());
        buf.extend_from_slice(&rows.to_be_bytes());
        buf.extend_from_slice(&cols.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    /// Build a raw (uncompressed) IDX labels byte buffer.
    fn idx_labels_bytes(n: u32, payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&IDX_MAGIC_LABELS.to_be_bytes());
        buf.extend_from_slice(&n.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn reads_plain_idx_images() {
        // 2 images of 2x2 = 4 pixels each, big-endian header + payload.
        let pixels: [u8; 8] = [0, 64, 128, 255, 1, 2, 3, 4];
        let data = idx_images_bytes(2, 2, 2, &pixels);

        let (values, n, per) = read_idx_images(&data).unwrap();
        assert_eq!(n, 2);
        assert_eq!(per, 4);
        assert_eq!(values.len(), 8);
        let expected = [
            0.0 / 255.0,
            64.0 / 255.0,
            128.0 / 255.0,
            255.0 / 255.0,
            1.0 / 255.0,
            2.0 / 255.0,
            3.0 / 255.0,
            4.0 / 255.0,
        ];
        assert_eq!(values, expected);
    }

    #[test]
    fn reads_plain_idx_labels() {
        let data = idx_labels_bytes(2, &[1, 7]);
        assert_eq!(read_idx_labels(&data).unwrap(), vec![1_u8, 7]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = idx_images_bytes(1, 1, 1, &[42]);
        data[0] = 0x99;
        assert!(read_idx_images(&data).is_err());

        let mut ldata = idx_labels_bytes(1, &[5]);
        ldata[3] = 0x55;
        assert!(read_idx_labels(&ldata).is_err());
    }

    #[test]
    fn reads_gzipped_idx_images() {
        // Build a raw images buffer, then gzip it.
        let pixels: [u8; 8] = [0, 64, 128, 255, 1, 2, 3, 4];
        let raw = idx_images_bytes(2, 2, 2, &pixels);

        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(&raw).unwrap();
        let compressed = gz.finish().unwrap();

        // Sanity: it really is gzipped.
        assert!(compressed.starts_with(&GZIP_MAGIC));

        let (values, n, per) = read_idx_images(&compressed).unwrap();
        assert_eq!((n, per), (2, 4));
        assert_eq!(
            values,
            [0.0 / 255.0, 64.0 / 255.0, 128.0 / 255.0, 255.0 / 255.0,
             1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 4.0 / 255.0]
        );
    }

    #[test]
    fn reads_gzipped_idx_labels() {
        let raw = idx_labels_bytes(2, &[1, 7]);
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(&raw).unwrap();
        let compressed = gz.finish().unwrap();
        assert_eq!(read_idx_labels(&compressed).unwrap(), vec![1_u8, 7]);
    }

    #[test]
    fn poisson_encode_fires_at_rate_p() {
        // A batch of 10_000 constant p = 0.3 probabilities, in_dim = 1.
        let p = 0.3_f32;
        let batch = 10_000usize;
        let flat = Tensor::<B, 1>::from_data(&vec![p; batch][..], &device());
        let probs = flat.reshape([batch, 1]);

        let spikes = poisson_encode(probs, 1); // shape (1, batch, 1)
        let data = spikes.to_data().to_vec::<f32>().unwrap();
        assert_eq!(data.len(), batch);

        // All values are 0 or 1.
        assert!(data.iter().all(|&v| v == 0.0 || v == 1.0));

        // Firing rate approximates p.
        let rate = data.iter().sum::<f32>() / batch as f32;
        assert!((rate - p).abs() < 0.03, "rate {rate} too far from {p}");
    }

    #[test]
    fn poisson_encode_shape_for_time_steps() {
        // 3 x (batch=4, in_dim=2) probs -> shape (t, batch, in_dim) = (3, 4, 2).
        let ones = Tensor::<B, 2>::ones([4, 2], &device());
        let spikes = poisson_encode(ones, 3);
        assert_eq!(spikes.shape().dims::<3>(), [3, 4, 2]);
    }

    #[test]
    fn fitness_and_accuracy_from_logits() {
        let logits = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 0.5], [3.0, 1.0, 2.0], [0.5, 0.5, 9.0]],
            &device(),
        );
        // argmax of each row: [1, 0, 2]. Labels: first two correct, last wrong.
        let labels = Tensor::<B, 1, burn::tensor::Int>::from_data([1_i32, 0, 9], &device());

        let fitness = fitness_from_logits(logits.clone(), labels.clone());
        let values = fitness.to_data().to_vec::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 1.0, 0.0]);

        let acc = accuracy_from_logits(logits, labels);
        assert!((acc - 2.0 / 3.0).abs() < 1e-6, "acc {acc}");
    }

    #[test]
    fn fitness_loglik_shape_and_finite() {
        // loglik 分支：返回形状应为 (batch,)，且所有元素有限。
        let logits = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 0.5], [3.0, 1.0, 2.0], [0.5, 0.5, 9.0]],
            &device(),
        );
        let labels = Tensor::<B, 1, burn::tensor::Int>::from_data([1_i32, 0, 2], &device());

        let fitness = fitness_from_logits_reward(logits, labels, Reward::Loglik);
        assert_eq!(fitness.shape().dims::<1>(), [3]);
        let values = fitness.to_data().to_vec::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "values {values:?}");
    }

    #[test]
    fn fitness_loglik_matches_hand_computed() {
        // 手算参考（数值稳定的 log_softmax）：
        //   log_softmax(x)[k] = x[k] - max(x) - ln(sum_j exp(x[j] - max(x)))
        fn log_softmax_row(row: &[f32], k: usize) -> f32 {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            row[k] - max - sum.ln()
        }

        let logits = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 0.5], [3.0, 1.0, 2.0], [0.5, 0.5, 9.0]],
            &device(),
        );
        let labels = Tensor::<B, 1, burn::tensor::Int>::from_data([1_i32, 0, 2], &device());

        let values = fitness_from_logits_reward(logits, labels, Reward::Loglik)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = [
            log_softmax_row(&[1.0, 2.0, 0.5], 1),
            log_softmax_row(&[3.0, 1.0, 2.0], 0),
            log_softmax_row(&[0.5, 0.5, 9.0], 2),
        ];
        for (v, e) in values.iter().zip(expected.iter()) {
            assert!((v - e).abs() < 1e-5, "got {v}, expected {e}");
        }
    }

    #[test]
    fn fitness_loglik_equal_for_same_row_label() {
        // 相同的 (logits 行, label) 组合必须产生完全相同的奖励元素。
        let logits = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 0.5], [1.0, 2.0, 0.5], [3.0, 1.0, 2.0]],
            &device(),
        );
        let labels = Tensor::<B, 1, burn::tensor::Int>::from_data([1_i32, 1, 0], &device());

        let values = fitness_from_logits_reward(logits, labels, Reward::Loglik)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        // 第 0、1 两行的 logits 与 label 完全相同。
        assert_eq!(values[0], values[1]);
    }

    #[test]
    fn fitness_loglik_softmax_normalization() {
        // softmax 归一化关系：任意一行的所有类别 exp(log_softmax) 之和应约为 1。
        let logits = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 0.5], [3.0, 1.0, 2.0], [0.5, 0.5, 9.0]],
            &device(),
        );
        let batch = 3usize;
        let n_class = 3usize;
        let mut row_sum = vec![0.0_f32; batch];
        for c in 0..n_class as i32 {
            let labels =
                Tensor::<B, 1, burn::tensor::Int>::from_data([c; 3], &device());
            let values = fitness_from_logits_reward(logits.clone(), labels, Reward::Loglik)
                .to_data()
                .to_vec::<f32>()
                .unwrap();
            for (acc, v) in row_sum.iter_mut().zip(values.iter()) {
                *acc += v.exp();
            }
        }
        for s in row_sum {
            assert!((s - 1.0).abs() < 1e-4, "softmax 行和 {s} 偏离 1");
        }
    }

    #[test]
    fn fitness_binary_matches_legacy() {
        // 回归：新 Reward::Binary 分支应与既有 binary 默认函数输出一致。
        let logits = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 0.5], [3.0, 1.0, 2.0], [0.5, 0.5, 9.0]],
            &device(),
        );
        let labels = Tensor::<B, 1, burn::tensor::Int>::from_data([1_i32, 0, 9], &device());

        let legacy = fitness_from_logits(logits.clone(), labels.clone());
        let new = fitness_from_logits_reward(logits, labels, Reward::Binary);
        assert_eq!(
            legacy.to_data().to_vec::<f32>().unwrap(),
            new.to_data().to_vec::<f32>().unwrap()
        );
    }
}
