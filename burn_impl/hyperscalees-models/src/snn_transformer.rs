//! 可训练的 **SNN Transformer**（逐 token 真自注意力，多头 + 位置编码 + 多块残差）。
//!
//! **批量并行**：`forward_batched` 把批量 `n` 放在张量第 0 维，用 rank-3 批量 matmul
//! 一次处理整个批次（对应 Python 参考的 `jax.vmap` 整批前向），使大批量（如 60000）
//! GPU 训练可行（逐样本循环在 GPU 上数千次内核启动不可行）。
//!
//! 设计目标：在 HyperScaleES 演化策略（ES）训练范式下**能实际学到判别信息**的
//! Transformer 式 SNN 模型，对齐 `src/hyperscalees/models/snn_self_attention_heads.py`
//! 的**架构形态**，但采用 `docs/es_selfattn_heads_train_failure_analysis.md` §5 给出的
//! 修复方向——**连续（非硬阈值）注意力核心 + 连续速率 Q/K/V 编码**：
//!
//! - 硬阈值 LIF 竞争会把「奖励对参数微扰的增益」压成零，使 ES 梯度
//!   `mean(scores * perturbation) ≈ 0`，模型停在随机初始化（Python `selfattn_heads`
//!   已复现）。本实现把注意力权重换成 **Boltzmann / Hopfield 连续松弛**：
//!   `u ← u + τ⁻¹(−u + h − g·mean(u))`，读数 `softmax(u)`，对扰动平滑，ES 可累积。
//! - Q/K/V 前端用 **连续 sigmoid 速率编码**（`rate_encode`，接入 `proj_gain`）。
//!
//! ```text
//! x: (T, n, in_dim)  in_dim = num_tokens·token_in_dim（patched 展平）
//!   -> 逐时间步逐 token Q/K/V 投影（in_q/in_k/in_v）-> 连续速率编码 (n, nt, d_model)
//!   -> + 位置编码 pos_emb + 逐样本 pos 噪声
//!   -> L 个块：多头连续自注意力（H=beta·(Q@Kᵀ)，softmax 行归一）-> O=@V；concat；
//!        b_o 投影；X += attn；swish 前馈；X += ffn
//!   -> 池化（mean over tokens）-> out -> *out_gain -> logits (n, num_classes)
//! ```
//!
//! 噪声注入：多头的 q/k/v/o/ff 权重形状大量重复，训练时按**参数索引**（[`TrainNoise`]）
//! 寻址，避免按权重**形状**路由的歧义。

use burn::tensor::activation;
use burn::tensor::{Device, Tensor};
use hyperscalees_core::B;

use crate::common::{Mm, Parameter, EMB_PARAM, MM_PARAM, PARAM};

// ---------------------------------------------------------------------------
// 噪声注入类型（ES 训练）
// ---------------------------------------------------------------------------

/// 可训练前向的全部噪声提供器（[`SnnTransformer::forward_batched`]）。
///
/// 使 ES 梯度 `mean(score · noise)` 有效（前向必须实际注入扰动）：
/// - [`Self::lora`]：所有 `Mm`（`MM_PARAM`）权重的**逐样本** LoRA 噪声，`lora[k]` =
///   `(A (n,r,a) 已乘 sign·base_sigma, B (n,r,b))`，`k` = [`Self::mm_indices`] 中该
///   参数索引的位置；
/// - [`Self::mm_indices`]：`params()` 中 `MM_PARAM` 参数的索引顺序列表；
/// - [`Self::pos_emb`]：`pos_emb` 逐样本加性噪声 `(n, num_tokens, d_model)`；
/// - [`Self::out_gain`]：`out_gain` 逐样本加性噪声 `(n, 1)`；
/// - [`Self::beta`]：`beta` 逐样本加性噪声 `(n, 1)`（加到 raw 再 softplus）。
pub struct TrainNoise<'a> {
    pub lora: &'a [(Tensor<B, 3>, Tensor<B, 3>)],
    pub mm_indices: &'a [usize],
    pub pos_emb: Option<Tensor<B, 3>>,
    pub out_gain: Option<Tensor<B, 2>>,
    pub beta: Option<Tensor<B, 2>>,
}

impl<'a> Default for TrainNoise<'a> {
    fn default() -> Self {
        Self {
            lora: &[],
            mm_indices: &[],
            pos_emb: None,
            out_gain: None,
            beta: None,
        }
    }
}

// ---------------------------------------------------------------------------
// 连续（可微）注意力核心（批量并行）
// ---------------------------------------------------------------------------

/// 多头连续 Hopfield->softmax 注意力（批量并行）。
///
/// `q,k,v`: (n, num_tokens, head_dim)；对每行 token 的 query 到所有 key 的相似度
/// `H = beta·(Q@Kᵀ)` (n, num_tokens, num_tokens)，逐 token 行做 Hopfield 松弛
/// `u ← u + (1/τ)(−u + H − g·mean_{key}(u))`，迭代 `n_iter` 步；稳态读数
/// `A = softmax(u, 沿 key 轴)`（行和 ≈ 1）。`O = A@V` (n, num_tokens, head_dim)。
fn continuous_attention(
    q: Tensor<B, 3>, // (n, nt, head_dim)
    k: Tensor<B, 3>, // (n, nt, head_dim)
    v: Tensor<B, 3>, // (n, nt, head_dim)
    beta: Tensor<B, 3>, // (n, 1, 1) 可逐样本
    g_inh: f32,
    tau_a: f32,
    n_iter: usize,
) -> Tensor<B, 3> {
    // H = q @ kᵀ：批量 matmul，(n, nt, head) @ (n, head, nt) -> (n, nt, nt)。
    let h = q.matmul(k.swap_dims(1, 2)).mul(beta); // (n, nt, nt)
    let one_over_tau = 1.0 / tau_a;
    let mut u = h.clone();
    for _ in 0..n_iter {
        let c = u.clone().mean_dim(2); // (n, nt, 1) 逐 token 行全局活动（divisive）
        u = u.clone() + (u.clone().neg().add(h.clone()) - c.mul_scalar(g_inh)).mul_scalar(one_over_tau);
    }
    // 稳定 softmax 沿 key 轴（dim 2）。
    let max_k = u.clone().max_dim(2); // (n, nt, 1)
    let e = (u - max_k).exp(); // (n, nt, nt)
    let a = e.clone() / e.sum_dim(2).add_scalar(1e-6); // (n, nt, nt) 行和 ≈1
    a.matmul(v) // (n, nt, head_dim)
}

/// softplus：把可训练 raw 参数映射为恒正值（沿样本广播）。
fn softplus(x: Tensor<B, 2>) -> Tensor<B, 2> {
    x.clone().exp().log1p()
}

/// 连续速率编码（批量并行）：`sigmoid(proj_gain · mean_T(p) / (1 + |mean_T(p)|))`。
///
/// `proj` 为逐时间步 Q/K/V 投影电流（见 `encode` 的循环累加），`p_mean` 已是对 T 的
/// 时间平均 (n, num_tokens, d)。返回 (0,1) 连续速率。
fn rate_encode(p_mean: Tensor<B, 3>, gain: f32) -> Tensor<B, 3> {
    let softsign = p_mean.clone() / p_mean.abs().add_scalar(1.0);
    activation::sigmoid(softsign * gain)
}

// ---------------------------------------------------------------------------
// 模型
// ---------------------------------------------------------------------------

/// 可训练的 SNN Transformer 分类器（批量并行，patched-MNIST 形态）。
pub struct SnnTransformer {
    pub in_q: Mm,
    pub in_k: Mm,
    pub in_v: Mm,
    pub pos_emb: Tensor<B, 2>,
    pub blocks: Vec<Block>,
    pub out: Mm,
    pub out_gain: Parameter,
    pub beta: Parameter,
    pub tau_m: f32,
    pub proj_gain: f32,
    pub g_inh: f32,
    pub tau_a: f32,
    pub n_iter: usize,
    pub num_heads: usize,
    pub num_tokens: usize,
    pub token_in_dim: usize,
}

/// 一个 Transformer 块：多头自注意力 + 前馈（残差在 forward 层添加）。
pub struct Block {
    pub q: Vec<Mm>,
    pub k: Vec<Mm>,
    pub v: Vec<Mm>,
    pub o: Mm,
    pub ff1: Mm,
    pub ff2: Mm,
}

impl SnnTransformer {
    /// 构建模型。`d_model` 必须能被 `num_heads` 整除。
    pub fn new(
        token_in_dim: usize,
        num_tokens: usize,
        num_classes: usize,
        d_model: usize,
        num_heads: usize,
        num_blocks: usize,
        device: &Device<B>,
    ) -> Self {
        assert!(d_model % num_heads == 0, "d_model({d_model}) 必须能被 num_heads({num_heads}) 整除");
        let head_dim = d_model / num_heads;
        let in_q = Mm::new(token_in_dim, d_model, device);
        let in_k = Mm::new(token_in_dim, d_model, device);
        let in_v = Mm::new(token_in_dim, d_model, device);
        let pos_emb = Tensor::<B, 2>::zeros([num_tokens, d_model], device);

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut q = Vec::with_capacity(num_heads);
            let mut k = Vec::with_capacity(num_heads);
            let mut v = Vec::with_capacity(num_heads);
            for _ in 0..num_heads {
                q.push(Mm::new(d_model, head_dim, device));
                k.push(Mm::new(d_model, head_dim, device));
                v.push(Mm::new(d_model, head_dim, device));
            }
            blocks.push(Block {
                q,
                k,
                v,
                o: Mm::new(d_model, d_model, device),
                ff1: Mm::new(d_model, d_model, device),
                ff2: Mm::new(d_model, d_model, device),
            });
        }

        let out = Mm::new(d_model, num_classes, device);
        let out_gain = Parameter::new(Tensor::<B, 1>::ones([1], device));
        let raw_beta = ((1.0 / (head_dim as f32).sqrt()).exp() - 1.0).ln();
        let beta = Parameter::new(Tensor::<B, 1>::from_data([raw_beta], device));

        Self {
            in_q,
            in_k,
            in_v,
            pos_emb,
            blocks,
            out,
            out_gain,
            beta,
            tau_m: 20.0,
            proj_gain: 2.0,
            g_inh: 0.5,
            tau_a: 5.0,
            n_iter: 8,
            num_heads,
            num_tokens,
            token_in_dim,
        }
    }

    /// 所有可训练参数（含位置编码），展平 rank-2（`pos_emb` 已是 rank-2；
    /// `out_gain`/`beta` rank-1 `(1,)` unsqueeze 为 `(1,1)`）。
    ///
    /// 顺序：`[in_q(0), in_k(1), in_v(2), pos_emb(3), 每块(3H+3), out, out_gain, beta]`。
    pub fn params(&self) -> Vec<Tensor<B, 2>> {
        let mut ps: Vec<Tensor<B, 2>> = Vec::new();
        ps.push(self.in_q.weight.clone());
        ps.push(self.in_k.weight.clone());
        ps.push(self.in_v.weight.clone());
        ps.push(self.pos_emb.clone());
        for blk in &self.blocks {
            for m in &blk.q {
                ps.push(m.weight.clone());
            }
            for m in &blk.k {
                ps.push(m.weight.clone());
            }
            for m in &blk.v {
                ps.push(m.weight.clone());
            }
            ps.push(blk.o.weight.clone());
            ps.push(blk.ff1.weight.clone());
            ps.push(blk.ff2.weight.clone());
        }
        ps.push(self.out.weight.clone());
        ps.push(self.out_gain.value.clone().unsqueeze::<2>());
        ps.push(self.beta.value.clone().unsqueeze::<2>());
        ps
    }

    /// ES 分类（es_map），顺序与 [`Self::params`] 一一对应。
    pub fn es_map(&self) -> Vec<i32> {
        let num_blocks = self.blocks.len();
        let h = self.num_heads;
        let total = 3 + 1 + num_blocks * (3 * h + 3) + 1 + 2;
        let mut map = vec![MM_PARAM; total];
        map[3] = EMB_PARAM;
        map[total - 1] = PARAM;
        map[total - 2] = PARAM;
        map
    }

    /// 参数列表中第 `b` 个块（0-based）的起始索引。
    #[inline]
    fn block_base(&self, b: usize) -> usize {
        4 + b * (3 * self.num_heads + 3)
    }

    /// `out`（读出）权重在 [`Self::params`] 中的索引。
    #[inline]
    fn out_param_index(&self) -> usize {
        4 + self.blocks.len() * (3 * self.num_heads + 3)
    }

    /// 单个矩阵参数的可选噪声 matmul（**批量并行**，批量 n 在第 0 维）。
    ///
    /// `x`: (n, tokens, in)；`w`: (out, in)；返回 (n, tokens, out)。
    /// LoRA：`base = x @ wᵀ + (x @ Bᵀ) @ A`，`A (n,r,out)`, `B (n,r,in)` 逐样本。
    fn nn(
        &self,
        idx: usize,
        x: Tensor<B, 3>,
        w: &Tensor<B, 2>,
        noise: Option<&TrainNoise>,
    ) -> Tensor<B, 3> {
        let [n, tokens, in_dim] = x.dims();
        let [out, _] = w.dims();
        let base = x
            .clone()
            .reshape([n * tokens, in_dim])
            .matmul(w.clone().transpose())
            .reshape([n, tokens, out]);
        if let Some(tn) = noise {
            if let Some(k) = tn.mm_indices.iter().position(|&i| i == idx) {
                let (a_t, b_t) = &tn.lora[k];
                // a_t (n, r, out), b_t (n, r, in)。
                let y = x.matmul(b_t.clone().swap_dims(1, 2)); // (n, tokens, in)@(n, in, r) -> (n, tokens, r)
                let noise_t = y.matmul(a_t.clone()); // (n, tokens, r)@(n, r, out) -> (n, tokens, out)
                return base + noise_t;
            }
        }
        base
    }

    /// 输入编码：逐时间步逐 token Q/K/V 投影 + 连续速率编码 -> 每 token 速率。
    ///
    /// `x`: (T, n, in_dim=num_tokens·token_in_dim)。对每个时间步把 `(n, in_dim)`
    /// 展平为 `(n·nt, tin)`，以 `(n'=n·nt, tokens=1, tin)` 形式经 [`Self::nn`]
    /// 投影（含 in_q/in_k/in_v 的 LoRA 噪声，参数索引 0/1/2），reshape 回
    /// `(n, nt, d_model)` 累计时间平均，过 `rate_encode`；三路平均为 token 表示。
    fn encode(
        &self,
        x: Tensor<B, 3>, // (T, n, in_dim)
        noise: Option<&TrainNoise>,
    ) -> Tensor<B, 3> {
        let [t, n, in_dim] = x.dims();
        let nt = self.num_tokens;
        let tin = self.token_in_dim;
        let d_model = self.in_q.weight.dims()[0];
        let device = x.device();
        let mut acc_q = Tensor::<B, 3>::zeros([n, nt, d_model], &device);
        let mut acc_k = Tensor::<B, 3>::zeros([n, nt, d_model], &device);
        let mut acc_v = Tensor::<B, 3>::zeros([n, nt, d_model], &device);
        for tt in 0..t {
            let x_t = x
                .clone()
                .slice([tt..tt + 1, 0..n, 0..in_dim])
                .squeeze_dim::<2>(0) // (n, in_dim)
                .reshape([n, nt, tin]); // (n, nt, tin)
            // 输入编码投影（in_q/in_k/in_v，共享权重 + 逐样本 LoRA 噪声）：批量 n 与
            // 噪声 a_t/b_t 的样本维对齐（tokens 不折进 batch，LoRA 是 per-sample）。
            // nn 返回 (n, nt, d_model)。
            let q_t = self.nn(0, x_t.clone(), &self.in_q.weight, noise);
            let k_t = self.nn(1, x_t.clone(), &self.in_k.weight, noise);
            let v_t = self.nn(2, x_t, &self.in_v.weight, noise);
            acc_q = acc_q.clone() + q_t;
            acc_k = acc_k.clone() + k_t;
            acc_v = acc_v.clone() + v_t;
        }
        let inv_t = 1.0 / t as f32;
        let qr = rate_encode(acc_q.mul_scalar(inv_t), self.proj_gain);
        let kr = rate_encode(acc_k.mul_scalar(inv_t), self.proj_gain);
        let vr = rate_encode(acc_v.mul_scalar(inv_t), self.proj_gain);
        (qr + kr + vr).div_scalar(3.0)
    }

    /// 批量前向：`(T, n, in_dim)` 泊松脉冲 -> `(n, num_classes)` logits。
    ///
    /// `in_dim = num_tokens·token_in_dim`（patched 展平）。批量并行（n 在第 0 维）。
    /// `noise` 提供全部可训练参数的逐样本扰动（[`TrainNoise`]；`None` 走干净路径）。
    pub fn forward_batched(
        &self,
        x: Tensor<B, 3>, // (T, n, in_dim)
        noise: Option<&TrainNoise>,
    ) -> Tensor<B, 2> {
        let [t, n, in_dim] = x.dims();
        assert_eq!(
            in_dim,
            self.num_tokens * self.token_in_dim,
            "输入维度必须为 num_tokens·token_in_dim = {}，实际 {in_dim}",
            self.num_tokens * self.token_in_dim
        );
        let d_model = self.in_q.weight.dims()[0];
        let device = x.device();

        // ---- 输入编码 ----
        let mut xh = self.encode(x, noise); // (n, nt, d_model)

        // ---- 位置编码 + 逐样本位置噪声 ----
        let pos = self.pos_emb.clone().unsqueeze_dim::<3>(0); // (1, nt, d)
        xh = xh + pos;
        if let Some(pn) = noise.and_then(|n| n.pos_emb.as_ref()) {
            xh = xh + pn.clone(); // (n, nt, d)
        }

        // ---- beta（逐样本）：(n, 1, 1) ----
        let beta_raw = match noise.and_then(|n| n.beta.as_ref()) {
            Some(bn) => self.beta.value.clone().unsqueeze::<2>() + bn.clone(), // (n,1)+(1,1)
            None => self.beta.value.clone().unsqueeze::<2>(), // (1,1)
        };
        let beta = softplus(beta_raw).unsqueeze_dim::<3>(2); // (n,1,1)
        let mm = noise;

        // ---- 各块 ----
        for b in 0..self.blocks.len() {
            let base = self.block_base(b);
            // 多头注意力
            let mut heads: Vec<Tensor<B, 3>> = Vec::with_capacity(self.num_heads);
            let blk = &self.blocks[b];
            for h in 0..self.num_heads {
                let q = self.nn(base + 3 * h, xh.clone(), &blk.q[h].weight, mm);
                let k = self.nn(base + 3 * h + 1, xh.clone(), &blk.k[h].weight, mm);
                let v = self.nn(base + 3 * h + 2, xh.clone(), &blk.v[h].weight, mm);
                let o = continuous_attention(q, k, v, beta.clone(), self.g_inh, self.tau_a, self.n_iter);
                heads.push(o); // (n, nt, head_dim)
            }
            let concat = Tensor::cat(heads, 2); // (n, nt, H·head_dim = d_model)
            let attn = self.nn(base + 3 * self.num_heads, concat, &blk.o.weight, mm);
            xh = xh + attn; // 残差 1
            // 前馈
            let ff1 = self.nn(base + 3 * self.num_heads + 1, xh.clone(), &blk.ff1.weight, mm);
            let act = ff1.clone().mul(activation::sigmoid(ff1)); // swish
            let ff2 = self.nn(base + 3 * self.num_heads + 2, act, &blk.ff2.weight, mm);
            xh = xh + ff2; // 残差 2
        }

        // ---- 池化 + 读出 ----
        let pooled = xh.mean_dim(1).squeeze_dim::<2>(1); // (n, d_model)
        let out_idx = self.out_param_index();
        let logits = self
            .nn(out_idx, pooled.unsqueeze_dim::<3>(1), &self.out.weight, mm)
            .squeeze_dim::<2>(1); // (n, num_classes)
        // out_gain（+ 逐样本噪声）。
        let gain = match noise.and_then(|n| n.out_gain.as_ref()) {
            Some(gn) => self.out_gain.value.clone().unsqueeze::<2>() + gn.clone(), // (n,1)
            None => self.out_gain.value.clone().unsqueeze::<2>(), // (1,1)
        };
        logits * gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    const TIN: usize = 4;
    const NTOK: usize = 3;
    const D_MODEL: usize = 4;
    const NHEADS: usize = 2;
    const NCLASSES: usize = 5;

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    }

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// 确定性输入：`(T=2, n, in_dim=NTOK*TIN)`。
    fn sample_batch(n: usize) -> Tensor<B, 3> {
        let in_dim = NTOK * TIN;
        let data: Vec<f32> = (0..(2 * n * in_dim))
            .map(|i| ((i % 3 == 0) || (i % 5 == 0)) as i32 as f32)
            .collect();
        Tensor::<B, 3>::from_data(TensorData::new(data, [2, n, in_dim].to_vec()), &device())
    }

    fn count_params(model: &SnnTransformer) -> usize {
        3 + 1 + model.blocks.len() * (3 * model.num_heads + 3) + 1 + 2
    }

    fn test_model(num_blocks: usize) -> SnnTransformer {
        SnnTransformer::new(TIN, NTOK, NCLASSES, D_MODEL, NHEADS, num_blocks, &device())
    }

    #[test]
    fn struct_and_es_map() {
        let model = test_model(1);
        assert_eq!(model.in_q.weight.dims(), [D_MODEL, TIN]);
        assert_eq!(model.pos_emb.dims(), [NTOK, D_MODEL]);
        assert_eq!(model.blocks[0].q.len(), NHEADS);
        assert_eq!(model.blocks[0].q[0].weight.dims(), [D_MODEL / NHEADS, D_MODEL]);
        assert_eq!(model.out.weight.dims(), [NCLASSES, D_MODEL]);

        let expected = count_params(&model);
        assert_eq!(model.params().len(), expected);
        let es = model.es_map();
        assert_eq!(es.len(), expected);
        assert_eq!(es[3], EMB_PARAM);
        assert_eq!(es[expected - 1], PARAM);
        assert_eq!(es[expected - 2], PARAM);
        for (i, &m) in es.iter().enumerate() {
            if i != 3 && i != expected - 1 && i != expected - 2 {
                assert_eq!(m, MM_PARAM, "index {i} 应为 MM_PARAM");
            }
        }
    }

    #[test]
    fn forward_batched_clean_deterministic_and_shaped() {
        let model = test_model(1);
        let x = sample_batch(3);
        let out1 = model.forward_batched(x.clone(), None);
        let out2 = model.forward_batched(x, None);
        assert_eq!(out1.dims(), [3, NCLASSES]);
        assert_eq!(to_vec(out1.clone()), to_vec(out2));
        assert!(to_vec(out1).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_batched_noised_differs_from_clean() {
        let model = test_model(1);
        let x = sample_batch(4);
        let clean = model.forward_batched(x.clone(), None);

        let es = model.es_map();
        let mm_indices: Vec<usize> = (0..es.len()).filter(|&i| es[i] == MM_PARAM).collect();
        let params = model.params();
        let mut lora = Vec::with_capacity(mm_indices.len());
        let r = 2usize;
        let n = 4usize;
        for &idx in &mm_indices {
            let [a, b] = params[idx].dims();
            lora.push((
                Tensor::<B, 3>::full([n, r, a], 0.5, &device()),
                Tensor::<B, 3>::full([n, r, b], 1.0, &device()),
            ));
        }
        let tn = TrainNoise {
            lora: &lora,
            mm_indices: &mm_indices,
            pos_emb: None,
            out_gain: None,
            beta: None,
        };
        let noised = model.forward_batched(x, Some(&tn));
        assert_eq!(noised.dims(), [4, NCLASSES]);
        assert!(to_vec(clean).iter().zip(to_vec(noised).iter()).any(|(a, b)| a != b));
    }

    #[test]
    fn multi_block_forward_shaped() {
        for nb in 1..=3usize {
            let model = test_model(nb);
            let x = sample_batch(2);
            let out = model.forward_batched(x, None);
            assert_eq!(out.dims(), [2, NCLASSES], "num_blocks={nb}");
            assert!(to_vec(out).iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn pos_emb_affects_output() {
        let mut model = test_model(1);
        let x = sample_batch(2);
        let base = to_vec(model.forward_batched(x.clone(), None));
        model.pos_emb = Tensor::<B, 2>::ones([NTOK, D_MODEL], &device());
        let changed = to_vec(model.forward_batched(x, None));
        assert!(base.iter().zip(changed.iter()).any(|(a, b)| a != b));
    }

    #[test]
    fn forward_batched_per_sample_independent() {
        let model = test_model(1);
        let tin = NTOK * TIN;
        let sol: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(
                [
                    1.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                    0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                ]
                .to_vec(),
                [2, 1, tin].to_vec(),
            ),
            &device(),
        );
        let sb: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(
                [
                    0.0_f32, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                ]
                .to_vec(),
                [2, 1, tin].to_vec(),
            ),
            &device(),
        );
        let x3 = Tensor::cat(
            vec![
                sol.clone().slice([0..2, 0..1, 0..tin]),
                sb.slice([0..2, 0..1, 0..tin]),
                sol.slice([0..2, 0..1, 0..tin]),
            ],
            1,
        );
        let out = model.forward_batched(x3, None);
        assert_eq!(out.dims(), [3, NCLASSES]);
        let o0 = to_vec(out.clone().slice([0..1, 0..NCLASSES]).squeeze_dim::<1>(0));
        let o2 = to_vec(out.slice([2..3, 0..NCLASSES]).squeeze_dim::<1>(0));
        assert_eq!(o0, o2, "相同输入的样本应得到相同 logits");
    }
}
