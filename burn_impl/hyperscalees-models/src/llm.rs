//! LLM components (RWKV-7 layer_norm / group_norm / ChannelMixing /
//! TimeMixing + BaseRWKV inner_loop, RWKV-6 Qwen2RMSNorm / Qwen2MLP /
//! RWKV6Attention + inner_loop, and the RWKV tokenizer pure-algorithm),
//! ported from `src/hyperscalees/models/llm/{llm,rwkv7,qrwkv6,tokenizer}.py`.
//!
//! Scope is the structural / pure-algorithm layer: the algebraic components
//! and the canonical state-recurrence inner loop. The end-to-end pretrained
//! transformer (embed / outhead / forward_seq over many layers) and the 20B
//! tokenizer file are intentionally NOT ported here.
//!
//! burn tensors are statically ranked, so each component is a Rust struct
//! holding named `Tensor` fields. Non-tensor state (the `new_starts` flags
//! and the `length` cutoff) are passed as plain Rust values; the per-timestep
//! recurrence is a plain `for` loop over a small concrete `T`.

#![allow(dead_code)]

use std::collections::HashMap;

use burn::tensor::activation;
use burn::tensor::{Device, Tensor};
use hyperscalees_core::B;

use crate::common::{Linear, Parameter, Tmm};

// ---------------------------------------------------------------------------
// Layer / group normalization
// ---------------------------------------------------------------------------

/// Layer normalization module (RWKV-7):
/// ``(x - mean) / sqrt(var + eps) * weight + bias``.
///
/// `weight` / `bias` are per-channel rank-1 `(C,)` [`Parameter`]s broadcast
/// against the last axis of `x`.
pub struct LayerNorm {
    /// Scale weight, shape `(C,)`.
    pub weight: Parameter,
    /// Shift bias, shape `(C,)`.
    pub bias: Parameter,
}

impl LayerNorm {
    /// Normalize `x` (rank `D`, last axis size `C`) over the last axis,
    /// scale/shift by the per-channel weight/bias. `mean_dim(-1)` keeps the
    /// reduced axis so the result broadcast against `x`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, eps: f32) -> Tensor<B, D> {
        let mean = x.clone().mean_dim(-1); // (..., 1)
        let mean_sq = mean.clone().powf_scalar(2.0);
        let x_sq = x.clone().powf_scalar(2.0);
        let var = x_sq.mean_dim(-1) - mean_sq; // E[x^2] - E[x]^2, (..., 1)
        let std = (var + eps).sqrt();
        let normed = (x - mean) / std;

        let dims = normed.dims();
        let c = dims[D - 1];
        let mut wshape = [1usize; D];
        wshape[D - 1] = c;
        let weight = self.weight.value.clone().reshape(wshape);
        let bias = self.bias.value.clone().reshape(wshape);
        normed * weight + bias
    }
}

/// Free-form layer norm: ``(x - mean) / sqrt(var + eps) * weight + bias``,
/// with explicit rank-1 `(C,)` `weight` / `bias` tensors.
pub fn layer_norm<const D: usize>(
    x: Tensor<B, D>,
    weight: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, D> {
    let mean = x.clone().mean_dim(-1);
    let mean_sq = mean.clone().powf_scalar(2.0);
    let x_sq = x.clone().powf_scalar(2.0);
    let var = x_sq.mean_dim(-1) - mean_sq;
    let std = (var + eps).sqrt();
    let normed = (x - mean) / std;

    let dims = normed.dims();
    let c = dims[D - 1];
    let mut wshape = [1usize; D];
    wshape[D - 1] = c;
    normed * weight.clone().reshape(wshape) + bias.clone().reshape(wshape)
}

/// Group normalization module (RWKV-7):
/// reshape `(N, C, ...)` to `(N, G, C//G, ...)`, normalize over the inner
/// group axes, reshape back, then scale/shift by weight/bias broadcast to
/// `(1, C, 1, ...)`.
pub struct GroupNorm {
    /// Scale weight, shape `(C,)`.
    pub weight: Parameter,
    /// Shift bias, shape `(C,)`.
    pub bias: Parameter,
}

impl GroupNorm {
    /// Normalize over `num_groups` across the channel axis. Supports the two
    /// ranks used by RWKV-7: rank-2 `(N, C)` (e.g. TimeMixing's `ln_x` on
    /// `(T, C)` with `G == H`) and rank-3 `(N, C, L)`.
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
        num_groups: usize,
        eps: f32,
    ) -> Tensor<B, D> {
        let dims = x.dims();
        let (n, c) = (dims[0], dims[1]);
        let g = num_groups;
        let inner = c / g;

        // Flatten to (N, G, C//G, rest...), normalize over the inner axes,
        // and restore the (N, C, ...) layout inside each branch so both arms
        // return a rank-`D` tensor.
        let xr = match D {
            2 => {
                // (N, G, C//G)
                let xg = x.clone().reshape([n, g, inner]);
                let mean = xg.clone().mean_dim(2);
                let var = xg.clone().powf_scalar(2.0).mean_dim(2)
                    - mean.clone().powf_scalar(2.0);
                let normed = (xg - mean) / (var + eps).sqrt();
                normed.reshape(dims) // (N, C)
            }
            3 => {
                // (N, G, C//G, L)
                let l = dims[2];
                let xg = x.clone().reshape([n, g, inner, l]);
                let mean = xg.clone().mean_dims(&[2, 3]);
                let var = xg.clone().powf_scalar(2.0).mean_dims(&[2, 3])
                    - mean.clone().powf_scalar(2.0);
                let normed = (xg - mean) / (var + eps).sqrt();
                normed.reshape(dims) // (N, C, L)
            }
            _ => panic!("GroupNorm rank unsupported: {D}"),
        };

        // gamma / beta reshaped to (1, C, 1, ...).
        let mut gb = [1usize; D];
        gb[1] = c;
        let gamma = self.weight.value.clone().reshape(gb);
        let beta = self.bias.value.clone().reshape(gb);
        gamma * xr + beta
    }
}

// ---------------------------------------------------------------------------
// RWKV-7: ChannelMixing / TimeMixing / base inner loop
// ---------------------------------------------------------------------------

/// Build the rank-2 Bool mask `(T, 1)` marking the `new_starts` timesteps.
fn new_starts_mask_2d(
    new_starts: &[bool],
    device: &Device<B>,
    _t: usize,
) -> Tensor<B, 2, burn::tensor::Bool> {
    let f: Vec<f32> = new_starts.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let f = Tensor::<B, 1>::from_data(f.as_slice(), device).unsqueeze_dim::<2>(1); // (T, 1)
    f.greater_equal_elem(0.5)
}

/// Zero out the `sx` rows where a new chunk starts, mirroring
/// `jnp.where(new_starts[:, None], 0, sx)`.
fn zero_at_new_starts(sx: Tensor<B, 2>, new_starts: &[bool], device: &Device<B>) -> Tensor<B, 2> {
    let (t, c) = (sx.dims()[0], sx.dims()[1]);
    let mask = new_starts_mask_2d(new_starts, device, t);
    let zeros = Tensor::<B, 2>::zeros([t, c], device);
    sx.mask_where(mask, zeros)
}

/// RWKV-7 ChannelMixing:
///
/// ```text
///   sx = concat([state, x[:-1]], 0); sx = where(new_starts, 0, sx); sx = sx - x
///   xk = x + sx * x_k
///   k  = square(relu(Linear_key(xk)))
///   return Linear_value(k), x[length - 1]
/// ```
///
/// `state` is `(1, C)` (the previous token history row); returns the channel
/// output `(T, C)` and the updated state `x[length - 1]` `(C,)`.
pub struct ChannelMixing {
    /// Per-channel mixing strength, `(C,)`.
    pub x_k: Parameter,
    /// Key projection.
    pub key: Linear,
    /// Value projection.
    pub value: Linear,
}

impl ChannelMixing {
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        state: Tensor<B, 2>,
        length: usize,
        new_starts: &[bool],
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (t, c) = (x.dims()[0], x.dims()[1]);
        let device = x.device().clone();

        let x_prev = x.clone().slice([0..t - 1, 0..c]); // (T-1, C)
        let sx = Tensor::cat(vec![state, x_prev], 0); // (T, C)
        let sx = zero_at_new_starts(sx, new_starts, &device);
        let sx = sx - x.clone();

        let xk = x.clone() + sx * self.x_k.value.clone().unsqueeze::<2>();
        let k = activation::relu(self.key.forward(xk)).powf_scalar(2.0);
        let out = self.value.forward(k);

        let last = x.slice([length - 1..length, 0..c]); // (1, C)
        (out, last)
    }
}

/// RWKV-7 BaseRWKV inner loop (the canonical state recurrence), operating on
/// per-timestep tensors:
///
/// ```text
///   w = exp(-exp(w))
///   for t in 0..T:
///     s    = where(new_starts[t], 0, s)
///     rt   = r[t, .., None]; wt = w[t, None]; kt = k[t, None]
///     vt   = v[t, .., None]; at = a[t, .., None]; bt = b[t, None]
///     sa   = s @ at
///     s    = s * wt + vt @ kt + sa @ bt
///     out[t] = (s @ rt).squeeze
///     if t < length: out_s = s
///   return out_s, out
/// ```
///
/// `r, w, k, v, a, b` are `(T, H, S)`; `s` is `(H, S, S)`; `new_starts` is a
/// length-`T` bool slice. Returns `(out_s, out)` with `out_s: (H, S, S)` and
/// `out: (T, H, S)`.
pub fn rwkv7_inner_loop(
    r: Tensor<B, 3>,
    w: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    s: Tensor<B, 3>,
    length: usize,
    new_starts: &[bool],
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let dims = r.dims();
    let (t, h, sd) = (dims[0], dims[1], dims[2]);
    let w = w.exp().neg().exp(); // exp(-exp(w)), (T, H, S)

    let reset_s = Tensor::<B, 3>::zeros([h, sd, sd], &s.device());
    let mut state = s;
    let mut out_s = state.clone();
    let mut outs: Vec<Tensor<B, 3>> = Vec::with_capacity(t);

    for tt in 0..t {
        if new_starts[tt] {
            state = reset_s.clone();
        }
        // Slice the t-th (H, S) and expand along the broadcast axis.
        let rt = r.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(2); // (H, S, 1)
        let wt = w.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1); // (H, 1, S)
        let kt = k.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1); // (H, 1, S)
        let vt = v.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(2); // (H, S, 1)
        let at = a.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(2); // (H, S, 1)
        let bt = b.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1); // (H, 1, S)

        let sa = state.clone().matmul(at.clone()); // (H,S,S)@(H,S,1) -> (H,S,1)
        state = state.clone().mul(wt).add(vt.matmul(kt)).add(sa.matmul(bt)); // (H,S,S)

        let out_t = state.clone().matmul(rt).squeeze_dim::<2>(2); // (H,S)
        if tt < length {
            out_s = state.clone();
        }
        outs.push(out_t.unsqueeze::<3>());
    }
    let out = Tensor::cat(outs, 0); // (T, H, S)
    (out_s, out)
}

/// RWKV-7 TimeMixing. This is the full per-layer mixer (see the module doc),
/// ported algebra-for-algebra. It returns `(out, state, v_first)` where
/// `out: (T, C)` (after the output projection), `state: (1 + S, C)` and
/// `v_first: (T, C)`.
///
/// The `T`, `H`, `S` are read from the tensor shapes / `state` shape.
pub struct TimeMixing {
    /// Per-channel time offsets (six `(C,)` parameters).
    pub x_r: Parameter,
    pub x_w: Parameter,
    pub x_k: Parameter,
    pub x_v: Parameter,
    pub x_a: Parameter,
    pub x_g: Parameter,
    /// Receptance projection.
    pub receptance: Linear,
    /// Decay network: `w0` + TMM_w2(tanh(TMM_w1))
    pub w0: Parameter,
    pub w1: Tmm,
    pub w2: Tmm,
    /// Key projection.
    pub key: Linear,
    /// Value projection.
    pub value: Linear,
    /// First-token value blend network.
    pub v0: Parameter,
    pub v1: Tmm,
    pub v2: Tmm,
    /// Soft-attention gating network.
    pub a0: Parameter,
    pub a1: Tmm,
    pub a2: Tmm,
    /// Output gate network.
    pub g1: Tmm,
    pub g2: Tmm,
    /// kk normalization scale / mixing.
    pub k_k: Parameter,
    pub k_a: Parameter,
    /// Per-head group normalization on the channel dim.
    pub ln_x: GroupNorm,
    /// Receptance-readout cross weight, `(C,)`.
    pub r_k: Parameter,
    /// Output projection.
    pub output: Linear,
}

impl TimeMixing {
    /// Run the full TimeMixing algebra. `x: (T, C)`, `state: (1 + S, C)`,
    /// `v_first: (T, C)`, `layer_id` selects the layer-0 first-token branch,
    /// `inner` is the state recurrence ([`rwkv7_inner_loop`]).
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        state: Tensor<B, 2>,
        v_first: Tensor<B, 2>,
        length: usize,
        new_starts: &[bool],
        h: usize,
        s_dim: usize,
        layer_id: usize,
        inner: fn(
            Tensor<B, 3>,
            Tensor<B, 3>,
            Tensor<B, 3>,
            Tensor<B, 3>,
            Tensor<B, 3>,
            Tensor<B, 3>,
            Tensor<B, 3>,
            usize,
            &[bool],
        ) -> (Tensor<B, 3>, Tensor<B, 3>),
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (t, c) = (x.dims()[0], x.dims()[1]);
        let device = x.device().clone();

        // ---- sx = concat([state[:1], x[:-1]], 0); zero on new starts; -x --
        let state_head = state.clone().slice([0..1, 0..c]); // (1, C)
        let x_prev = x.clone().slice([0..t - 1, 0..c]); // (T-1, C)
        let sx = Tensor::cat(vec![state_head, x_prev], 0); // (T, C)
        let sx = zero_at_new_starts(sx, new_starts, &device);
        let sx = sx - x.clone();

        // ---- per-gate inputs x + sx * x_* ---------------------------------
        let xr = x.clone()
            + sx.clone() * self.x_r.value.clone().unsqueeze::<2>();
        let xw = x.clone()
            + sx.clone() * self.x_w.value.clone().unsqueeze::<2>();
        let xk = x.clone()
            + sx.clone() * self.x_k.value.clone().unsqueeze::<2>();
        let xv = x.clone()
            + sx.clone() * self.x_v.value.clone().unsqueeze::<2>();
        let xa = x.clone()
            + sx.clone() * self.x_a.value.clone().unsqueeze::<2>();
        let xg = x.clone()
            + sx.clone() * self.x_g.value.clone().unsqueeze::<2>();

        // ---- r, w, k, v ---------------------------------------------------
        let r = self.receptance.forward(xr); // (T, C)

        let w1_out = self.w1.forward(xw).tanh(); // (T, k)
        let w2_out = self.w2.forward(w1_out); // (T, C)
        let inner_w = self.w0.value.clone().unsqueeze::<2>() + w2_out; // (T, C)
        let w = activation::softplus(inner_w.neg(), 1.0).neg().sub_scalar(0.5); // (T, C)

        let k = self.key.forward(xk.clone()); // (T, C)
        let v = self.value.forward(xv.clone()); // (T, C)

        // ---- v_first update (only layer 0 uses the raw v) -----------------
        let mut v_first = v_first;
        let v = if layer_id == 0 {
            v_first = v.clone();
            v.clone()
        } else {
            let v1_out = self.v1.forward(xv).tanh(); // (T, k)
            let v2_out = self.v2.forward(v1_out); // (T, C)
            let sig = activation::sigmoid(self.v0.value.clone().unsqueeze::<2>() + v2_out); // (T, C)
            v.clone() + (v_first.clone() - v.clone()) * sig
        };

        // ---- a, g ---------------------------------------------------------
        let a1_out = self.a1.forward(xa).tanh(); // (T, k)
        let a2_out = self.a2.forward(a1_out); // (T, C)
        let a = activation::sigmoid(self.a0.value.clone().unsqueeze::<2>() + a2_out); // (T, C)

        let g1_out = self.g1.forward(xg); // (T, k)
        let g = self.g2.forward(activation::sigmoid(g1_out)); // (T, C)

        // ---- kk normalization --------------------------------------------
        let kk2 = k.clone() * self.k_k.value.clone().unsqueeze::<2>(); // (T, C)
        let kk3 = kk2.clone().reshape([t, h, s_dim]); // (T, H, S)
        let kk_norm = kk3
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-12); // (T, H, 1)
        let kk3n = kk3 / kk_norm; // (T, H, S)
        let kk = kk3n.reshape([t, c]); // (T, C)

        let k = k * (self.k_a.value.clone().unsqueeze::<2>() * (a.clone() - 1.0) + 1.0); // (T, C)

        // ---- state bookkeeping for the recurrence -------------------------
        let last_row = x.slice([length - 1..length, 0..c]); // (1, C)
        let mut state = state.slice_assign([0..1, 0..c], last_row); // state[0] = x[length-1]

        let s_init_row = state.clone().slice([1..1 + s_dim, 0..c]); // (S, C)
        let s_init = s_init_row.reshape([h, s_dim, s_dim]); // (H, S, S)

        // ---- reshape to (T, H, S) and run the recurrence ------------------
        let r3 = r.clone().reshape([t, h, s_dim]);
        let w3 = w.reshape([t, h, s_dim]);
        let k3 = k.clone().reshape([t, h, s_dim]);
        let v3 = v.clone().reshape([t, h, s_dim]);
        let a_i = kk.clone().neg().reshape([t, h, s_dim]);
        let b_i = (kk.clone() * a.clone()).reshape([t, h, s_dim]);

        let (state_new, out) = inner(r3, w3, k3, v3, a_i, b_i, s_init, length, new_starts);

        let state_flat = state_new.reshape([s_dim, c]); // (S, C)
        state = state.slice_assign([1..1 + s_dim, 0..c], state_flat); // state[1:] = ...
        let mut x = out.reshape([t, h * s_dim]); // (T, C)

        // ---- group norm per head, then residual + gate --------------------
        x = self.ln_x.forward(x, h, 64e-5);

        let r4 = r.clone().reshape([1, t, h, s_dim]); // (1, T, H, S)
        let k4 = k.clone().reshape([1, t, h, s_dim]); // (1, T, H, S)
        let v4 = v.clone().reshape([1, t, h, s_dim]); // (1, T, H, S)
        let rk4 = self
            .r_k
            .value
            .clone()
            .reshape([1, 1, h, s_dim]); // (1, 1, H, S)
        let readout = r4.mul(k4).mul(rk4).sum_dim(3).mul(v4); // (1,T,H,1)*(1,T,H,S) -> (1,T,H,S)
        let readout = readout.reshape([t, c]); // (T, C)
        x = x + readout;

        x = x * g; // (T, C)

        let out = self.output.forward(x);
        (out, state, v_first)
    }
}

// ---------------------------------------------------------------------------
// RWKV-6 (Qwen2) components
// ---------------------------------------------------------------------------

/// Qwen2 RMS normalization:
/// ``hidden * rsqrt(mean(x^2, axis=-1) + eps) * weight``.
pub struct Qwen2RMSNorm {
    /// Scale weight, shape `(C,)`.
    pub weight: Parameter,
}

impl Qwen2RMSNorm {
    pub fn forward(&self, x: Tensor<B, 2>, eps: f32) -> Tensor<B, 2> {
        let var = x.clone().powf_scalar(2.0).mean_dim(-1); // (T, 1)
        let hidden = x * (var + eps).sqrt().recip(); // * rsqrt(variance + eps)
        hidden * self.weight.value.clone().unsqueeze::<2>()
    }
}

/// Qwen2 MLP: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.
pub struct Qwen2MLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Qwen2MLP {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// RWKV-6 inner loop (qrwkv6):
///
/// ```text
///   scale = S^-0.5;  w = exp(w)
///   for t in 0..T:
///     s    = where(new_starts[t], 0, s)
///     rt   = r[t, None] * scale; kt = k[t, .., None]; vt = v[t, None]
///     at   = kt * vt
///     s    = at + w[t] * s
///     out[t] = (rt @ s).squeeze(1)
///     if t < length: out_s = s
/// ```
///
/// `r, k, v` are `(T, H, S)`; `w` is `(T, H, S, 1)` (the `log_w` decay); `s`
/// is `(H, S, S)`. Returns `(out_s, out)` with `out_s: (H,S,S)`, `out: (T,H,S)`.
pub fn rwkv6_inner_loop(
    r: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 4>,
    s: Tensor<B, 3>,
    length: usize,
    new_starts: &[bool],
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let dims = r.dims();
    let (t, h, sd) = (dims[0], dims[1], dims[2]);
    let scale = (sd as f32).powf(-0.5);
    let w = w.exp(); // (T, H, S, 1)

    let reset_s = Tensor::<B, 3>::zeros([h, sd, sd], &s.device());
    let mut state = s;
    let mut out_s = state.clone();
    let mut outs: Vec<Tensor<B, 3>> = Vec::with_capacity(t);

    for tt in 0..t {
        if new_starts[tt] {
            state = reset_s.clone();
        }
        let rt = r.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1)
            .mul_scalar(scale); // (H, 1, S)
        let kt = k.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(2); // (H, S, 1)
        let vt = v.clone()
            .slice([tt..tt + 1, 0..h, 0..sd])
            .squeeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1); // (H, 1, S)
        let at = kt.mul(vt); // (H, S, S)
        let wt = w.clone()
            .slice([tt..tt + 1, 0..h, 0..sd, 0..1])
            .squeeze_dim::<3>(0); // (H, S, 1)
        state = at + wt.mul(state.clone()); // (H, S, S)

        let out_t = rt.matmul(state.clone()).squeeze_dim::<2>(1); // (H, 1, S) -> (H, S)
        if tt < length {
            out_s = state.clone();
        }
        outs.push(out_t.unsqueeze::<3>());
    }
    let out = Tensor::cat(outs, 0); // (T, H, S)
    (out_s, out)
}

/// Repeat a `(T, K, S)` tensor `n` times along axis 1, mirroring
/// `jnp.repeat(k, n, axis=-2)`.
fn repeat_axis1(t: Tensor<B, 3>, n: usize) -> Tensor<B, 3> {
    let dims = t.dims();
    let (a, b, cc) = (dims[0], dims[1], dims[2]);
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        parts.push(t.clone().slice([0..a, 0..b, 0..cc]));
    }
    Tensor::cat(parts, 1) // (T, n*K, S)
}

/// RWKV-6 attention (qrwkv6 `RWKV6Attention`), algebra-for-algebra. Returns
/// `(out, state)` with `out: (T, C)` and `state: (1 + S, C)`.
pub struct RWKV6Attention {
    /// Per-channel time mixing parameters `(C,)`.
    pub time_maa_x: Parameter,
    pub time_maa_r: Parameter,
    pub time_maa_k: Parameter,
    pub time_maa_v: Parameter,
    pub time_maa_w: Parameter,
    pub time_maa_g: Parameter,
    /// 5-way channel mixer network: `time_maa_w1: (C, 5C)`, `time_maa_w2: (C, C)`.
    pub time_maa_w1: Tmm,
    pub time_maa_w2: Tmm,
    /// Decay network.
    pub time_decay: Parameter,
    pub time_decay_w1: Tmm,
    pub time_decay_w2: Tmm,
    /// Q / K / V / gate / output projections.
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub gate: Linear,
    pub o_proj: Linear,
}

impl RWKV6Attention {
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        state: Tensor<B, 2>,
        length: usize,
        new_starts: &[bool],
        h: usize,
        s_dim: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (t, c) = (x.dims()[0], x.dims()[1]);
        let device = x.device().clone();

        // ---- sx -----------------------------------------------------------
        let sx = Tensor::cat(
            vec![
                state.clone().slice([0..1, 0..c]),
                x.clone().slice([0..t - 1, 0..c]),
            ],
            0,
        ); // (T, C)
        let sx = zero_at_new_starts(sx, new_starts, &device);
        let sx = sx - x.clone();

        // ---- xxx = 5-way time mixing -------------------------------------
        let maa_in = x.clone() + sx.clone() * self.time_maa_x.value.clone().unsqueeze::<2>();
        let w1 = self.time_maa_w1.forward(maa_in).tanh(); // (T, 5C)
        let five = self.time_maa_w1.weight.dims()[1] / c; // 5
        let w1_3 = w1.reshape([t, five, c]); // (T, 5, C)
        let w1_t = w1_3.swap_dims(0, 1); // (5, T, C)

        // Per-slice (T, C) @ time_maa_w2 (C, C) -> (5, T, C).
        let mut parts: Vec<Tensor<B, 3>> = Vec::with_capacity(five);
        for i in 0..five {
            let sl = w1_t
                .clone()
                .slice([i..i + 1, 0..t, 0..c])
                .squeeze_dim::<2>(0); // (T, C)
            parts.push(self.time_maa_w2.forward(sl).unsqueeze::<3>());
        }
        let xxx = Tensor::cat(parts, 0); // (5, T, C)

        let mr = xxx.clone().slice([0..1, 0..t, 0..c]).squeeze_dim::<2>(0);
        let mk = xxx.clone().slice([1..2, 0..t, 0..c]).squeeze_dim::<2>(0);
        let mv = xxx.clone().slice([2..3, 0..t, 0..c]).squeeze_dim::<2>(0);
        let mw = xxx.clone().slice([3..4, 0..t, 0..c]).squeeze_dim::<2>(0);
        let mg = xxx.slice([4..5, 0..t, 0..c]).squeeze_dim::<2>(0);

        // ---- per-gate inputs x + sx * (time_maa_* + m*) -------------------
        let xr = x.clone() + sx.clone() * (self.time_maa_r.value.clone().unsqueeze::<2>() + mr);
        let xk = x.clone() + sx.clone() * (self.time_maa_k.value.clone().unsqueeze::<2>() + mk);
        let xv = x.clone() + sx.clone() * (self.time_maa_v.value.clone().unsqueeze::<2>() + mv);
        let xw = x.clone() + sx.clone() * (self.time_maa_w.value.clone().unsqueeze::<2>() + mw);
        let xg = x.clone() + sx.clone() * (self.time_maa_g.value.clone().unsqueeze::<2>() + mg);

        // ---- projections --------------------------------------------------
        let r = self.q_proj.forward(xr).reshape([t, h, s_dim]); // (T, H, S)
        let k = self.k_proj.forward(xk).reshape([t, c / s_dim, s_dim]); // (T, K, S)
        let v = self.v_proj.forward(xv).reshape([t, c / s_dim, s_dim]); // (T, K, S)

        // ---- decay / gate -------------------------------------------------
        let dec1 = self.time_decay_w1.forward(xw).tanh(); // (T, k)
        let dec2 = self.time_decay_w2.forward(dec1); // (T, C)
        let dec2_4 = dec2.reshape([t, h, s_dim, 1]); // (T, H, S, 1)
        let decay = self
            .time_decay
            .value
            .clone()
            .reshape([1, h, s_dim, 1]); // (1, H, S, 1)
        let w_lora = decay + dec2_4; // (T, H, S, 1)

        let g = activation::sigmoid(self.gate.forward(xg)); // (T, C)

        // ---- repeat K / V to H heads, log-space decay ---------------------
        let num_kv = k.dims()[1];
        let num_kv_reps = h / num_kv;
        let k = if num_kv_reps > 1 {
            repeat_axis1(k, num_kv_reps)
        } else {
            k
        };
        let v = if num_kv_reps > 1 {
            repeat_axis1(v, num_kv_reps)
        } else {
            v
        };

        let log_w = w_lora.exp().neg().clamp_min(-5.0); // (T, H, S, 1), = clip(-exp(w_lora))
        let log_w_3 = log_w.clone().squeeze_dim::<3>(3); // (T, H, S)
        // k = k * (1 - exp(log_w[..., 0]))
        let k = k * log_w_3.exp().neg().add_scalar(1.0);

        // ---- state bookkeeping -------------------------------------------
        let mut state = state.slice_assign(
            [0..1, 0..c],
            x.clone().slice([length - 1..length, 0..c]),
        ); // state[0] = x[length-1]
        let s_init = state
            .clone()
            .slice([1..1 + s_dim, 0..c])
            .reshape([h, s_dim, s_dim]);

        let (state_new, out) =
            rwkv6_inner_loop(r, k, v, log_w, s_init, length, new_starts);

        state = state.slice_assign(
            [1..1 + s_dim, 0..c],
            state_new.reshape([s_dim, c]),
        ); // state[1:]

        let x = out.reshape([t, h * s_dim]) * g; // (T, C)
        let out = self.o_proj.forward(x);
        (out, state)
    }
}

// ---------------------------------------------------------------------------
// RWKV tokenizer (pure algorithm, no tensors)
// ---------------------------------------------------------------------------

/// The RWKV tokenizer pure algorithm, ported from `tokenizer.py`.
///
/// Construction precomputes a `table[s0][s1]` of candidate multi-byte tokens
/// (longest first), a `good[s0]` bitmap of valid second bytes and a `wlen[s0]`
/// maximum token length per first byte. `encode_bytes` greedily matches the
/// longest token starting at each position; `decode_bytes` concatenates the
/// token byte-strings.
pub struct RWKVTokenizer {
    /// `idx -> bytes`, token id equals the vector index.
    idx2token: Vec<Vec<u8>>,
    /// `bytes -> idx`.
    token2idx: HashMap<Vec<u8>, u32>,
    /// `table[s0][s1]` = candidate multi-byte tokens, longest first.
    table: [Vec<Vec<Vec<u8>>>; 256],
    /// `good[s0][s1]` = whether a multi-byte token starts with byte pair `s0 s1`.
    good: [[bool; 256]; 256],
    /// `wlen[s0]` = max token length starting with byte `s0`.
    wlen: [usize; 256],
}

fn empty_table() -> [Vec<Vec<Vec<u8>>>; 256] {
    std::array::from_fn(|_| vec![Vec::new(); 256])
}

impl RWKVTokenizer {
    /// Build a tokenizer from explicit token-id / bytes pairs. The candidate
    /// `table` rows are ordered longest-first so the greedy encoder picks the
    /// longest match, mirroring the reverse-iteration in `__init__`.
    pub fn from_pairs(tokens: &[(u32, Vec<u8>)]) -> Result<Self, String> {
        // Order candidates longest-first for the `table` rows.
        let mut ordered: Vec<(u32, Vec<u8>)> = tokens.to_vec();
        ordered.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        let max_id = tokens.iter().map(|(id, _)| *id).max().unwrap_or(0) as usize;
        let mut idx2token = vec![Vec::new(); max_id + 1];
        for (id, bytes) in tokens.iter() {
            idx2token[*id as usize] = bytes.clone();
        }

        let mut token2idx = HashMap::new();
        for (id, bytes) in tokens.iter() {
            token2idx.insert(bytes.clone(), *id);
        }

        let mut table = empty_table();
        let mut good = [[false; 256]; 256];
        let mut wlen = [0usize; 256];

        for (_, bytes) in ordered.iter() {
            if bytes.len() >= 2 {
                let s0 = bytes[0] as usize;
                let s1 = bytes[1] as usize;
                table[s0][s1].push(bytes.clone());
                good[s0][s1] = true;
                wlen[s0] = wlen[s0].max(bytes.len());
            }
        }

        Ok(Self {
            idx2token,
            token2idx,
            table,
            good,
            wlen,
        })
    }

    /// Build a tokenizer from vocab-file lines in the `idx <space> <len>
    /// <space> <hex-bytes>` format. Each line is split on whitespace into
    /// `[idx, len, hex]`; `hex` is the hex encoding of the token bytes. This
    /// mirrors the file-line construction (a hex payload instead of a Python
    /// literal) so a small synthetic vocab can drive the tables.
    pub fn from_vocab(lines: &[&str]) -> Result<Self, String> {
        let mut pairs = Vec::new();
        for (lineno, line) in lines.iter().enumerate() {
            let mut it = line.split_whitespace();
            let idx: u32 = it
                .next()
                .ok_or_else(|| format!("line {lineno}: missing idx"))?
                .parse()
                .map_err(|e| format!("line {lineno}: bad idx: {e}"))?;
            let len: usize = it
                .next()
                .ok_or_else(|| format!("line {lineno}: missing len"))?
                .parse()
                .map_err(|e| format!("line {lineno}: bad len: {e}"))?;
            let hex = it
                .next()
                .ok_or_else(|| format!("line {lineno}: missing hex"))?;
            let bytes = hex_to_bytes(hex).ok_or_else(|| format!("line {lineno}: bad hex"))?;
            if bytes.len() != len {
                return Err(format!("line {lineno}: len {} != bytes {}", len, bytes.len()));
            }
            pairs.push((idx, bytes));
        }
        Self::from_pairs(&pairs)
    }

    /// Greedy longest-match encoding of raw bytes, mirroring `encodeBytes`.
    pub fn encode_bytes(&self, src: &[u8]) -> Vec<u32> {
        let src_len = src.len();
        let mut tokens = Vec::new();
        let mut i = 0usize;
        while i < src_len {
            let mut matched: &[u8] = &src[i..i + 1];
            if i < src_len - 1 {
                let s0 = src[i] as usize;
                let s1 = src[i + 1] as usize;
                if self.good[s0][s1] {
                    let end = (i + self.wlen[s0]).min(src_len);
                    let sss = &src[i..end];
                    // First token in `table[s0][s1]` that `sss` starts with is
                    // the longest match (rows are longest-first).
                    if let Some(tok) = self.table[s0][s1]
                        .iter()
                        .find(|t| sss.starts_with(t.as_slice()))
                    {
                        matched = tok.as_slice();
                    }
                }
            }
            if let Some(&tok) = self.token2idx.get(matched) {
                tokens.push(tok);
            } else {
                // The single-byte fallback must always be present for valid
                // inputs; if it is not, push nothing (keeps i/length aligned).
            }
            i += matched.len();
        }
        tokens
    }

    /// Concatenate the stored bytes for each token id, mirroring `decodeBytes`.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &t in tokens {
            if let Some(bytes) = self.idx2token.get(t as usize) {
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Encode a string via its UTF-8 bytes.
    pub fn encode(&self, src: &str) -> Vec<u32> {
        self.encode_bytes(src.as_bytes())
    }

    /// Decode tokens back to a string (UTF-8).
    pub fn decode(&self, tokens: &[u32]) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(self.decode_bytes(tokens))
    }
}

fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for chunk in bytes.chunks(2) {
        let hi = (chunk[0] as char).to_digit(16)?;
        let lo = (chunk[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use crate::common::Mm;

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    }

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    const TOL: f32 = 1e-4;

    // -- layer_norm ---------------------------------------------------------

    #[test]
    fn layer_norm_matches_hand_computed() {
        // x (2,4): rows with differing mean/std.
        let x = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            &device(),
        );
        let w = Tensor::<B, 1>::from_data([2.0_f32, 1.0, 0.5, 1.0], &device());
        let b = Tensor::<B, 1>::from_data([1.0_f32, 0.0, 0.5, -1.0], &device());

        // Row 0: mean 2.5, var 1.25, std = sqrt(1.25+eps) ~ 1.1180.
        //   normalized = [-1.3416, -0.4472,  0.4472,  1.3416]
        //   * w + b    = [ -1.6832, -0.4472,  0.7236,  0.3416]
        // Row 1: mean 25, var 125, std ~ 11.1803.
        //   normalized = [-1.3416, -0.4472,  0.4472,  1.3416]
        //   * w + b    = [ -1.6832, -0.4472,  0.7236,  0.3416]
        let out = layer_norm(x, &w, &b, 1e-5);
        let data = to_vec(out);
        let expected = [-1.6832, -0.4472, 0.7236, 0.3416, -1.6832, -0.4472, 0.7236, 0.3416];
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(near(*a, *b, TOL), "expected {b}, got {a}");
        }
    }

    #[test]
    fn layer_norm_module_matches_free_fn() {
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device());
        let weight = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 1.0], &device());
        let bias = Tensor::<B, 1>::zeros([3], &device());
        let module = LayerNorm {
            weight: Parameter::new(weight.clone()),
            bias: Parameter::new(bias.clone()),
        };
        let a = to_vec(module.forward(x.clone(), 1e-5));
        let b = to_vec(layer_norm(x, &weight, &bias, 1e-5));
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(near(*x, *y, TOL));
        }
    }

    // -- group_norm ---------------------------------------------------------

    #[test]
    fn group_norm_matches_hand_computed() {
        // x (2,4,2), groups=2 -> reshape (2,2,2,2), normalize over the last 2.
        let x = Tensor::<B, 3>::from_data(
            [
                [[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[9.0_f32, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            ],
            &device(),
        );
        let weight = Parameter::new(Tensor::<B, 1>::from_data([1.0_f32, 1.0, 1.0, 1.0], &device()));
        let bias = Parameter::new(Tensor::<B, 1>::zeros([4], &device()));
        let gn = GroupNorm { weight, bias };

        let out = gn.forward(x, 2, 1e-5);
        let data = to_vec(out);

        // Each group of 4 has mean {2.5,6.5,10.5,14.5} and var 1.25.
        // normalized per group: [-1.3416,-0.4472,0.4472,1.3416].
        let expected = [
            -1.34164, -0.44721, 0.44721, 1.34164, // n=0 group0 (c0,c1)
            -1.34164, -0.44721, 0.44721, 1.34164, // n=0 group1 (c2,c3)
            -1.34164, -0.44721, 0.44721, 1.34164, // n=1 group0
            -1.34164, -0.44721, 0.44721, 1.34164, // n=1 group1
        ];
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(near(*a, *b, TOL), "expected {b}, got {a}");
        }
    }

    // -- ChannelMixing ------------------------------------------------------

    #[test]
    fn channel_mixing_single_step_hand_computed() {
        // T=2, C=1 for trivially hand-computable math.
        // x = [[3.0], [5.0]]; state (1,1) = [[1.0]]; x_k = [[2.0]]
        // length = 2 (state_out = x[1]).
        let x = Tensor::<B, 2>::from_data([[3.0_f32], [5.0]], &device());
        let state = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let x_k = Parameter::new(Tensor::<B, 1>::from_data([2.0_f32], &device()));
        // Identity key/value weights (1x1).
        let key = Linear {
            weight: Mm { weight: Tensor::<B, 2>::from_data([[1.0_f32]], &device()) },
            bias: None,
        };
        let value = Linear {
            weight: Mm { weight: Tensor::<B, 2>::from_data([[1.0_f32]], &device()) },
            bias: None,
        };
        let cm = ChannelMixing { x_k, key, value };
        let new_starts = [false, false];

        let (out, new_state) = cm.forward(x.clone(), state, 2, &new_starts);

        // sx[0] = state(1) - x(3) = -2; sx[1] = x[0](3) - x[1](5) = -2.
        // xk = x + sx*x_k = [3 + (-4), 5 + (-4)] = [-1, 1].
        // k = relu(xk)^2 = relu([-1,1])^2 = [0, 1].
        // out = k (identity value) = [0, 1].
        let ov = to_vec(out);
        assert!(near(ov[0], 0.0, TOL) && near(ov[1], 1.0, TOL));
        // state = x[length-1] = x[1] = 5.
        let st = to_vec(new_state);
        assert_eq!(st.len(), 1);
        assert!(near(st[0], 5.0, TOL));
    }

    // -- rwkv7_inner_loop ---------------------------------------------------

    #[test]
    fn rwkv7_inner_loop_matches_hand_computed() {
        let device = &device();
        // T=2, H=2, S=2.
        // s0 (2,2,2) heads.
        let s = Tensor::<B, 3>::from_data(
            [[[1.0_f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            device,
        );
        // r: t0 h0=[1,0], h1=[0,1]; t1 both [1,1].
        let r = Tensor::<B, 3>::from_data(
            [[[1.0_f32, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            device,
        );
        // k=v=a=b = 0 so the recurrence is pure decay: s = s*exp(-exp(w)).
        let z = Tensor::<B, 3>::zeros([2, 2, 2], device);
        // decay 0.5 <-> w input = ln(ln(2)) ~ -0.36651290...
        let wdec = 2.0_f32.ln().ln();
        let w = Tensor::<B, 3>::from_data(
            [[[wdec; 2]; 2]; 2],
            device,
        );
        let (out_s, out) = super::rwkv7_inner_loop(r, w, z.clone(), z.clone(), z.clone(), z, s, 2, &[false, false]);

        // out[t=0] = (s0*0.5 @ r[0]); out[t=1] = (s0*0.25 @ r[1]).
        let out_expected = [
            [[0.5_f32, 1.5], [3.0, 4.0]],     // t=0
            [[0.75_f32, 1.75], [2.75, 3.75]], // t=1
        ];
        let ov = to_vec(out);
        let mut idx = 0;
        for t in 0..2 {
            for h in 0..2 {
                for s in 0..2 {
                    assert!(near(ov[idx], out_expected[t][h][s], TOL), "out[{t}][{h}][{s}] got {}", ov[idx]);
                    idx += 1;
                }
            }
        }

        // out_s = final state (t=1) = s0 * 0.25.
        let s_expected = [
            [[0.25_f32, 0.5], [0.75, 1.0]],
            [[1.25_f32, 1.5], [1.75, 2.0]],
        ];
        let sv = to_vec(out_s);
        let mut idx = 0;
        for h in 0..2 {
            for a in 0..2 {
                for b in 0..2 {
                    assert!(near(sv[idx], s_expected[h][a][b], TOL), "state[{h}][{a}][{b}] got {}", sv[idx]);
                    idx += 1;
                }
            }
        }
    }

    // -- TimeMixing structural (shapes / runs) ------------------------------

    struct TimeMixingBuilder {
        c: usize,
        k: usize,
    }

    impl TimeMixingBuilder {
        fn param(&self) -> Parameter {
            Parameter::new(Tensor::<B, 1>::zeros([self.c], &device()))
        }
        fn lin(&self) -> Linear {
            Linear {
                weight: Mm {
                    weight: Tensor::<B, 2>::zeros([self.c, self.c], &device()),
                },
                bias: None,
            }
        }
        fn tmm(&self) -> Tmm {
            Tmm {
                weight: Tensor::<B, 2>::zeros([self.c, self.k], &device()),
            }
        }
        fn tmm_out(&self) -> Tmm {
            Tmm {
                weight: Tensor::<B, 2>::zeros([self.k, self.c], &device()),
            }
        }
        fn group_norm(&self) -> GroupNorm {
            GroupNorm {
                weight: Parameter::new(Tensor::<B, 1>::ones([self.c], &device())),
                bias: Parameter::new(Tensor::<B, 1>::zeros([self.c], &device())),
            }
        }
        fn build(&self) -> TimeMixing {
            TimeMixing {
                x_r: self.param(),
                x_w: self.param(),
                x_k: self.param(),
                x_v: self.param(),
                x_a: self.param(),
                x_g: self.param(),
                receptance: self.lin(),
                w0: self.param(),
                w1: self.tmm(),
                w2: self.tmm_out(),
                key: self.lin(),
                value: self.lin(),
                v0: self.param(),
                v1: self.tmm(),
                v2: self.tmm_out(),
                a0: self.param(),
                a1: self.tmm(),
                a2: self.tmm_out(),
                g1: self.tmm(),
                g2: self.tmm_out(),
                k_k: self.param(),
                k_a: self.param(),
                ln_x: self.group_norm(),
                r_k: self.param(),
                output: self.lin(),
            }
        }
    }

    #[test]
    fn time_mixing_runs_and_shapes() {
        // T=2, H=2, S=2, C=4. Intermediate k=4.
        let b = TimeMixingBuilder { c: 4, k: 4 };
        let tm = b.build();

        let x = Tensor::<B, 2>::from_data([[1.0_f32, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], &device());
        // state (1 + S, C) = (3, 4).
        let state = Tensor::<B, 2>::zeros([3, 4], &device());
        let v_first = Tensor::<B, 2>::zeros([2, 4], &device());
        let new_starts = [false, false];

        let (out, state_out, v_first_out) = tm.forward(x.clone(), state, v_first.clone(), 2, &new_starts, 2, 2, 0, super::rwkv7_inner_loop);
        assert_eq!(out.dims(), [2, 4]);
        assert_eq!(state_out.dims(), [3, 4]);
        assert_eq!(v_first_out.dims(), [2, 4]);
        assert!(to_vec(out).iter().all(|v| v.is_finite()));

        // Layer id > 0 keeps v_first unchanged (stays the input zeros).
        let (_, _, vf2) = tm.forward(
            x,
            Tensor::<B, 2>::zeros([3, 4], &device()),
            v_first.clone(),
            2,
            &new_starts,
            2,
            2,
            1,
            super::rwkv7_inner_loop,
        );
        assert_eq!(to_vec(vf2), to_vec(v_first));
    }

    // -- Qwen2RMSNorm / Qwen2MLP ---------------------------------------------

    #[test]
    fn qwen2_rmsnorm_matches_hand_computed() {
        // x = [[1, 2], [3, 4]]; eps=1e-6.
        // mean_sq row0 = (1+4)/2 = 2.5; row1 = (9+16)/2 = 12.5.
        // rsqrt(2.5+1e-6) ~ 0.63246; x*rsqrt = [0.63246, 1.26491].
        // rsqrt(12.5) ~ 0.28284; x*rsqrt = [0.84853, 1.13137].
        // weight = [1, 2] -> multiply -> row0 = [0.63246, 2.52982].
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let rm = Qwen2RMSNorm {
            weight: Parameter::new(Tensor::<B, 1>::from_data([1.0_f32, 2.0], &device())),
        };
        let out = to_vec(rm.forward(x, 1e-6));
        assert!(near(out[0], 0.63246, TOL));
        assert!(near(out[1], 2.52982, TOL));
        assert!(near(out[2], 0.84853, TOL));
        assert!(near(out[3], 2.26274, TOL));
    }

    #[test]
    fn qwen2_mlp_forward_matches_hand_computed() {
        // gate/up projected to a single unit with unit weights (C=1, in=1).
        let one = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let zero_bias = None;
        let gate_proj = Linear { weight: Mm { weight: one.clone() }, bias: zero_bias.clone() };
        let up_proj = Linear { weight: Mm { weight: one.clone() }, bias: zero_bias.clone() };
        let down_proj = Linear { weight: Mm { weight: one }, bias: None };
        let mlp = Qwen2MLP { gate_proj, up_proj, down_proj };

        // x = [[3.0]]: gate = silu(3)*up(3) = (3*sigmoid(3))*3; down identity.
        let x = Tensor::<B, 2>::from_data([[3.0_f32]], &device());
        let sig3 = 1.0 / (1.0 + (-3.0_f32).exp());
        let expected = 3.0 * sig3 * 3.0;
        let out = to_vec(mlp.forward(x));
        assert!(near(out[0], expected, TOL), "expected {expected}, got {}", out[0]);
    }

    // -- rwkv6_inner_loop ---------------------------------------------------

    #[test]
    fn rwkv6_inner_loop_matches_hand_computed() {
        let device = &device();
        // T=2, H=2, S=2. k = v = 0 -> at = 0, pure decay s = w[t]*s.
        // decay 0.5 <-> w input ln(0.5) = -0.6931472.
        let s = Tensor::<B, 3>::from_data(
            [[[1.0_f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            device,
        );
        let r = Tensor::<B, 3>::from_data(
            [[[1.0_f32, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            device,
        );
        let z = Tensor::<B, 3>::zeros([2, 2, 2], device);
        let wln = 0.5_f32.ln();
        let w = Tensor::<B, 4>::from_data([[[[wln; 1]; 2]; 2]; 2], device);

        let scale = (2.0_f32).powf(-0.5);
        let (out_s, out) = super::rwkv6_inner_loop(r, z.clone(), z, w, s, 2, &[false, false]);

        // t=0: state = s0*0.5; out = scale * (rt-weighted read of s*0.5)
        // t=1: state = s0*0.25; out = scale*(sum over rows of s0*0.25) (rt=[1,1])
        let out_expected = [
            [
                [scale * 0.5, scale * 1.0],   // h0, rt=[1,0] -> row0
                [scale * 3.5, scale * 4.0],   // h1, rt=[0,1] -> row1
            ],
            [
                [scale * 1.0, scale * 1.5],   // h0 sum rows of [0.25,0.5],[0.75,1.0]
                [scale * 3.0, scale * 3.5],   // h1 sum rows of [1.25,1.5],[1.75,2.0]
            ],
        ];
        let ov = to_vec(out);
        let mut idx = 0;
        for t in 0..2 {
            for h in 0..2 {
                for s in 0..2 {
                    assert!(near(ov[idx], out_expected[t][h][s], TOL), "t{t} h{h} s{s} got {}", ov[idx]);
                    idx += 1;
                }
            }
        }

        let s_expected = [
            [[0.25_f32, 0.5], [0.75, 1.0]],
            [[1.25_f32, 1.5], [1.75, 2.0]],
        ];
        let sv = to_vec(out_s);
        let mut idx = 0;
        for h in 0..2 {
            for a in 0..2 {
                for b in 0..2 {
                    assert!(near(sv[idx], s_expected[h][a][b], TOL), "state h{h} got {}", sv[idx]);
                    idx += 1;
                }
            }
        }
    }

    // -- RWKV6Attention structural ------------------------------------------

    #[test]
    fn rwkv6_attention_runs_and_shapes() {
        let c = 4; // H=2, S=2
        let h = 2;
        let s_dim = 2;
        // time_maa_w1: (C, 5*C) = (4, 20); time_maa_w2: (C, C) = (4, 4).
        let lin = || Linear {
            weight: Mm { weight: Tensor::<B, 2>::zeros([c, c], &device()) },
            bias: None,
        };
        let attention = RWKV6Attention {
            time_maa_x: Parameter::new(Tensor::<B, 1>::zeros([c], &device())),
            time_maa_r: Parameter::new(Tensor::<B, 1>::zeros([c], &device())),
            time_maa_k: Parameter::new(Tensor::<B, 1>::zeros([c], &device())),
            time_maa_v: Parameter::new(Tensor::<B, 1>::zeros([c], &device())),
            time_maa_w: Parameter::new(Tensor::<B, 1>::zeros([c], &device())),
            time_maa_g: Parameter::new(Tensor::<B, 1>::zeros([c], &device())),
            time_maa_w1: Tmm { weight: Tensor::<B, 2>::zeros([c, 5 * c], &device()) },
            time_maa_w2: Tmm { weight: Tensor::<B, 2>::zeros([c, c], &device()) },
            time_decay: Parameter::new(Tensor::<B, 1>::zeros([c], &device())),
            time_decay_w1: Tmm { weight: Tensor::<B, 2>::zeros([c, c], &device()) },
            time_decay_w2: Tmm { weight: Tensor::<B, 2>::zeros([c, c], &device()) },
            q_proj: lin(),
            k_proj: lin(),
            v_proj: lin(),
            gate: lin(),
            o_proj: lin(),
        };

        let x = Tensor::<B, 2>::from_data([[1.0_f32, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], &device());
        let state = Tensor::<B, 2>::zeros([1 + s_dim, c], &device());
        let (out, state_out) = attention.forward(x, state, 2, &[false, false], h, s_dim);
        assert_eq!(out.dims(), [2, c]);
        assert_eq!(state_out.dims(), [1 + s_dim, c]);
        assert!(to_vec(out).iter().all(|v| v.is_finite()));
    }

    // -- RWKV tokenizer ------------------------------------------------------

    fn tiny_vocab() -> RWKVTokenizer {
        RWKVTokenizer::from_pairs(&[
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"\n".to_vec()),
        ])
        .unwrap()
    }

    #[test]
    fn tokenizer_round_trip() {
        let tok = tiny_vocab();
        for s in ["ab", "ba", "ab\n", "bab", "a\nb\n", ""] {
            let enc = tok.encode(s);
            let dec = tok.decode(&enc).unwrap();
            assert_eq!(dec, s, "round-trip failed for {s:?}");
        }
    }

    #[test]
    fn tokenizer_greedy_longest_match() {
        let tok = tiny_vocab();
        // "ab" must encode to the single "ab" token (id 2), not ["a","b"].
        assert_eq!(tok.encode("ab"), vec![2]);
        // "aba" -> "ab"(2) + "a"(0).
        assert_eq!(tok.encode("aba"), vec![2, 0]);
        // "bab" -> "b"(1) + "ab"(2).
        assert_eq!(tok.encode("bab"), vec![1, 2]);
    }

    #[test]
    fn tokenizer_decode_reverses_encode() {
        let tok = tiny_vocab();
        assert_eq!(tok.decode_bytes(&[2, 3]), b"ab\n");
        assert_eq!(tok.decode(&[2]), Ok(String::from("ab")));
    }

    #[test]
    fn tokenizer_from_vocab_lines() {
        // idx <len> <hex>.
        let lines = ["0 1 61", "1 1 62", "2 2 6162", "3 1 0a"];
        let tok = RWKVTokenizer::from_vocab(&lines).unwrap();
        assert_eq!(tok.encode("ab"), vec![2]);
        assert_eq!(tok.encode("ba"), vec![1, 0]);
        assert_eq!(tok.decode(&[2, 3]).unwrap(), "ab\n");
    }

    #[test]
    fn tokenizer_hex_helper() {
        assert_eq!(hex_to_bytes("6162").unwrap(), b"ab");
        assert_eq!(hex_to_bytes("0a").unwrap(), b"\n");
        assert!(hex_to_bytes("abc").is_none());
    }
}
