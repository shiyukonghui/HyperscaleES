"""真正的逐 token 脉冲自注意力（Transformer 式）—— 序列到序列的注意力核心。

与 ``snn_attention.py`` / ``snn_attention_lif.py`` 的本质区别：

  snn_attention(.lif).        本文件（真正的 self-attention）
  -----------------------     -------------------------------------------------
  q_center = mean(Q)          Q 未平均：每个 token 用自己的 query
  H = q_center @ K.T (N,)     相似度矩阵 H = Q@K.T ∈ (N,N)（逐 token 成对相似度）
  p = softmax(H) (N,)         逐行脉冲 softmax：A ∈ (N,N) 注意力矩阵
  o = p[:,None]*v; mean(o)    序列输出 O = A@V ∈ (N,d)，保留 token 结构

因此本文件实现的是**真正的自注意力**（每个 token 独立 query，输出仍是 token 序列），
而旧版只是"全局 query + 标量权重"的 **Attention Pooling**（模拟 Softmax 池化，
不等价于 Transformer 注意力）。

脉冲化两层：
  1. Q/K/V 编码：LIF 时间积分 -> spike-count rate（∈ [0,1]），同 snn_attention_lif；
  2. 注意力矩阵：对每行 token i，其 query 对全部 key 的相似度电流 h[i,:] 注入
     一组 LIF "注意力神经元"（每 key 一个），全局抑制神经元提供 divisive
     归一化，突触迹低通后按行 softmax，得到 A ∈ (N,N)。

输出保留 token 维：O = A@V ∈ (N,d)；下游按需用 CLS token / 均值 / 或再接层。
训练仍走 HyperscaleES Noiser（演化策略），硬阈值不可微不影响更新。
"""

import jax
import jax.numpy as jnp

from .base_model import Model, CommonInit
from .common import merge_inits, call_submodule, MM, Parameter
from .snn_attention import lif_step, run_lif

DEFAULT_CORE_ARGS = {
    "g_inh": 0.5,      # 注意力神经元全局抑制强度
    "tau_syn": 5.0,    # 突触迹低通时间常数（注意：用 tau_syn 而非 tau_a，因无 tau_a 概念）
    "n_iter": 8,       # 注意力竞争迭代步数
}

DEFAULT_POOL = "mean"  # 序列输出的池化方式


# ----------------------------------------------------------------------------
# 逐行脉冲 softmax：对单个 token 的 query，其 N 个 key 的注意力权重
# ----------------------------------------------------------------------------
def row_softmax_lif(h_i, vth, g_inh, tau_m, tau_syn, n_iter):
    """单个 token query 的脉冲注意力权重（对全部 key 的 softmax 近似）。

    ``h_i``: (N,) 该 token 的 query 对所有 key 的相似度电流 beta*q_i^T k_j。
    用 N 个 LIF 注意力神经元（每 key 一个）竞争：全局抑制强度 G=sum(计数)
    提供 divisive normalization；统计 n_iter 步内各 key 神经元的**发放计数**，
    归一化后 p_j = count_j / sum(count) 即该行 softmax 权重（发放即参与，
    行和恒≈1 只要该行有任一 key 在竞争期发放）。

    Returns p_i: (N,) 归一化权重。
    """
    vth = jnp.asarray(vth, dtype=h_i.dtype)
    g_inh = jnp.asarray(g_inh, dtype=h_i.dtype)
    n = h_i.shape[0]
    # 行内归一化：把该 token 的相似度电流缩放到 ~[-1,1]，
    # 使 LIF 注意力神经元能进入发放区间（原始 beta*Q@K.T 数量级太小会全不发放）。
    h_i = h_i / (jnp.max(jnp.abs(h_i)) + 1e-6)

    def step(state, _):
        u, count = state                              # 膜电位、累计发放计数
        G = jnp.sum(count)                            # 全局抑制强度（用计数）
        u_new, s = lif_step({"tau_m": tau_m, "v_th": vth},
                            u, h_i - g_inh * G)       # 真脉冲 + hard reset
        return (u_new, count + s.astype(count.dtype)), count

    u0 = jnp.zeros((n,), dtype=h_i.dtype)
    count0 = jnp.zeros((n,), dtype=h_i.dtype)
    (_, count_final), _ = jax.lax.scan(step, (u0, count0), jnp.arange(n_iter))
    # 发放计数归一化 => 行 softmax（只要该行有任一 key 发放，行和即≈1）
    p = count_final / (jnp.sum(count_final) + 1e-6)
    return p


# ----------------------------------------------------------------------------
# 真正逐 token 的自注意力核心：N 个 token，N×N 注意力矩阵，序列输出
# ----------------------------------------------------------------------------
def self_attention_lif(Q, K, V, vth, g_inh, tau_m, tau_syn, beta, n_iter):
    """真正的 Transformer 式脉冲自注意力。

    Args:
        Q, K, V: (N, d) 各 token 的 query/key/value（未平均，逐 token 独立）。
    Returns:
        (A, O)：A ∈ (N,N) 注意力矩阵（A[i,j]=token i 的 query 关注 key j 的权重），
                O ∈ (N,d) 序列输出（O = A@V，保留 token 结构）。
    """
    beta = jnp.asarray(beta, dtype=Q.dtype)
    # 相似度矩阵：H[i,j] = beta * Q[i]^T K[j]，每行是该 token 的 query 对全部 key
    H = beta * (Q @ K.T)                                # (N, N)
    # 逐行脉冲 softmax -> N×N 注意力矩阵
    A = jax.vmap(lambda h_i: row_softmax_lif(
        h_i, vth=vth, g_inh=g_inh, tau_m=tau_m,
        tau_syn=tau_syn, n_iter=n_iter))(H)             # (N, N)
    O = A @ V                                           # (N, d) 序列输出
    return A, O


def softmax_self_attention(Q, K, V, beta=1.0):
    """参考：标准 Transformer 自注意力（连续 softmax），用于等价性对比。"""
    beta = jnp.asarray(beta, dtype=Q.dtype)
    H = beta * (Q @ K.T)                                # (N, N)
    e = jnp.exp(H - jnp.max(H, axis=-1, keepdims=True))
    A = e / jnp.sum(e, axis=-1, keepdims=True)          # (N, N)
    O = A @ V                                           # (N, d)
    return A, O


# ----------------------------------------------------------------------------
# Q/K/V 编码：真正的 LIF 时间积分（与 snn_attention_lif._qkv_lif 相同）
# ----------------------------------------------------------------------------
def _qkv_lif(common_params, x, vths):
    """把 token 的泊松输入脉冲编码成 Q/K/V 的 spike-count rate 向量。

    x: (T, N, token_in_dim) 泊松脉冲；返回逐 token (N, d) 的 Q/K/V 发放率。
    注意返回的 Q/K/V 是**逐 token 独立**的 (N, d)，未做任何 token 间平均。
    """
    tau_m = common_params.frozen_params["tau_m"]

    def encode(proj, vth):
        lif_p = {"tau_m": tau_m, "v_th": vth}
        v0 = jnp.zeros(proj.shape[1:], dtype=proj.dtype)   # (N, d)
        spikes = run_lif(lif_p, proj, v0)                  # (T, N, d) 0/1 脉冲
        return jnp.mean(spikes, axis=0)                    # (N, d) spike-count rate

    q_proj = jax.vmap(lambda xt: call_submodule(MM, "q", common_params, xt))(x)
    k_proj = jax.vmap(lambda xt: call_submodule(MM, "k", common_params, xt))(x)
    v_proj = jax.vmap(lambda xt: call_submodule(MM, "v", common_params, xt))(x)
    return encode(q_proj, vths[0]), encode(k_proj, vths[1]), encode(v_proj, vths[2])


# ----------------------------------------------------------------------------
# 模型：逐 token 脉冲自注意力分类器（patched-MNIST）
# ----------------------------------------------------------------------------
class SNNSelfAttentionModel(Model):
    """基于真正逐 token 脉冲自注意力的分类模型。

    输入 x: (T, N, token_in_dim) 泊松脉冲（N 个 token）；
    前向：LIF QKV 编码 -> 逐 token 自注意力（N×N 矩阵）-> 序列输出 O ∈ (N,d)
         -> 池化（CLS 或均值）-> readout logits。
    """

    @staticmethod
    def _attention(Q, K, V, beta, frozen, vth_attn):
        return self_attention_lif(
            Q, K, V, vth=vth_attn, g_inh=frozen["g_inh"],
            tau_syn=frozen["tau_syn"], tau_m=frozen["tau_m"],
            beta=beta, n_iter=frozen["n_iter"])

    @classmethod
    def rand_init(cls, key, token_in_dim, num_tokens, num_classes, d_head,
                  tau_m=20.0, proj_gain=2.0, trainable_beta=True,
                  v_th=0.05, pool=DEFAULT_POOL, dtype=jnp.float32, **core_args):
        core = {k: core_args.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
        keys = jax.random.split(key, 6)
        layers = dict(
            q=MM.rand_init(keys[0], token_in_dim, d_head, dtype),
            k=MM.rand_init(keys[1], token_in_dim, d_head, dtype),
            v=MM.rand_init(keys[2], token_in_dim, d_head, dtype),
            out=MM.rand_init(keys[3], d_head, num_classes, dtype),
            out_gain=Parameter.rand_init(keys[4], None, None, jnp.ones((1,)), dtype),
        )
        # 可训练阈值：softplus 恒正（Q/K/V/注意力各一）
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        for name in ("q_th", "k_th", "v_th", "attn_th"):
            layers[name] = Parameter.rand_init(None, None, None, raw_vth0, dtype)
        raw_beta = jnp.log(jnp.exp(jnp.asarray(1.0 / jnp.sqrt(d_head), dtype)) - 1.0)
        layers["beta"] = Parameter.rand_init(None, None, None, raw_beta, dtype)

        frozen_params = {
            "tau_m": jnp.asarray(tau_m, dtype=dtype),
            "proj_gain": jnp.asarray(proj_gain, dtype=dtype),
            "trainable_beta": bool(trainable_beta),
            "pool": pool,
            **core,
        }
        if core_args:
            raise TypeError(f"unexpected core args: {sorted(core_args)}")
        merged = merge_inits(**layers)
        return CommonInit(frozen_params, merged.params, merged.scan_map, merged.es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        x = x.astype(common_params.params["q"].dtype)
        vths = [jax.nn.softplus(call_submodule(Parameter, name, common_params))
                for name in ("q_th", "k_th", "v_th", "attn_th")]
        Q, K, V = _qkv_lif(common_params, x, vths[:3])

        if common_params.frozen_params["trainable_beta"]:
            beta = jax.nn.softplus(call_submodule(Parameter, "beta", common_params))
        else:
            beta = 1.0 / jnp.sqrt(Q.shape[-1])

        _, O = cls._attention(Q, K, V, beta, common_params.frozen_params, vths[3])
        # O ∈ (N, d)，池化到单向量做分类
        pool = common_params.frozen_params["pool"]
        if pool == "mean":
            pooled = jnp.mean(O, axis=0)
        elif pool == "last":
            pooled = O[-1]
        elif pool == "max":
            pooled = jnp.max(O, axis=0)
        else:
            raise ValueError(f"unknown pool: {pool}")

        logits = call_submodule(MM, "out", common_params, pooled)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


def model_rand_init(key, token_in_dim, num_tokens, num_classes, d_head, **kwargs):
    """构建 SNNSelfAttentionModel 的 CommonInit（逐 token 脉冲自注意力）。

    额外 kwargs（DEFAULT_CORE_ARGS）覆盖路由超参；``v_th``/``pool`` 传入模型。
    """
    base_kwargs = {k: kwargs.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
    return SNNSelfAttentionModel.rand_init(
        key, token_in_dim, num_tokens, num_classes, d_head,
        **base_kwargs, **kwargs)
