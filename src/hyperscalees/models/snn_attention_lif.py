"""真正的脉冲神经元版 SNN 注意力模型 —— LIF Q/K/V 编码 + 脉冲注意力核心。

与 ``snn_attention.py``（连续速率近似）的对照：

  snn_attention.py          snn_attention_lif.py（本文件）
  ------------------------- ------------------------------------------------
  _rate_encode(静态 sigmoid) _qkv_lif(真 LIF 时间积分 + spike-count rate)
  hopfield_attention(连续u)  hopfield_attention_lif(LIF 注意力神经元 + 突触迹)
  meanfield_attention(连续r) meanfield_attention_lif(LIF 发放率近似群体率)

改动要点（对应 docs/es_attention_accumulation_equivalence.md §5.3 根因的修复）：
  1. Q/K/V 编码：泊松输入脉冲 x: (T, N, D) 经 MM 投影后注入 LIF 膜电位
     （run_lif：泄漏 -> 积分 -> 阈值发放 -> hard reset），时间平均发放率
     （spike-count rate）作为 token 的 rate 向量；
  2. Hopfield 注意力：每个 key token 一个 LIF "注意力神经元"，相似度电流
     I_j = beta*q̄^T k_j 注入，全局抑制神经元 G=sum(z) 提供 divisive
     normalization，突触迹 z_j 低通滤波发放，注意力权重 p_j = z_j/sum(z_l)
     （固定点近似 Softmax，现代 Hopfield 理论的脉冲实现）；
  3. Mean-field 注意力：Wilson-Cowan 群体率用 LIF 发放率近似，
     r 经突触迹低通，A_j = exp(beta q̄^T k_j) r_j / sum(...)；
  4. 阈值 v_th 逐模块可训练（q_th / k_th / v_th / attn_th，softplus 恒正），
     与标准 SNN（snn_mnist_train_accumulate.py 的 TrainableVthSNN）一致。

训练仍走 HyperscaleES Noiser（演化策略）：硬阈值不可微不影响更新，
这正是"无反向传播"训练脉冲网络的天然适配点。
"""

import jax
import jax.numpy as jnp

from .base_model import Model, CommonInit
from .common import merge_inits, call_submodule, MM, Parameter
from .snn_attention import (
    DEFAULT_CORE_ARGS,
    lif_step,
    run_lif,
    softmax_attention,
)


# ----------------------------------------------------------------------------
# Q/K/V 编码：真正的 LIF 时间积分（替代 snn_attention._rate_encode 的静态 sigmoid）
# ----------------------------------------------------------------------------
def _qkv_lif(common_params, x, vths):
    """把 token 的泊松输入脉冲编码成 Q/K/V 的 spike-count rate 向量。

    Args:
        common_params: CommonParams（含 'q'/'k'/'v' MM 模块与 frozen tau_m）。
        x:    (T, num_tokens, token_in_dim) 二值泊松脉冲输入（一个样本）。
        vths: (q_th, k_th, v_th) 三个可训练阈值（已 softplus 恒正）。
    Returns:
        (q_rate, k_rate, v_rate) 各 (num_tokens, d_head)，取值 [0,1] 的发放率。
    """
    tau_m = common_params.frozen_params["tau_m"]

    def encode(proj, vth):
        # proj: (T, N, d) 投影电流
        lif_p = {"tau_m": tau_m, "v_th": vth}
        v0 = jnp.zeros(proj.shape[1:], dtype=proj.dtype)   # (N, d) 初始膜电位
        spikes = run_lif(lif_p, proj, v0)                  # (T, N, d) 0/1 脉冲
        return jnp.mean(spikes, axis=0)                    # (N, d) spike-count rate

    q_proj = jax.vmap(lambda xt: call_submodule(MM, "q", common_params, xt))(x)
    k_proj = jax.vmap(lambda xt: call_submodule(MM, "k", common_params, xt))(x)
    v_proj = jax.vmap(lambda xt: call_submodule(MM, "v", common_params, xt))(x)
    return encode(q_proj, vths[0]), encode(k_proj, vths[1]), encode(v_proj, vths[2])


# ----------------------------------------------------------------------------
# Route 1: Hopfield 能量竞争的脉冲实现（LIF 注意力神经元 + 全局抑制 + 突触迹）
# ----------------------------------------------------------------------------
def hopfield_attention_lif(q, k, v, vth_attn, g_inh, tau_a, tau_m, beta, n_iter):
    """脉冲 Hopfield 注意力（现代 Hopfield 理论的 SNN 实现）。

    每个 key token 一个 LIF 注意力神经元，注入相似度电流
    ``I_j = beta * q̄^T k_j``；全局抑制神经元强度 ``G = sum(z)`` 提供
    divisive normalization；突触迹 ``z`` 对发放做一阶低通（时间常数 tau_a）。
    迭代 ``n_iter`` 步后归一化突触迹即注意力权重，读出 ``o = sum_j p_j v_j``。

    Returns (p, o): (num_tokens,) 归一化权重, (num_tokens, d) 值读出。
    """
    beta = jnp.asarray(beta, dtype=q.dtype)
    g_inh = jnp.asarray(g_inh, dtype=q.dtype)
    vth = jnp.asarray(vth_attn, dtype=q.dtype)
    q_center = jnp.mean(q, axis=0, keepdims=True)          # (1, d)
    h = (beta * (q_center @ k.T))[0]                       # (n,) 相似度电流
    n = h.shape[0]

    def step(state, _):
        u, z = state                                       # 膜电位、突触迹
        G = jnp.sum(z)                                     # 全局抑制强度
        u_new, s = lif_step({"tau_m": tau_m, "v_th": vth},
                            u, h - g_inh * G)              # 真发放 + hard reset
        z_new = z + (1.0 / tau_a) * (-z + s.astype(z.dtype))  # 突触迹低通
        return (u_new, z_new), z_new

    u0 = jnp.zeros((n,), dtype=q.dtype)
    z0 = jnp.zeros((n,), dtype=q.dtype)
    (_, z_final), _ = jax.lax.scan(step, (u0, z0), jnp.arange(n_iter))
    p = z_final / (jnp.sum(z_final) + 1e-6)                # 归一化注意力权重
    o = p[:, None] * v
    return p, o


# ----------------------------------------------------------------------------
# Route 2: Mean-field 群体动力学的脉冲实现（LIF 发放率近似 Wilson-Cowan 群体率）
# ----------------------------------------------------------------------------
def meanfield_attention_lif(q, k, v, vth_attn, gamma, tau_m, beta, n_iter):
    """LIF 版 Wilson-Cowan 群体注意力。

    每个 value group 的可用度 ``r_j`` 用 LIF 发放率近似：注入电流
    ``I_j = h_j - gamma*R``（R=sum(r) 为总活动，divisive 抑制），发放经
    突触迹低通得到群体率；迭代 ``n_iter`` 步后与指数相似度组合成权重
    ``A_j = exp(beta q̄^T k_j) r_j / sum(...)``。

    Returns (A, o): (num_tokens,) 归一化权重, (num_tokens, d) 值读出。
    """
    beta = jnp.asarray(beta, dtype=q.dtype)
    gamma = jnp.asarray(gamma, dtype=q.dtype)
    vth = jnp.asarray(vth_attn, dtype=q.dtype)
    q_center = jnp.mean(q, axis=0, keepdims=True)
    h = (beta * (q_center @ k.T))[0]                       # (n,)
    n = h.shape[0]

    def step(state, _):
        u, r = state                                       # 膜电位、群体率(突触迹)
        R = jnp.sum(r)                                     # 总活动
        u_new, s = lif_step({"tau_m": tau_m, "v_th": vth},
                            u, h - gamma * R)              # 真发放 + hard reset
        r_new = r + (1.0 / tau_m) * (-r + s.astype(r.dtype))  # 发放率低通
        return (u_new, r_new), r_new

    u0 = jnp.zeros((n,), dtype=q.dtype)
    r0 = jnp.zeros((n,), dtype=q.dtype)
    (_, r_final), _ = jax.lax.scan(step, (u0, r0), jnp.arange(n_iter))
    r = r_final / (jnp.sum(r_final) + 1e-6)

    e = jnp.exp(beta * q_center @ k.T)                     # (1, n)
    numer = e[0] * r                                       # (n,)
    A = numer / (jnp.sum(numer) + 1e-6)                    # (n,)
    o = A[:, None] * v
    return A, o


# ----------------------------------------------------------------------------
# 模型类：与 snn_attention.SNNAttentionModel 同构，但 QKV/注意力走真脉冲路径
# ----------------------------------------------------------------------------
class SNNAttentionLIFModel(Model):
    """基于真脉冲神经元的 SNN 注意力分类器（patched-token MNIST）。

    输入 ``x``: (T, num_tokens, token_in_dim) 二值泊松脉冲；
    输出: logits (num_classes,)。

    ``rand_init`` args: key, token_in_dim, num_tokens, num_classes, d_head,
    tau_m, v_th（可训练阈值初始值）, proj_gain, trainable_beta, dtype,
    以及路由超参（g_inh/tau_a/gamma/n_iter）。
    """

    @staticmethod
    def _attention(q, k, v, beta, frozen, vth_attn):
        raise NotImplementedError

    @classmethod
    def rand_init(cls, key, token_in_dim, num_tokens, num_classes, d_head,
                  tau_m=20.0, proj_gain=2.0, trainable_beta=True,
                  v_th=0.05, dtype=jnp.float32, **core_args):
        core = {k: core_args.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
        keys = jax.random.split(key, 6)
        layers = dict(
            q=MM.rand_init(keys[0], token_in_dim, d_head, dtype),
            k=MM.rand_init(keys[1], token_in_dim, d_head, dtype),
            v=MM.rand_init(keys[2], token_in_dim, d_head, dtype),
            out=MM.rand_init(keys[3], d_head, num_classes, dtype),
            out_gain=Parameter.rand_init(keys[4], None, None, jnp.ones((1,)), dtype),
        )
        # 可训练阈值：softplus 恒正参数化（与 TrainableVthSNN 一致）
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype=dtype)) - 1.0)
        for name in ("q_th", "k_th", "v_th", "attn_th"):
            layers[name] = Parameter.rand_init(None, None, None, raw_vth0, dtype)
        raw_beta = jnp.log(jnp.exp(jnp.asarray(1.0 / jnp.sqrt(d_head), dtype)) - 1.0)
        layers["beta"] = Parameter.rand_init(None, None, None, raw_beta, dtype)

        frozen_params = {
            "tau_m": jnp.asarray(tau_m, dtype=dtype),
            "proj_gain": jnp.asarray(proj_gain, dtype=dtype),
            "trainable_beta": bool(trainable_beta),
            **core,
        }
        if core_args:
            raise TypeError(f"unexpected core args: {sorted(core_args)}")
        merged = merge_inits(**layers)
        return CommonInit(frozen_params, merged.params, merged.scan_map, merged.es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        x = x.astype(common_params.params["q"].dtype)
        # 读取可训练阈值（softplus 恒正）
        vths = [jax.nn.softplus(call_submodule(Parameter, name, common_params))
                for name in ("q_th", "k_th", "v_th", "attn_th")]
        q_rate, k_rate, v_rate = _qkv_lif(common_params, x, vths[:3])

        if common_params.frozen_params["trainable_beta"]:
            beta = jax.nn.softplus(call_submodule(Parameter, "beta", common_params))
        else:
            beta = 1.0 / jnp.sqrt(q_rate.shape[-1])

        p, o = cls._attention(q_rate, k_rate, v_rate, beta,
                              common_params.frozen_params, vth_attn=vths[3])
        pooled = jnp.mean(o, axis=0)                       # (d_head,)
        logits = call_submodule(MM, "out", common_params, pooled)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


class HopfieldAttnSNNLIF(SNNAttentionLIFModel):
    """Route 1：脉冲 Hopfield 注意力（LIF 注意力神经元 + 突触迹）。"""

    @staticmethod
    def _attention(q, k, v, beta, frozen, vth_attn):
        return hopfield_attention_lif(
            q, k, v, vth_attn=vth_attn,
            g_inh=frozen["g_inh"], tau_a=frozen["tau_a"],
            tau_m=frozen["tau_m"], beta=beta, n_iter=frozen["n_iter"],
        )


class MeanFieldAttnSNNLIF(SNNAttentionLIFModel):
    """Route 2：脉冲 Mean-field 注意力（LIF 发放率近似 Wilson-Cowan 群体率）。"""

    @staticmethod
    def _attention(q, k, v, beta, frozen, vth_attn):
        return meanfield_attention_lif(
            q, k, v, vth_attn=vth_attn,
            gamma=frozen["gamma"], tau_m=frozen["tau_m"],
            beta=beta, n_iter=frozen["n_iter"],
        )


def model_rand_init(route, key, token_in_dim, num_tokens, num_classes, d_head,
                    **kwargs):
    """Factory：构建 HopfieldAttnSNNLIF 或 MeanFieldAttnSNNLIF 的 CommonInit。

    ``route`` in {"hopfield", "meanfield"}。额外 kwargs（DEFAULT_CORE_ARGS）覆盖
    路由超参；``v_th`` 等传入模型 rand_init。
    """
    base_kwargs = {k: kwargs.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
    model = HopfieldAttnSNNLIF if route == "hopfield" else MeanFieldAttnSNNLIF
    return model.rand_init(key, token_in_dim, num_tokens, num_classes, d_head,
                           **base_kwargs, **kwargs)
