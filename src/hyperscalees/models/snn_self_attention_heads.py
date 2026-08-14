"""增强版逐 token 脉冲自注意力：多头 + 位置编码 + 多块残差加深。

在 ``snn_self_attention.py`` 基础上补齐 Transformer 三要素：

  1. **多头（multi-head）**：``num_heads`` 个头，每头独立 LIF Q/K/V 投影 +
     独立脉冲自注意力，头输出拼接后投影回 d_model；
  2. **位置编码（positional encoding）**：可训练位置嵌入 ``pos_emb ∈ (N, d_model)``，
     在每块输入叠加，给脉冲 token 序列注入位置信息；
  3. **加深（multi-block residual）**：L 个 self-attention 块堆叠，每块 =
     多头脉冲自注意力 -> 残差 -> LIF 前馈 MLP -> 残差（标准 Transformer 块结构）。

前向：
  x: (T, N, token_in) 泊松脉冲
    -> LIF 编码（逐 token 独立）  得到 token 表示 h0 ∈ (N, d_model)
    -> 堆叠 L 个块（每块先加位置编码 pos_emb，再多头脉冲自注意力+残差，再前馈+残差）
    -> 池化 -> readout logits

所有可训练权重一律经 ``call_submodule(MM, ...)`` 访问，使 Noiser 的 LoRA
噪声（do_mm）正确注入更新；硬阈值不可微不影响演化策略训练。
"""

import jax
import jax.numpy as jnp

from .base_model import Model, CommonInit
from .common import merge_inits, call_submodule, MM, Parameter
from .snn_attention import lif_step, run_lif
from .snn_self_attention import row_softmax_lif

DEFAULT_CORE_ARGS = {
    "g_inh": 0.5,       # 注意力神经元全局抑制强度
    "tau_syn": 5.0,     # 突触迹低通时间常数
    "n_iter": 8,        # 注意力竞争迭代步数
}

DEFAULT_CFG = {
    "d_model": 32,      # token 表示维度
    "num_heads": 4,     # 多头数（每头维度 = d_model // num_heads）
    "num_blocks": 2,    # self-attention 块数（加深）
}


def _rate(proj, vth, tau_m):
    """沿第一个轴（时间 T）做 LIF 积分，返回其余轴的 spike-count rate。

    ``proj``: (T, ...)，如 (T, N, d)。v0 形状为 proj.shape[1:]，spikes 沿 T 平均。
    """
    v0 = jnp.zeros(proj.shape[1:], dtype=proj.dtype)
    spikes = run_lif({"tau_m": tau_m, "v_th": vth}, proj, v0)
    return jnp.mean(spikes, axis=0)                       # (N, d)


def _softplus_param(common_params, name):
    return jax.nn.softplus(call_submodule(Parameter, name, common_params))


# 真正逐 token 自注意力核心（复用单头的 row_softmax_lif，逐行竞争）
def multihead_attention_once(common_params, X, b, H, beta):
    """对第 b 块的输入 X ∈ (N, d_model)，多头脉冲自注意力 + 输出投影。

    每头：Q/K/V 线性投影（do_mm 注入 LoRA 噪声，X 已是 rate 表示，无需再时间积分）
    -> 逐行脉冲 softmax（LIF 竞争模拟 softmax）-> O_h = A_h @ V_h。
    H 头拼接 -> 投影回 d_model。返回该块注意力输出 (N, d_model)。

    脉冲性体现在**注意力竞争**（row_softmax_lif 的发放计数 winner-take-all），
    与单头 snn_self_attention.py 一致。
    """
    head_dim = common_params.frozen_params["head_dim"]
    vth_attn = _softplus_param(common_params, "attn_th")
    g_inh = common_params.frozen_params["g_inh"]
    tau_m = common_params.frozen_params["tau_m"]
    tau_syn = common_params.frozen_params["tau_syn"]
    n_iter = common_params.frozen_params["n_iter"]

    heads = []
    for h in range(H):
        # 逐头 Q/K/V 线性投影（rate 表示 X，do_mm 注入 LoRA 噪声）
        qr = jax.vmap(lambda xt: call_submodule(MM, f"b{b}_q{h}", common_params, xt))(X)
        kr = jax.vmap(lambda xt: call_submodule(MM, f"b{b}_k{h}", common_params, xt))(X)
        vr = jax.vmap(lambda xt: call_submodule(MM, f"b{b}_v{h}", common_params, xt))(X)
        # 逐 token query 的逐行脉冲 softmax -> (N, N) 注意力矩阵
        A = jax.vmap(lambda qi: row_softmax_lif(
            beta * (qi @ kr.T), vth=vth_attn, g_inh=g_inh,
            tau_m=tau_m, tau_syn=tau_syn, n_iter=n_iter))(qr)
        heads.append(A @ vr)                                # (N, head_dim)

    concat = jnp.concatenate(heads, axis=-1)                # (N, H*head_dim = d_model)
    out = jax.vmap(lambda ct: call_submodule(MM, f"b{b}_o", common_params, ct))(concat)
    return out                                              # (N, d_model)


def ffn_once(common_params, X, b):
    """前馈 MLP（块内，普通带噪声线性 + LIF 单步发放入口）。X: (N, d_model)。"""
    tau_m = common_params.frozen_params["tau_m"]
    vth = _softplus_param(common_params, f"b{b}_ff_th")
    cur1 = jax.vmap(lambda xt: call_submodule(MM, f"b{b}_ff1", common_params, xt))(X)
    # LIF 单步（把 -1..1 的线性激活过膜电位 -> 发放率），保留非线性门控
    def lif_single(c):
        v = c / jnp.maximum(jnp.max(jnp.abs(c)), 1e-6)     # 归一化激活
        spike = (v >= vth).astype(v.dtype)
        return spike
    act = jax.vmap(lif_single)(cur1)                        # (N, d_model) 0/1
    out = jax.vmap(lambda st: call_submodule(MM, f"b{b}_ff2", common_params, st))(act)
    return out


class SNNSelfAttentionHeadsModel(Model):
    """多头 + 位置编码 + 多块残差的逐 token 脉冲自注意力分类器。"""

    @classmethod
    def rand_init(cls, key, token_in_dim, num_tokens, num_classes, d_head,
                  tau_m=20.0, proj_gain=2.0, trainable_beta=True,
                  v_th=0.05, dtype=jnp.float32, **core_args):
        core = {k: core_args.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
        cfg = {k: core_args.pop(k, v) for k, v in DEFAULT_CFG.items()}
        if core_args:
            raise TypeError(f"unexpected core args: {sorted(core_args)}")
        d_model, H, L = cfg["d_model"], cfg["num_heads"], cfg["num_blocks"]
        head_dim = d_model // H
        assert head_dim >= 1, "d_model 必须能被 num_heads 整除"

        # 分配 keys：输入编码3 + 位置 + 每块(3H + 1 o + 2 ffn) + readout2 + 阈值若干
        n_keys = 3 + 1 + L * (3 * H + 3) + 2
        keys = jax.random.split(key, n_keys)
        layers, ki = {}, 0

        layers["in_q"] = MM.rand_init(keys[ki], token_in_dim, d_model, dtype); ki += 1
        layers["in_k"] = MM.rand_init(keys[ki], token_in_dim, d_model, dtype); ki += 1
        layers["in_v"] = MM.rand_init(keys[ki], token_in_dim, d_model, dtype); ki += 1
        layers["pos_emb"] = Parameter.rand_init(
            None, None, None, jnp.zeros((num_tokens, d_model), dtype), dtype)

        for b in range(L):
            for h in range(H):
                layers[f"b{b}_q{h}"] = MM.rand_init(keys[ki], d_model, head_dim, dtype); ki += 1
                layers[f"b{b}_k{h}"] = MM.rand_init(keys[ki], d_model, head_dim, dtype); ki += 1
                layers[f"b{b}_v{h}"] = MM.rand_init(keys[ki], d_model, head_dim, dtype); ki += 1
            layers[f"b{b}_o"] = MM.rand_init(keys[ki], d_model, d_model, dtype); ki += 1
            layers[f"b{b}_ff1"] = MM.rand_init(keys[ki], d_model, d_model, dtype); ki += 1
            layers[f"b{b}_ff2"] = MM.rand_init(keys[ki], d_model, d_model, dtype); ki += 1

        layers["out"] = MM.rand_init(keys[ki], d_model, num_classes, dtype); ki += 1
        layers["out_gain"] = Parameter.rand_init(
            keys[ki], None, None, jnp.ones((1,)), dtype)

        # 可训练阈值（softplus 恒正）
        raw_vth0 = jnp.log(jnp.exp(jnp.asarray(v_th, dtype)) - 1.0)
        for name in ("q_th", "k_th", "v_th", "attn_th"):
            layers[name] = Parameter.rand_init(None, None, None, raw_vth0, dtype)
        for b in range(L):
            layers[f"b{b}_ff_th"] = Parameter.rand_init(None, None, None, raw_vth0, dtype)

        raw_beta = jnp.log(jnp.exp(jnp.asarray(1.0 / jnp.sqrt(head_dim), dtype)) - 1.0)
        layers["beta"] = Parameter.rand_init(None, None, None, raw_beta, dtype)

        frozen_params = {
            "tau_m": jnp.asarray(tau_m, dtype),
            "proj_gain": jnp.asarray(proj_gain, dtype),
            "trainable_beta": bool(trainable_beta),
            "d_model": d_model, "num_heads": H, "num_blocks": L, "head_dim": head_dim,
            **core,
        }
        merged = merge_inits(**layers)
        return CommonInit(frozen_params, merged.params, merged.scan_map, merged.es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        x = x.astype(common_params.params["in_q"].dtype)
        tau_m = common_params.frozen_params["tau_m"]
        L = common_params.frozen_params["num_blocks"]

        # 输入 LIF 编码 -> token 表示 (N, d_model)：Q/K/V 三路发放率取平均
        def enc(name, th_name):
            # x: (T, N, token_in) 泊松脉冲；沿 T,N 逐 token 投影 -> (T, N, d)，
            # 再沿时间积分得到逐 token rate (N, d)。
            proj = jax.vmap(jax.vmap(lambda xt: call_submodule(MM, name, common_params, xt)))(x)
            return _rate(proj, _softplus_param(common_params, th_name), tau_m)
        X = (enc("in_q", "q_th") + enc("in_k", "k_th") + enc("in_v", "v_th")) / 3.0

        # 位置编码
        X = X + call_submodule(Parameter, "pos_emb", common_params)

        beta = (jax.nn.softplus(call_submodule(Parameter, "beta", common_params))
                if common_params.frozen_params["trainable_beta"]
                else 1.0 / jnp.sqrt(common_params.frozen_params["head_dim"]))

        # 堆叠 L 个块（残差：X <- X + attn(X) + ffn(X)）
        for b in range(L):
            attn_out = multihead_attention_once(common_params, X, b,
                                                common_params.frozen_params["num_heads"], beta)
            X = X + attn_out
            X = X + ffn_once(common_params, X, b)

        pooled = jnp.mean(X, axis=0)                        # (d_model,)
        logits = call_submodule(MM, "out", common_params, pooled)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


def model_rand_init(key, token_in_dim, num_tokens, num_classes, d_head, **kwargs):
    base_kwargs = {k: kwargs.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
    return SNNSelfAttentionHeadsModel.rand_init(
        key, token_in_dim, num_tokens, num_classes, d_head,
        **base_kwargs, **kwargs)
