"""SNN Attention models — HyperscaleES ``Model`` implementations.

Based on ``docs/注意力机制数学等价snn迁移.md``. That document argues that a
"逐时刻严格等价" ANN→SNN Attention copy is unrealistic, and instead proposes three
coarser equivalences (fixed-point / time-average / mean-field). Two of the routes
are implemented here as trainable attention blocks:

1. **Hopfield energy competition** (Section 一):  Softmax attention weight
   ≈ free-energy minimizer ≈ SNN attractor competition. A per-token population of
   LIF "attention neurons" integrates similarity currents :math:`I_j = \\beta q^\\top k_j`
   under a *global inhibition* neuron ``G`` (≈ sum of activities) that provides the
   divisive normalization; the spike trains are low-pass filtered into a synaptic
   trace ``z_j`` and the attention estimate is :math:`p_j = z_j / (\\varepsilon + \\sum_l z_l)`.
   The value readout is :math:`o = \\sum_j p_j v_j`.

2. **Mean-field population dynamics** (Section 二):  A Wilson–Cowan style population
   rate for each key/value token, :math:`\\tau \\dot r_j = -r_j + \\phi(h_j - \\gamma R)`
   with :math:`R = \\sum_l r_l`. After S updates the converged rate ``r_j`` is combined
   with the exponential similarities (the document's "value group availability"
   reading) into :math:`A_j = \\exp(\\beta q^\\top k_j) r_j / \\sum_l \\exp(\\beta q^\\top k_l) r_l`
   and the output is :math:`o = \\sum_j A_j v_j`.

Both blocks share the same Q/K/V front-end (noised ``MM`` projections + LIF encoding
into rate vectors) and attach to the HyperscaleES noiser (EggRoll), so **every trainable
weight is updated by the evolution strategy, not by back-prop through spikes**.

A reference "softmax attention" forward is provided to *measure* the equivalence
(Section 五: weight error :math:`\\|\\hat p - p^*\\|`, output error/cosine).
"""

import jax
import jax.numpy as jnp

from .base_model import Model, CommonInit
from .common import merge_inits, call_submodule, MM, Parameter


# ----------------------------------------------------------------------------
# LIF dynamics (shared semantics with models/snn.py)
# ----------------------------------------------------------------------------
def lif_step(params, v, current, dt=1.0):
    """Single LIF update: leak -> charge -> fire -> reset.
    ``params`` must contain 'tau_m' and 'v_th'. Returns (new_v, spike)."""
    tau_m = params["tau_m"]
    v_th = params["v_th"]
    v = v + (dt / tau_m) * (-v + current)
    spike = (v >= v_th).astype(v.dtype)
    v = v * (1.0 - spike)  # hard reset
    return v, spike


def run_lif(params, input_current, v0):
    """Run LIF over time; ``input_current``: (T, ...). Returns spikes (T, ...)."""

    def step(carry, current_t):
        v = carry
        v, spike = lif_step(params, v, current_t)
        return v, spike

    _, spikes = jax.lax.scan(step, v0, input_current)
    return spikes


# ----------------------------------------------------------------------------
# Shared Q/K/V front end (robust rate encoding)
# ----------------------------------------------------------------------------
def _rate_encode(proj, gain):
    """Temporal spike-count rate encoding -> bounded (0,1) rates.

    ``proj``: (T, ..., d) projection currents.  The **temporal mean** over the Poisson
    spike window (the canonical SNN spike-count rate code) is passed through a
    softsign-normalised sigmoid, giving a dense, bounded, monotone rate in (0,1).
    A sum reduction (rather than ``jax.lax.scan``) is used so the encoder stays
    compatible with outer ``jax.vmap`` over the sample/batch axis.
    """
    mean_p = jnp.mean(proj, axis=0)                       # (…, d) temporal mean
    return jax.nn.sigmoid(gain * mean_p / (1.0 + jnp.abs(mean_p)))


def _mk_qkv(common_params, x):
    """Project and encode token spikes into per-token Q/K/V **rate** vectors.

    Args:
        common_params: CommonParams for this model (contains 'q'/'k'/'v' MM modules).
        x:    (T, num_tokens, token_in_dim) spike input for one sample.
    Returns:
        (q, k, v) each (num_tokens, d_head) firing rates in (0,1).
    """
    q_proj = jax.vmap(lambda xt: call_submodule(MM, "q", common_params, xt))(x)
    k_proj = jax.vmap(lambda xt: call_submodule(MM, "k", common_params, xt))(x)
    v_proj = jax.vmap(lambda xt: call_submodule(MM, "v", common_params, xt))(x)

    gain = common_params.frozen_params["proj_gain"]
    q_rate = _rate_encode(q_proj, gain)
    k_rate = _rate_encode(k_proj, gain)
    v_rate = _rate_encode(v_proj, gain)
    return q_rate, k_rate, v_rate


# ----------------------------------------------------------------------------
# Route 1: Hopfield energy competition (graded attractor -> softmax)
# ----------------------------------------------------------------------------
def hopfield_attention(q, k, v, g_inh, tau_a, beta, n_iter):
    """Graded (continuous) Hopfield relaxation -> softmax-like attention weights.

    This is the Section 一/三 energy route: the Softmax weight
    :math:`p_j \\propto \\exp(\\beta \\bar q^\\top k_j)` is the fixed point of a free-energy
    descent (the replicator/attractor dynamics). Here a continuous Hopfield relaxation
    of the per-token similarity energies
    :math:`u \\leftarrow u + \\tau_a^{-1}(-u + h - g_{inh}\\cdot\\mathrm{mean}(u))`
    is iterated ``n_iter`` times; ``mean(u)`` plays the role of the global-inhibitory
    normalization. The steady state is, up to an additive constant, the similarity
    vector ``h``, and a stable Boltzmann readout (:math:`p_j = e^{u_j}/\\sum_l e^{u_l}`)
    recovers the Softmax attention weights (the attractor retrieval). This avoids the
    fragile binary-spike threshold so the ES-trained model does not collapse.

    Returns (p, o): (num_tokens,) normalized weights, (num_tokens, d) value readout.
    """
    beta = jnp.asarray(beta, dtype=q.dtype)
    g_inh = jnp.asarray(g_inh, dtype=q.dtype)
    q_center = jnp.mean(q, axis=0, keepdims=True)   # (1, d)
    h = (beta * (q_center @ k.T))[0]                # (n,) similarity energies

    def step(u, _):
        c = jnp.mean(u)                              # global activity (divisive norm)
        u = u + (1.0 / tau_a) * (-u + h - g_inh * c)
        return u, None

    u, _ = jax.lax.scan(step, h, jnp.arange(n_iter))
    e = jnp.exp(u - jnp.max(u))                     # stable Boltzmann readout
    p = e / (jnp.sum(e) + 1e-6)                     # (n,)
    o = p[:, None] * v
    return p, o


# ----------------------------------------------------------------------------
# Route 2: mean-field population dynamics (Wilson-Cowan + divisive norm)
# ----------------------------------------------------------------------------
def meanfield_attention(q, k, v, gamma, beta, n_iter):
    """Wilson–Cowan population approach to the attention weights.

    The "availability" of each value group :math:`r_j` follows the recurrent
    :math:`r \\leftarrow \\mathrm{ReLU}(h_j - \\gamma R)` population law (leaky,
    divisively normalized); iterated ``n_iter`` times. The Section 二.4 estimator is
    :math:`A_j = e^{\\beta q^\\top k_j} r_j / \\sum_l e^{\\beta q^\\top k_l} r_l` and the
    readout :math:`o = \\sum_j A_j v_j`.

    Returns (A, o): (num_tokens,) normalized weights, (num_tokens, d) readout.
    """
    beta = jnp.asarray(beta, dtype=q.dtype)
    gamma = jnp.asarray(gamma, dtype=q.dtype)
    q_center = jnp.mean(q, axis=0, keepdims=True)
    h = (beta * (q_center @ k.T))[0]               # (n,)

    def step(r, _):
        R = jnp.sum(r)
        r_new = jax.nn.relu(h - gamma * R)
        r_new = r_new / (jnp.maximum(jnp.sum(r_new), 1e-6))
        return r_new, r_new

    r0 = jax.nn.relu(h)
    r, _ = jax.lax.scan(step, r0, jnp.arange(n_iter))
    r = r / (jnp.sum(r) + 1e-6)

    e = jnp.exp(beta * q_center @ k.T)             # (1, n)
    numer = e[0] * r                               # (n,)
    A = numer / (jnp.sum(numer) + 1e-6)            # (n,)
    o = A[:, None] * v
    return A, o


# ----------------------------------------------------------------------------
# Reference softmax attention (the equivalence target, Section 0)
# ----------------------------------------------------------------------------
def softmax_attention(q, k, v, beta=1.0):
    """Standard Softmax self-attention output used as the equivalence target."""
    q_center = jnp.mean(q, axis=0, keepdims=True)
    e = jnp.exp(beta * (q_center @ k.T))[0]        # (n,)
    p = e / (jnp.sum(e) + 1e-6)                    # (n,)
    o = p[:, None] * v
    return p, o


DEFAULT_CORE_ARGS = {
    "g_inh": 0.5,      # Hopfield global-inhibition strength
    "tau_a": 5.0,      # Hopfield attractor relaxation time constant
    "gamma": 0.5,      # Mean-field divisive normalization strength
    "n_iter": 8,       # number of recurrent / population iterations
}


class SNNAttentionModel(Model):
    """Base class for an SNN attention classifier (patched-token MNIST).

    Input per sample ``x``: (T, num_tokens, token_in_dim) binary Poisson spikes of the
    patched tokens. Output: logits (num_classes,).

    ``rand_init`` args: key, token_in_dim, num_tokens, num_classes, d_head, tau_m,
    v_th, trainable_beta, dtype, plus the route hyper-parameters from DEFAULT_CORE_ARGS.
    """

    @staticmethod
    def _attention(q, k, v, beta, frozen):
        raise NotImplementedError

    @classmethod
    def rand_init(cls, key, token_in_dim, num_tokens, num_classes, d_head,
                  tau_m=20.0, proj_gain=2.0, trainable_beta=True,
                  dtype=jnp.float32, **core_args):
        core = {k: core_args.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
        keys = jax.random.split(key, 5)
        layers = dict(
            q=MM.rand_init(keys[0], token_in_dim, d_head, dtype),
            k=MM.rand_init(keys[1], token_in_dim, d_head, dtype),
            v=MM.rand_init(keys[2], token_in_dim, d_head, dtype),
            out=MM.rand_init(keys[3], d_head, num_classes, dtype),
            out_gain=Parameter.rand_init(keys[4], None, None, jnp.ones((1,)), dtype),
        )
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
        q_rate, k_rate, v_rate = _mk_qkv(common_params, x)

        if common_params.frozen_params["trainable_beta"]:
            beta = jax.nn.softplus(call_submodule(Parameter, "beta", common_params))
        else:
            beta = 1.0 / jnp.sqrt(q_rate.shape[-1])

        p, o = cls._attention(q_rate, k_rate, v_rate, beta,
                              common_params.frozen_params)
        pooled = jnp.mean(o, axis=0)                 # (d_head,)
        logits = call_submodule(MM, "out", common_params, pooled)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain

class HopfieldAttnSNN(SNNAttentionModel):
    """Route 1: graded Hopfield attractor competition (Section 一 / 三)."""

    @staticmethod
    def _attention(q, k, v, beta, frozen):
        return hopfield_attention(
            q, k, v,
            g_inh=frozen["g_inh"], tau_a=frozen["tau_a"],
            beta=beta, n_iter=frozen["n_iter"],
        )


class MeanFieldAttnSNN(SNNAttentionModel):
    """Route 2: Wilson-Cowan population (Section 二)."""

    @staticmethod
    def _attention(q, k, v, beta, frozen):
        return meanfield_attention(
            q, k, v, gamma=frozen["gamma"], beta=beta, n_iter=frozen["n_iter"],
        )


def model_rand_init(route, key, token_in_dim, num_tokens, num_classes, d_head,
                    **kwargs):
    """Factory: build a HopfieldAttnSNN or MeanFieldAttnSNN CommonInit.

    ``route`` in {"hopfield", "meanfield"}. Extra kwargs (DEFAULT_CORE_ARGS) override
    the per-route hyper-parameters.
    """
    base_kwargs = {k: kwargs.pop(k, v) for k, v in DEFAULT_CORE_ARGS.items()}
    model = HopfieldAttnSNN if route == "hopfield" else MeanFieldAttnSNN
    return model.rand_init(key, token_in_dim, num_tokens, num_classes, d_head,
                           **base_kwargs, **kwargs)
