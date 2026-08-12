"""Leaky Integrate-and-Fire (LIF) SNN model, conforming to the HyperscaleES ``Model`` interface.

The model is trained with the noiser (evolutionary strategy) abstraction, so it does
NOT rely on gradients through the non-differentiable spike function — the spike
thresholding / reset are hard operations and the noiser updates weights via fitness.

Architecture (two LIF hidden layers + rate readout head):

    x: (T, batch, in_dim)  binary Poisson spikes
      -> Linear(noised via noiser.do_mm)         (per timestep, same iterinfo)
      -> LIF hidden layer 1   (recurrence over T via jax.lax.scan)
      -> Linear(noised via noiser.do_mm)
      -> LIF hidden layer 2
      -> Linear(noised via noiser.do_mm)
      -> mean firing rate over time
      -> readout logits (batch, num_classes)
"""

from functools import partial

import jax
import jax.numpy as jnp

from .base_model import Model, CommonInit
from .common import merge_inits, call_submodule, MM, Parameter


# ----------------------------------------------------------------------------
# LIF neuron dynamics (pure JAX, no Model wrapper needed)
# ----------------------------------------------------------------------------
def lif_step(params, v, current, dt=1.0):
    """Single LIF update: leak -> charge -> fire -> reset.

    Args:
        params: dict with 'tau_m', 'v_th' (membrane time constant, threshold).
                These come from frozen_params so they are not evolved.
        v:      membrane potential (..., hidden) at previous step.
        current: input current (..., hidden) at this step.
    Returns:
        new_v, spike (0/1).
    """
    tau_m = params["tau_m"]
    v_th = params["v_th"]
    # sub-threshold decay (leak) + input
    v = v + (dt / tau_m) * (-v + current)
    spike = (v >= v_th).astype(v.dtype)
    # hard reset to 0 after firing
    v = v * (1.0 - spike)
    return v, spike


def run_lif(params, input_current, v0):
    """Run LIF dynamics over time.

    Args:
        params: LIF params (tau_m, v_th).
        input_current: (T, batch, hidden).
        v0: initial membrane potential (batch, hidden).
    Returns:
        spikes: (T, batch, hidden) 0/1
    """
    def step(carry, current_t):
        v = carry
        v, spike = lif_step(params, v, current_t)
        return v, spike

    final_v, spikes = jax.lax.scan(step, v0, input_current)
    return spikes


# ----------------------------------------------------------------------------
# SNN Model
# ----------------------------------------------------------------------------
class SNNModel(Model):
    """A small fully-connected LIF SNN.

    ``rand_init`` args: key, in_dim, hidden_dims=(h1, h2), num_classes,
    tau_m, v_th, dtype.
    """

    @classmethod
    def rand_init(cls, key, in_dim, hidden_dims, num_classes, tau_m=20.0,
                  v_th=0.3, dtype=jnp.float32):
        """Initialize the SNN.

        ``v_th`` should be tuned so LIF neurons actually fire: with weights scaled by
        1/sqrt(fan_in) and inputs in [0,1], a threshold around 0.3 keeps the hidden
        layers active (a naive 1.0 silences the network).
        """
        in_key, h1_key, h2_key, out_key = jax.random.split(key, 4)

        layers = merge_inits(
            fc1=MM.rand_init(in_key, in_dim, hidden_dims[0], dtype),
            fc2=MM.rand_init(h1_key, hidden_dims[0], hidden_dims[1], dtype),
            fc3=MM.rand_init(h2_key, hidden_dims[1], num_classes, dtype),
            # trainable bias/scale parameter(s) as an example of a PARAM; kept
            # simple here as a scalar gain on the readout.
            out_gain=Parameter.rand_init(out_key, None, None,
                                         jnp.ones((1,)), dtype=dtype),
        )
        frozen_params = {"tau_m": jnp.asarray(tau_m, dtype=dtype),
                         "v_th": jnp.asarray(v_th, dtype=dtype)}
        scan_map = layers.scan_map
        es_map = layers.es_map
        return CommonInit(frozen_params, layers.params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        """Forward pass over a SINGLE sample.

        x: (T, in_dim) binary Poisson spikes for one sample.
        Batch parallelization is done by vmapping `Model.forward` over the batch axis.
        Returns logits: (num_classes,).
        """
        x = x.astype(common_params.params["fc1"].dtype)
        # fc1 weight is (h1, in_dim); fc2 weight is (h2, h1)
        hidden1 = common_params.params["fc1"].shape[0]
        hidden2 = common_params.params["fc2"].shape[0]

        # --- Layer 1: Linear projection per timestep, then LIF over T ---
        # The noised matmul is applied identically at each timestep (same iterinfo).
        def proj1(x_t):
            return call_submodule(MM, "fc1", common_params, x_t)
        cur1 = jax.vmap(proj1)(x)                       # (T, h1)
        v0 = jnp.zeros((hidden1,), dtype=cur1.dtype)
        spikes1 = run_lif(common_params.frozen_params, cur1, v0)  # (T, h1)

        # --- Layer 2 ---
        def proj2(x_t):
            return call_submodule(MM, "fc2", common_params, x_t)
        cur2 = jax.vmap(proj2)(spikes1)                 # (T, h2)
        v0_2 = jnp.zeros((hidden2,), dtype=cur2.dtype)
        spikes2 = run_lif(common_params.frozen_params, cur2, v0_2)  # (T, h2)

        # --- Readout: mean firing rate over time -> fc3 ---
        rate = jnp.mean(spikes2, axis=0)                # (h2,) in [0,1]
        logits = call_submodule(MM, "fc3", common_params, rate)  # (C,)
        gain = call_submodule(Parameter, "out_gain", common_params)
        return logits * gain


# Backwards-friendly alias kept for clarity in scripts.
LIFLayer = lif_step
