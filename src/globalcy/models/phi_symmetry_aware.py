from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from globalcy.models.phi_local_mlp import apply_local_mlp, init_local_mlp


def init_symmetry_aware_phi(rng: jax.Array, input_dim: int, hidden_dims: Sequence[int]) -> dict[str, object]:
    base_dim = input_dim - 1
    base_key, gate_key = jax.random.split(rng)
    return {
        "base_mlp": init_local_mlp(base_key, base_dim, hidden_dims),
        "gate_w": jax.random.normal(gate_key, (1, hidden_dims[0]), dtype=jnp.float32) * 0.05,
        "gate_b": jnp.zeros((hidden_dims[0],), dtype=jnp.float32),
        "readout_w": jax.random.normal(gate_key, (hidden_dims[0], 1), dtype=jnp.float32) * 0.05,
    }


def apply_symmetry_aware_phi(params: dict[str, object], inputs: jnp.ndarray) -> jnp.ndarray:
    base_dim = params["base_mlp"][0]["w"].shape[0]
    base_inputs = inputs[:, :base_dim]
    orbit_size = inputs[:, base_dim: base_dim + 1]
    base_hidden = base_inputs
    base_layers = params["base_mlp"]
    for layer in base_layers[:-1]:
        base_hidden = jnp.tanh(base_hidden @ layer["w"] + layer["b"])
    gate = jnp.tanh(orbit_size @ params["gate_w"] + params["gate_b"])
    fused_hidden = base_hidden * (1.0 + 0.1 * gate)
    output = fused_hidden @ base_layers[-1]["w"] + base_layers[-1]["b"]
    output = output + gate @ params["readout_w"]
    return jnp.squeeze(output, axis=-1)
