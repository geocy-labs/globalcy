from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp


def init_local_mlp(rng: jax.Array, input_dim: int, hidden_dims: Sequence[int]) -> list[dict[str, jnp.ndarray]]:
    dims = [input_dim, *hidden_dims, 1]
    params: list[dict[str, jnp.ndarray]] = []
    keys = jax.random.split(rng, len(dims) - 1)
    for key, fan_in, fan_out in zip(keys, dims[:-1], dims[1:], strict=True):
        weight = jax.random.normal(key, (fan_in, fan_out), dtype=jnp.float32) / jnp.sqrt(float(fan_in))
        bias = jnp.zeros((fan_out,), dtype=jnp.float32)
        params.append({"w": weight, "b": bias})
    return params


def apply_local_mlp(params: list[dict[str, jnp.ndarray]], inputs: jnp.ndarray) -> jnp.ndarray:
    hidden = inputs
    for layer in params[:-1]:
        hidden = jnp.tanh(hidden @ layer["w"] + layer["b"])
    output = hidden @ params[-1]["w"] + params[-1]["b"]
    return jnp.squeeze(output, axis=-1)
