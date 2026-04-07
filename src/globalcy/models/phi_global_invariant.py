from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from globalcy.models.phi_local_mlp import apply_local_mlp, init_local_mlp


def init_global_invariant_phi(rng: jax.Array, input_dim: int, hidden_dims: Sequence[int]) -> list[dict[str, jnp.ndarray]]:
    return init_local_mlp(rng, input_dim, hidden_dims)


def apply_global_invariant_phi(params: list[dict[str, jnp.ndarray]], inputs: jnp.ndarray) -> jnp.ndarray:
    return apply_local_mlp(params, inputs)
