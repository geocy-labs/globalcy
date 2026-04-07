from __future__ import annotations

import jax
import jax.numpy as jnp

from globalcy.models.phi_symmetry_aware import apply_symmetry_aware_phi, init_symmetry_aware_phi


def test_symmetry_aware_model_smoke():
    features = jnp.ones((4, 33), dtype=jnp.float32)
    params = init_symmetry_aware_phi(jax.random.PRNGKey(7), input_dim=33, hidden_dims=[16, 16])
    outputs = apply_symmetry_aware_phi(params, features)
    assert outputs.shape == (4,)
