from __future__ import annotations

import jax.numpy as jnp


def canonicalize_symmetry_features(features: jnp.ndarray) -> jnp.ndarray:
    return features


def symmetry_orbit_size_default(batch_size: int) -> jnp.ndarray:
    return jnp.ones((batch_size, 1), dtype=jnp.float32)
