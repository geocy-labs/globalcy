from __future__ import annotations

import jax.numpy as jnp

from globalcy.geometry.projective import affine_to_homogeneous, homogeneous_to_invariants


def invariant_features_from_affine(affine_complex: jnp.ndarray, chart_id: int = 0) -> jnp.ndarray:
    homogeneous = affine_to_homogeneous(affine_complex, chart_id=chart_id)
    return homogeneous_to_invariants(homogeneous)
