from __future__ import annotations

import jax.numpy as jnp

from globalcy.geometry.projective import homogeneous_to_invariants, normalize_homogeneous, rescale_points


def test_projective_invariance_smoke():
    point = jnp.array([1.0 + 0.0j, 0.2 + 0.1j, -0.1 + 0.3j, 0.4 - 0.2j], dtype=jnp.complex64)
    original = homogeneous_to_invariants(normalize_homogeneous(point))
    scaled = homogeneous_to_invariants(normalize_homogeneous(rescale_points(point, 1.7 + 0.4j)))
    assert jnp.allclose(original, scaled, atol=1e-6)
