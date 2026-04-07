from __future__ import annotations

import jax.numpy as jnp

from globalcy.diagnostics.consistency import symmetry_consistency_score


def test_symmetry_consistency_smoke():
    base = jnp.array([0.2, 0.4, 0.6], dtype=jnp.float32)
    assert symmetry_consistency_score(base, base) == 0.0
