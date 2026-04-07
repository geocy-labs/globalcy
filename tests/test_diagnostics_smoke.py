from __future__ import annotations

import jax.numpy as jnp

from globalcy.diagnostics.characteristic_forms import characteristic_summary
from globalcy.diagnostics.positivity import positivity_summary


def test_diagnostics_smoke():
    metrics = jnp.stack([jnp.eye(3, dtype=jnp.complex64), 1.2 * jnp.eye(3, dtype=jnp.complex64)], axis=0)
    positivity = positivity_summary(metrics)
    characteristic = characteristic_summary(metrics)
    assert positivity["negative_fraction"] == 0.0
    assert characteristic["determinant_mean"] > 0.0
