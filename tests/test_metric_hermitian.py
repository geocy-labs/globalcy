from __future__ import annotations

import jax.numpy as jnp

from globalcy.geometry.kahler import kahler_metric


def test_metric_hermitian():
    point = jnp.array([0.1, -0.2, 0.2, 0.05, 0.15, -0.1], dtype=jnp.float32)
    metric = kahler_metric(lambda x: jnp.sum(x * x) * 0.01, point)
    assert jnp.allclose(metric, jnp.conjugate(metric.T), atol=1e-6)
