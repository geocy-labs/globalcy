from __future__ import annotations

import jax.numpy as jnp


def characteristic_summary(metrics: jnp.ndarray) -> dict[str, float]:
    determinants = jnp.real(jnp.linalg.det(metrics))
    return {
        "determinant_mean": float(jnp.mean(determinants)),
        "determinant_std": float(jnp.std(determinants)),
        "euler_proxy": float(jnp.mean(jnp.log(jnp.maximum(determinants, 1e-8)))),
    }
