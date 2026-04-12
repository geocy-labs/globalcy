from __future__ import annotations

import jax
import jax.numpy as jnp


def positivity_summary(metrics: jnp.ndarray) -> dict[str, float]:
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(metrics)
    min_eigs = jnp.real(eigenvalues[:, 0])
    sample_count = int(min_eigs.shape[0])
    tail_count = max((sample_count + 9) // 10, 1)
    spectral_tail = jnp.sort(min_eigs)[:tail_count]
    return {
        "min_eigenvalue_mean": float(jnp.mean(min_eigs)),
        "min_eigenvalue_min": float(jnp.min(min_eigs)),
        "negative_fraction": float(jnp.mean(min_eigs <= 0.0)),
        "spectral_tail_mean": float(jnp.mean(spectral_tail)),
    }
