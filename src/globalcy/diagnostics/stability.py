from __future__ import annotations

import jax.numpy as jnp


def seed_stability_summary(metric_values: jnp.ndarray) -> dict[str, float]:
    return {
        "mean": float(jnp.mean(metric_values)),
        "std": float(jnp.std(metric_values)),
        "min": float(jnp.min(metric_values)),
        "max": float(jnp.max(metric_values)),
    }
