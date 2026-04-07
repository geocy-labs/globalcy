from __future__ import annotations

import jax
import jax.numpy as jnp

from globalcy.geometry.kahler import fs_metric_affine


def fs_logdet_target(affine_real: jnp.ndarray) -> jnp.ndarray:
    n = affine_real.shape[0] // 2
    z = affine_real[:n] + 1j * affine_real[n:]
    metric = fs_metric_affine(z)
    determinant = jnp.real(jnp.linalg.det(metric))
    return jnp.log(jnp.maximum(determinant, 1e-8))


v_fs_logdet_target = jax.vmap(fs_logdet_target)
