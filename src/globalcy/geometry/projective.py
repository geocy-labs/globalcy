from __future__ import annotations

import jax.numpy as jnp


def affine_to_homogeneous(affine_complex: jnp.ndarray, chart_id: int = 0) -> jnp.ndarray:
    n = affine_complex.shape[-1] + 1
    components = []
    source_idx = 0
    for target_idx in range(n):
        if target_idx == chart_id:
            components.append(jnp.ones((), dtype=affine_complex.dtype))
        else:
            components.append(affine_complex[source_idx])
            source_idx += 1
    return jnp.stack(components, axis=0)


def normalize_homogeneous(homogeneous: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(homogeneous)
    return homogeneous / jnp.maximum(norm, 1e-8)


def rescale_points(homogeneous: jnp.ndarray, scale: complex) -> jnp.ndarray:
    return homogeneous * jnp.asarray(scale, dtype=homogeneous.dtype)


def homogeneous_to_invariants(homogeneous: jnp.ndarray) -> jnp.ndarray:
    normalized = normalize_homogeneous(homogeneous)
    outer = normalized[:, None] * jnp.conjugate(normalized[None, :])
    return jnp.concatenate([jnp.real(outer).reshape(-1), jnp.imag(outer).reshape(-1)], axis=0)


def homogeneous_to_affine_real(homogeneous: jnp.ndarray, chart_id: int = 0) -> jnp.ndarray:
    pivot = homogeneous[chart_id]
    coords = []
    for idx in range(homogeneous.shape[0]):
        if idx == chart_id:
            continue
        value = homogeneous[idx] / pivot
        coords.extend([jnp.real(value), jnp.imag(value)])
    return jnp.asarray(coords, dtype=jnp.float32)
