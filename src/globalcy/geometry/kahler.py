from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def fs_metric_affine(z: jnp.ndarray) -> jnp.ndarray:
    norm_sq = jnp.real(jnp.vdot(z, z))
    scale = 1.0 + norm_sq
    identity = jnp.eye(z.shape[0], dtype=jnp.complex64)
    outer = jnp.conjugate(z)[:, None] * z[None, :]
    return ((scale * identity) - outer) / (scale**2)


def _complex_hessian_from_real_hessian(real_hessian: jnp.ndarray) -> jnp.ndarray:
    n = real_hessian.shape[0] // 2
    xx = real_hessian[:n, :n]
    xy = real_hessian[:n, n:]
    yx = real_hessian[n:, :n]
    yy = real_hessian[n:, n:]
    return 0.25 * (xx + yy + 1j * (yx - xy))


def kahler_metric(phi_fn: Callable[[jnp.ndarray], jnp.ndarray], affine_real: jnp.ndarray) -> jnp.ndarray:
    n = affine_real.shape[0] // 2
    affine_complex = affine_real[:n] + 1j * affine_real[n:]
    fs = fs_metric_affine(affine_complex)
    real_hessian = jax.hessian(phi_fn)(affine_real)
    correction = _complex_hessian_from_real_hessian(real_hessian)
    return fs + correction


def metric_statistics(metric: jnp.ndarray) -> dict[str, jnp.ndarray]:
    hermitian_residual = jnp.max(jnp.abs(metric - jnp.conjugate(metric.T)))
    eigenvalues = jnp.linalg.eigvalsh(metric)
    determinant = jnp.real(jnp.linalg.det(metric))
    return {
        "determinant": determinant,
        "min_eigenvalue": jnp.min(jnp.real(eigenvalues)),
        "max_eigenvalue": jnp.max(jnp.real(eigenvalues)),
        "hermitian_residual": hermitian_residual,
    }
