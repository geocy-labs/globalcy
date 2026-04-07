from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from globalcy.geometry.projective import normalize_homogeneous, rescale_points


def chart_consistency_score(predictions: jnp.ndarray, point_ids: jnp.ndarray) -> float:
    prediction_array = np.asarray(predictions)
    point_id_array = np.asarray(point_ids)
    if prediction_array.size == 0:
        return 0.0
    group_values: list[float] = []
    for point_id in np.unique(point_id_array):
        values = prediction_array[point_id_array == point_id]
        group_values.append(float(np.std(values)) if values.size > 1 else 0.0)
    return float(np.mean(group_values))


def projective_invariance_drift(
    model_apply,
    params,
    homogeneous_batch: jnp.ndarray,
    feature_builder,
    scale: complex = 1.7 + 0.0j,
) -> float:
    def one_drift(homogeneous: jnp.ndarray) -> jnp.ndarray:
        original = feature_builder(normalize_homogeneous(homogeneous))
        rescaled = feature_builder(normalize_homogeneous(rescale_points(homogeneous, scale)))
        return jnp.abs(model_apply(params, original[None, :])[0] - model_apply(params, rescaled[None, :])[0])

    return float(jnp.mean(jax.vmap(one_drift)(homogeneous_batch)))


def symmetry_consistency_score(predictions: jnp.ndarray, symmetry_predictions: jnp.ndarray) -> float:
    if predictions.size == 0:
        return 0.0
    return float(jnp.mean(jnp.abs(predictions - symmetry_predictions)))
