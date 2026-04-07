from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from globalcy.diagnostics.characteristic_forms import characteristic_summary
from globalcy.diagnostics.consistency import chart_consistency_score, projective_invariance_drift, symmetry_consistency_score
from globalcy.diagnostics.positivity import positivity_summary
from globalcy.geometry.kahler import kahler_metric


def evaluate_model(
    model_apply: Callable,
    params,
    features: jnp.ndarray,
    affine_real: jnp.ndarray,
    homogeneous: jnp.ndarray,
    point_ids: jnp.ndarray,
    transform_inputs: Callable,
    projective_feature_builder: Callable,
    symmetry_features: jnp.ndarray | None = None,
) -> dict[str, float]:
    predictions = model_apply(params, features)
    metrics = jax.vmap(lambda point: kahler_metric(lambda x: model_apply(params, transform_inputs(x)[None, :])[0], point))(affine_real)
    summary = {}
    summary.update(positivity_summary(metrics))
    summary.update(characteristic_summary(metrics))
    summary["chart_consistency"] = chart_consistency_score(predictions, point_ids)
    summary["projective_invariance_drift"] = projective_invariance_drift(model_apply, params, homogeneous, projective_feature_builder)
    if symmetry_features is not None:
        symmetry_predictions = model_apply(params, symmetry_features)
        summary["symmetry_consistency"] = symmetry_consistency_score(predictions, symmetry_predictions)
    return summary
