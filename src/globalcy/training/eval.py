from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from globalcy.diagnostics.characteristic_forms import characteristic_summary
from globalcy.diagnostics.consistency import chart_consistency_score, projective_invariance_drift, symmetry_consistency_score
from globalcy.diagnostics.positivity import positivity_summary
from globalcy.geometry.kahler import kahler_metric


@dataclass(slots=True)
class EvaluationResult:
    summary: dict[str, float]
    pointwise: pd.DataFrame


def evaluate_model(
    model_apply: Callable,
    params,
    features: jnp.ndarray,
    targets: jnp.ndarray,
    affine_real: jnp.ndarray,
    homogeneous: jnp.ndarray,
    point_ids: jnp.ndarray,
    weights: jnp.ndarray,
    transform_inputs: Callable,
    projective_feature_builder: Callable,
    *,
    case_id: str,
    model_name: str,
    seed: int,
    symmetry_features: jnp.ndarray | None = None,
) -> EvaluationResult:
    predictions = model_apply(params, features)
    metrics = jax.vmap(lambda point: kahler_metric(lambda x: model_apply(params, transform_inputs(x)[None, :])[0], point))(affine_real)
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(metrics)
    min_eigenvalues = jnp.real(eigenvalues[:, 0])
    determinants = jnp.real(jnp.linalg.det(metrics))
    clipped_determinants = jnp.maximum(determinants, 1e-8)
    logdet_g = jnp.log(clipped_determinants)
    summary = {}
    summary.update(positivity_summary(metrics))
    summary.update(characteristic_summary(metrics))
    summary["chart_consistency"] = chart_consistency_score(predictions, point_ids)
    summary["projective_invariance_drift"] = projective_invariance_drift(model_apply, params, homogeneous, projective_feature_builder)
    if symmetry_features is not None:
        symmetry_predictions = model_apply(params, symmetry_features)
        summary["symmetry_consistency"] = symmetry_consistency_score(predictions, symmetry_predictions)
    else:
        summary["symmetry_consistency"] = 0.0
    pointwise = pd.DataFrame(
        {
            "point_id": np.asarray(point_ids),
            "case_id": [case_id] * int(point_ids.shape[0]),
            "model_name": [model_name] * int(point_ids.shape[0]),
            "seed": [int(seed)] * int(point_ids.shape[0]),
            "prediction": np.asarray(predictions),
            "target": np.asarray(targets),
            "min_eigenvalue": np.asarray(min_eigenvalues),
            "determinant_g": np.asarray(determinants),
            "logdet_g": np.asarray(logdet_g),
            "quadrature_weight": np.asarray(weights),
        }
    )
    pointwise["logdet_target"] = np.asarray(targets)
    pointwise["logdet_residual_proxy"] = pointwise["logdet_g"] - pointwise["logdet_target"]
    pointwise["negative_flag"] = pointwise["min_eigenvalue"] <= 0.0
    pointwise["euler_density_proxy"] = pointwise["logdet_g"]
    return EvaluationResult(summary=summary, pointwise=pointwise)
