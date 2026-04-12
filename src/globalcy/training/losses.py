from __future__ import annotations

import jax
import jax.numpy as jnp

from globalcy.geometry.kahler import kahler_metric
from globalcy.geometry.projective import normalize_homogeneous, rescale_points


def weighted_mse(predictions: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    residual = predictions - targets
    return jnp.sum(weights * residual * residual) / jnp.maximum(jnp.sum(weights), 1e-8)


def positivity_penalty(metrics: jnp.ndarray) -> jnp.ndarray:
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(metrics)
    return jnp.mean(jnp.maximum(0.0, -jnp.real(eigenvalues[:, 0])))


def determinant_penalty(metrics: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    determinants = jnp.real(jnp.linalg.det(metrics))
    target_scale = jnp.exp(targets)
    return jnp.mean((determinants - target_scale) ** 2)


def chart_penalty(predictions: jnp.ndarray) -> jnp.ndarray:
    return jnp.var(predictions)


def symmetry_penalty(predictions: jnp.ndarray, symmetry_predictions: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((predictions - symmetry_predictions) ** 2)


def projective_consistency_penalty(
    model_apply,
    params,
    homogeneous: jnp.ndarray,
    projective_feature_builder,
    *,
    scale: complex = 1.7 + 0.0j,
) -> jnp.ndarray:
    def one_drift(point: jnp.ndarray) -> jnp.ndarray:
        original = projective_feature_builder(normalize_homogeneous(point))
        rescaled = projective_feature_builder(normalize_homogeneous(rescale_points(point, scale)))
        return jnp.abs(model_apply(params, original[None, :])[0] - model_apply(params, rescaled[None, :])[0])

    return jnp.mean(jax.vmap(one_drift)(homogeneous))


def make_loss_fn(
    model_apply,
    affine_real: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray,
    loss_config: dict[str, float],
    transform_inputs=None,
    projective_homogeneous: jnp.ndarray | None = None,
    projective_feature_builder=None,
):
    negativity_weight = float(loss_config.get("negativity_weight", loss_config.get("positivity_weight", 0.0)))
    projective_weight = float(loss_config.get("projective_consistency_weight", 0.0))
    projective_scale = complex(loss_config.get("projective_consistency_scale", 1.7))

    def loss_fn(params, features: jnp.ndarray, symmetry_features: jnp.ndarray | None = None):
        predictions = model_apply(params, features)
        metric_fn = lambda point: kahler_metric(lambda x: model_apply(params, transform_inputs(x)[None, :])[0], point)
        metrics = jax.vmap(metric_fn)(affine_real)
        loss = loss_config["target_weight"] * weighted_mse(predictions, targets, weights)
        loss = loss + loss_config["determinant_weight"] * determinant_penalty(metrics, targets)
        loss = loss + negativity_weight * positivity_penalty(metrics)
        if projective_homogeneous is not None and projective_feature_builder is not None and projective_weight > 0.0:
            loss = loss + projective_weight * projective_consistency_penalty(
                model_apply,
                params,
                projective_homogeneous,
                projective_feature_builder,
                scale=projective_scale,
            )
        if symmetry_features is not None:
            symmetry_predictions = model_apply(params, symmetry_features)
            loss = loss + loss_config["symmetry_weight"] * symmetry_penalty(predictions, symmetry_predictions)
        else:
            loss = loss + loss_config["chart_weight"] * chart_penalty(predictions)
        return loss, {"predictions": predictions, "metrics": metrics}

    return loss_fn
