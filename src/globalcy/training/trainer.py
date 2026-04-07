from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp

from globalcy.geometry.pullbacks import v_fs_logdet_target
from globalcy.training.losses import make_loss_fn, weighted_mse


@dataclass(slots=True)
class TrainResult:
    params: object
    predictions: jnp.ndarray
    targets: jnp.ndarray
    train_loss: float
    runtime_seconds: float


def _adam_init(params):
    zeros_like = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros_like, "v": zeros_like, "t": jnp.array(0)}


def _adam_step(params, grads, state, learning_rate: float):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state["m"], grads)
    v = jax.tree_util.tree_map(lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad), state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**t), m)
    vhat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**t), v)
    new_params = jax.tree_util.tree_map(lambda param, m_value, v_value: param - learning_rate * m_value / (jnp.sqrt(v_value) + eps), params, mhat, vhat)
    return new_params, {"m": m, "v": v, "t": t}


def train_model(
    model_apply,
    params,
    features: jnp.ndarray,
    affine_real: jnp.ndarray,
    weights: jnp.ndarray,
    config: dict,
    transform_inputs,
    symmetry_features: jnp.ndarray | None = None,
) -> TrainResult:
    targets = v_fs_logdet_target(affine_real)
    loss_fn = make_loss_fn(model_apply, affine_real, targets, weights, config["loss"], transform_inputs=transform_inputs)
    state = _adam_init(params)
    start = perf_counter()

    for _ in range(config["training"]["epochs"]):
        (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, features, symmetry_features)
        params, state = _adam_step(params, grads, state, config["training"]["learning_rate"])

    predictions = model_apply(params, features)
    runtime_seconds = perf_counter() - start
    final_loss = float(weighted_mse(predictions, targets, weights))
    return TrainResult(
        params=params,
        predictions=predictions,
        targets=targets,
        train_loss=final_loss,
        runtime_seconds=runtime_seconds,
    )
