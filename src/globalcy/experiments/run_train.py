from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from globalcy.data.bundle_adapter import load_bundle_batch
from globalcy.geometry.invariants import invariant_features_from_affine
from globalcy.geometry.projective import homogeneous_to_affine_real, homogeneous_to_invariants
from globalcy.models.phi_global_invariant import apply_global_invariant_phi, init_global_invariant_phi
from globalcy.models.phi_local_mlp import apply_local_mlp, init_local_mlp
from globalcy.models.phi_symmetry_aware import apply_symmetry_aware_phi, init_symmetry_aware_phi
from globalcy.reports.export import write_json, write_predictions, write_summary
from globalcy.training.eval import evaluate_model
from globalcy.training.trainer import train_model
from globalcy.utils.config import load_config
from globalcy.utils.logging import configure_logging
from globalcy.utils.reproducibility import git_commit, package_versions


def _model_bundle(model_type: str):
    if model_type == "local":
        return init_local_mlp, apply_local_mlp
    if model_type == "global":
        return init_global_invariant_phi, apply_global_invariant_phi
    if model_type == "symmetry_aware":
        return init_symmetry_aware_phi, apply_symmetry_aware_phi
    raise ValueError(f"Unsupported model type: {model_type}")


def _features_for_model(batch, model_type: str):
    if model_type == "local":
        return batch.local_features, None
    if model_type == "global":
        return batch.invariant_features, None
    return batch.symmetry_features, batch.symmetry_features


def _transform_for_model(model_type: str):
    if model_type == "local":
        return lambda affine_real: affine_real
    return lambda affine_real: invariant_features_from_affine(affine_real[: affine_real.shape[0] // 2] + 1j * affine_real[affine_real.shape[0] // 2 :], chart_id=0)


def _projective_feature_builder(model_type: str):
    if model_type == "local":
        return lambda homogeneous: homogeneous_to_affine_real(homogeneous, chart_id=0)
    return homogeneous_to_invariants


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[3]
    bundle_path = repo_root.joinpath(config["bundle_path"]).resolve()
    batch = load_bundle_batch(bundle_path)
    model_type = config["model"]["type"]
    init_fn, apply_fn = _model_bundle(model_type)
    features, symmetry_features = _features_for_model(batch, model_type)

    rng = jax.random.PRNGKey(config["training"]["seed"])
    params = init_fn(rng, int(features.shape[-1]), config["model"]["hidden_dims"])

    affine_real = batch.local_features
    result = train_model(
        model_apply=apply_fn,
        params=params,
        features=features,
        affine_real=affine_real,
        weights=batch.weights,
        config=config,
        transform_inputs=_transform_for_model(model_type),
        symmetry_features=symmetry_features,
    )
    diagnostics = evaluate_model(
        model_apply=apply_fn,
        params=result.params,
        features=features,
        affine_real=affine_real,
        homogeneous=batch.homogeneous[..., 0] + 1j * batch.homogeneous[..., 1],
        point_ids=jnp.asarray(batch.point_ids),
        transform_inputs=_transform_for_model(model_type),
        projective_feature_builder=_projective_feature_builder(model_type),
        symmetry_features=symmetry_features,
    )

    run_dir = Path(args.out or Path("outputs") / config["run_name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": model_type,
        "train_loss": result.train_loss,
        "runtime_seconds": result.runtime_seconds,
        **batch.metadata,
        **diagnostics,
    }
    provenance = {
        "config": config,
        "git_commit": git_commit(repo_root),
        "package_versions": package_versions(),
        "seed": config["training"]["seed"],
    }

    write_json(run_dir / "config.json", provenance)
    write_json(run_dir / "metrics.json", metrics)
    np.savez(run_dir / "params.npz", *jax.tree_util.tree_leaves(result.params))
    write_predictions(run_dir / "predictions.parquet", batch.point_ids, np.asarray(result.predictions), np.asarray(result.targets))
    write_summary(
        run_dir / "summary.md",
        [
            f"# {config['run_name']}",
            "",
            f"- model: `{model_type}`",
            f"- geometry: `{batch.metadata['geometry']}`",
            f"- case_id: `{batch.metadata['case_id']}`",
            f"- train_loss: `{result.train_loss:.6f}`",
            f"- min_eigenvalue_mean: `{metrics['min_eigenvalue_mean']:.6f}`",
            f"- projective_invariance_drift: `{metrics['projective_invariance_drift']:.6f}`",
        ],
    )
    print(f"Wrote run outputs to {run_dir}")


if __name__ == "__main__":
    main()
