from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from globalcy.data.loaders import load_bundle_tables


def _sorted_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([column for column in frame.columns if column.startswith(prefix)])


def _paired_complex_columns(frame: pd.DataFrame, prefix: str, count: int) -> list[str]:
    columns: list[str] = []
    for idx in range(count):
        columns.extend([f"{prefix}{idx}_re", f"{prefix}{idx}_im"])
    return [column for column in columns if column in frame.columns]


def _real_feature_block(frame: pd.DataFrame, prefixes: list[str]) -> np.ndarray:
    columns: list[str] = []
    for prefix in prefixes:
        columns.extend(_sorted_columns(frame, prefix))
    return frame[columns].to_numpy(dtype=np.float32)


@dataclass(slots=True)
class BundleBatch:
    local_features: jnp.ndarray
    invariant_features: jnp.ndarray
    symmetry_features: jnp.ndarray
    weights: jnp.ndarray
    point_ids: np.ndarray
    chart_ids: np.ndarray
    homogeneous: jnp.ndarray
    affine_complex: jnp.ndarray
    metadata: dict[str, Any]
    evaluation_summary: dict[str, Any]


def load_bundle_batch(bundle_path: str | Path) -> BundleBatch:
    tables = load_bundle_tables(bundle_path)
    points = tables["points"].sort_values("point_id").reset_index(drop=True)
    invariants = tables["invariants"].sort_values("point_id").reset_index(drop=True)

    local_columns = _paired_complex_columns(points, "affine_", 3)
    homogeneous_columns = _paired_complex_columns(points, "z", 4)
    local_features = points[local_columns].to_numpy(dtype=np.float32)
    homogeneous_features = points[homogeneous_columns].to_numpy(dtype=np.float32)
    affine_pairs = local_features.reshape(len(points), -1, 2)
    affine_complex = affine_pairs[..., 0] + 1j * affine_pairs[..., 1]

    invariant_features = _real_feature_block(invariants, ["m_"])
    canonical_frame = tables.get("canonical_invariants", invariants).sort_values("point_id").reset_index(drop=True)
    symmetry_features = _real_feature_block(canonical_frame, ["m_"])

    if "sample_weights" in tables:
        weights_frame = tables["sample_weights"].sort_values("point_id").reset_index(drop=True)
        weight_array = weights_frame.get("combined_weight", pd.Series(np.ones(len(points), dtype=np.float32))).to_numpy(dtype=np.float32)
    else:
        weight_array = np.ones(len(points), dtype=np.float32)

    manifest = tables["manifest"]
    case_metadata = tables.get("case_metadata", {})
    metadata = {
        "bundle_path": str(Path(bundle_path).resolve()),
        "geometry": manifest.get("geometry"),
        "case_id": manifest.get("case_id"),
        "seed": manifest.get("seed", manifest.get("parameters", {}).get("seed")),
        "lambda": manifest.get("parameters", {}).get("lambda", case_metadata.get("lambda")),
        "chart_count": int(points["chart_id"].nunique()),
        "point_count": int(len(points)),
    }

    return BundleBatch(
        local_features=jnp.asarray(local_features),
        invariant_features=jnp.asarray(invariant_features),
        symmetry_features=jnp.asarray(symmetry_features),
        weights=jnp.asarray(weight_array),
        point_ids=points["point_id"].to_numpy(),
        chart_ids=points["chart_id"].to_numpy(),
        homogeneous=jnp.asarray(homogeneous_features.reshape(len(points), -1, 2)),
        affine_complex=jnp.asarray(affine_complex),
        metadata=metadata,
        evaluation_summary=tables.get("evaluation_summary", {}),
    )
