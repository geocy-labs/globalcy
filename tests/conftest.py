from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    n = 6
    point_ids = np.arange(n)
    chart_ids = np.array([0, 1, 0, 1, 0, 1])
    affine = np.stack(
        [
            np.linspace(0.1, 0.6, n),
            np.linspace(-0.2, 0.3, n),
            np.linspace(0.05, 0.3, n),
            np.linspace(0.2, -0.1, n),
            np.linspace(-0.1, 0.2, n),
            np.linspace(0.15, -0.15, n),
        ],
        axis=1,
    )
    homogeneous = np.stack(
        [
            np.ones(n),
            affine[:, 0],
            affine[:, 2],
            affine[:, 4],
            np.zeros(n),
            affine[:, 1],
            affine[:, 3],
            affine[:, 5],
        ],
        axis=1,
    )

    points = pd.DataFrame(
        {
            "point_id": point_ids,
            "chart_id": chart_ids,
            "geometry": ["fermat_quartic"] * n,
            "case_id": ["fermat_quartic"] * n,
            "seed": [7] * n,
            "z0_re": homogeneous[:, 0],
            "z0_im": homogeneous[:, 4],
            "z1_re": homogeneous[:, 1],
            "z1_im": homogeneous[:, 5],
            "z2_re": homogeneous[:, 2],
            "z2_im": homogeneous[:, 6],
            "z3_re": homogeneous[:, 3],
            "z3_im": homogeneous[:, 7],
            "affine_0_re": affine[:, 0],
            "affine_0_im": affine[:, 1],
            "affine_1_re": affine[:, 2],
            "affine_1_im": affine[:, 3],
            "affine_2_re": affine[:, 4],
            "affine_2_im": affine[:, 5],
        }
    )
    points.to_parquet(bundle / "points.parquet", index=False)

    invariant_data: dict[str, np.ndarray] = {"point_id": point_ids}
    for i in range(4):
        for j in range(4):
            invariant_data[f"m_{i}_{j}_re"] = np.full(n, 0.1 * (i + j + 1), dtype=np.float32)
            invariant_data[f"m_{i}_{j}_im"] = np.zeros(n, dtype=np.float32)
    invariants = pd.DataFrame(invariant_data)
    invariants["geometry"] = "fermat_quartic"
    invariants["case_id"] = "fermat_quartic"
    invariants["seed"] = 7
    invariants.to_parquet(bundle / "invariants.parquet", index=False)

    weights = pd.DataFrame(
        {
            "point_id": point_ids,
            "geometry": ["fermat_quartic"] * n,
            "case_id": ["fermat_quartic"] * n,
            "seed": [7] * n,
            "combined_weight": np.ones(n, dtype=np.float32),
        }
    )
    weights.to_parquet(bundle / "sample_weights.parquet", index=False)

    manifest = {
        "geometry": "fermat_quartic",
        "case_id": "fermat_quartic",
        "seed": 7,
        "parameters": {"seed": 7},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "case_metadata.json").write_text(json.dumps({"case_id": "fermat_quartic", "geometry": "fermat_quartic", "seed": 7, "n": n}), encoding="utf-8")
    (bundle / "evaluation_summary.json").write_text(json.dumps({"chart_consistency": {}, "projective_invariance_drift": {}}), encoding="utf-8")
    return bundle
