from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_config(path: Path, *, bundle_path: str, model_type: str) -> None:
    path.write_text(
        "\n".join(
            [
                "run_name: pointwise_diag_test",
                f"bundle_path: \"{bundle_path}\"",
                "training:",
                "  seed: 7",
                "  epochs: 2",
                "  batch_size: 4",
                "  learning_rate: 0.001",
                "  train_fraction: 0.8",
                "model:",
                f"  type: {model_type}",
                "  hidden_dims: [8, 8]",
                "loss:",
                "  target_weight: 1.0",
                "  determinant_weight: 0.1",
                "  positivity_weight: 0.05",
                "  chart_weight: 0.0",
                "  symmetry_weight: 0.05",
                "evaluation:",
                "  projective_scale: 1.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_train_exports_pointwise_diagnostics_for_local_and_global(synthetic_bundle: Path, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_path = synthetic_bundle.as_posix()
    required_columns = {
        "point_id",
        "case_id",
        "model_name",
        "seed",
        "prediction",
        "target",
        "min_eigenvalue",
        "logdet_g",
        "quadrature_weight",
        "logdet_target",
        "logdet_residual_proxy",
        "negative_flag",
        "euler_density_proxy",
    }
    for model_type in ("local", "global"):
        config_path = tmp_path / f"{model_type}.yaml"
        out_dir = tmp_path / model_type
        _write_config(config_path, bundle_path=bundle_path, model_type=model_type)
        subprocess.run(
            [sys.executable, "-m", "globalcy.experiments.run_train", "--config", str(config_path), "--out", str(out_dir)],
            check=True,
            cwd=repo_root,
        )
        pointwise_path = out_dir / "pointwise_diagnostics.parquet"
        assert pointwise_path.exists()
        frame = pd.read_parquet(pointwise_path)
        assert required_columns.issubset(frame.columns)
        assert set(frame["model_name"]) == {model_type}
        assert set(frame["case_id"]) == {"fermat_quartic"}
        assert len(frame) == 6
