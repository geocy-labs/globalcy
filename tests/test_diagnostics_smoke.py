from __future__ import annotations

import jax.numpy as jnp

from globalcy.diagnostics.characteristic_forms import characteristic_summary
from globalcy.diagnostics.positivity import positivity_summary
from pathlib import Path

from globalcy.experiments.run_compare import compare_metrics, main as run_compare_main


def test_diagnostics_smoke():
    metrics = jnp.stack([jnp.eye(3, dtype=jnp.complex64), 1.2 * jnp.eye(3, dtype=jnp.complex64)], axis=0)
    positivity = positivity_summary(metrics)
    characteristic = characteristic_summary(metrics)
    assert positivity["negative_fraction"] == 0.0
    assert characteristic["determinant_mean"] > 0.0


def test_model_comparison_diagnostics_smoke():
    frame, aggregated, summary, lines = compare_metrics(
        [
            {
                "run_dir": "local",
                "seed": 7,
                "model": "local",
                "geometry": "cefalu_quartic",
                "case_id": "cefalu_lambda_0_75",
                "lambda": 0.75,
                "train_loss": 1.0,
                "negative_fraction": 0.2,
                "chart_consistency": 0.3,
                "projective_invariance_drift": 0.4,
                "symmetry_consistency": 0.5,
                "min_eigenvalue_mean": 0.1,
                "determinant_mean": 1.2,
                "euler_proxy": 0.7,
                "runtime_seconds": 2.0,
            },
            {
                "run_dir": "global",
                "seed": 7,
                "model": "global",
                "geometry": "cefalu_quartic",
                "case_id": "cefalu_lambda_0_75",
                "lambda": 0.75,
                "train_loss": 0.9,
                "negative_fraction": 0.1,
                "chart_consistency": 0.2,
                "projective_invariance_drift": 0.1,
                "symmetry_consistency": 0.2,
                "min_eigenvalue_mean": 0.2,
                "determinant_mean": 1.1,
                "euler_proxy": 0.6,
                "runtime_seconds": 2.2,
            },
        ]
    )
    assert list(frame["model"]) == ["global", "local"]
    assert aggregated.loc[aggregated["model"] == "global", "projective_invariance_drift_mean"].iloc[0] == 0.1
    assert summary["best_by_metric"]["projective_invariance_drift"] == "global"
    assert any("best `projective_invariance_drift`" in line for line in lines)


def test_multiseed_aggregation_smoke():
    _, aggregated, summary, _ = compare_metrics(
        [
            {
                "run_dir": "seed7_local",
                "seed": 7,
                "model": "local",
                "geometry": "cefalu_quartic",
                "case_id": "cefalu_lambda_1_0",
                "lambda": 1.0,
                "train_loss": 1.2,
                "negative_fraction": 0.2,
                "chart_consistency": 0.1,
                "projective_invariance_drift": 0.4,
                "symmetry_consistency": 0.0,
                "min_eigenvalue_mean": 0.1,
                "determinant_mean": 1.0,
                "euler_proxy": 0.7,
                "runtime_seconds": 2.0,
            },
            {
                "run_dir": "seed11_local",
                "seed": 11,
                "model": "local",
                "geometry": "cefalu_quartic",
                "case_id": "cefalu_lambda_1_0",
                "lambda": 1.0,
                "train_loss": 1.0,
                "negative_fraction": 0.1,
                "chart_consistency": 0.2,
                "projective_invariance_drift": 0.3,
                "symmetry_consistency": 0.0,
                "min_eigenvalue_mean": 0.15,
                "determinant_mean": 1.1,
                "euler_proxy": 0.6,
                "runtime_seconds": 2.3,
            },
        ]
    )
    assert aggregated["seed_count"].iloc[0] == 2
    assert summary["stability"]["projective_invariance_drift"]["mean"] > 0.0


def test_comparison_output_files(tmp_path: Path, monkeypatch):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    out_dir = tmp_path / "comparison"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "metrics.json").write_text(
        """{"seed": 7, "model": "local", "geometry": "cefalu_quartic", "case_id": "cefalu_lambda_0_75", "lambda": 0.75, "train_loss": 1.0, "negative_fraction": 0.2, "chart_consistency": 0.1, "projective_invariance_drift": 0.3, "symmetry_consistency": 0.0, "min_eigenvalue_mean": 0.1, "determinant_mean": 1.0, "euler_proxy": 0.2, "runtime_seconds": 2.0}""",
        encoding="utf-8",
    )
    (run_b / "metrics.json").write_text(
        """{"seed": 11, "model": "global", "geometry": "cefalu_quartic", "case_id": "cefalu_lambda_0_75", "lambda": 0.75, "train_loss": 0.8, "negative_fraction": 0.1, "chart_consistency": 0.05, "projective_invariance_drift": 0.1, "symmetry_consistency": 0.2, "min_eigenvalue_mean": 0.15, "determinant_mean": 1.1, "euler_proxy": 0.25, "runtime_seconds": 2.1}""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_compare",
            "--run-dir",
            str(run_a),
            "--run-dir",
            str(run_b),
            "--out",
            str(out_dir),
        ],
    )
    run_compare_main()
    assert (out_dir / "comparison.csv").exists()
    assert (out_dir / "comparison_aggregated.csv").exists()
    assert (out_dir / "comparison_aggregated.md").exists()
