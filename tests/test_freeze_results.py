from __future__ import annotations

from pathlib import Path

import pandas as pd

from globalcy.experiments.freeze_results import freeze_results


def _write_comparison_dir(base: Path, case_id: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        [
            {
                "run_dir": f"{case_id}_seed7_local",
                "seed": 7,
                "model": "local",
                "geometry": "cefalu_quartic",
                "case_id": case_id,
                "lambda": 0.75 if case_id.endswith("0_75") else 1.0,
                "train_loss": 1.2,
                "min_eigenvalue_mean": 0.1,
                "negative_fraction": 0.2,
                "chart_consistency": 0.0,
                "projective_invariance_drift": 0.4,
                "symmetry_consistency": 0.0,
                "determinant_mean": 1.0,
                "euler_proxy": 0.5,
                "runtime_seconds": 2.0,
            },
            {
                "run_dir": f"{case_id}_seed11_global",
                "seed": 11,
                "model": "global",
                "geometry": "cefalu_quartic",
                "case_id": case_id,
                "lambda": 0.75 if case_id.endswith("0_75") else 1.0,
                "train_loss": 1.0,
                "min_eigenvalue_mean": 0.2,
                "negative_fraction": 0.1,
                "chart_consistency": 0.0,
                "projective_invariance_drift": 0.1,
                "symmetry_consistency": 0.2,
                "determinant_mean": 1.1,
                "euler_proxy": 0.6,
                "runtime_seconds": 2.1,
            },
        ]
    )
    comparison.to_csv(base / "comparison.csv", index=False)

    aggregated = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "geometry": "cefalu_quartic",
                "lambda": 0.75 if case_id.endswith("0_75") else 1.0,
                "model": "local",
                "train_loss_mean": 1.2,
                "train_loss_std": 0.1,
                "min_eigenvalue_mean_mean": 0.1,
                "min_eigenvalue_mean_std": 0.01,
                "negative_fraction_mean": 0.2,
                "negative_fraction_std": 0.02,
                "chart_consistency_mean": 0.0,
                "chart_consistency_std": 0.0,
                "projective_invariance_drift_mean": 0.4,
                "projective_invariance_drift_std": 0.03,
                "symmetry_consistency_mean": 0.0,
                "symmetry_consistency_std": 0.0,
                "determinant_mean_mean": 1.0,
                "determinant_mean_std": 0.02,
                "euler_proxy_mean": 0.5,
                "euler_proxy_std": 0.03,
                "runtime_seconds_mean": 2.0,
                "runtime_seconds_std": 0.1,
                "seed_count": 3,
            },
            {
                "case_id": case_id,
                "geometry": "cefalu_quartic",
                "lambda": 0.75 if case_id.endswith("0_75") else 1.0,
                "model": "global",
                "train_loss_mean": 1.0,
                "train_loss_std": 0.1,
                "min_eigenvalue_mean_mean": 0.2,
                "min_eigenvalue_mean_std": 0.02,
                "negative_fraction_mean": 0.1,
                "negative_fraction_std": 0.01,
                "chart_consistency_mean": 0.0,
                "chart_consistency_std": 0.0,
                "projective_invariance_drift_mean": 0.1,
                "projective_invariance_drift_std": 0.01,
                "symmetry_consistency_mean": 0.2,
                "symmetry_consistency_std": 0.02,
                "determinant_mean_mean": 1.1,
                "determinant_mean_std": 0.03,
                "euler_proxy_mean": 0.6,
                "euler_proxy_std": 0.04,
                "runtime_seconds_mean": 2.1,
                "runtime_seconds_std": 0.1,
                "seed_count": 3,
            },
            {
                "case_id": case_id,
                "geometry": "cefalu_quartic",
                "lambda": 0.75 if case_id.endswith("0_75") else 1.0,
                "model": "symmetry_aware",
                "train_loss_mean": 1.3,
                "train_loss_std": 0.2,
                "min_eigenvalue_mean_mean": 0.15,
                "min_eigenvalue_mean_std": 0.01,
                "negative_fraction_mean": 0.12,
                "negative_fraction_std": 0.02,
                "chart_consistency_mean": 0.0,
                "chart_consistency_std": 0.0,
                "projective_invariance_drift_mean": 0.12,
                "projective_invariance_drift_std": 0.01,
                "symmetry_consistency_mean": 0.1,
                "symmetry_consistency_std": 0.01,
                "determinant_mean_mean": 1.05,
                "determinant_mean_std": 0.02,
                "euler_proxy_mean": 0.55,
                "euler_proxy_std": 0.03,
                "runtime_seconds_mean": 2.4,
                "runtime_seconds_std": 0.1,
                "seed_count": 3,
            },
        ]
    )
    aggregated.to_csv(base / "comparison_aggregated.csv", index=False)
    return base


def test_freeze_results_outputs(tmp_path: Path):
    comparison_a = _write_comparison_dir(tmp_path / "case_a", "cefalu_lambda_0_75")
    comparison_b = _write_comparison_dir(tmp_path / "case_b", "cefalu_lambda_1_0")
    out_dir = tmp_path / "freeze"

    freeze_results([comparison_a, comparison_b], out_dir)

    expected_files = [
        "paper1_core_results.csv",
        "paper1_core_results.md",
        "paper1_robustness.csv",
        "paper1_robustness.md",
        "paper1_summary.md",
        "paper1_results.json",
        "fig_core_comparison.png",
        "fig_hardest_case.png",
    ]
    for name in expected_files:
        assert (out_dir / name).exists()

    core = pd.read_csv(out_dir / "paper1_core_results.csv")
    robustness = pd.read_csv(out_dir / "paper1_robustness.csv")
    assert set(core["model"]) == {"local", "global", "symmetry_aware"}
    assert robustness["seed_count"].min() == 3
