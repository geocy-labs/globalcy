from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from globalcy.experiments.generate_paper2_artifacts import generate_paper2_artifacts


def _write_regime_frozen(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    casewise = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "geometry": "cefalu_quartic",
                "lambda": lambda_value,
                "model": model,
                "seed_count": 3,
                "negative_fraction_mean": neg,
                "negative_fraction_std": 0.01,
                "projective_invariance_drift_mean": drift,
                "projective_invariance_drift_std": drift * 0.1,
                "min_eigenvalue_mean_mean": min_eig,
                "min_eigenvalue_mean_std": 0.01,
                "spectral_tail_mean_mean": tail,
                "spectral_tail_mean_std": 0.01,
                "train_loss_mean": loss,
                "train_loss_std": 0.1,
                "determinant_mean_mean": 1.0,
                "determinant_mean_std": 0.02,
                "euler_proxy_mean": 0.6 + lambda_value,
                "euler_proxy_std": 0.03,
                "runtime_seconds_mean": 2.0 + lambda_value,
                "runtime_seconds_std": 0.1,
            }
            for case_id, lambda_value, model, neg, drift, min_eig, tail, loss in [
                ("cefalu_lambda_0_50", 0.5, "local", 0.08, 2.5e-8, 0.13, 0.09, 8.2),
                ("cefalu_lambda_0_50", 0.5, "global", 0.05, 1.2e-8, 0.15, 0.11, 7.9),
                ("cefalu_lambda_0_75", 0.75, "local", 0.11, 3.0e-8, 0.12, 0.08, 8.5),
                ("cefalu_lambda_0_75", 0.75, "global", 0.07, 1.5e-8, 0.14, 0.10, 8.1),
                ("cefalu_lambda_0_90", 0.9, "local", 0.14, 3.8e-8, 0.11, 0.07, 8.8),
                ("cefalu_lambda_0_90", 0.9, "global", 0.09, 1.8e-8, 0.13, 0.09, 8.2),
                ("cefalu_lambda_1_0", 1.0, "local", 0.16, 4.2e-8, 0.10, 0.06, 9.0),
                ("cefalu_lambda_1_0", 1.0, "global", 0.11, 2.0e-8, 0.12, 0.08, 8.4),
                ("cefalu_lambda_1_10", 1.1, "local", 0.19, 5.0e-8, 0.09, 0.05, 9.4),
                ("cefalu_lambda_1_10", 1.1, "global", 0.13, 2.4e-8, 0.11, 0.07, 8.7),
            ]
        ]
    )
    sweep = (
        casewise.groupby("model", dropna=False)[
            ["negative_fraction_mean", "projective_invariance_drift_mean", "spectral_tail_mean_mean", "train_loss_mean"]
        ]
        .mean()
        .reset_index()
    )
    sweep["case_count"] = 5
    sweep["run_count"] = 15
    casewise.to_csv(base / "paper2_casewise_results.csv", index=False)
    sweep.to_csv(base / "paper2_sweep_results.csv", index=False)
    (base / "paper2_results.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (base / "paper2_summary.md").write_text("# summary\n", encoding="utf-8")
    return base


def _write_ablation_frozen(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "geometry": "cefalu_quartic",
                "lambda": lambda_value,
                "objective_variant": variant,
                "seed_count": 3,
                "negative_fraction_mean": neg,
                "negative_fraction_std": 0.01,
                "projective_invariance_drift_mean": drift,
                "projective_invariance_drift_std": drift * 0.1,
                "spectral_tail_mean_mean": tail,
                "spectral_tail_mean_std": 0.01,
                "train_loss_mean": loss,
                "train_loss_std": 0.1,
                "min_eigenvalue_mean_mean": 0.12,
                "determinant_mean_mean": 1.0,
                "euler_proxy_mean": 1.6,
                "chart_consistency_mean": 0.0,
                "runtime_seconds_mean": 2.0,
            }
            for case_id, lambda_value, variant, neg, drift, tail, loss in [
                ("cefalu_lambda_0_90", 0.9, "baseline", 0.11, 2.2e-8, 0.08, 8.3),
                ("cefalu_lambda_0_90", 0.9, "baseline_plus_negativity", 0.09, 2.2e-8, 0.09, 8.2),
                ("cefalu_lambda_0_90", 0.9, "baseline_plus_projective", 0.10, 1.6e-8, 0.08, 8.2),
                ("cefalu_lambda_0_90", 0.9, "baseline_plus_both", 0.08, 1.5e-8, 0.10, 8.1),
                ("cefalu_lambda_1_0", 1.0, "baseline", 0.13, 2.5e-8, 0.07, 8.5),
                ("cefalu_lambda_1_0", 1.0, "baseline_plus_negativity", 0.11, 2.4e-8, 0.08, 8.4),
                ("cefalu_lambda_1_0", 1.0, "baseline_plus_projective", 0.12, 1.8e-8, 0.07, 8.4),
                ("cefalu_lambda_1_0", 1.0, "baseline_plus_both", 0.10, 1.7e-8, 0.09, 8.2),
                ("cefalu_lambda_1_10", 1.1, "baseline", 0.15, 2.9e-8, 0.06, 8.8),
                ("cefalu_lambda_1_10", 1.1, "baseline_plus_negativity", 0.13, 2.8e-8, 0.07, 8.7),
                ("cefalu_lambda_1_10", 1.1, "baseline_plus_projective", 0.14, 2.0e-8, 0.06, 8.6),
                ("cefalu_lambda_1_10", 1.1, "baseline_plus_both", 0.11, 1.9e-8, 0.08, 8.4),
            ]
        ]
    )
    per_objective = (
        per_case.groupby("objective_variant", dropna=False)[
            ["negative_fraction_mean", "projective_invariance_drift_mean", "spectral_tail_mean_mean", "train_loss_mean"]
        ]
        .mean()
        .reset_index()
    )
    per_objective["case_count"] = 3
    hardest_case = per_case.loc[per_case["case_id"] == "cefalu_lambda_1_10"].reset_index(drop=True)
    per_case.to_csv(base / "paper2_ablation_results.csv", index=False)
    per_objective.to_csv(base / "per_objective_results.csv", index=False)
    hardest_case.to_csv(base / "hardest_case_ablation.csv", index=False)
    (base / "paper2_ablation_results.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (base / "paper2_ablation_summary.md").write_text("# ablation\n", encoding="utf-8")
    return base


def test_generate_paper2_artifacts_outputs(tmp_path: Path) -> None:
    regime_dir = _write_regime_frozen(tmp_path / "regime" / "frozen")
    ablation_dir = _write_ablation_frozen(tmp_path / "ablation" / "frozen")
    out_dir = tmp_path / "paper_artifacts"

    result = generate_paper2_artifacts(regime_dir, ablation_dir, out_dir=out_dir)

    figures_dir = Path(result["figures_dir"])
    tables_dir = Path(result["tables_dir"])
    expected_figures = [
        "fig_paper2_diagnostic_trajectories.png",
        "fig_paper2_degradation_profiles.png",
        "fig_paper2_objective_ablation.png",
        "fig_paper2_hardest_case.png",
    ]
    expected_tables = [
        "table_paper2_core_sweep_results.csv",
        "table_paper2_core_sweep_results.md",
        "table_paper2_ablation_summary.csv",
        "table_paper2_ablation_summary.md",
        "table_paper2_degradation_summary.csv",
        "table_paper2_degradation_summary.md",
    ]
    for name in expected_figures:
        assert (figures_dir / name).exists()
    for name in expected_tables:
        assert (tables_dir / name).exists()
    assert (out_dir / "paper2_artifact_manifest.json").exists()
    assert (out_dir / "paper2_artifact_summary.md").exists()

    manifest = json.loads((out_dir / "paper2_artifact_manifest.json").read_text(encoding="utf-8"))
    assert "spectral_tail_mean" in manifest["metric_fields"]
    degradation = pd.read_csv(tables_dir / "table_paper2_degradation_summary.csv")
    assert set(degradation["metric"]) == {
        "negative_fraction_mean",
        "projective_invariance_drift_mean",
        "spectral_tail_mean_mean",
    }
