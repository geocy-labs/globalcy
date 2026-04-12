from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from globalcy.experiments.run_objective_ablation import run_objective_ablation


def _bundle_manifest(case_id: str, seed: int, lambda_value: float) -> dict[str, object]:
    return {
        "geometry": "cefalu_quartic",
        "case_id": case_id,
        "seed": seed,
        "parameters": {"lambda": lambda_value, "seed": seed},
        "protocol_metadata": {
            "available_model_facing_views": [
                "local_chart",
                "invariant",
                "symmetry_metadata",
                "sampling_metadata",
            ]
        },
    }


def test_objective_ablation_freezes_variant_outputs(tmp_path: Path, synthetic_bundle: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    bundles_root = benchmark_dir / "bundles"
    cases = [
        ("cefalu_lambda_0_90", 0.9),
        ("cefalu_lambda_1_0", 1.0),
    ]
    seeds = [7, 11]
    for case_id, lambda_value in cases:
        for seed in seeds:
            bundle_dir = bundles_root / case_id / f"seed_{seed}"
            shutil.copytree(synthetic_bundle, bundle_dir)
            (bundle_dir / "manifest.json").write_text(
                json.dumps(_bundle_manifest(case_id, seed, lambda_value)),
                encoding="utf-8",
            )

    preset_manifest = {
        "preset_name": "cefalu_hard_regime_sweep_v1",
        "benchmark_version": "paper2_hard_regime_v1",
        "geometry_family": "cefalu_quartic",
        "target_name": "hypersurface_fs_scalar",
        "seeds": seeds,
        "n_samples": 6,
        "cases": [
            {
                "case_id": case_id,
                "lambda_value": lambda_value,
                "geometry_family": "cefalu_quartic",
                "benchmark_version": "paper2_hard_regime_v1",
                "available_model_facing_views": [
                    "local_chart",
                    "invariant",
                    "symmetry_metadata",
                    "sampling_metadata",
                ],
            }
            for case_id, lambda_value in cases
        ],
    }
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "benchmark_preset_manifest.json").write_text(json.dumps(preset_manifest), encoding="utf-8")

    def fake_executor(config_path: Path, out_dir: Path) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_manifest = json.loads(Path(config["bundle_path"], "manifest.json").read_text(encoding="utf-8"))
        seed = int(config["training"]["seed"])
        variant = config["objective_variant"]
        penalty_bonus = {
            "baseline": 0.0,
            "baseline_plus_negativity": -0.02,
            "baseline_plus_projective": -0.015,
            "baseline_plus_both": -0.03,
        }[variant]
        projective_bonus = {
            "baseline": 0.0,
            "baseline_plus_negativity": 0.0,
            "baseline_plus_projective": -1.5e-8,
            "baseline_plus_both": -2.0e-8,
        }[variant]
        lambda_value = float(bundle_manifest["parameters"]["lambda"])
        metrics = {
            "model": "global",
            "objective_variant": variant,
            "geometry": "cefalu_quartic",
            "case_id": bundle_manifest["case_id"],
            "seed": seed,
            "train_loss": 8.0 + 0.1 * lambda_value + penalty_bonus,
            "negative_fraction": 0.10 + 0.01 * lambda_value + penalty_bonus,
            "min_eigenvalue_mean": 0.12 - penalty_bonus,
            "spectral_tail_mean": 0.08 - 0.5 * penalty_bonus,
            "chart_consistency": 0.0,
            "projective_invariance_drift": 5e-8 + projective_bonus,
            "determinant_mean": 0.9 - penalty_bonus,
            "euler_proxy": 1.1 + lambda_value,
            "runtime_seconds": 0.4 + 0.1 * seed,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    config = {
        "run_name": "paper2_objective_smoke",
        "benchmark_dir": str(benchmark_dir),
        "cases": [case_id for case_id, _ in cases],
        "model": {"type": "global", "hidden_dims": [8, 8]},
        "training": {
            "seed": 7,
            "seeds": seeds,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "train_fraction": 0.8,
        },
        "loss": {
            "target_weight": 1.0,
            "determinant_weight": 0.1,
            "negativity_weight": 0.0,
            "projective_consistency_weight": 0.0,
            "projective_consistency_scale": 1.7,
            "chart_weight": 0.0,
            "symmetry_weight": 0.0,
        },
        "objective_variants": [
            {"name": "baseline", "loss_overrides": {"negativity_weight": 0.0, "projective_consistency_weight": 0.0}},
            {"name": "baseline_plus_negativity", "loss_overrides": {"negativity_weight": 0.05, "projective_consistency_weight": 0.0}},
            {"name": "baseline_plus_projective", "loss_overrides": {"negativity_weight": 0.0, "projective_consistency_weight": 0.05}},
            {"name": "baseline_plus_both", "loss_overrides": {"negativity_weight": 0.05, "projective_consistency_weight": 0.05}},
        ],
        "evaluation": {"projective_scale": 1.7},
    }

    result = run_objective_ablation(config, out_dir=tmp_path / "ablation_out", run_executor=fake_executor)

    frozen_root = Path(result["frozen_root"])
    summaries_root = Path(result["summaries_root"])
    assert (summaries_root / "per_run_results.csv").exists()
    assert (summaries_root / "per_case_results.csv").exists()
    assert (summaries_root / "per_objective_results.csv").exists()
    assert (summaries_root / "hardest_case_ablation.csv").exists()
    assert (frozen_root / "paper2_ablation_results.csv").exists()
    assert (frozen_root / "paper2_ablation_results.json").exists()
    assert (frozen_root / "paper2_ablation_summary.md").exists()

    per_case = pd.read_csv(frozen_root / "paper2_ablation_results.csv")
    per_objective = pd.read_csv(frozen_root / "per_objective_results.csv")
    hardest_case = pd.read_csv(frozen_root / "hardest_case_ablation.csv")
    assert set(per_case["objective_variant"]) == {
        "baseline",
        "baseline_plus_negativity",
        "baseline_plus_projective",
        "baseline_plus_both",
    }
    assert "spectral_tail_mean_mean" in per_case.columns
    assert set(per_objective["objective_variant"]) == set(per_case["objective_variant"])
    assert set(hardest_case["case_id"]) == {"cefalu_lambda_1_0"}

    payload = json.loads((frozen_root / "paper2_ablation_results.json").read_text(encoding="utf-8"))
    assert len(payload["per_run_records"]) == 16
    assert payload["objective_variants"][0]["name"] == "baseline"
