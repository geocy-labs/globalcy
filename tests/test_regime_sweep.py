from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from globalcy.experiments.run_regime_sweep import run_regime_sweep


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


def test_regime_sweep_writes_paper2_outputs(tmp_path: Path, synthetic_bundle: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    bundles_root = benchmark_dir / "bundles"
    cases = [
        ("cefalu_lambda_0_50", 0.5),
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
            (bundle_dir / "case_metadata.json").write_text(
                json.dumps(
                    {
                        "case_id": case_id,
                        "geometry": "cefalu_quartic",
                        "seed": seed,
                        "lambda": lambda_value,
                    }
                ),
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
        model = config["model"]["type"]
        seed = int(config["training"]["seed"])
        bundle_path = Path(config["bundle_path"])
        bundle_manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
        lambda_value = float(bundle_manifest["parameters"]["lambda"])
        base = 0.04 + 0.01 * lambda_value + 0.001 * seed
        model_shift = {"local": 0.02, "global": 0.0}[model]
        metrics = {
            "model": model,
            "geometry": "cefalu_quartic",
            "case_id": bundle_manifest["case_id"],
            "seed": seed,
            "train_loss": 8.0 + model_shift + 0.1 * lambda_value,
            "negative_fraction": base + model_shift,
            "min_eigenvalue_mean": 0.15 - model_shift,
            "spectral_tail_mean": 0.05 + 0.5 * base - model_shift,
            "chart_consistency": 0.0,
            "projective_invariance_drift": 1e-8 + model_shift * 1e-8,
            "symmetry_consistency": 0.03 + model_shift,
            "determinant_mean": 0.8 - model_shift,
            "euler_proxy": 1.1 + lambda_value,
            "runtime_seconds": 0.5 + 0.1 * seed,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    config = {
        "run_name": "paper2_regime_smoke",
        "benchmark_dir": str(benchmark_dir),
        "models": ["local", "global"],
        "training": {
            "seed": 7,
            "seeds": seeds,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "train_fraction": 0.8,
        },
        "model": {"type": "global", "hidden_dims": [8, 8]},
        "loss": {
            "target_weight": 1.0,
            "determinant_weight": 0.1,
            "positivity_weight": 0.05,
            "chart_weight": 0.0,
            "symmetry_weight": 0.05,
        },
        "evaluation": {"projective_scale": 1.7},
    }

    result = run_regime_sweep(config, out_dir=tmp_path / "regime_out", run_executor=fake_executor)

    frozen_root = Path(result["frozen_root"])
    summaries_root = Path(result["summaries_root"])
    assert (summaries_root / "per_run_results.csv").exists()
    assert (summaries_root / "casewise_results.csv").exists()
    assert (summaries_root / "sweep_results.csv").exists()
    assert (frozen_root / "paper2_casewise_results.csv").exists()
    assert (frozen_root / "paper2_sweep_results.csv").exists()
    assert (frozen_root / "paper2_results.json").exists()
    assert (frozen_root / "paper2_summary.md").exists()

    casewise = pd.read_csv(frozen_root / "paper2_casewise_results.csv")
    sweep = pd.read_csv(frozen_root / "paper2_sweep_results.csv")
    assert set(casewise["case_id"]) == {"cefalu_lambda_0_50", "cefalu_lambda_1_0"}
    assert set(casewise["model"]) == {"local", "global"}
    assert set(sweep["model"]) == {"local", "global"}
    assert (casewise["seed_count"] == 2).all()
    assert "spectral_tail_mean_mean" in casewise.columns
    assert "spectral_tail_mean_mean" in sweep.columns
    assert set(sweep["case_count"]) == {2}

    payload = json.loads((frozen_root / "paper2_results.json").read_text(encoding="utf-8"))
    assert payload["preset_manifest"]["preset_name"] == "cefalu_hard_regime_sweep_v1"
    assert len(payload["per_run_records"]) == 8
    assert "spectral_tail_mean" in payload["per_run_records"][0]

    summary_text = (frozen_root / "paper2_summary.md").read_text(encoding="utf-8")
    assert "spectral tail" in summary_text.lower()
