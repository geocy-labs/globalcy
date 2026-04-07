from __future__ import annotations

import jax.numpy as jnp

from globalcy.diagnostics.characteristic_forms import characteristic_summary
from globalcy.diagnostics.positivity import positivity_summary
from globalcy.experiments.run_compare import compare_metrics


def test_diagnostics_smoke():
    metrics = jnp.stack([jnp.eye(3, dtype=jnp.complex64), 1.2 * jnp.eye(3, dtype=jnp.complex64)], axis=0)
    positivity = positivity_summary(metrics)
    characteristic = characteristic_summary(metrics)
    assert positivity["negative_fraction"] == 0.0
    assert characteristic["determinant_mean"] > 0.0


def test_model_comparison_diagnostics_smoke():
    frame, summary, lines = compare_metrics(
        [
            {
                "run_dir": "local",
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
    assert summary["best_by_metric"]["projective_invariance_drift"] == "global"
    assert any("best `projective_invariance_drift`" in line for line in lines)
