from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from globalcy.reports.export import write_json, write_summary


COMPARISON_COLUMNS = [
    "model",
    "geometry",
    "case_id",
    "lambda",
    "train_loss",
    "min_eigenvalue_mean",
    "negative_fraction",
    "chart_consistency",
    "projective_invariance_drift",
    "symmetry_consistency",
    "determinant_mean",
    "euler_proxy",
    "runtime_seconds",
]


def load_run_metrics(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    metrics["run_dir"] = str(run_path)
    return metrics


def compare_metrics(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    frame = pd.DataFrame(records)
    for column in COMPARISON_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    ordered = frame[["run_dir", *COMPARISON_COLUMNS]].sort_values("model").reset_index(drop=True)

    score_columns = [
        "negative_fraction",
        "chart_consistency",
        "projective_invariance_drift",
        "symmetry_consistency",
    ]
    best_by_metric: dict[str, str] = {}
    for column in score_columns:
        available = ordered.dropna(subset=[column])
        if not available.empty:
            best_by_metric[column] = str(available.loc[available[column].astype(float).idxmin(), "model"])

    numeric_columns = [
        "train_loss",
        "min_eigenvalue_mean",
        "negative_fraction",
        "chart_consistency",
        "projective_invariance_drift",
        "symmetry_consistency",
    ]
    stability: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        available = ordered.dropna(subset=[column])
        if not available.empty:
            values = available[column].astype(float)
            stability[column] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    summary_lines = [
        "# Model Comparison",
        "",
        f"- case_id: `{ordered['case_id'].iloc[0]}`",
        f"- geometry: `{ordered['geometry'].iloc[0]}`",
        f"- models compared: {', '.join(f'`{model}`' for model in ordered['model'])}",
    ]
    for metric, model in best_by_metric.items():
        summary_lines.append(f"- best `{metric}`: `{model}`")
    return ordered, {"best_by_metric": best_by_metric, "stability": stability}, summary_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", dest="run_dirs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = [load_run_metrics(run_dir) for run_dir in args.run_dirs]
    frame, summary, summary_lines = compare_metrics(records)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "comparison.csv", index=False)
    write_json(out_dir / "comparison.json", summary | {"records": frame.to_dict(orient="records")})
    write_summary(out_dir / "comparison.md", summary_lines)
    print(f"Wrote comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
