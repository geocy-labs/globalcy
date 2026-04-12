from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from globalcy.reports.export import write_json, write_summary
from globalcy.reports.figures import save_paper1_core_figure, save_paper1_hardest_case_figure


CORE_COLUMNS = [
    "case_id",
    "model",
    "seed_count",
    "train_loss_mean",
    "train_loss_std",
    "negative_fraction_mean",
    "negative_fraction_std",
    "min_eigenvalue_mean_mean",
    "min_eigenvalue_mean_std",
    "projective_invariance_drift_mean",
    "projective_invariance_drift_std",
    "symmetry_consistency_mean",
    "symmetry_consistency_std",
    "determinant_mean_mean",
    "determinant_mean_std",
    "euler_proxy_mean",
    "euler_proxy_std",
]


def load_comparison_dirs(comparison_dirs: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_frames: list[pd.DataFrame] = []
    aggregated_frames: list[pd.DataFrame] = []
    for directory in comparison_dirs:
        comparison_dir = Path(directory).resolve()
        comparison_frames.append(pd.read_csv(comparison_dir / "comparison.csv"))
        aggregated_frames.append(pd.read_csv(comparison_dir / "comparison_aggregated.csv"))
    return (
        pd.concat(comparison_frames, ignore_index=True).sort_values(["case_id", "seed", "model"]).reset_index(drop=True),
        pd.concat(aggregated_frames, ignore_index=True).sort_values(["case_id", "model"]).reset_index(drop=True),
    )


def build_core_results_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    table = aggregated[CORE_COLUMNS].copy()
    table["case"] = table["case_id"].map(
        {
            "cefalu_lambda_0_75": "Cefalu lambda=0.75",
            "cefalu_lambda_1_0": "Cefalu lambda=1.0",
        }
    ).fillna(table["case_id"])
    return table[["case", "model", *[column for column in CORE_COLUMNS if column not in {"case_id", "model"}]]]


def build_robustness_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, group in aggregated.groupby("case_id", dropna=False):
        indexed = group.set_index("model")
        row: dict[str, object] = {
            "case_id": case_id,
            "seed_count": int(group["seed_count"].max()),
        }
        for metric in ["negative_fraction_mean", "projective_invariance_drift_mean", "symmetry_consistency_mean"]:
            if metric in group.columns:
                best_model = group.loc[group[metric].astype(float).idxmin(), "model"]
                row[f"best_{metric}"] = str(best_model)
        if {"global", "local"}.issubset(indexed.index):
            row["global_minus_local_negative_fraction"] = float(
                indexed.loc["global", "negative_fraction_mean"] - indexed.loc["local", "negative_fraction_mean"]
            )
            row["global_minus_local_projective_drift"] = float(
                indexed.loc["global", "projective_invariance_drift_mean"] - indexed.loc["local", "projective_invariance_drift_mean"]
            )
        if {"symmetry_aware", "global"}.issubset(indexed.index):
            row["symmetry_minus_global_symmetry_consistency"] = float(
                indexed.loc["symmetry_aware", "symmetry_consistency_mean"] - indexed.loc["global", "symmetry_consistency_mean"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_results_memo(aggregated: pd.DataFrame, robustness: pd.DataFrame) -> list[str]:
    lines = [
        "# Paper 1 Results Memo",
        "",
        "This freeze summarizes the first multi-seed GlobalCY comparison benchmark for the two key hard Cefalu cases.",
        "",
        "Compared models:",
        "- `local`",
        "- `global`",
        "- `symmetry_aware`",
        "",
        "Cases:",
        "- `cefalu_lambda_0_75`",
        "- `cefalu_lambda_1_0`",
        "",
    ]
    for _, row in robustness.iterrows():
        lines.append(f"## {row['case_id']}")
        if "best_negative_fraction_mean" in row:
            lines.append(f"- best negative-fraction mean: `{row['best_negative_fraction_mean']}`")
        if "best_projective_invariance_drift_mean" in row:
            lines.append(f"- best projective-drift mean: `{row['best_projective_invariance_drift_mean']}`")
        if "best_symmetry_consistency_mean" in row:
            lines.append(f"- best symmetry-consistency mean: `{row['best_symmetry_consistency_mean']}`")
        if "global_minus_local_negative_fraction" in row:
            lines.append(f"- global minus local negative-fraction mean: `{row['global_minus_local_negative_fraction']:.6f}`")
        if "symmetry_minus_global_symmetry_consistency" in row:
            lines.append(f"- symmetry-aware minus global symmetry-consistency mean: `{row['symmetry_minus_global_symmetry_consistency']:.6f}`")
        lines.append("")

    lines.extend(
        [
            "## Current reading",
            "- this is a lightweight Paper-1 result freeze built from the Milestone 3 multi-seed benchmark",
            "- gains should be interpreted as first controlled comparisons, not final paper claims",
            "- the symmetry-aware model is still intentionally modest and may improve with stronger equivariant structure",
        ]
    )
    return lines


def write_markdown_table(path: Path, title: str, frame: pd.DataFrame) -> None:
    lines = [f"# {title}", "", "```text", frame.to_string(index=False), "```"]
    write_summary(path, lines)


def freeze_results(comparison_dirs: list[str | Path], out_dir: str | Path) -> None:
    run_records, aggregated = load_comparison_dirs(comparison_dirs)
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    core_results = build_core_results_table(aggregated)
    robustness = build_robustness_table(aggregated)

    core_results.to_csv(out_path / "paper1_core_results.csv", index=False)
    robustness.to_csv(out_path / "paper1_robustness.csv", index=False)
    write_markdown_table(out_path / "paper1_core_results.md", "Paper 1 Core Results", core_results)
    write_markdown_table(out_path / "paper1_robustness.md", "Paper 1 Robustness", robustness)

    save_paper1_core_figure(aggregated, str(out_path / "fig_core_comparison.png"))
    save_paper1_hardest_case_figure(aggregated, str(out_path / "fig_hardest_case.png"))

    memo_lines = build_results_memo(aggregated, robustness)
    write_summary(out_path / "paper1_summary.md", memo_lines)
    write_json(
        out_path / "paper1_results.json",
        {
            "comparison_dirs": [str(Path(directory).resolve()) for directory in comparison_dirs],
            "run_records": run_records.to_dict(orient="records"),
            "aggregated_records": aggregated.to_dict(orient="records"),
            "core_results": core_results.to_dict(orient="records"),
            "robustness": robustness.to_dict(orient="records"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", action="append", dest="comparison_dirs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    freeze_results(args.comparison_dirs, args.out)
    print(f"Wrote frozen Paper 1 results to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

