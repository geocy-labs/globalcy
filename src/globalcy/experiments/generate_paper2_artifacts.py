from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from globalcy.reports.export import write_json, write_summary
from globalcy.reports.figures import (
    save_paper2_degradation_profiles,
    save_paper2_diagnostic_trajectories,
    save_paper2_hardest_case,
    save_paper2_objective_ablation,
)
from globalcy.reports.tables import (
    build_paper2_ablation_table,
    build_paper2_core_sweep_table,
    build_paper2_degradation_summary,
    write_markdown_table,
)


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _resolve_out_dir(regime_frozen_dir: Path, out_dir: str | Path | None) -> Path:
    if out_dir is not None:
        return Path(out_dir).resolve()
    base_name = regime_frozen_dir.parent.parent.name.replace("_regime_sweep", "")
    return (Path("outputs") / f"{base_name}_paper_artifacts").resolve()


def _hardest_case_id(casewise: pd.DataFrame, allowed_case_ids: set[str] | None = None) -> str:
    global_rows = casewise.loc[casewise["model"] == "global"].copy()
    if global_rows.empty:
        global_rows = casewise.copy()
    if allowed_case_ids is not None:
        global_rows = global_rows.loc[global_rows["case_id"].astype(str).isin(allowed_case_ids)].copy()
    if global_rows.empty:
        raise ValueError("No overlapping cases available to choose a hardest-case figure.")
    ordered = global_rows.sort_values(["negative_fraction_mean", "lambda"], ascending=[False, False]).reset_index(drop=True)
    return str(ordered.iloc[0]["case_id"])


def _summary_lines(
    *,
    regime_casewise: pd.DataFrame,
    ablation_casewise: pd.DataFrame,
    degradation_table: pd.DataFrame,
    hardest_case_id: str,
) -> list[str]:
    best_global = regime_casewise.loc[regime_casewise["model"] == "global"].sort_values("negative_fraction_mean").iloc[0]
    best_ablation = ablation_casewise.sort_values(["negative_fraction_mean", "projective_invariance_drift_mean"]).iloc[0]
    return [
        "# Paper II Artifact Summary",
        "",
        "This directory contains manuscript-facing figures and tables generated from frozen Paper II outputs only.",
        "",
        "## Inputs",
        "- regime-sweep frozen outputs",
        "- objective-ablation frozen outputs",
        "",
        "## Highlights",
        f"- strongest global sweep row by negative fraction: `{best_global['case_id']}`",
        f"- best ablation variant across the frozen ablation table: `{best_ablation['objective_variant']}`",
        f"- hardest-case focus used for Figure D: `{hardest_case_id}`",
        "",
        "## Degradation computation",
        "- lower-is-better metrics (`negative_fraction`, `projective_invariance_drift`) use deterioration ratio = end_value / start_value",
        "- higher-is-better metrics (`spectral_tail_mean`) use deterioration ratio = start_value / end_value",
        "- start and end are defined by the lambda-ordered sweep endpoints",
        "",
        "## Frozen outputs used",
        "- `paper2_casewise_results.csv`",
        "- `paper2_sweep_results.csv`",
        "- `paper2_ablation_results.csv`",
        "- `per_objective_results.csv`",
        "- `hardest_case_ablation.csv`",
    ]


def generate_paper2_artifacts(
    regime_frozen_dir: str | Path,
    ablation_frozen_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, str]:
    regime_dir = Path(regime_frozen_dir).resolve()
    ablation_dir = Path(ablation_frozen_dir).resolve()
    output_root = _resolve_out_dir(regime_dir, out_dir)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    regime_casewise = _load_frame(regime_dir / "paper2_casewise_results.csv")
    regime_sweep = _load_frame(regime_dir / "paper2_sweep_results.csv")
    ablation_casewise = _load_frame(ablation_dir / "paper2_ablation_results.csv")
    per_objective = _load_frame(ablation_dir / "per_objective_results.csv")
    hardest_case_ablation = _load_frame(ablation_dir / "hardest_case_ablation.csv")

    core_table = build_paper2_core_sweep_table(regime_casewise)
    ablation_table = build_paper2_ablation_table(ablation_casewise)
    degradation_table = build_paper2_degradation_summary(regime_casewise)
    ablation_case_ids = set(ablation_casewise["case_id"].astype(str).unique())
    hardest_case_id = _hardest_case_id(regime_casewise, allowed_case_ids=ablation_case_ids)

    core_csv = tables_dir / "table_paper2_core_sweep_results.csv"
    core_md = tables_dir / "table_paper2_core_sweep_results.md"
    ablation_csv = tables_dir / "table_paper2_ablation_summary.csv"
    ablation_md = tables_dir / "table_paper2_ablation_summary.md"
    degradation_csv = tables_dir / "table_paper2_degradation_summary.csv"
    degradation_md = tables_dir / "table_paper2_degradation_summary.md"

    core_table.to_csv(core_csv, index=False)
    ablation_table.to_csv(ablation_csv, index=False)
    degradation_table.to_csv(degradation_csv, index=False)
    write_summary(core_md, write_markdown_table("Paper 2 Core Sweep Results", core_table))
    write_summary(ablation_md, write_markdown_table("Paper 2 Ablation Summary", ablation_table))
    write_summary(degradation_md, write_markdown_table("Paper 2 Degradation Summary", degradation_table))

    fig_a = figures_dir / "fig_paper2_diagnostic_trajectories.png"
    fig_b = figures_dir / "fig_paper2_degradation_profiles.png"
    fig_c = figures_dir / "fig_paper2_objective_ablation.png"
    fig_d = figures_dir / "fig_paper2_hardest_case.png"
    save_paper2_diagnostic_trajectories(regime_casewise, str(fig_a))
    save_paper2_degradation_profiles(regime_casewise, str(fig_b))
    save_paper2_objective_ablation(ablation_casewise, str(fig_c))
    save_paper2_hardest_case(regime_casewise, ablation_casewise, str(fig_d), hardest_case_id=hardest_case_id)

    manifest = {
        "run_name": output_root.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_frozen_artifacts": {
            "regime_frozen_dir": str(regime_dir),
            "ablation_frozen_dir": str(ablation_dir),
            "regime_files": [
                str(regime_dir / "paper2_casewise_results.csv"),
                str(regime_dir / "paper2_sweep_results.csv"),
                str(regime_dir / "paper2_results.json"),
                str(regime_dir / "paper2_summary.md"),
            ],
            "ablation_files": [
                str(ablation_dir / "paper2_ablation_results.csv"),
                str(ablation_dir / "paper2_ablation_results.json"),
                str(ablation_dir / "paper2_ablation_summary.md"),
                str(ablation_dir / "per_objective_results.csv"),
                str(ablation_dir / "hardest_case_ablation.csv"),
            ],
        },
        "figures": [str(fig_a), str(fig_b), str(fig_c), str(fig_d)],
        "tables": [str(core_csv), str(core_md), str(ablation_csv), str(ablation_md), str(degradation_csv), str(degradation_md)],
        "metric_fields": [
            "negative_fraction",
            "projective_invariance_drift",
            "min_eigenvalue_mean",
            "spectral_tail_mean",
            "train_loss",
            "determinant_mean",
            "euler_proxy",
            "runtime_seconds",
        ],
        "degradation_definition": {
            "lower_is_better_metrics": "deterioration_ratio = end_value / start_value",
            "higher_is_better_metrics": "deterioration_ratio = start_value / end_value",
            "lambda_ordering": "ascending lambda across canonical case ids",
        },
    }
    write_json(output_root / "paper2_artifact_manifest.json", manifest)
    write_summary(
        output_root / "paper2_artifact_summary.md",
        _summary_lines(
            regime_casewise=regime_casewise,
            ablation_casewise=ablation_casewise,
            degradation_table=degradation_table,
            hardest_case_id=hardest_case_id,
        ),
    )

    return {
        "output_root": str(output_root),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-frozen-dir", required=True)
    parser.add_argument("--ablation-frozen-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = generate_paper2_artifacts(
        args.regime_frozen_dir,
        args.ablation_frozen_dir,
        out_dir=args.out,
    )
    print(f"Wrote Paper II paper-facing artifacts to {result['output_root']}")


if __name__ == "__main__":
    main()
