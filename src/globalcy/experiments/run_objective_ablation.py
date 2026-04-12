from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from globalcy.reports.export import write_json, write_summary
from globalcy.utils.config import load_config


PER_RUN_COLUMNS = [
    "case_id",
    "geometry",
    "lambda",
    "seed",
    "model",
    "objective_variant",
    "bundle_path",
    "run_dir",
    "train_loss",
    "negative_fraction",
    "min_eigenvalue_mean",
    "spectral_tail_mean",
    "chart_consistency",
    "projective_invariance_drift",
    "determinant_mean",
    "euler_proxy",
    "runtime_seconds",
]

AGGREGATE_METRICS = [
    "train_loss",
    "negative_fraction",
    "min_eigenvalue_mean",
    "spectral_tail_mean",
    "chart_consistency",
    "projective_invariance_drift",
    "determinant_mean",
    "euler_proxy",
    "runtime_seconds",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_root(config: dict[str, Any], out_override: str | Path | None = None) -> Path:
    if out_override is not None:
        return Path(out_override).resolve()
    return (Path("outputs") / f"{config['run_name']}_ablation").resolve()


def _executor_from_subprocess(config_path: Path, out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "globalcy.experiments.run_train", "--config", str(config_path), "--out", str(out_dir)],
        check=True,
    )


def _default_variants() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "loss_overrides": {"negativity_weight": 0.0, "projective_consistency_weight": 0.0}},
        {"name": "baseline_plus_negativity", "loss_overrides": {"negativity_weight": 0.05, "projective_consistency_weight": 0.0}},
        {"name": "baseline_plus_projective", "loss_overrides": {"negativity_weight": 0.0, "projective_consistency_weight": 0.05}},
        {"name": "baseline_plus_both", "loss_overrides": {"negativity_weight": 0.05, "projective_consistency_weight": 0.05}},
    ]


def _selected_cases(config: dict[str, Any], preset_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    available = {str(case["case_id"]): dict(case) for case in preset_manifest.get("cases", [])}
    requested = list(config.get("cases") or ["cefalu_lambda_0_90", "cefalu_lambda_1_0", "cefalu_lambda_1_10"])
    cases: list[dict[str, Any]] = []
    for case_id in requested:
        if case_id not in available:
            continue
        cases.append(available[case_id])
    if not cases:
        available_keys = ", ".join(sorted(available))
        raise ValueError(f"No requested ablation cases were found in the preset manifest. Available cases: {available_keys}.")
    return cases


def _build_run_config(
    config: dict[str, Any],
    *,
    seed: int,
    bundle_path: Path,
    objective_variant: dict[str, Any],
) -> dict[str, Any]:
    derived = json.loads(json.dumps(config))
    derived["bundle_path"] = str(bundle_path)
    derived["training"]["seed"] = int(seed)
    derived["model"]["type"] = "global"
    derived["objective_variant"] = objective_variant["name"]
    loss_overrides = dict(config.get("loss", {}))
    loss_overrides.update(objective_variant.get("loss_overrides", {}))
    derived["loss"] = loss_overrides
    return derived


def _collect_run_record(metrics_path: Path, *, case_entry: dict[str, Any], bundle_dir: Path, run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    return {
        "case_id": case_entry["case_id"],
        "geometry": metrics.get("geometry", case_entry.get("geometry_family")),
        "lambda": case_entry.get("lambda_value"),
        "seed": metrics.get("seed"),
        "model": metrics.get("model"),
        "objective_variant": metrics.get("objective_variant", "default"),
        "bundle_path": str(bundle_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "train_loss": metrics.get("train_loss"),
        "negative_fraction": metrics.get("negative_fraction"),
        "min_eigenvalue_mean": metrics.get("min_eigenvalue_mean"),
        "spectral_tail_mean": metrics.get("spectral_tail_mean"),
        "chart_consistency": metrics.get("chart_consistency"),
        "projective_invariance_drift": metrics.get("projective_invariance_drift"),
        "determinant_mean": metrics.get("determinant_mean"),
        "euler_proxy": metrics.get("euler_proxy"),
        "runtime_seconds": metrics.get("runtime_seconds"),
    }


def _aggregate(frame: pd.DataFrame, group_columns: list[str], *, count_column: str, count_name: str) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)[AGGREGATE_METRICS].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    counts = frame.groupby(group_columns, dropna=False)[count_column].nunique().reset_index(name=count_name)
    return grouped.merge(counts, on=group_columns, how="left")


def _hardest_case_summary(casewise: pd.DataFrame) -> pd.DataFrame:
    hardest_case = casewise.sort_values("lambda", ascending=False)["case_id"].iloc[0]
    return casewise.loc[casewise["case_id"] == hardest_case].reset_index(drop=True)


def _build_summary_lines(
    *,
    preset_manifest: dict[str, Any],
    per_run: pd.DataFrame,
    per_case: pd.DataFrame,
    per_objective: pd.DataFrame,
    hardest_case: pd.DataFrame,
) -> list[str]:
    best_negativity = per_case.loc[per_case["negative_fraction_mean"].astype(float).idxmin()]
    best_drift = per_case.loc[per_case["projective_invariance_drift_mean"].astype(float).idxmin()]
    best_tail = per_case.loc[per_case["spectral_tail_mean_mean"].astype(float).idxmax()]
    hardest_case_id = hardest_case["case_id"].iloc[0]
    return [
        "# Paper II Objective Ablation Summary",
        "",
        f"- preset: `{preset_manifest['preset_name']}`",
        f"- benchmark version: `{preset_manifest['benchmark_version']}`",
        f"- cases: {', '.join(f'`{case_id}`' for case_id in per_case['case_id'].drop_duplicates())}",
        f"- objective variants: {', '.join(f'`{name}`' for name in per_case['objective_variant'].drop_duplicates())}",
        f"- seeds: {', '.join(str(seed) for seed in sorted(per_run['seed'].dropna().astype(int).unique()))}",
        "",
        "## Best observed means",
        f"- lowest mean negative fraction: `{best_negativity['objective_variant']}` on `{best_negativity['case_id']}`",
        f"- lowest mean projective-invariance drift: `{best_drift['objective_variant']}` on `{best_drift['case_id']}`",
        f"- strongest mean spectral tail: `{best_tail['objective_variant']}` on `{best_tail['case_id']}`",
        "",
        f"## Hardest-case view: `{hardest_case_id}`",
        "",
        "```text",
        hardest_case.to_string(index=False),
        "```",
        "",
        "## Objective-level aggregates",
        "",
        "```text",
        per_objective.to_string(index=False),
        "```",
    ]


def run_objective_ablation(
    config: dict[str, Any],
    *,
    out_dir: str | Path | None = None,
    run_executor: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_dir = repo_root.joinpath(config["benchmark_dir"]).resolve()
    preset_manifest_path = benchmark_dir / "benchmark_preset_manifest.json"
    if not preset_manifest_path.exists():
        raise FileNotFoundError(f"Missing benchmark_preset_manifest.json: {preset_manifest_path}")
    preset_manifest = _load_json(preset_manifest_path)

    seeds = list(config["training"].get("seeds") or preset_manifest.get("seeds") or [7, 11, 19])
    objective_variants = list(config.get("objective_variants") or _default_variants())
    selected_cases = _selected_cases(config, preset_manifest)

    output_root = _resolve_output_root(config, out_override=out_dir)
    runs_root = output_root / "runs"
    summaries_root = output_root / "summaries"
    frozen_root = output_root / "frozen"
    for path in (runs_root, summaries_root, frozen_root):
        path.mkdir(parents=True, exist_ok=True)

    write_json(output_root / "benchmark_preset_manifest.json", preset_manifest)
    write_json(output_root / "config.json", config)

    executor = run_executor or _executor_from_subprocess
    records: list[dict[str, Any]] = []

    for case_entry in selected_cases:
        case_id = str(case_entry["case_id"])
        for seed in seeds:
            bundle_dir = benchmark_dir / "bundles" / case_id / f"seed_{seed}"
            manifest_path = bundle_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing bundle manifest for case '{case_id}' seed {seed}: {manifest_path}")
            for objective_variant in objective_variants:
                derived_config = _build_run_config(
                    config,
                    seed=int(seed),
                    bundle_path=bundle_dir.resolve(),
                    objective_variant=objective_variant,
                )
                derived_config_path = summaries_root / f"{case_id}_seed_{seed}_{objective_variant['name']}.json"
                derived_config_path.write_text(json.dumps(derived_config, indent=2), encoding="utf-8")
                run_dir = runs_root / case_id / f"seed_{seed}" / objective_variant["name"]
                executor(derived_config_path, run_dir)
                records.append(
                    _collect_run_record(
                        run_dir / "metrics.json",
                        case_entry=case_entry,
                        bundle_dir=bundle_dir,
                        run_dir=run_dir,
                    )
                )

    if not records:
        raise ValueError("Objective ablation produced no run records.")

    per_run = pd.DataFrame(records)[PER_RUN_COLUMNS].sort_values(["case_id", "seed", "objective_variant"]).reset_index(drop=True)
    per_case = _aggregate(per_run, ["case_id", "geometry", "lambda", "objective_variant"], count_column="seed", count_name="seed_count")
    per_case = per_case.sort_values(["case_id", "objective_variant"]).reset_index(drop=True)
    per_objective = _aggregate(per_run, ["objective_variant"], count_column="case_id", count_name="case_count")
    per_objective = per_objective.sort_values("objective_variant").reset_index(drop=True)
    hardest_case = _hardest_case_summary(per_case)

    per_run.to_csv(summaries_root / "per_run_results.csv", index=False)
    per_case.to_csv(summaries_root / "per_case_results.csv", index=False)
    per_objective.to_csv(summaries_root / "per_objective_results.csv", index=False)
    hardest_case.to_csv(summaries_root / "hardest_case_ablation.csv", index=False)

    per_case.to_csv(frozen_root / "paper2_ablation_results.csv", index=False)
    per_objective.to_csv(frozen_root / "per_objective_results.csv", index=False)
    hardest_case.to_csv(frozen_root / "hardest_case_ablation.csv", index=False)
    write_json(
        frozen_root / "paper2_ablation_results.json",
        {
            "preset_manifest": preset_manifest,
            "objective_variants": objective_variants,
            "per_run_records": per_run.to_dict(orient="records"),
            "per_case_records": per_case.to_dict(orient="records"),
            "per_objective_records": per_objective.to_dict(orient="records"),
            "hardest_case_records": hardest_case.to_dict(orient="records"),
        },
    )
    write_summary(
        frozen_root / "paper2_ablation_summary.md",
        _build_summary_lines(
            preset_manifest=preset_manifest,
            per_run=per_run,
            per_case=per_case,
            per_objective=per_objective,
            hardest_case=hardest_case,
        ),
    )

    return {
        "output_root": str(output_root),
        "runs_root": str(runs_root),
        "summaries_root": str(summaries_root),
        "frozen_root": str(frozen_root),
        "per_run": per_run,
        "per_case": per_case,
        "per_objective": per_objective,
        "hardest_case": hardest_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_objective_ablation(config, out_dir=args.out)
    print(f"Wrote Paper II objective ablation outputs to {result['output_root']}")


if __name__ == "__main__":
    main()
