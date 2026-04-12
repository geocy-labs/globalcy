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
    "bundle_path",
    "run_dir",
    "train_loss",
    "negative_fraction",
    "min_eigenvalue_mean",
    "spectral_tail_mean",
    "chart_consistency",
    "projective_invariance_drift",
    "symmetry_consistency",
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
    "symmetry_consistency",
    "determinant_mean",
    "euler_proxy",
    "runtime_seconds",
]

OPTIONAL_METRICS: list[str] = []


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_root(config: dict[str, Any], out_override: str | Path | None = None) -> Path:
    if out_override is not None:
        return Path(out_override).resolve()
    return (Path("outputs") / f"{config['run_name']}_regime_sweep").resolve()


def _executor_from_subprocess(config_path: Path, out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "globalcy.experiments.run_train", "--config", str(config_path), "--out", str(out_dir)],
        check=True,
    )


def _case_entries_from_manifest(
    benchmark_dir: Path,
    benchmark_manifest: dict[str, Any],
    requested_seeds: list[int],
    requested_cases: list[str] | None = None,
) -> list[dict[str, Any]]:
    available_cases = {str(case["case_id"]): dict(case) for case in benchmark_manifest.get("cases", [])}
    selected_case_ids = requested_cases or list(available_cases)
    entries: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        if case_id not in available_cases:
            available = ", ".join(sorted(available_cases))
            raise ValueError(f"Unknown benchmark case '{case_id}'. Available cases: {available}.")
        case_entry = dict(available_cases[case_id])
        for seed in requested_seeds:
            bundle_dir = benchmark_dir / "bundles" / case_id / f"seed_{seed}"
            manifest_path = bundle_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing bundle manifest for case '{case_id}' seed {seed}: {manifest_path}")
            bundle_manifest = _load_json(manifest_path)
            entries.append(
                {
                    "case": case_entry,
                    "seed": seed,
                    "bundle_dir": bundle_dir,
                    "bundle_manifest": bundle_manifest,
                }
            )
    return entries


def _build_run_config(config: dict[str, Any], *, model_name: str, seed: int, bundle_path: Path) -> dict[str, Any]:
    derived = json.loads(json.dumps(config))
    derived["bundle_path"] = str(bundle_path)
    derived["model"]["type"] = model_name
    derived["training"]["seed"] = int(seed)
    return derived


def _collect_run_record(
    metrics_path: Path,
    *,
    case_entry: dict[str, Any],
    bundle_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    record = {
        "case_id": case_entry["case_id"],
        "geometry": metrics.get("geometry", case_entry.get("geometry_family")),
        "lambda": case_entry.get("lambda_value"),
        "seed": metrics.get("seed"),
        "model": metrics.get("model"),
        "bundle_path": str(bundle_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
    }
    for key in AGGREGATE_METRICS + OPTIONAL_METRICS:
        record[key] = metrics.get(key)
    return record


def _aggregate_frame(frame: pd.DataFrame, group_columns: list[str], *, count_column: str, count_name: str) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)[AGGREGATE_METRICS].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    counts = frame.groupby(group_columns, dropna=False)[count_column].nunique().reset_index(name=count_name)
    return grouped.merge(counts, on=group_columns, how="left")


def _aggregate_casewise(frame: pd.DataFrame) -> pd.DataFrame:
    casewise = _aggregate_frame(
        frame,
        ["case_id", "geometry", "lambda", "model"],
        count_column="seed",
        count_name="seed_count",
    )
    return casewise.sort_values(["case_id", "model"]).reset_index(drop=True)


def _aggregate_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    sweep = _aggregate_frame(frame, ["model"], count_column="case_id", count_name="case_count")
    run_counts = frame.groupby("model", dropna=False).size().reset_index(name="run_count")
    return sweep.merge(run_counts, on="model", how="left").sort_values("model").reset_index(drop=True)


def _build_summary_lines(
    *,
    preset_manifest: dict[str, Any],
    per_run: pd.DataFrame,
    casewise: pd.DataFrame,
    sweep: pd.DataFrame,
    models: list[str],
) -> list[str]:
    case_list = ", ".join(f"`{case_id}`" for case_id in casewise["case_id"].drop_duplicates())
    best_negativity = casewise.loc[casewise["negative_fraction_mean"].astype(float).idxmin()]
    best_drift = casewise.loc[casewise["projective_invariance_drift_mean"].astype(float).idxmin()]
    lines = [
        "# Paper II Regime Sweep Summary",
        "",
        f"- preset: `{preset_manifest['preset_name']}`",
        f"- benchmark version: `{preset_manifest['benchmark_version']}`",
        f"- cases: {case_list}",
        f"- models: {', '.join(f'`{model}`' for model in models)}",
        f"- seeds: {', '.join(str(seed) for seed in sorted(per_run['seed'].dropna().astype(int).unique()))}",
        "",
        "## Best observed means",
        f"- lowest mean negative fraction: `{best_negativity['model']}` on `{best_negativity['case_id']}`",
        f"- lowest mean projective-invariance drift: `{best_drift['model']}` on `{best_drift['case_id']}`",
        "",
        "## Sweep-level aggregates",
        "",
        "```text",
        sweep.to_string(index=False),
        "```",
    ]
    if "spectral_tail_mean_mean" in casewise.columns:
        best_tail = casewise.loc[casewise["spectral_tail_mean_mean"].astype(float).idxmax()]
        lines.insert(
            11,
            f"- strongest mean spectral tail: `{best_tail['model']}` on `{best_tail['case_id']}`",
        )
    else:
        lines.extend(
            [
                "",
                "## Blob 3 hook",
                "- spectral-tail summaries are not frozen yet in this sweep; add them to per-run metrics and aggregation when the diagnostic is ready",
            ]
        )
    return lines


def run_regime_sweep(
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

    requested_seeds = list(config["training"].get("seeds") or preset_manifest.get("seeds") or [7, 11, 19])
    models = list(config.get("models") or ["local", "global"])
    requested_cases = list(config.get("cases") or [])
    case_entries = _case_entries_from_manifest(
        benchmark_dir,
        preset_manifest,
        requested_seeds=requested_seeds,
        requested_cases=requested_cases or None,
    )

    output_root = _resolve_output_root(config, out_override=out_dir)
    runs_root = output_root / "runs"
    summaries_root = output_root / "summaries"
    frozen_root = output_root / "frozen"
    for path in (runs_root, summaries_root, frozen_root):
        path.mkdir(parents=True, exist_ok=True)

    write_json(output_root / "benchmark_preset_manifest.json", preset_manifest)
    write_json(output_root / "regime_sweep_config.json", config)

    executor = run_executor or _executor_from_subprocess
    records: list[dict[str, Any]] = []

    for entry in case_entries:
        case_entry = entry["case"]
        seed = int(entry["seed"])
        bundle_dir = Path(entry["bundle_dir"]).resolve()
        for model_name in models:
            derived_config = _build_run_config(config, model_name=model_name, seed=seed, bundle_path=bundle_dir)
            derived_config_path = summaries_root / f"{case_entry['case_id']}_seed_{seed}_{model_name}.json"
            derived_config_path.write_text(json.dumps(derived_config, indent=2), encoding="utf-8")
            run_dir = runs_root / case_entry["case_id"] / f"seed_{seed}" / model_name
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
        raise ValueError("Regime sweep produced no run records.")

    ordered_columns = PER_RUN_COLUMNS + [metric for metric in OPTIONAL_METRICS if metric in records[0]]
    per_run = pd.DataFrame(records)
    per_run = per_run[ordered_columns].sort_values(["case_id", "seed", "model"]).reset_index(drop=True)
    casewise = _aggregate_casewise(per_run)
    sweep = _aggregate_sweep(per_run)

    per_run.to_csv(summaries_root / "per_run_results.csv", index=False)
    casewise.to_csv(summaries_root / "casewise_results.csv", index=False)
    sweep.to_csv(summaries_root / "sweep_results.csv", index=False)

    casewise.to_csv(frozen_root / "paper2_casewise_results.csv", index=False)
    sweep.to_csv(frozen_root / "paper2_sweep_results.csv", index=False)
    payload = {
        "preset_manifest": preset_manifest,
        "models": models,
        "per_run_records": per_run.to_dict(orient="records"),
        "casewise_records": casewise.to_dict(orient="records"),
        "sweep_records": sweep.to_dict(orient="records"),
    }
    write_json(frozen_root / "paper2_results.json", payload)
    write_summary(
        frozen_root / "paper2_summary.md",
        _build_summary_lines(
            preset_manifest=preset_manifest,
            per_run=per_run,
            casewise=casewise,
            sweep=sweep,
            models=models,
        ),
    )

    return {
        "output_root": str(output_root),
        "runs_root": str(runs_root),
        "summaries_root": str(summaries_root),
        "frozen_root": str(frozen_root),
        "preset_manifest": preset_manifest,
        "per_run": per_run,
        "casewise": casewise,
        "sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_regime_sweep(config, out_dir=args.out)
    print(f"Wrote Paper II regime sweep outputs to {result['output_root']}")


if __name__ == "__main__":
    main()
