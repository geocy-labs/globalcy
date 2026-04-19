from __future__ import annotations

import pandas as pd


def metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([metrics])


def write_markdown_table(title: str, frame: pd.DataFrame) -> list[str]:
    return [f"# {title}", "", "```text", frame.to_string(index=False), "```"]


def build_paper2_core_sweep_table(casewise: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id",
        "lambda",
        "model",
        "seed_count",
        "negative_fraction_mean",
        "projective_invariance_drift_mean",
        "min_eigenvalue_mean_mean",
        "spectral_tail_mean_mean",
        "train_loss_mean",
    ]
    available = [column for column in columns if column in casewise.columns]
    return casewise[available].sort_values(["lambda", "model"]).reset_index(drop=True)


def build_paper2_ablation_table(ablation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id",
        "lambda",
        "objective_variant",
        "seed_count",
        "negative_fraction_mean",
        "projective_invariance_drift_mean",
        "spectral_tail_mean_mean",
        "train_loss_mean",
    ]
    available = [column for column in columns if column in ablation.columns]
    return ablation[available].sort_values(["lambda", "objective_variant"]).reset_index(drop=True)


def build_paper2_degradation_summary(casewise: pd.DataFrame) -> pd.DataFrame:
    if casewise.empty:
        return pd.DataFrame()
    ordered = casewise.sort_values(["model", "lambda"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    metrics = [
        ("negative_fraction_mean", "lower_better"),
        ("projective_invariance_drift_mean", "lower_better"),
        ("spectral_tail_mean_mean", "higher_better"),
    ]
    for model, group in ordered.groupby("model", dropna=False):
        group = group.sort_values("lambda").reset_index(drop=True)
        start_case = str(group.iloc[0]["case_id"])
        end_case = str(group.iloc[-1]["case_id"])
        for metric, direction in metrics:
            if metric not in group.columns:
                continue
            start_value = float(group.iloc[0][metric])
            end_value = float(group.iloc[-1][metric])
            if direction == "lower_better":
                deterioration_ratio = float(end_value / max(start_value, 1e-12))
            else:
                deterioration_ratio = float(start_value / max(end_value, 1e-12))
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "start_case": start_case,
                    "end_case": end_case,
                    "start_value": start_value,
                    "end_value": end_value,
                    "delta": end_value - start_value,
                    "deterioration_ratio": deterioration_ratio,
                    "direction": direction,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "model"]).reset_index(drop=True)
