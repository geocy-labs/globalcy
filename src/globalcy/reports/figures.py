from __future__ import annotations

import pandas as pd


def save_metric_bar_chart(frame: pd.DataFrame, metric: str, path: str, x_column: str = "model") -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    frame.plot.bar(x=x_column, y=metric, ax=axis, legend=False)
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} by model")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


MODEL_LABELS = {
    "local": "Local",
    "global": "Global",
    "symmetry_aware": "Symmetry-aware",
}

MODEL_COLORS = {
    "local": "#7A8DA6",
    "global": "#2A6F97",
    "symmetry_aware": "#C66B3D",
}

CASE_LABELS = {
    "cefalu_lambda_0_50": "Cefalú lambda=0.50",
    "cefalu_lambda_0_75": "Cefalú lambda=0.75",
    "cefalu_lambda_0_90": "Cefalú lambda=0.90",
    "cefalu_lambda_1_0": "Cefalú lambda=1.0",
    "cefalu_lambda_1_10": "Cefalú lambda=1.10",
}


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def _case_label(case_id: str) -> str:
    return CASE_LABELS.get(case_id, case_id)


def _case_title_label(case_id: str) -> str:
    return {
        "cefalu_lambda_0_50": r"Cefalú $\lambda = .50$",
        "cefalu_lambda_0_75": r"Cefalú $\lambda = .75$",
        "cefalu_lambda_0_90": r"Cefalú $\lambda = .90$",
        "cefalu_lambda_1_0": r"Cefalú $\lambda = 1.0$",
        "cefalu_lambda_1_10": r"Cefalú $\lambda = 1.10$",
    }.get(case_id, _case_label(case_id))


def _lambda_tick_labels(frame: pd.DataFrame) -> list[str]:
    ordered = frame[["case_id", "lambda"]].drop_duplicates().sort_values(["lambda", "case_id"])
    return [f"{float(value):.2f}" for value in ordered["lambda"].tolist()]


def _case_order(frame: pd.DataFrame) -> list[str]:
    ordered = frame[["case_id", "lambda"]].drop_duplicates().sort_values(["lambda", "case_id"])
    return ordered["case_id"].astype(str).tolist()


def _metric_panel(
    axis,
    frame: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    title: str,
    ylabel: str,
    use_log: bool = False,
) -> None:
    cases = list(frame["case_id"].drop_duplicates())
    models = list(MODEL_LABELS)
    x_positions = range(len(cases))
    width = 0.24

    for index, model in enumerate(models):
        subset = frame.loc[frame["model"] == model]
        means = [float(subset.loc[subset["case_id"] == case_id, metric_mean].iloc[0]) for case_id in cases]
        stds = [float(subset.loc[subset["case_id"] == case_id, metric_std].iloc[0]) for case_id in cases]
        offsets = [position + (index - 1) * width for position in x_positions]
        axis.bar(
            offsets,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.6,
            label=_model_label(model),
        )

    axis.set_title(title, fontsize=11)
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels([_case_label(case_id) for case_id in cases], rotation=10, ha="right")
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    if use_log:
        axis.set_yscale("log")


def save_paper1_core_figure(frame: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(["case_id", "model"]).reset_index(drop=True)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    _metric_panel(
        axes[0],
        ordered,
        "negative_fraction_mean",
        "negative_fraction_std",
        "Negativity",
        "Negative fraction",
    )
    _metric_panel(
        axes[1],
        ordered,
        "projective_invariance_drift_mean",
        "projective_invariance_drift_std",
        "Projective-invariance drift",
        "Drift",
        use_log=True,
    )
    _metric_panel(
        axes[2],
        ordered,
        "train_loss_mean",
        "train_loss_std",
        "Train loss",
        "Loss",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.96))
    figure.suptitle("Core comparison across hard Cefalú cases", fontsize=13, y=1.04)
    figure.subplots_adjust(top=0.82, wspace=0.25)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_paper1_hardest_case_figure(frame: pd.DataFrame, path: str, case_id: str = "cefalu_lambda_1_0") -> None:
    import matplotlib.pyplot as plt

    subset = frame.loc[frame["case_id"] == case_id].copy()
    ordered_models = [model for model in MODEL_LABELS if model in set(subset["model"])]
    subset["model"] = pd.Categorical(subset["model"], categories=ordered_models, ordered=True)
    subset = subset.sort_values("model").reset_index(drop=True)
    models = list(subset["model"].astype(str))
    labels = [_model_label(model) for model in models]
    colors = [MODEL_COLORS.get(model, "#666666") for model in models]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    metrics = [
        ("negative_fraction_mean", "negative_fraction_std", "Negativity", "Negative fraction", False),
        ("projective_invariance_drift_mean", "projective_invariance_drift_std", "Projective-invariance drift", "Drift", True),
        ("train_loss_mean", "train_loss_std", "Train loss", "Loss", False),
    ]

    x_positions = range(len(models))
    for axis, (metric_mean, metric_std, title, ylabel, use_log) in zip(axes, metrics):
        means = [float(subset.loc[subset["model"] == model, metric_mean].iloc[0]) for model in models]
        stds = [float(subset.loc[subset["model"] == model, metric_std].iloc[0]) for model in models]
        axis.bar(
            list(x_positions),
            means,
            yerr=stds,
            capsize=3,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        if use_log:
            axis.set_yscale("log")

    figure.suptitle(f"Hardest-case comparison: {_case_title_label(case_id)}", fontsize=13, y=1.02)
    figure.subplots_adjust(top=0.84, wspace=0.25)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_paper2_diagnostic_trajectories(frame: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(["lambda", "model"]).reset_index(drop=True)
    cases = _case_order(ordered)
    lambda_labels = _lambda_tick_labels(ordered)
    x_positions = list(range(len(cases)))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    metrics = [
        ("negative_fraction_mean", "negative_fraction_std", "Negative fraction", False),
        ("projective_invariance_drift_mean", "projective_invariance_drift_std", "Projective-invariance drift", True),
        ("spectral_tail_mean_mean", "spectral_tail_mean_std", "Spectral-tail mean", False),
    ]

    for axis, (metric_mean, metric_std, ylabel, use_log) in zip(axes, metrics):
        for model in [model for model in MODEL_LABELS if model in set(ordered["model"])]:
            subset = ordered.loc[ordered["model"] == model]
            means = [float(subset.loc[subset["case_id"] == case_id, metric_mean].iloc[0]) for case_id in cases]
            stds = [float(subset.loc[subset["case_id"] == case_id, metric_std].iloc[0]) for case_id in cases]
            axis.errorbar(
                x_positions,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=MODEL_COLORS[model],
                label=_model_label(model),
            )
        axis.set_title(ylabel, fontsize=11)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(lambda_labels)
        axis.set_xlabel(r"Cefalú ($\lambda=$)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        if use_log:
            axis.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1), frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("GlobalCY II diagnostic trajectories across the Cefalú hard-regime sweep", fontsize=13, y=1.05)
    figure.subplots_adjust(top=0.8, wspace=0.28)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_paper2_degradation_profiles(frame: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(["lambda", "model"]).reset_index(drop=True)
    cases = _case_order(ordered)
    lambda_labels = _lambda_tick_labels(ordered)
    x_positions = list(range(len(cases)))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    metrics = [
        ("negative_fraction_mean", "Negative-fraction deterioration", "lower_better"),
        ("projective_invariance_drift_mean", "Projective-drift deterioration", "lower_better"),
        ("spectral_tail_mean_mean", "Spectral-tail deterioration", "higher_better"),
    ]

    for axis, (metric, title, direction) in zip(axes, metrics):
        for model in [model for model in ("local", "global") if model in set(ordered["model"])]:
            subset = ordered.loc[ordered["model"] == model].sort_values("lambda")
            values = subset[metric].astype(float).tolist()
            baseline = max(values[0], 1e-12)
            if direction == "lower_better":
                profile = [value / baseline for value in values]
            else:
                profile = [baseline / max(value, 1e-12) for value in values]
            axis.plot(
                x_positions,
                profile,
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[model],
                label=_model_label(model),
            )
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Relative deterioration vs lambda=0.50")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(lambda_labels)
        axis.set_xlabel(r"Cefalú ($\lambda=$)")
        axis.axhline(1.0, linestyle="--", linewidth=0.8, color="#666666")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1), frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("GlobalCY II degradation profiles across the Cefalú hard-regime sweep", fontsize=13, y=1.05)
    figure.subplots_adjust(top=0.8, wspace=0.28)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_paper2_objective_ablation(frame: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(["lambda", "objective_variant"]).reset_index(drop=True)
    cases = _case_order(ordered.rename(columns={"objective_variant": "model"}))
    variants = ordered["objective_variant"].drop_duplicates().astype(str).tolist()
    width = 0.18
    x_positions = list(range(len(cases)))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    metrics = [
        ("negative_fraction_mean", "negative_fraction_std", "Negative fraction", False),
        ("projective_invariance_drift_mean", "projective_invariance_drift_std", "Projective-invariance drift", True),
        ("spectral_tail_mean_mean", "spectral_tail_mean_std", "Spectral-tail mean", False),
    ]
    palette = ["#6C757D", "#2A6F97", "#8E7DBE", "#C66B3D"]

    for axis, (metric_mean, metric_std, title, use_log) in zip(axes, metrics):
        for index, variant in enumerate(variants):
            subset = ordered.loc[ordered["objective_variant"] == variant]
            means = [float(subset.loc[subset["case_id"] == case_id, metric_mean].iloc[0]) for case_id in cases]
            stds = [float(subset.loc[subset["case_id"] == case_id, metric_std].iloc[0]) for case_id in cases]
            offsets = [position + (index - (len(variants) - 1) / 2) * width for position in x_positions]
            axis.bar(offsets, means, width=width, yerr=stds, capsize=3, color=palette[index % len(palette)], edgecolor="black", linewidth=0.6, label=variant)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([_case_label(case_id) for case_id in cases], rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        if use_log:
            axis.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    figure.suptitle("GlobalCY II geometry-aware objective ablation", fontsize=13, y=1.06)
    figure.subplots_adjust(top=0.78, wspace=0.28)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_paper2_hardest_case(frame: pd.DataFrame, ablation_frame: pd.DataFrame, path: str, hardest_case_id: str) -> None:
    import matplotlib.pyplot as plt

    sweep_subset = frame.loc[frame["case_id"] == hardest_case_id].copy()
    ablation_subset = ablation_frame.loc[ablation_frame["case_id"] == hardest_case_id].copy()
    best_variant = ablation_subset.sort_values(["negative_fraction_mean", "projective_invariance_drift_mean"]).iloc[0]
    comparison_rows = [
        {
            "label": _model_label(model),
            "negative_fraction_mean": float(row["negative_fraction_mean"]),
            "negative_fraction_std": float(row["negative_fraction_std"]),
            "projective_invariance_drift_mean": float(row["projective_invariance_drift_mean"]),
            "projective_invariance_drift_std": float(row["projective_invariance_drift_std"]),
            "spectral_tail_mean_mean": float(row["spectral_tail_mean_mean"]),
            "spectral_tail_mean_std": float(row["spectral_tail_mean_std"]),
            "color": MODEL_COLORS[model],
        }
        for model, row in (
            ("local", sweep_subset.loc[sweep_subset["model"] == "local"].iloc[0]),
            ("global", sweep_subset.loc[sweep_subset["model"] == "global"].iloc[0]),
        )
        if model in set(sweep_subset["model"])
    ]
    comparison_rows.append(
        {
            "label": f"Global + {best_variant['objective_variant']}",
            "negative_fraction_mean": float(best_variant["negative_fraction_mean"]),
            "negative_fraction_std": float(best_variant["negative_fraction_std"]),
            "projective_invariance_drift_mean": float(best_variant["projective_invariance_drift_mean"]),
            "projective_invariance_drift_std": float(best_variant["projective_invariance_drift_std"]),
            "spectral_tail_mean_mean": float(best_variant["spectral_tail_mean_mean"]),
            "spectral_tail_mean_std": float(best_variant["spectral_tail_mean_std"]),
            "color": "#8E7DBE",
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("negative_fraction_mean", "negative_fraction_std", "Negative fraction", False),
        ("projective_invariance_drift_mean", "projective_invariance_drift_std", "Projective-invariance drift", True),
        ("spectral_tail_mean_mean", "spectral_tail_mean_std", "Spectral-tail mean", False),
    ]
    labels = [row["label"] for row in comparison_rows]
    x_positions = list(range(len(comparison_rows)))
    colors = [row["color"] for row in comparison_rows]
    for axis, (metric_mean, metric_std, title, use_log) in zip(axes, metrics):
        means = [row[metric_mean] for row in comparison_rows]
        stds = [row[metric_std] for row in comparison_rows]
        axis.bar(x_positions, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        if use_log:
            axis.set_yscale("log")

    figure.suptitle(f"GlobalCY II hardest-case comparison: {_case_title_label(hardest_case_id)}", fontsize=13, y=1.04)
    figure.subplots_adjust(top=0.82, wspace=0.28)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
