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
    "cefalu_lambda_0_75": "Cefalu lambda=0.75",
    "cefalu_lambda_1_0": "Cefalu lambda=1.0",
}


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def _case_label(case_id: str) -> str:
    return CASE_LABELS.get(case_id, case_id)


def _case_title_label(case_id: str) -> str:
    return {
        "cefalu_lambda_0_75": r"Cefalu $\lambda = .75$",
        "cefalu_lambda_1_0": r"Cefalu $\lambda = 1.0$",
    }.get(case_id, _case_label(case_id))


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
    figure.suptitle("Core comparison across hard Cefalu cases", fontsize=13, y=1.04)
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
