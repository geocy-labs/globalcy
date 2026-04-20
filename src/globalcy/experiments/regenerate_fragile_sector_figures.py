from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["local", "global"]
FRAGILE_ORDER = ["regular", "fragile"]
COLORS = {
    "regular": "#2a6f97",
    "fragile": "#b63a3a",
}
LAMBDA_TICKS = [0.50, 0.75, 0.90, 1.00, 1.10]
LAMBDA_LABELS = ["0.50", "0.75", "0.90", "1.00", "1.10"]


def _resolve_default_summary_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "artifacts" / "paper2_fragile_vs_regular_summary.csv"


def _resolve_default_out_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "figures"


def _make_two_panel_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True)

    for ax, model_name in zip(axes, MODEL_ORDER):
        model_frame = frame.loc[frame["model_name"] == model_name].sort_values(
            ["lambda_value", "fragile_flag"]
        )
        for fragile_label in FRAGILE_ORDER:
            subset = model_frame.loc[model_frame["fragile_label"] == fragile_label].sort_values(
                "lambda_value"
            )
            ax.plot(
                subset["lambda_value"],
                subset[metric],
                marker="o",
                linewidth=2,
                color=COLORS[fragile_label],
                label=fragile_label.capitalize(),
            )
        ax.set_title(model_name.capitalize())
        ax.set_xlabel("Cefal\\'u ($\\lambda=$)")
        ax.set_xticks(LAMBDA_TICKS)
        ax.set_xticklabels(LAMBDA_LABELS)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def regenerate_figures(summary_csv: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(summary_csv)

    outputs = [
        (
            "negative_frequency",
            "Negative frequency",
            out_dir / "fig_paper2_fragile_vs_regular_negfreq.png",
            "GlobalCY II fragile vs regular negative-frequency support shadow",
        ),
        (
            "spectral_tail_mean",
            "Spectral tail mean",
            out_dir / "fig_paper2_fragile_vs_regular_tailmean.png",
            "GlobalCY II fragile vs regular lower-tail support shadow",
        ),
        (
            "logdet_residual_q90",
            "logdet residual q90",
            out_dir / "fig_paper2_fragile_vs_regular_residual.png",
            "GlobalCY II equation-facing residual concentration by support sector",
        ),
        (
            "euler_density_proxy_weighted_mean",
            "Weighted Euler density proxy",
            out_dir / "fig_paper2_fragile_vs_regular_euler_proxy.png",
            "GlobalCY II class-shadow style concentration by support sector",
        ),
    ]

    written: list[Path] = []
    for metric, ylabel, out_path, title in outputs:
        _make_two_panel_figure(frame, metric=metric, ylabel=ylabel, out_path=out_path, title=title)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default=str(_resolve_default_summary_path()))
    parser.add_argument("--out-dir", default=str(_resolve_default_out_dir()))
    args = parser.parse_args()

    written = regenerate_figures(Path(args.summary_csv), Path(args.out_dir))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
