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
