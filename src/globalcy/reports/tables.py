from __future__ import annotations

import pandas as pd


def metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([metrics])
