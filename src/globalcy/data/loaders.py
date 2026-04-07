from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle_tables(bundle_path: str | Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_path).resolve()
    manifest = _read_json(bundle_dir / "manifest.json")
    tables: dict[str, Any] = {
        "bundle_dir": bundle_dir,
        "manifest": manifest,
        "points": pd.read_parquet(bundle_dir / "points.parquet"),
        "invariants": pd.read_parquet(bundle_dir / "invariants.parquet"),
    }

    optional_tables = {
        "sample_weights": "sample_weights.parquet",
        "canonical_representatives": "canonical_representatives.parquet",
        "canonical_invariants": "canonical_invariants.parquet",
        "orbits": "orbits.parquet",
    }
    for key, name in optional_tables.items():
        path = bundle_dir / name
        if path.exists():
            tables[key] = pd.read_parquet(path)

    optional_json = {
        "case_metadata": "case_metadata.json",
        "evaluation_summary": "evaluation_summary.json",
        "symmetry_report": "symmetry_report.json",
    }
    for key, name in optional_json.items():
        path = bundle_dir / name
        if path.exists():
            tables[key] = _read_json(path)

    return tables
