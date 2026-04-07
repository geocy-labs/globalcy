"""Reproducibility helpers."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path


def package_versions() -> dict[str, str]:
    """Return key package versions used by GlobalCY."""

    versions: dict[str, str] = {}
    for module_name in ("jax", "jaxlib", "numpy", "pandas", "yaml"):
        module = importlib.import_module(module_name)
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_json(payload: dict[str, object], path: str | Path) -> None:
    """Write one JSON payload."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
