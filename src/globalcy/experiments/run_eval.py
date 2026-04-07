from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
