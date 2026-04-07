from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from globalcy.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    model_types = ["local", "global", "symmetry_aware"]
    run_root = Path("outputs") / f"{config['run_name']}_ablation"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []

    for model_type in model_types:
        temp_config = dict(config)
        temp_config["model"] = dict(config["model"])
        temp_config["model"]["type"] = model_type
        derived_config = run_root / f"{model_type}.json"
        derived_config.write_text(json.dumps(temp_config), encoding="utf-8")
        model_run_dir = run_root / model_type
        subprocess.run([sys.executable, "-m", "globalcy.experiments.run_train", "--config", str(derived_config), "--out", str(model_run_dir)], check=True)
        run_dirs.append(model_run_dir)

    compare_command = [sys.executable, "-m", "globalcy.experiments.run_compare", "--out", str(run_root / "comparison")]
    for run_dir in run_dirs:
        compare_command.extend(["--run-dir", str(run_dir)])
    subprocess.run(compare_command, check=True)


if __name__ == "__main__":
    main()
