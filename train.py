"""Train a ResNet-50 classifier from a YAML experiment config.

Examples:
    python train.py --config configs/ant_bee_scratch.yaml
    python train.py --config configs/ant_bee_transfer.yaml --set train.epochs=5
"""

from __future__ import annotations

import argparse
import json

from dlresnet.config import load_config
from dlresnet.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to an experiment YAML file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=5 (repeatable)",
    )
    args = parser.parse_args()

    result = train(load_config(args.config, args.overrides))

    summary = {
        "run_dir": str(result.run_dir),
        "model_path": str(result.model_path),
        "epochs_run": len(result.history.get("loss", [])),
        "final": {k: v[-1] for k, v in result.history.items() if v},
    }
    if result.test_metrics is not None:
        summary["test"] = {k: result.test_metrics[k] for k in ("loss", "accuracy", "num_samples")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
