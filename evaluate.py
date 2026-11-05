"""Evaluate a trained model on an image folder (one sub-folder per class).

Example:
    python evaluate.py runs/birds_scratch --data-dir data/100-bird-species/test
"""

from __future__ import annotations

import argparse
import json

from dlresnet.evaluation import evaluate_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="run directory or .keras file")
    parser.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", help="optional path to write the metrics JSON")
    args = parser.parse_args()

    metrics = evaluate_dir(args.model, args.data_dir, batch_size=args.batch_size)
    text = json.dumps(metrics, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    keys = ("model", "num_samples", "loss", "accuracy", "top5")
    summary = {k: metrics[k] for k in keys if k in metrics}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
