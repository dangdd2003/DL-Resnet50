"""Command-line interface: ``dlresnet {train,evaluate,predict,inspect}`` (JSON on stdout)."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlresnet", description="ResNet-50 image classification (Keras 3)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model from a YAML config")
    p.add_argument("--config", required=True, help="path to an experiment YAML file")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=5 (repeatable)",
    )

    p = sub.add_parser("evaluate", help="evaluate a saved model on an image folder")
    p.add_argument("model", help="run directory or .keras file")
    p.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    p.add_argument("--batch-size", type=int, default=32)

    p = sub.add_parser("predict", help="classify images with a saved model")
    p.add_argument("model", help="run directory or .keras file")
    p.add_argument("images", nargs="+", help="image files")
    p.add_argument("--top-k", type=int, default=5)

    p = sub.add_parser("inspect", help="show input shape, classes and parameter counts")
    p.add_argument("model", help="run directory or .keras file")
    return parser


def _run(args: argparse.Namespace) -> Any:
    if args.command == "train":
        from dlresnet.config import load_config
        from dlresnet.train import train

        result = train(load_config(args.config, args.overrides))
        return {
            "run_dir": str(result.run_dir),
            "model_path": str(result.model_path),
            "epochs_run": len(result.history.get("loss", [])),
            "final": {k: v[-1] for k, v in result.history.items() if v},
            "test_metrics": (
                None
                if result.test_metrics is None
                else {k: result.test_metrics[k] for k in ("loss", "accuracy", "num_samples")}
            ),
        }
    if args.command == "evaluate":
        from dlresnet.evaluate import evaluate_dir

        return evaluate_dir(args.model, args.data_dir, batch_size=args.batch_size)
    if args.command == "predict":
        from dlresnet.evaluate import load_run
        from dlresnet.predict import predict

        model, class_names = load_run(args.model)
        return predict(model, class_names, args.images, top_k=args.top_k)
    if args.command == "inspect":
        from dlresnet.inspect_model import describe

        return describe(args.model)
    raise ValueError(f"Unknown command {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = _parser().parse_args(argv)
    out = sys.stdout
    try:
        # Keras/TF progress output goes to stderr so stdout stays machine-readable JSON.
        with contextlib.redirect_stdout(sys.stderr):
            result = _run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out.write(json.dumps(result, indent=2) + "\n")
    out.flush()
    return 0
