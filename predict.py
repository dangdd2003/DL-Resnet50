"""Classify images with a trained model (top-k labels and probabilities).

Example:
    python predict.py runs/ant_bee_transfer path/to/bee.jpg path/to/ant.jpg --top-k 2
"""

from __future__ import annotations

import argparse
import json

from dlresnet.evaluation import load_run
from dlresnet.inference import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="run directory or .keras file")
    parser.add_argument("images", nargs="+", help="image files")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model, class_names = load_run(args.model)
    print(json.dumps(predict(model, class_names, args.images, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
