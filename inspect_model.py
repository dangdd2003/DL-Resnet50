"""Show a saved model's input shape, classes and parameter counts.

Example:
    python inspect_model.py runs/ant_bee_scratch
"""

from __future__ import annotations

import argparse
import json

from dlresnet.inspection import describe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="run directory or .keras file")
    args = parser.parse_args()
    print(json.dumps(describe(args.model), indent=2))


if __name__ == "__main__":
    main()
