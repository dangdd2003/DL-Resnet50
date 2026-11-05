"""Summarize training runs into a GitHub-flavoured markdown table.

Each run is a directory under ``runs/<run_name>/``. Only directories containing
``history.json`` are summarised. ``config.yaml`` (for the model architecture) and
``metrics.json`` (for test-set results) are optional.

Usage::

    python scripts/summarize_runs.py [runs_dir] [--output FILE.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

HEADER = "| Run | Model | Epochs | Best val acc (epoch) | Final val acc | Test acc | Test top-5 |"
MISSING = "—"  # em dash, used for absent values


def _read_json(path: Path) -> Any | None:
    """Return parsed JSON from ``path``, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_arch(config_path: Path) -> str | None:
    """Return ``model.arch`` from a run's config.yaml, or None if unavailable."""
    if not config_path.is_file():
        return None
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(cfg, dict):
        return None
    model = cfg.get("model")
    arch = model.get("arch") if isinstance(model, dict) else None
    return str(arch) if arch is not None else None


def _is_number(value: Any) -> bool:
    """Return True for int/float values, excluding bools."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _best_epoch(values: list[Any]) -> tuple[float, int] | None:
    """Return (max value, 1-based epoch of its first occurrence), ignoring non-numbers."""
    best: tuple[float, int] | None = None
    for index, value in enumerate(values):
        if not _is_number(value):
            continue
        if best is None or value > best[0]:
            best = (float(value), index + 1)
    return best


def _summarise_run(run_dir: Path, history: dict[str, Any]) -> dict[str, Any]:
    """Build one summary row for a run from its already-loaded history."""
    loss = history.get("loss")
    val_acc = history.get("val_accuracy")
    val_acc = val_acc if isinstance(val_acc, list) else []
    best = _best_epoch(val_acc)
    final_val = val_acc[-1] if val_acc and _is_number(val_acc[-1]) else None

    metrics = _read_json(run_dir / "metrics.json")
    if not isinstance(metrics, dict):
        metrics = {}
    test_acc = metrics.get("accuracy")
    test_top5 = metrics.get("top5")

    return {
        "run": run_dir.name,
        "arch": _read_arch(run_dir / "config.yaml"),
        "epochs": len(loss) if isinstance(loss, list) else None,
        "best_val_acc": best[0] if best else None,
        "best_epoch": best[1] if best else None,
        "final_val_acc": float(final_val) if final_val is not None else None,
        "test_acc": float(test_acc) if _is_number(test_acc) else None,
        "test_top5": float(test_top5) if _is_number(test_top5) else None,
    }


def collect(runs_dir: Path) -> list[dict]:
    """Collect one summary row per run sub-directory that has a history.json.

    Args:
        runs_dir: Directory whose immediate sub-directories are training runs.

    Returns:
        Rows sorted by run name. Each row has the keys ``run``, ``arch``, ``epochs``,
        ``best_val_acc``, ``best_epoch``, ``final_val_acc``, ``test_acc`` and
        ``test_top5``; values that cannot be determined are None.
    """
    rows: list[dict] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        history = _read_json(run_dir / "history.json")
        if not isinstance(history, dict):
            continue
        rows.append(_summarise_run(run_dir, history))
    return rows


def _pct(value: float | None) -> str:
    """Format a 0-1 accuracy as a percentage with one decimal place."""
    return MISSING if value is None else f"{value * 100:.1f}%"


def _cell(text: str) -> str:
    """Escape a value for use inside a markdown table cell."""
    return text.replace("|", "\\|")


def to_markdown(rows: list[dict]) -> str:
    """Render summary rows as a GitHub markdown table.

    Args:
        rows: Rows as returned by :func:`collect`.

    Returns:
        The table text, one line per row, terminated by a newline.
    """
    lines = [HEADER, "|" + "---|" * 7]
    for row in rows:
        if row["best_val_acc"] is None:
            best = MISSING
        else:
            best = f"{_pct(row['best_val_acc'])} ({row['best_epoch']})"
        cells = [
            _cell(str(row["run"])),
            _cell(str(row["arch"])) if row["arch"] is not None else MISSING,
            str(row["epochs"]) if row["epochs"] is not None else MISSING,
            best,
            _pct(row["final_val_acc"]),
            _pct(row["test_acc"]),
            _pct(row["test_top5"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, 1 if the runs directory is missing or contains no runs.
    """
    parser = argparse.ArgumentParser(description="Summarize training runs as a markdown table.")
    parser.add_argument(
        "runs_dir",
        nargs="?",
        default="runs",
        type=Path,
        help="directory containing one sub-directory per run (default: runs)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="also write the markdown table to this file"
    )
    args = parser.parse_args(argv)

    if not args.runs_dir.is_dir():
        print(f"error: runs directory not found: {args.runs_dir}", file=sys.stderr)
        return 1

    rows = collect(args.runs_dir)
    if not rows:
        print(f"error: no runs with history.json found in {args.runs_dir}", file=sys.stderr)
        return 1

    table = to_markdown(rows)
    print(table, end="")
    if args.output is not None:
        try:
            args.output.write_text(table, encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {args.output}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
