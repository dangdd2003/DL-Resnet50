#!/usr/bin/env python3
"""Download the Kaggle datasets used by the ResNet-50 training configs.

Layouts produced under ``--data-dir`` (default ``data``)::

    data/hymenoptera/{train,val}/{ants,bees}/...
    data/100-bird-species/{train,valid,test}/<class>/...

Requirements: ``pip install kagglehub`` and Kaggle credentials, either
``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables.

Usage::

    python scripts/download_datasets.py {hymenoptera,birds,all} [--data-dir data] [--force]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif"})
MAX_SEARCH_DEPTH = 4
MAX_LISTED_DIRS = 20

KAGGLEHUB_HELP = (
    "kagglehub is required. Install it with `pip install kagglehub`, then set up Kaggle "
    "credentials via ~/.kaggle/kaggle.json or the KAGGLE_USERNAME and KAGGLE_KEY "
    "environment variables."
)


@dataclass(frozen=True)
class Dataset:
    """A Kaggle dataset, the local folder it is copied to, and the splits it must provide."""

    handle: str
    target: str
    splits: tuple[str, ...]


DATASETS: dict[str, Dataset] = {
    "hymenoptera": Dataset("thedatasith/hymenoptera", "hymenoptera", ("train", "val")),
    "birds": Dataset(
        "gpiosenka/100-bird-species", "100-bird-species", ("train", "valid", "test")
    ),
}


def _import_kagglehub() -> ModuleType:
    """Import kagglehub lazily so that the rest of the script works without it."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(KAGGLEHUB_HELP) from exc
    return kagglehub


def download_dataset(handle: str) -> Path:
    """Download a Kaggle dataset with kagglehub and return its local cache path."""
    kagglehub = _import_kagglehub()
    try:
        return Path(kagglehub.dataset_download(handle))
    except Exception as exc:
        raise RuntimeError(
            f"failed to download '{handle}': {exc}. Check your Kaggle credentials "
            "and that your account can access the dataset."
        ) from exc


def _subdirs(path: Path) -> list[Path]:
    """Return the visible sub-directories of ``path`` (sorted), or [] if unreadable."""
    try:
        with os.scandir(path) as entries:
            return sorted(
                Path(entry.path)
                for entry in entries
                if entry.is_dir() and not entry.name.startswith(".")
            )
    except OSError:
        return []


def find_split_root(root: Path, splits: Sequence[str]) -> Path:
    """Locate the shallowest directory under ``root`` that contains all ``splits``.

    The search is breadth-first and limited to ``MAX_SEARCH_DEPTH`` levels, so the
    shallowest match wins. Raises ``FileNotFoundError`` if no directory matches.
    """
    required = set(splits)
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    while queue:
        current, depth = queue.popleft()
        children = _subdirs(current)
        if required <= {child.name for child in children}:
            return current
        if depth < MAX_SEARCH_DEPTH:
            queue.extend((child, depth + 1) for child in children)

    top_level = [child.name for child in _subdirs(root)][:MAX_LISTED_DIRS]
    found = ", ".join(top_level) or "(no sub-directories)"
    raise FileNotFoundError(
        f"could not find a directory containing {', '.join(splits)} within "
        f"{MAX_SEARCH_DEPTH} levels of {root}. Top-level directories found: {found}"
    )


def count_images(folder: Path) -> int:
    """Count image files (case-insensitive extension match) anywhere under ``folder``."""
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def verify_layout(root: Path, splits: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Check the ``root/<split>/<class>/`` layout and return ``{split: (classes, images)}``.

    Raises ``ValueError`` if a split is missing, the splits do not share the same class
    folders, or any class folder contains no images.
    """
    class_names: dict[str, set[str]] = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise ValueError(f"split '{split}' is missing under {root}")
        class_names[split] = {child.name for child in _subdirs(split_dir)}
        if not class_names[split]:
            raise ValueError(f"split '{split}' under {root} has no class folders")

    reference_split = splits[0]
    reference = class_names[reference_split]
    for split in splits[1:]:
        if class_names[split] != reference:
            missing = sorted(reference - class_names[split])[:5]
            extra = sorted(class_names[split] - reference)[:5]
            raise ValueError(
                f"class folders differ between '{reference_split}' and '{split}' "
                f"(first missing from '{split}': {missing}; first extra in '{split}': {extra})"
            )

    summary: dict[str, tuple[int, int]] = {}
    for split in splits:
        images = 0
        for class_dir in _subdirs(root / split):
            count = count_images(class_dir)
            if count == 0:
                raise ValueError(f"class folder '{split}/{class_dir.name}' contains no images")
            images += count
        summary[split] = (len(class_names[split]), images)
    return summary


def copy_splits(src_root: Path, dest: Path, splits: Sequence[str]) -> None:
    """Copy only the required split folders from ``src_root`` into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    for split in splits:
        shutil.copytree(src_root / split, dest / split, dirs_exist_ok=True)


def format_summary(name: str, dest: Path, summary: dict[str, tuple[int, int]]) -> str:
    """Render the per-split table printed after each dataset is verified."""
    lines = [f"{name}: {dest}", f"{'split':<10}{'classes':>10}{'images':>10}"]
    for split, (classes, images) in summary.items():
        lines.append(f"{split:<10}{classes:>10}{images:>10}")
    return "\n".join(lines)


def process_dataset(name: str, data_dir: Path, force: bool) -> dict[str, tuple[int, int]]:
    """Download (unless already present), copy, verify and summarise one dataset."""
    spec = DATASETS[name]
    dest = data_dir / spec.target
    if dest.exists() and not force:
        print(f"[{name}] {dest} already exists; skipping (use --force to replace it).")
    else:
        print(f"[{name}] downloading {spec.handle} via kagglehub ...")
        src = download_dataset(spec.handle)
        split_root = find_split_root(src, spec.splits)
        if dest.exists():
            shutil.rmtree(dest)
        copy_splits(split_root, dest, spec.splits)
    summary = verify_layout(dest, spec.splits)
    print(format_summary(name, dest, summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Download the Kaggle training datasets.")
    parser.add_argument(
        "dataset",
        choices=[*DATASETS, "all"],
        help="which dataset to fetch: hymenoptera, birds, or all",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory to place datasets in (default: data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing dataset folder instead of skipping it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on error."""
    args = build_parser().parse_args(argv)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    try:
        for name in names:
            process_dataset(name, args.data_dir, args.force)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
