"""Training-curve plots (headless matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_history(history: dict[str, list[float]], path: str | Path) -> Path:
    """Save a two-panel (loss, accuracy) train-vs-validation plot to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    try:
        for ax, metric, title in (
            (axes[0], "loss", "Sparse categorical cross-entropy"),
            (axes[1], "accuracy", "Accuracy"),
        ):
            for key, label in ((metric, "train"), (f"val_{metric}", "validation")):
                values = history.get(key)
                if values:
                    ax.plot(range(1, len(values) + 1), values, marker="o", label=label)
            ax.set_title(title)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            if ax.get_lines():
                ax.legend()
        axes[1].set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return path
