"""Plot helpers for experiment figures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_bars(df, x: str, y: str, group: str, out_path: str | Path, title: str | None = None) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, grp in df.groupby(group):
        ax.plot(grp[x], grp[y], marker="o", label=str(key))
    if title:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    return str(out_path)


def plot_box(df, x: str, y: str, out_path: str | Path, title: str | None = None) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = [grp[y].values for _, grp in df.groupby(x)]
    labels = [str(k) for k, _ in df.groupby(x)]
    ax.boxplot(groups, labels=labels)
    if title:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    return str(out_path)
