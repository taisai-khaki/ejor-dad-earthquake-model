"""Tabular exporters used by experiment drivers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd


def save_results_csv(df, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def append_markdown_table(df, output_path: str | Path) -> str:
    output_path = Path(output_path)
    md = df.to_markdown(index=False)
    output_path.write_text(md, encoding="utf-8")
    return md


def summary_by_instance(rows: Iterable[dict], group_cols=("instance_name",), value_col="objective") -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    return df.groupby(list(group_cols))[value_col].agg(["count", "mean", "std", "min", "max"]).reset_index()
