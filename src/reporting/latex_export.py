"""LaTeX export helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def dataframe_to_latex(df: pd.DataFrame, output_path: str | Path, caption: str = "", label: str = "") -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, float_format="%.4f")
    if caption:
        latex = f"% {caption}\n" + latex
    if label:
        latex = f"% {label}\n" + latex
    output_path.write_text(latex, encoding="utf-8")
    return latex
