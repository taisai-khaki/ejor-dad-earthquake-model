"""Statistical utilities for experiment result aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ComparisonResult:
    test: str
    statistic: float
    p_value: float
    effect_size: float
    effect_name: str
    n: int


def grouped_summary(
    df: pd.DataFrame,
    group_by: List[str],
    metrics: List[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return mean/std/count summaries by group."""
    if df.empty:
        return pd.DataFrame()
    groups = df.groupby(group_by)
    out = groups[metrics].agg(["mean", "std", "count"]).reset_index()
    if confidence is None:
        return out
    return out


def bootstrap_ci(values: Iterable[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 2026) -> tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1.0 - alpha / 2)))
    return lower, upper


def paired_comparison(a: Iterable[float], b: Iterable[float]) -> ComparisonResult:
    a_arr = np.array(list(a), dtype=float)
    b_arr = np.array(list(b), dtype=float)
    n = min(a_arr.size, b_arr.size)
    if n == 0:
        return ComparisonResult("wilcoxon", float("nan"), float("nan"), float("nan"), "rank_biserial", 0)

    a_arr = a_arr[:n]
    b_arr = b_arr[:n]

    # Paired non-parametric test where possible; fallback to t-test on edge cases.
    try:
        stat, p = stats.wilcoxon(a_arr, b_arr)
        test_name = "wilcoxon"
        diff = a_arr - b_arr
        nz = diff[np.abs(diff) > 1e-12]
        if nz.size == 0:
            effect = 0.0
        else:
            ranks = stats.rankdata(np.abs(nz))
            w_pos = float(np.sum(ranks[nz > 0]))
            w_neg = float(np.sum(ranks[nz < 0]))
            denom = w_pos + w_neg
            effect = 0.0 if denom == 0 else float((w_pos - w_neg) / denom)
        effect_name = "rank_biserial"
    except ValueError:
        stat, p = stats.ttest_rel(a_arr, b_arr, nan_policy="omit")
        test_name = "ttest_rel"
        diff = a_arr - b_arr
        denom = 0.0 if n < 2 else np.std(diff, ddof=1)
        effect = 0.0 if denom == 0 else float(np.mean(diff) / denom)
        effect_name = "cohens_dz"
    return ComparisonResult(test_name, float(stat), float(p), effect, effect_name, int(n))


def effect_size(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    """Compute Cohen's d and relative mean effect."""
    a_arr = np.array(list(a), dtype=float)
    b_arr = np.array(list(b), dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return {"mean_gap": float("nan"), "rel_gap": float("nan"), "cohens_d": float("nan")}

    n = min(a_arr.size, b_arr.size)
    a_arr = a_arr[:n]
    b_arr = b_arr[:n]
    diff = b_arr - a_arr
    mean_a = float(np.mean(a_arr))
    mean_b = float(np.mean(b_arr))
    pooled_std = float(np.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)) if n > 1 else 0.0
    cohens_d = 0.0 if pooled_std == 0 else float(np.mean(diff) / pooled_std)
    return {"mean_gap": mean_b - mean_a, "rel_gap": (mean_b - mean_a) / (abs(mean_a) + 1e-9), "cohens_d": cohens_d}


def robustness_summary(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    method_col: str = "method",
) -> pd.DataFrame:
    """Summarize results over multiple random seeds or replications."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    rows = []
    for method, sub in df.groupby(method_col):
        group_values = sub.groupby(group_col)[metric].apply(list)
        for key, vals in group_values.items():
            lo, hi = bootstrap_ci(vals)
            rows.append(
                {
                    "method": method,
                    group_col: key,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "count": int(len(vals)),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)
