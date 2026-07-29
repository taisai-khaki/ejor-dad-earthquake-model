from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PolicySelection:
    selected: Any
    numerical_minimum: float
    selected_objective: float
    objective_tolerance: float
    tied_policy_count: int


def select_lexicographic_best(
    results: Sequence[Any], objective_tolerance: float | None = None
) -> PolicySelection:
    """Select a reproducible policy among numerically tied objective values."""
    if not results:
        raise ValueError("results must contain at least one policy.")
    numerical_minimum = min(float(result.objective) for result in results)
    tolerance = (
        max(1e-8, 1e-10 * max(1.0, abs(numerical_minimum)))
        if objective_tolerance is None
        else float(objective_tolerance)
    )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and nonnegative.")
    tied = [result for result in results if float(result.objective) <= numerical_minimum + tolerance]
    selected = min(tied, key=lambda result: tuple(float(value) for value in np.asarray(result.y, dtype=float)))
    return PolicySelection(
        selected=selected,
        numerical_minimum=numerical_minimum,
        selected_objective=float(selected.objective),
        objective_tolerance=tolerance,
        tied_policy_count=len(tied),
    )
