from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

import numpy as np

from ejor_dad.model import DADInstance, PiecewiseLinearResponseParams, ThresholdResponseParams


@dataclass(frozen=True)
class GridCell:
    """An axis-aligned retrofit cell used by the upper-corner certificate."""

    index: int
    lower: np.ndarray
    upper: np.ndarray
    lower_budget_used: float
    upper_budget_used: float


@dataclass(frozen=True)
class ContinuousGridCertificate:
    """Valid lower/upper bounds for the continuous road-retrofit problem."""

    grid_upper_bound: float
    continuous_lower_bound: float
    absolute_gap: float
    relative_gap_percent: float
    evaluated_cell_count: int
    lower_bound_cell: GridCell


def validate_upper_corner_certificate_instance(instance: DADInstance) -> None:
    """Validate the assumptions required by the monotonic upper-corner bound.

    This certificate is intentionally narrow: complete regime-labelled no-tail
    states, monotone timely-access response, nonnegative disruption penalties,
    and capped-TV ambiguity without moment restrictions. Those conditions make
    the robust fixed-y value nonincreasing in each retrofit component.
    """
    if not instance.links:
        raise ValueError("The continuous grid certificate requires at least one retrofit link.")
    if any(state.is_tail for state in instance.states):
        raise ValueError("The continuous grid certificate is defined only for complete no-tail state sets.")
    expected_states = {
        tuple(sorted(combo))
        for count in range(len(instance.links) + 1)
        for combo in combinations(instance.link_ids, count)
    }
    if instance.hazard_regimes is None:
        actual_states = {tuple(state.failed_links) for state in instance.states}
        if len(actual_states) != len(instance.states) or actual_states != expected_states:
            raise ValueError("The continuous grid certificate requires every independent failure state exactly once.")
    else:
        regime_lookup = {regime.id: regime for regime in instance.hazard_regimes}
        for regime_id, regime in regime_lookup.items():
            regime_states = [state for state in instance.states if state.hazard_regime_id == regime_id]
            actual_patterns = {tuple(state.failed_links) for state in regime_states}
            if len(regime_states) != len(expected_states) or actual_patterns != expected_states:
                raise ValueError("Each hazard regime must contain every road-failure pattern exactly once.")
            if any(tuple(state.failed_centers) != tuple(regime.failed_centers) for state in regime_states):
                raise ValueError("Facility availability must be fixed within each hazard regime.")
    if instance.failure_moment_envelope is not None:
        raise ValueError("Moment-constrained ambiguity is not covered by the law-invariant upper-corner certificate.")
    if not isinstance(instance.survival, (ThresholdResponseParams, PiecewiseLinearResponseParams)):
        raise ValueError("The continuous grid certificate requires a monotone timely-access response.")
    if (
        isinstance(instance.survival, ThresholdResponseParams)
        and instance.survival.timely_access_value + 1e-12 < instance.survival.late_access_value
    ):
        raise ValueError("The timely-access response must be nonincreasing in travel time.")


def budget_intersecting_grid_cells(
    retrofit_costs: Sequence[float],
    retrofit_budget: float,
    levels: Sequence[float],
    atol: float = 1e-10,
) -> tuple[GridCell, ...]:
    """Return all cells whose lower corner is road-budget feasible.

    Every continuous feasible retrofit vector lies in one of these cells. Its
    upper corner can exceed the road budget; that corner is evaluated only as a
    monotonicity lower bound, never as a feasible policy.
    """
    costs = np.asarray(retrofit_costs, dtype=float)
    grid = np.asarray(levels, dtype=float)
    if costs.ndim != 1 or costs.size == 0 or not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("retrofit_costs must be a nonempty finite nonnegative vector.")
    if not np.isfinite(retrofit_budget) or retrofit_budget < 0.0:
        raise ValueError("retrofit_budget must be finite and nonnegative.")
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("levels must be a finite one-dimensional grid with at least two entries.")
    if not np.isclose(grid[0], 0.0, atol=atol) or not np.isclose(grid[-1], 1.0, atol=atol):
        raise ValueError("The certificate grid must start at 0 and end at 1.")
    if np.any(np.diff(grid) <= atol) or np.any(grid < -atol) or np.any(grid > 1.0 + atol):
        raise ValueError("The certificate grid must be strictly increasing within [0, 1].")

    cells: list[GridCell] = []
    for index, level_indices in enumerate(product(range(grid.size - 1), repeat=costs.size), start=1):
        lower = np.asarray([grid[level_index] for level_index in level_indices], dtype=float)
        upper = np.asarray([grid[level_index + 1] for level_index in level_indices], dtype=float)
        lower_budget_used = float(costs @ lower)
        if lower_budget_used > retrofit_budget + atol:
            continue
        cells.append(
            GridCell(
                index=index,
                lower=lower,
                upper=upper,
                lower_budget_used=lower_budget_used,
                upper_budget_used=float(costs @ upper),
            )
        )
    if not cells:
        raise RuntimeError("No budget-intersecting grid cells were generated.")
    return tuple(cells)


def continuous_grid_certificate(
    cells: Sequence[GridCell],
    upper_corner_objectives: Sequence[float],
    grid_upper_bound: float,
    atol: float = 1e-7,
) -> ContinuousGridCertificate:
    """Combine exact upper-corner evaluations into a continuous-domain bound."""
    objectives = np.asarray(upper_corner_objectives, dtype=float)
    if not cells or objectives.shape != (len(cells),):
        raise ValueError("One finite upper-corner objective is required for every grid cell.")
    if not np.all(np.isfinite(objectives)) or not np.isfinite(grid_upper_bound):
        raise ValueError("Certificate objectives and grid upper bound must be finite.")
    lower_index = int(np.argmin(objectives))
    continuous_lower_bound = float(objectives[lower_index])
    if continuous_lower_bound > grid_upper_bound + atol:
        raise RuntimeError("The upper-corner lower bound exceeded the feasible grid upper bound.")
    absolute_gap = max(0.0, float(grid_upper_bound - continuous_lower_bound))
    return ContinuousGridCertificate(
        grid_upper_bound=float(grid_upper_bound),
        continuous_lower_bound=continuous_lower_bound,
        absolute_gap=absolute_gap,
        relative_gap_percent=100.0 * absolute_gap / max(1.0, abs(float(grid_upper_bound))),
        evaluated_cell_count=len(cells),
        lower_bound_cell=cells[lower_index],
    )



