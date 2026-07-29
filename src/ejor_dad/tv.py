from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class TVWorstCaseResult:
    distribution: np.ndarray
    value: float
    moved_mass: float


@dataclass(frozen=True)
class TVSaturationResult:
    cap_only_value: float
    minimum_radius: float
    distribution: np.ndarray


@dataclass(frozen=True)
class TVProfileSegment:
    """One linear transfer segment of a capped-TV risk profile."""

    start_radius: float
    end_radius: float
    source_index: int
    target_index: int
    value_slope: float
    start_value: float
    end_value: float


@dataclass(frozen=True)
class CappedTVProfile:
    """Exact fixed-law capped-TV value profile over the ambiguity radius."""

    nominal: np.ndarray
    values: np.ndarray
    maximize: bool
    density_cap: float | None
    segments: tuple[TVProfileSegment, ...]
    terminal_radius: float
    terminal_value: float
    atol: float = 1e-12

    @property
    def breakpoints(self) -> np.ndarray:
        return np.asarray([0.0, *(segment.end_radius for segment in self.segments)], dtype=float)

    def evaluate(self, rho: float) -> TVWorstCaseResult:
        """Evaluate the exact profile while retaining the optimal distribution."""
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1].")
        distribution = self.nominal.copy()
        for segment in self.segments:
            transferred = min(
                max(rho - segment.start_radius, 0.0),
                segment.end_radius - segment.start_radius,
            )
            if transferred <= self.atol:
                continue
            distribution[segment.source_index] -= transferred
            distribution[segment.target_index] += transferred
        distribution = np.maximum(distribution, 0.0)
        distribution /= distribution.sum()
        return TVWorstCaseResult(
            distribution=distribution,
            value=float(distribution @ self.values),
            moved_mass=float(min(rho, self.terminal_radius)),
        )


def capped_tv_profile(
    nominal: Sequence[float],
    values: Sequence[float],
    maximize: bool = True,
    density_cap: float | None = None,
    atol: float = 1e-12,
) -> CappedTVProfile:
    """Build the law-invariant spectral profile for a finite capped-TV set.

    The profile applies only when the ambiguity set contains total variation and
    optional density-cap constraints. Additional moment constraints require the
    LP path in :func:`worst_case_tv_distribution` and are intentionally not
    represented here.
    """
    if density_cap is not None and (not np.isfinite(density_cap) or density_cap < 1.0):
        raise ValueError("density_cap must be finite and at least 1 when provided.")
    p0 = np.asarray(nominal, dtype=float)
    v = np.asarray(values, dtype=float)
    if p0.shape != v.shape:
        raise ValueError("nominal and values must have the same shape.")
    if p0.ndim != 1 or p0.size == 0:
        raise ValueError("nominal and values must be nonempty one-dimensional arrays.")
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(v)):
        raise ValueError("nominal and values must be finite.")
    if np.any(p0 < -atol):
        raise ValueError("nominal probabilities must be nonnegative.")
    p0 = np.maximum(p0, 0.0)
    total = float(p0.sum())
    if total <= atol:
        raise ValueError("nominal probabilities must have positive total mass.")
    p0 = p0 / total
    p = p0.copy()
    upper_bounds = np.ones_like(p) if density_cap is None else np.minimum(1.0, density_cap * p0)
    effective_values = v if maximize else -v
    source_order = list(np.argsort(effective_values))
    target_order = list(np.argsort(-effective_values))
    source_pointer = 0
    target_pointer = 0
    radius = 0.0
    current_value = float(p0 @ v)
    segments: list[TVProfileSegment] = []

    while source_pointer < len(p) and target_pointer < len(p):
        source = source_order[source_pointer]
        target = target_order[target_pointer]
        if effective_values[target] <= effective_values[source] + atol:
            break
        source_capacity = p[source]
        target_capacity = upper_bounds[target] - p[target]
        transferred = min(source_capacity, target_capacity)
        if transferred <= atol:
            if source_capacity <= atol:
                source_pointer += 1
            if target_capacity <= atol:
                target_pointer += 1
            continue
        start_radius = radius
        end_radius = radius + float(transferred)
        slope = float(v[target] - v[source])
        end_value = current_value + float(transferred) * slope
        segments.append(
            TVProfileSegment(
                start_radius=start_radius,
                end_radius=end_radius,
                source_index=int(source),
                target_index=int(target),
                value_slope=slope,
                start_value=current_value,
                end_value=end_value,
            )
        )
        p[source] -= transferred
        p[target] += transferred
        radius = end_radius
        current_value = end_value
        if p[source] <= atol:
            source_pointer += 1
        if upper_bounds[target] - p[target] <= atol:
            target_pointer += 1

    nominal_copy = p0.copy()
    values_copy = v.copy()
    nominal_copy.setflags(write=False)
    values_copy.setflags(write=False)
    return CappedTVProfile(
        nominal=nominal_copy,
        values=values_copy,
        maximize=maximize,
        density_cap=density_cap,
        segments=tuple(segments),
        terminal_radius=radius,
        terminal_value=current_value,
        atol=atol,
    )


def worst_case_tv_distribution(
    nominal: Sequence[float],
    values: Sequence[float],
    rho: float,
    maximize: bool = True,
    density_cap: float | None = None,
    inequality_matrix: Sequence[Sequence[float]] | None = None,
    inequality_rhs: Sequence[float] | None = None,
    atol: float = 1e-12,
) -> TVWorstCaseResult:
    """Exact finite-support total-variation reweighting.

    With the convention ``0.5 * ||p - p_hat||_1 <= rho``, moving ``delta`` mass
    from one state to another consumes ``delta`` units of TV budget. A finite
    density cap imposes ``p_s <= density_cap * p_hat_s`` and therefore preserves
    nominal support. When no moment constraints are supplied, the result is the
    law-invariant spectral profile evaluated at ``rho``.
    """
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1].")
    if density_cap is not None and (not np.isfinite(density_cap) or density_cap < 1.0):
        raise ValueError("density_cap must be finite and at least 1 when provided.")
    p = np.asarray(nominal, dtype=float).copy()
    v = np.asarray(values, dtype=float)
    if p.shape != v.shape:
        raise ValueError("nominal and values must have the same shape.")
    if np.any(p < -atol):
        raise ValueError("nominal probabilities must be nonnegative.")
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if total <= atol:
        raise ValueError("nominal probabilities must have positive total mass.")
    p = p / total
    if inequality_matrix is not None or inequality_rhs is not None:
        if inequality_matrix is None or inequality_rhs is None:
            raise ValueError("inequality_matrix and inequality_rhs must be provided together.")
        return constrained_tv_distribution(
            p,
            v,
            rho,
            maximize,
            density_cap,
            inequality_matrix,
            inequality_rhs,
            atol,
        )
    return capped_tv_profile(
        p,
        v,
        maximize=maximize,
        density_cap=density_cap,
        atol=atol,
    ).evaluate(rho)


def minimum_saturation_radius(
    nominal: Sequence[float],
    values: Sequence[float],
    density_cap: float,
    maximize: bool = True,
    value_tolerance: float = 1e-9,
) -> TVSaturationResult:
    """Find the least TV radius attaining the density-cap-only optimum."""
    p0 = np.asarray(nominal, dtype=float)
    v = np.asarray(values, dtype=float)
    if p0.shape != v.shape or p0.ndim != 1 or p0.size == 0:
        raise ValueError("nominal and values must be equally sized nonempty vectors.")
    if not np.isfinite(density_cap) or density_cap < 1.0:
        raise ValueError("density_cap must be finite and at least 1.")
    if np.any(p0 < 0.0) or float(p0.sum()) <= 0.0:
        raise ValueError("nominal probabilities must be nonnegative with positive mass.")
    p0 = p0 / p0.sum()
    upper = np.minimum(1.0, density_cap * p0)
    cap_solution = linprog(
        c=(-v if maximize else v),
        A_eq=np.ones((1, p0.size)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, float(bound)) for bound in upper],
        method="highs",
    )
    if not cap_solution.success:
        raise RuntimeError(f"Density-cap-only LP failed: {cap_solution.message}")
    cap_value = float(cap_solution.x @ v)
    num_states = p0.size
    objective = np.concatenate([np.zeros(num_states), 0.5 * np.ones(num_states)])
    rows = []
    rhs = []
    for index in range(num_states):
        positive = np.zeros(2 * num_states)
        positive[index] = 1.0
        positive[num_states + index] = -1.0
        rows.append(positive)
        rhs.append(float(p0[index]))
        negative = np.zeros(2 * num_states)
        negative[index] = -1.0
        negative[num_states + index] = -1.0
        rows.append(negative)
        rhs.append(float(-p0[index]))
    value_row = np.zeros(2 * num_states)
    value_row[:num_states] = -v if maximize else v
    rows.append(value_row)
    rhs.append((-cap_value if maximize else cap_value) + value_tolerance)
    equality = np.zeros((1, 2 * num_states))
    equality[0, :num_states] = 1.0
    solution = linprog(
        c=objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs),
        A_eq=equality,
        b_eq=np.array([1.0]),
        bounds=[(0.0, float(bound)) for bound in upper] + [(0.0, None)] * num_states,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Saturation-radius LP failed: {solution.message}")
    distribution = np.maximum(solution.x[:num_states], 0.0)
    distribution /= distribution.sum()
    return TVSaturationResult(
        cap_only_value=cap_value,
        minimum_radius=0.5 * float(np.abs(distribution - p0).sum()),
        distribution=distribution,
    )


def constrained_tv_distribution(
    nominal: np.ndarray,
    values: np.ndarray,
    rho: float,
    maximize: bool,
    density_cap: float | None,
    inequality_matrix: Sequence[Sequence[float]],
    inequality_rhs: Sequence[float],
    atol: float,
) -> TVWorstCaseResult:
    num_states = len(nominal)
    moment_matrix = np.asarray(inequality_matrix, dtype=float)
    moment_rhs = np.asarray(inequality_rhs, dtype=float)
    if moment_matrix.ndim != 2 or moment_matrix.shape[1] != num_states:
        raise ValueError("inequality_matrix must have one column per state.")
    if moment_rhs.shape != (moment_matrix.shape[0],):
        raise ValueError("inequality_rhs must have one entry per inequality row.")

    objective = np.concatenate([(-values if maximize else values), np.zeros(num_states)])
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for state_index in range(num_states):
        positive = np.zeros(2 * num_states)
        positive[state_index] = 1.0
        positive[num_states + state_index] = -1.0
        rows.append(positive)
        rhs.append(nominal[state_index])
        negative = np.zeros(2 * num_states)
        negative[state_index] = -1.0
        negative[num_states + state_index] = -1.0
        rows.append(negative)
        rhs.append(-nominal[state_index])
    tv_row = np.zeros(2 * num_states)
    tv_row[num_states:] = 1.0
    rows.append(tv_row)
    rhs.append(2.0 * rho)
    for row, bound in zip(moment_matrix, moment_rhs):
        augmented = np.zeros(2 * num_states)
        augmented[:num_states] = row
        rows.append(augmented)
        rhs.append(float(bound))

    equality = np.zeros((1, 2 * num_states))
    equality[0, :num_states] = 1.0
    probability_upper = (
        np.ones(num_states)
        if density_cap is None
        else np.minimum(1.0, density_cap * nominal)
    )
    solution = linprog(
        c=objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs, dtype=float),
        A_eq=equality,
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, float(upper)) for upper in probability_upper]
        + [(0.0, None)] * num_states,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Moment-constrained total-variation LP failed: {solution.message}")
    distribution = np.maximum(solution.x[:num_states], 0.0)
    total = float(distribution.sum())
    if abs(total - 1.0) > 1e-8:
        raise RuntimeError("Moment-constrained total-variation LP returned invalid probability mass.")
    distribution /= total
    if moment_matrix.size and np.max(moment_matrix @ distribution - moment_rhs) > max(atol, 1e-8):
        raise RuntimeError("Moment-constrained total-variation LP violated a moment bound.")
    moved_mass = 0.5 * float(np.abs(distribution - nominal).sum())
    return TVWorstCaseResult(
        distribution=distribution,
        value=float(distribution @ values),
        moved_mass=moved_mass,
    )

