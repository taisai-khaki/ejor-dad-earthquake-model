from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import inf
from typing import Sequence

import numpy as np

from ejor_dad.model import DADInstance, FailureMomentEnvelope


@dataclass(frozen=True)
class MomentBound:
    name: str
    coefficients: np.ndarray
    nominal_value: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class FailureMomentSystem:
    bounds: tuple[MomentBound, ...]
    inequality_matrix: np.ndarray
    inequality_rhs: np.ndarray


def build_failure_moment_system(
    instance: DADInstance,
    nominal: Sequence[float],
) -> FailureMomentSystem:
    envelope = instance.failure_moment_envelope
    if envelope is None:
        return empty_moment_system(len(instance.states))
    if any(state.is_tail for state in instance.states):
        raise ValueError("Failure-moment constraints require an explicit no-tail state support.")
    nominal_array = np.asarray(nominal, dtype=float)
    if nominal_array.shape != (len(instance.states),):
        raise ValueError("Nominal distribution must have one entry per state.")
    if np.any(nominal_array < -1e-10) or not np.isclose(nominal_array.sum(), 1.0, atol=1e-8):
        raise ValueError("Nominal distribution must be nonnegative and sum to one.")

    indicators = failure_indicator_matrix(instance)
    bounds: list[MomentBound] = []
    if envelope.marginal_active:
        for link_index, link in enumerate(instance.links):
            coefficients = indicators[:, link_index]
            nominal_value = float(nominal_array @ coefficients)
            lower, upper = probability_band(
                nominal_value,
                envelope.marginal_relative_tolerance,
                envelope.marginal_absolute_tolerance,
            )
            bounds.append(
                MomentBound(
                    name=f"marginal::{link.id}",
                    coefficients=coefficients,
                    nominal_value=nominal_value,
                    lower_bound=lower,
                    upper_bound=upper,
                )
            )

    if envelope.joint_active:
        for left_index, right_index in combinations(range(len(instance.links)), 2):
            coefficients = indicators[:, left_index] * indicators[:, right_index]
            nominal_value = float(nominal_array @ coefficients)
            lower, upper = probability_band(
                nominal_value,
                envelope.joint_relative_tolerance,
                envelope.joint_absolute_tolerance,
            )
            bounds.append(
                MomentBound(
                    name=f"joint::{instance.links[left_index].id}::{instance.links[right_index].id}",
                    coefficients=coefficients,
                    nominal_value=nominal_value,
                    lower_bound=lower,
                    upper_bound=upper,
                )
            )

    failure_count = indicators.sum(axis=1)
    nominal_count_mean = float(nominal_array @ failure_count)
    if envelope.count_mean_absolute_tolerance is not None:
        tolerance = envelope.count_mean_absolute_tolerance
        bounds.append(
            MomentBound(
                name="failure_count_mean",
                coefficients=failure_count,
                nominal_value=nominal_count_mean,
                lower_bound=max(0.0, nominal_count_mean - tolerance),
                upper_bound=min(float(len(instance.links)), nominal_count_mean + tolerance),
            )
        )

    if envelope.count_second_moment_active:
        coefficients = (failure_count - nominal_count_mean) ** 2
        nominal_value = float(nominal_array @ coefficients)
        relative = envelope.count_second_moment_relative_tolerance or 0.0
        tolerance = envelope.count_second_moment_absolute_tolerance + relative * nominal_value
        bounds.append(
            MomentBound(
                name="failure_count_fixed_center_second_moment",
                coefficients=coefficients,
                nominal_value=nominal_value,
                lower_bound=max(0.0, nominal_value - tolerance),
                upper_bound=min(float(np.max(coefficients)), nominal_value + tolerance),
            )
        )

    return assemble_moment_system(bounds, len(instance.states))


def failure_indicator_matrix(instance: DADInstance) -> np.ndarray:
    link_index = {link.id: index for index, link in enumerate(instance.links)}
    indicators = np.zeros((len(instance.states), len(instance.links)), dtype=float)
    for state_index, state in enumerate(instance.states):
        for link_id in state.failed_links:
            indicators[state_index, link_index[link_id]] = 1.0
    return indicators


def moment_bound_diagnostics(
    system: FailureMomentSystem,
    distribution: Sequence[float],
    tolerance: float = 1e-8,
) -> list[dict[str, float | str | bool]]:
    probabilities = np.asarray(distribution, dtype=float)
    diagnostics: list[dict[str, float | str | bool]] = []
    for bound in system.bounds:
        value = float(probabilities @ bound.coefficients)
        structural_zero = (
            abs(bound.nominal_value) <= tolerance
            and abs(bound.lower_bound) <= tolerance
            and abs(bound.upper_bound) <= tolerance
        )
        diagnostics.append(
            {
                "name": bound.name,
                "nominal_value": bound.nominal_value,
                "value": value,
                "lower_bound": bound.lower_bound,
                "upper_bound": bound.upper_bound,
                "lower_slack": value - bound.lower_bound,
                "upper_slack": bound.upper_bound - value,
                "structural_zero": structural_zero,
                "active": not structural_zero
                and min(value - bound.lower_bound, bound.upper_bound - value) <= tolerance,
            }
        )
    return diagnostics


def probability_band(
    nominal_value: float,
    relative_tolerance: float | None,
    absolute_tolerance: float,
) -> tuple[float, float]:
    width = absolute_tolerance + (relative_tolerance or 0.0) * nominal_value
    return max(0.0, nominal_value - width), min(1.0, nominal_value + width)


def assemble_moment_system(bounds: Sequence[MomentBound], num_states: int) -> FailureMomentSystem:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for bound in bounds:
        if bound.upper_bound < inf:
            rows.append(np.asarray(bound.coefficients, dtype=float))
            rhs.append(bound.upper_bound)
        if bound.lower_bound > -inf:
            rows.append(-np.asarray(bound.coefficients, dtype=float))
            rhs.append(-bound.lower_bound)
    matrix = np.vstack(rows) if rows else np.empty((0, num_states), dtype=float)
    return FailureMomentSystem(
        bounds=tuple(bounds),
        inequality_matrix=matrix,
        inequality_rhs=np.asarray(rhs, dtype=float),
    )


def empty_moment_system(num_states: int) -> FailureMomentSystem:
    return FailureMomentSystem(
        bounds=(),
        inequality_matrix=np.empty((0, num_states), dtype=float),
        inequality_rhs=np.empty(0, dtype=float),
    )
