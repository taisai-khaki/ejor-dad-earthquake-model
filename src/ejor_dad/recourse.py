from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from ejor_dad.model import DADInstance, State


@dataclass(frozen=True)
class RecourseResult:
    survivors: float
    dispatch: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    survival: np.ndarray


def solve_recourse(instance: DADInstance, state: State, z: Sequence[float], w: Sequence[float], y: Sequence[float] | None = None) -> RecourseResult:
    """Solve statewise dispatch with hard minimum zone-service requirements."""
    num_centers = len(instance.centers)
    num_zones = len(instance.zones)
    if state.is_tail:
        if np.any(instance.service_fractions_for_state(state) > 1e-12):
            raise ValueError("Positive zone-service requirements are incompatible with a zero-service tail state.")
        return RecourseResult(0.0, np.zeros((num_centers, num_zones)), np.zeros(num_centers), np.zeros(num_zones), np.zeros(num_zones), np.zeros((num_centers, num_zones)))
    demand = instance.demand_after_renovation(z)
    capacity = instance.capacity_after_investment(w) * instance.facility_availability(state)
    survival = instance.survival_matrix(state, y=y)
    objective = -survival.reshape(-1)
    rows, rhs = [], []
    for center_index in range(num_centers):
        row = np.zeros(num_centers * num_zones)
        row[center_index * num_zones:(center_index + 1) * num_zones] = 1.0
        rows.append(row)
        rhs.append(capacity[center_index])
    for zone_index in range(num_zones):
        row = np.zeros(num_centers * num_zones)
        row[zone_index::num_zones] = 1.0
        rows.append(row)
        rhs.append(demand[zone_index])
    for zone_index, service_fraction in enumerate(instance.service_fractions_for_state(state)):
        row = np.zeros(num_centers * num_zones)
        row[zone_index::num_zones] = -survival[:, zone_index]
        rows.append(row)
        rhs.append(-service_fraction * demand[zone_index])
    primal = linprog(c=objective, A_ub=np.vstack(rows), b_ub=np.asarray(rhs), bounds=[(0.0, None)] * (num_centers * num_zones), method="highs")
    if not primal.success:
        raise RuntimeError(f"Recourse primal failed for state {state.id}: {primal.message}")
    marginals = np.asarray(primal.ineqlin.marginals, dtype=float)
    expected_marginals = num_centers + 2 * num_zones
    if marginals.shape != (expected_marginals,):
        raise RuntimeError(
            f"Recourse primal returned {marginals.size} row marginals; "
            f"expected {expected_marginals}."
        )
    alpha = np.maximum(0.0, -marginals[:num_centers])
    beta = np.maximum(0.0, -marginals[num_centers:num_centers + num_zones])
    gamma = np.maximum(0.0, -marginals[num_centers + num_zones:])
    return RecourseResult(float(-primal.fun), primal.x.reshape((num_centers, num_zones)), alpha, beta, gamma, survival)


def solve_recourse_dual(instance: DADInstance, state: State, z: Sequence[float], w: Sequence[float], survival: np.ndarray | None = None, y: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_centers = len(instance.centers)
    num_zones = len(instance.zones)
    if state.is_tail:
        return np.zeros(num_centers), np.zeros(num_zones), np.zeros(num_zones)
    if survival is None:
        survival = instance.survival_matrix(state, y=y)
    demand = instance.demand_after_renovation(z)
    capacity = instance.capacity_after_investment(w) * instance.facility_availability(state)
    service = instance.service_fractions_for_state(state)
    objective = np.concatenate([capacity, demand, -service * demand])
    rows, rhs = [], []
    for center_index in range(num_centers):
        for zone_index in range(num_zones):
            row = np.zeros(num_centers + 2 * num_zones)
            row[center_index] = -1.0
            row[num_centers + zone_index] = -1.0
            row[num_centers + num_zones + zone_index] = survival[center_index, zone_index]
            rows.append(row)
            rhs.append(-survival[center_index, zone_index])
    dual = linprog(c=objective, A_ub=np.vstack(rows), b_ub=np.asarray(rhs), bounds=[(0.0, None)] * (num_centers + 2 * num_zones), method="highs")
    if not dual.success:
        raise RuntimeError(f"Recourse dual failed for state {state.id}: {dual.message}")
    return dual.x[:num_centers], dual.x[num_centers:num_centers + num_zones], dual.x[num_centers + num_zones:]
