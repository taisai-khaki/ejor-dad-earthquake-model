from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from ejor_dad.fixed_y import RecourseCut, _append_capability_constraints, _design_basis_states, _road_signature, _state_signature, aggregate_cut, evaluate_fixed_y, evaluate_plan_losses
from ejor_dad.model import DADInstance
from ejor_dad.moments import build_failure_moment_system
from ejor_dad.states import nominal_probabilities
from ejor_dad.tv import worst_case_tv_distribution


@dataclass(frozen=True)
class CapacityRange:
    center_id: str
    minimum: float
    maximum: float
    minimum_iterations: int
    maximum_iterations: int


def fixed_y_capacity_ranges(
    instance: DADInstance,
    y: Sequence[float],
    objective_tolerance: float = 1e-6,
    separation_tolerance: float = 1e-7,
    max_iterations: int = 200,
) -> tuple[CapacityRange, ...]:
    """Range each capacity decision among objective-equivalent fixed-y plans."""
    baseline = evaluate_fixed_y(instance, y, epsilon=separation_tolerance, max_iterations=max_iterations)
    threshold = baseline.objective + objective_tolerance
    ranges = []
    for center_index, center in enumerate(instance.centers):
        minimum, min_iterations = _range_endpoint(instance, baseline, threshold, center_index, False, separation_tolerance, max_iterations)
        maximum, max_iterations_used = _range_endpoint(instance, baseline, threshold, center_index, True, separation_tolerance, max_iterations)
        ranges.append(CapacityRange(center.id, minimum, maximum, min_iterations, max_iterations_used))
    return tuple(ranges)


def _range_endpoint(instance, baseline, threshold, center_index, maximize, tolerance, max_iterations):
    cuts = list(baseline.cuts)
    if not cuts:
        raise RuntimeError("Fixed-y capacity ranging requires at least one recourse cut.")
    nominal = nominal_probabilities(instance.links, instance.states, baseline.y, instance.hazard_regimes)
    moment_system = build_failure_moment_system(instance, nominal)
    for iteration in range(1, max_iterations + 1):
        z, w = _solve_range_master(instance, cuts, threshold, center_index, maximize, baseline.y)
        losses, _, recourse = evaluate_plan_losses(instance, z, w, y=baseline.y)
        worst_case = worst_case_tv_distribution(
            nominal, losses, instance.ambiguity_radius, maximize=True,
            density_cap=instance.ambiguity_density_cap,
            inequality_matrix=moment_system.inequality_matrix if moment_system.bounds else None,
            inequality_rhs=moment_system.inequality_rhs if moment_system.bounds else None,
        )
        if worst_case.value <= threshold + tolerance:
            return float(w[center_index]), iteration
        cuts.append(RecourseCut(
            distribution=worst_case.distribution,
            alphas=np.vstack([result.alpha for result in recourse]),
            betas=np.vstack([result.beta for result in recourse]),
            road_signature=_road_signature(baseline.y),
            state_signature=_state_signature(instance),
        ))
    raise RuntimeError("Capacity-range separation did not converge.")


def _solve_range_master(instance, cuts, threshold, center_index, maximize, y):
    num_zones = len(instance.zones)
    num_centers = len(instance.centers)
    block_size = num_centers * num_zones
    h_start = num_zones + num_centers
    num_design_states = len(_design_basis_states(instance))
    num_vars = h_start + num_design_states * block_size
    objective = np.zeros(num_vars)
    objective[num_zones + center_index] = -1.0 if maximize else 1.0
    rows = []
    rhs = []
    renovation = np.zeros(num_vars)
    renovation[:num_zones] = instance.renovation_costs
    rows.append(renovation)
    rhs.append(instance.budget_renovation)
    if instance.minimum_protected_population > 0.0:
        protected = np.zeros(num_vars)
        protected[:num_zones] = -instance.protected_population_coefficients
        rows.append(protected)
        rhs.append(-instance.minimum_protected_population)
    capacity = np.zeros(num_vars)
    capacity[num_zones:h_start] = instance.capacity_costs
    rows.append(capacity)
    rhs.append(instance.budget_capacity)
    _append_capability_constraints(instance, y, num_vars, h_start, rows, rhs)
    for cut in cuts:
        constant, z_coefficients, w_coefficients = aggregate_cut(instance, cut)
        row = np.zeros(num_vars)
        row[:num_zones] = z_coefficients
        row[num_zones:h_start] = w_coefficients
        rows.append(row)
        rhs.append(threshold - constant)
    upper = [instance.budget_capacity / cost if cost > 0 else None for cost in instance.capacity_costs]
    solution = linprog(
        c=objective, A_ub=np.vstack(rows), b_ub=np.asarray(rhs),
        bounds=[(0.0, 1.0)] * num_zones + [(0.0, bound) for bound in upper] + [(0.0, None)] * (num_design_states * block_size), method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Capacity-range master failed: {solution.message}")
    return solution.x[:num_zones], solution.x[num_zones:h_start]
