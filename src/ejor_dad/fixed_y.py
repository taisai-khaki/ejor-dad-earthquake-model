from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from ejor_dad.model import DADInstance
from ejor_dad.moments import build_failure_moment_system
from ejor_dad.recourse import RecourseResult, solve_recourse
from ejor_dad.states import nominal_probabilities
from ejor_dad.tv import worst_case_tv_distribution


@dataclass(frozen=True)
class RecourseCut:
    distribution: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray
    gammas: np.ndarray


@dataclass(frozen=True)
class FixedYResult:
    objective: float
    lower_bound: float
    z: np.ndarray
    w: np.ndarray
    y: np.ndarray
    nominal_distribution: np.ndarray
    worst_case_distribution: np.ndarray
    state_losses: np.ndarray
    state_survivors: np.ndarray
    iterations: int
    cuts: tuple[RecourseCut, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FixedPlanResult:
    """Exact ambiguity evaluation of a complete, non-reoptimized decision plan."""

    objective: float
    nominal_objective: float
    z: np.ndarray
    w: np.ndarray
    y: np.ndarray
    nominal_distribution: np.ndarray
    worst_case_distribution: np.ndarray
    state_losses: np.ndarray
    state_survivors: np.ndarray


def evaluate_fixed_y(
    instance: DADInstance,
    y: Sequence[float],
    epsilon: float = 1e-6,
    max_iterations: int = 200,
    initial_z: Sequence[float] | None = None,
    initial_w: Sequence[float] | None = None,
    nominal_distribution_override: Sequence[float] | None = None,
    enforce_retrofit_budget: bool = True,
    initial_cuts: Sequence[RecourseCut] | None = None,
) -> FixedYResult:
    """Algorithm 1 from Section 4.1: exact fixed-retrofitting evaluation.

    ``enforce_retrofit_budget=False`` is reserved for monotonicity certificates:
    the supplied ``y`` remains in ``[0, 1]``, while the exposure and capacity
    budgets remain enforced by the master LP. It must not be used to report a
    feasible policy.
    """
    y_vec = np.asarray(y, dtype=float)
    if y_vec.shape != (len(instance.links),):
        raise ValueError("Fixed retrofit plan y must have one value for every link.")
    if np.any(y_vec < -1e-9) or np.any(y_vec > 1 + 1e-9):
        raise ValueError("Fixed retrofit plan y must lie in [0, 1].")
    if enforce_retrofit_budget and float(np.dot(instance.retrofit_costs, y_vec)) > instance.budget_retrofit + 1e-7:
        raise ValueError("Fixed retrofit plan y violates the retrofit budget.")
    if nominal_distribution_override is None:
        nominal = nominal_probabilities(instance.links, instance.states, y_vec, instance.hazard_regimes)
    else:
        nominal = np.asarray(nominal_distribution_override, dtype=float)
        if nominal.shape != (len(instance.states),):
            raise ValueError(
                "nominal_distribution_override must have one entry per state."
            )
        if np.any(nominal < -1e-10):
            raise ValueError("nominal_distribution_override must be nonnegative.")
        total = float(nominal.sum())
        if total <= 0:
            raise ValueError("nominal_distribution_override must have positive mass.")
        nominal = np.maximum(nominal, 0.0) / total
    moment_system = build_failure_moment_system(instance, nominal)
    z0 = np.zeros(len(instance.zones), dtype=float) if initial_z is None else np.asarray(initial_z, dtype=float)
    w0 = np.zeros(len(instance.centers), dtype=float) if initial_w is None else np.asarray(initial_w, dtype=float)
    cuts: list[RecourseCut] = list(initial_cuts or ())
    if not cuts:
        recourse_at_start = solve_recourse_states(instance, z0, w0, y=y_vec)
        cuts.append(
            RecourseCut(
                distribution=nominal,
                alphas=np.vstack([result.alpha for result in recourse_at_start]),
                betas=np.vstack([result.beta for result in recourse_at_start]),
                gammas=np.vstack([result.gamma for result in recourse_at_start]),
            )
        )
    latest_result: FixedYResult | None = None
    for iteration in range(1, max_iterations + 1):
        z_hat, w_hat, eta = solve_master(instance, cuts)
        recourse = solve_recourse_states(instance, z_hat, w_hat, y=y_vec)
        survivors = np.array([result.survivors for result in recourse], dtype=float)
        losses = instance.loss_after_response(z_hat, survivors)
        tv_result = worst_case_tv_distribution(
            nominal,
            losses,
            instance.ambiguity_radius,
            maximize=True,
            density_cap=instance.ambiguity_density_cap,
            inequality_matrix=(
                moment_system.inequality_matrix if moment_system.bounds else None
            ),
            inequality_rhs=(moment_system.inequality_rhs if moment_system.bounds else None),
        )
        objective = tv_result.value
        alphas = np.vstack([result.alpha for result in recourse])
        betas = np.vstack([result.beta for result in recourse])
        gammas = np.vstack([result.gamma for result in recourse])
        latest_result = FixedYResult(
            objective=objective,
            lower_bound=float(eta),
            z=z_hat,
            w=w_hat,
            y=y_vec,
            nominal_distribution=nominal,
            worst_case_distribution=tv_result.distribution,
            state_losses=losses,
            state_survivors=survivors,
            iterations=iteration,
            cuts=tuple(cuts),
        )
        if objective - eta <= epsilon:
            return latest_result
        cuts.append(RecourseCut(distribution=tv_result.distribution, alphas=alphas, betas=betas, gammas=gammas))
    if latest_result is None:
        raise RuntimeError("Fixed-y evaluation failed before the first iteration.")
    final_gap = latest_result.objective - latest_result.lower_bound
    raise RuntimeError(
        "Fixed-y cut generation did not converge within "
        f"{max_iterations} iterations; final objective gap was {final_gap:.6g}."
    )



def evaluate_fixed_plan(
    instance: DADInstance,
    z: Sequence[float],
    w: Sequence[float],
    y: Sequence[float],
    nominal_distribution_override: Sequence[float] | None = None,
) -> FixedPlanResult:
    """Evaluate a complete decision without reoptimizing its exposure or capacity actions."""
    z_vec = np.asarray(z, dtype=float)
    w_vec = np.asarray(w, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    if z_vec.shape != (len(instance.zones),) or np.any(z_vec < -1e-9) or np.any(z_vec > 1.0 + 1e-9):
        raise ValueError("Fixed renovation plan z must have one value in [0, 1] for every zone.")
    if w_vec.shape != (len(instance.centers),) or np.any(w_vec < -1e-9):
        raise ValueError("Fixed capacity plan w must have one nonnegative value for every center.")
    if y_vec.shape != (len(instance.links),) or np.any(y_vec < -1e-9) or np.any(y_vec > 1.0 + 1e-9):
        raise ValueError("Fixed retrofit plan y must have one value in [0, 1] for every link.")
    if float(np.dot(instance.renovation_costs, z_vec)) > instance.budget_renovation + 1e-7:
        raise ValueError("Fixed renovation plan z violates the renovation budget.")
    if float(np.dot(instance.capacity_costs, w_vec)) > instance.budget_capacity + 1e-7:
        raise ValueError("Fixed capacity plan w violates the capacity budget.")
    if float(np.dot(instance.retrofit_costs, y_vec)) > instance.budget_retrofit + 1e-7:
        raise ValueError("Fixed retrofit plan y violates the retrofit budget.")
    if nominal_distribution_override is None:
        nominal = nominal_probabilities(instance.links, instance.states, y_vec, instance.hazard_regimes)
    else:
        nominal = np.asarray(nominal_distribution_override, dtype=float)
        if nominal.shape != (len(instance.states),):
            raise ValueError(
                "nominal_distribution_override must have one entry per state."
            )
        if np.any(nominal < -1e-10):
            raise ValueError("nominal_distribution_override must be nonnegative.")
        total = float(nominal.sum())
        if total <= 0.0:
            raise ValueError("nominal_distribution_override must have positive mass.")
        nominal = np.maximum(nominal, 0.0) / total
    losses, survivors, _ = evaluate_plan_losses(instance, z_vec, w_vec, y=y_vec)
    moment_system = build_failure_moment_system(instance, nominal)
    worst_case = worst_case_tv_distribution(
        nominal,
        losses,
        instance.ambiguity_radius,
        maximize=True,
        density_cap=instance.ambiguity_density_cap,
        inequality_matrix=(
            moment_system.inequality_matrix if moment_system.bounds else None
        ),
        inequality_rhs=(
            moment_system.inequality_rhs if moment_system.bounds else None
        ),
    )
    return FixedPlanResult(
        objective=float(worst_case.value),
        nominal_objective=float(nominal @ losses),
        z=z_vec,
        w=w_vec,
        y=y_vec,
        nominal_distribution=nominal,
        worst_case_distribution=worst_case.distribution,
        state_losses=losses,
        state_survivors=survivors,
    )

def solve_master(instance: DADInstance, cuts: Sequence[RecourseCut]) -> tuple[np.ndarray, np.ndarray, float]:
    num_zones = len(instance.zones)
    num_centers = len(instance.centers)
    eta_index = num_zones + num_centers
    num_vars = eta_index + 1
    c = np.zeros(num_vars)
    c[eta_index] = 1.0
    rows = []
    rhs = []
    renovation_row = np.zeros(num_vars)
    renovation_row[:num_zones] = instance.renovation_costs
    rows.append(renovation_row)
    rhs.append(instance.budget_renovation)
    if instance.minimum_protected_population > 0.0:
        protected_row = np.zeros(num_vars)
        protected_row[:num_zones] = -instance.protected_population_coefficients
        rows.append(protected_row)
        rhs.append(-instance.minimum_protected_population)
    capacity_row = np.zeros(num_vars)
    capacity_row[num_zones:eta_index] = instance.capacity_costs
    rows.append(capacity_row)
    rhs.append(instance.budget_capacity)
    for cut in cuts:
        constant, z_coefficients, w_coefficients = aggregate_cut(instance, cut)
        row = np.zeros(num_vars)
        row[:num_zones] = z_coefficients
        row[num_zones:eta_index] = w_coefficients
        row[eta_index] = -1.0
        rows.append(row)
        rhs.append(-constant)
    w_upper = [
        instance.budget_capacity / cost if cost > 0 else None
        for cost in instance.capacity_costs
    ]
    bounds = [(0.0, 1.0)] * num_zones + [(0.0, upper) for upper in w_upper] + [(0.0, instance.d_max)]
    solution = linprog(
        c=c,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Fixed-y master LP failed: {solution.message}")
    z = solution.x[:num_zones]
    w = solution.x[num_zones:eta_index]
    eta = float(solution.x[eta_index])
    return z, w, eta

def aggregate_cut(instance: DADInstance, cut: RecourseCut) -> tuple[float, np.ndarray, np.ndarray]:
    base = instance.base_demands
    immediate = instance.base_immediate_losses
    existing = instance.existing_capacities
    probabilities = np.asarray(cut.distribution, dtype=float)
    alphas = np.asarray(cut.alphas, dtype=float)
    betas = np.asarray(cut.betas, dtype=float)
    gammas = np.asarray(cut.gammas, dtype=float)
    service_fractions = np.vstack([instance.service_fractions_for_state(state) for state in instance.states])
    one_minus_beta = 1.0 - betas + service_fractions * gammas
    loss_exposure_coefficients = immediate[np.newaxis, :] + one_minus_beta * base[np.newaxis, :]
    availability = np.vstack([instance.facility_availability(state) for state in instance.states])
    constant_by_state = loss_exposure_coefficients.sum(axis=1) - np.sum(alphas * availability * existing[np.newaxis, :], axis=1)
    z_by_state = -loss_exposure_coefficients
    w_by_state = -alphas * availability
    constant = float(np.dot(probabilities, constant_by_state))
    z_coefficients = probabilities @ z_by_state
    w_coefficients = probabilities @ w_by_state
    return constant, z_coefficients, w_coefficients


def solve_recourse_states(
    instance: DADInstance,
    z: Sequence[float],
    w: Sequence[float],
    y: Sequence[float] | None = None,
) -> list[RecourseResult]:
    """Solve one LP per distinct physical recourse state and expand by label."""
    cache: dict[tuple[object, ...], RecourseResult] = {}
    results: list[RecourseResult] = []
    for state in instance.states:
        availability = instance.facility_availability(state)
        service_requirement = instance.service_fractions_for_state(state)
        if state.is_tail:
            key = (
                True,
                availability.tobytes(),
                service_requirement.tobytes(),
            )
        else:
            survival = instance.survival_matrix(state, y=y)
            key = (
                False,
                survival.shape,
                survival.tobytes(),
                availability.tobytes(),
                service_requirement.tobytes(),
            )
        result = cache.get(key)
        if result is None:
            result = solve_recourse(instance, state, z, w, y=y)
            cache[key] = result
        results.append(result)
    return results


def evaluate_plan_losses(
    instance: DADInstance,
    z: Sequence[float],
    w: Sequence[float],
    y: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[RecourseResult]]:
    recourse = solve_recourse_states(instance, z, w, y=y)
    survivors = np.array([result.survivors for result in recourse], dtype=float)
    losses = instance.loss_after_response(z, survivors)
    return losses, survivors, recourse




