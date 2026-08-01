from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
import time
from collections.abc import Callable
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import linprog

from ejor_dad.fixed_y import FixedYResult, aggregate_cut, evaluate_fixed_y, evaluate_plan_losses
from ejor_dad.model import DADInstance, State
from ejor_dad.recourse import solve_recourse

BranchClass = Literal["y", "v", "pi", "theta", "density", "removal", "omega"]
ProbabilityRelaxation = Literal["product_tree", "corner_boxes", "corner_link_cuts"]


@dataclass(frozen=True)
class BranchKey:
    variable_class: BranchClass
    index: tuple[int, ...]


@dataclass
class SBBNode:
    y_bounds: np.ndarray
    v_bounds: dict[tuple[int, int], tuple[float, float]]
    pi_bounds: np.ndarray
    theta_bounds: np.ndarray
    density_bounds: np.ndarray
    removal_bounds: np.ndarray
    omega_bounds: tuple[float, float]
    lower_bound: float = -np.inf
    depth: int = 0
    relaxation: "NodeRelaxationResult | None" = None


@dataclass(frozen=True)
class NodeRelaxationResult:
    lower_bound: float
    z: np.ndarray
    w: np.ndarray
    y: np.ndarray
    pi: np.ndarray
    theta: np.ndarray
    density_dual: np.ndarray
    removal_dual: np.ndarray
    omega: float
    kappa: float
    success: bool
    message: str = ""
    cut_iterations: int = 0


@dataclass(frozen=True)
class SBBResult:
    objective: float
    lower_bound: float
    gap: float
    z: np.ndarray
    w: np.ndarray
    y: np.ndarray
    fixed_y_result: FixedYResult
    nodes_processed: int
    nodes_remaining: int
    converged: bool
    pruned_nodes: int = 0
    max_tree_depth: int = 0
    fixed_y_oracle_calls: int = 0
    recourse_cuts_generated: int = 0
    relaxation_failures: int = 0
    runtime_sec: float = 0.0
    termination_reason: str = ""


@dataclass(frozen=True)
class VariableIndex:
    z: list[int]
    w: list[int]
    y: list[int]
    pi: list[int]
    v: dict[tuple[int, int], int]
    theta: list[int]
    chi: list[int]
    density_dual: list[int]
    density_chi: list[int]
    removal_dual: list[int]
    removal_chi: list[int]
    omega: int
    kappa: int
    size: int


@dataclass(frozen=True)
class LinearExpr:
    constant: float = 0.0
    coefficients: dict[int, float] = field(default_factory=dict)

    @staticmethod
    def variable(index: int, coefficient: float = 1.0) -> "LinearExpr":
        return LinearExpr(0.0, {index: coefficient})

    def scaled(self, scale: float) -> "LinearExpr":
        return LinearExpr(
            self.constant * scale,
            {index: value * scale for index, value in self.coefficients.items()},
        )

    def plus(self, other: "LinearExpr") -> "LinearExpr":
        coefficients = dict(self.coefficients)
        for index, value in other.coefficients.items():
            coefficients[index] = coefficients.get(index, 0.0) + value
        return LinearExpr(self.constant + other.constant, coefficients)


def validate_probability_relaxation(probability_relaxation: str) -> None:
    valid = {"product_tree", "corner_boxes", "corner_link_cuts"}
    if probability_relaxation not in valid:
        raise ValueError(
            "probability_relaxation must be one of "
            f"{sorted(valid)}, got {probability_relaxation!r}."
        )


def solve_global_sbb(
    instance: DADInstance,
    epsilon: float = 1e-3,
    fixed_y_epsilon: float = 1e-6,
    cut_epsilon: float = 1e-7,
    max_nodes: int = 200,
    initial_y: Sequence[float] | None = None,
    time_limit_sec: float | None = None,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
    initial_root: SBBNode | None = None,
    probability_relaxation: ProbabilityRelaxation = "product_tree",
) -> SBBResult:
    """Section 4.6 spatial branch-and-bound over the retrofitting layer.

    The method is exact as max_nodes -> infinity with zero tolerances. For real
    Nepal networks, use this on reduced/candidate corridors first, then use the
    incumbent as a high-quality warm start for larger heuristic searches.
    """
    validate_probability_relaxation(probability_relaxation)
    if np.any(instance.zone_service_fractions > 1e-12):
        raise NotImplementedError(
            "Continuous SBB does not include the separate design-basis capability block; "
            "use fixed-y enumeration for service-constrained instances."
        )
    if instance.hazard_regimes is not None:
        raise NotImplementedError("Continuous SBB does not yet include spatial hazard-regime mixtures; use exact fixed-y enumeration.")
    if instance.has_retrofit_performance_effects:
        raise NotImplementedError(
            "Continuous SBB does not yet include retrofit-dependent conditional delays; "
            "use exact fixed-y enumeration for performance-adjusted instances."
        )
    if instance.failure_moment_envelope is not None:
        raise NotImplementedError(
            "Continuous SBB does not yet include failure-moment constraints; "
            "use exact fixed-y enumeration for moment-constrained instances."
        )
    start_time = time.time()
    fixed_y_oracle_calls = 0
    pruned_nodes = 0
    max_tree_depth = 0

    def emit(event: str, **payload: Any) -> None:
        if trace_callback is None:
            return
        record = {
            "event": event,
            "time_sec": time.time() - start_time,
            "processed_nodes": int(nodes_processed),
            "incumbent_UB": float(upper_bound),
            "global_LB": float(global_lower_bound),
        }
        record.update(payload)
        trace_callback(record)

    if initial_y is None:
        y0 = np.zeros(len(instance.links), dtype=float)
    else:
        y0 = np.asarray(initial_y, dtype=float)
    incumbent = evaluate_fixed_y(instance, y0, epsilon=fixed_y_epsilon)
    fixed_y_oracle_calls += 1
    upper_bound = incumbent.objective
    root = (
        clone_node(initial_root)
        if initial_root is not None
        else root_node(instance, probability_relaxation=probability_relaxation)
    )
    cut_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {i: [] for i in range(len(instance.states))}
    initialize_recourse_cuts(instance, cut_sets, incumbent)
    active: list[tuple[float, int, SBBNode]] = []
    counter = 0
    heappush(active, (root.lower_bound, counter, root))
    nodes_processed = 0
    global_lower_bound = root.lower_bound
    converged = False
    termination_reason = "node_limit"
    relaxation_failures = 0
    valid_lower_bound_seen = False
    emit("start", active_nodes=len(active))
    while active:
        if time_limit_sec is not None and time.time() - start_time >= time_limit_sec:
            termination_reason = "time_limit"
            break
        priority_bound, queue_order, node = heappop(active)
        max_tree_depth = max(max_tree_depth, node.depth)
        if node.relaxation is None:
            if nodes_processed >= max_nodes:
                heappush(active, (priority_bound, queue_order, node))
                termination_reason = "node_limit"
                break
            relaxation = solve_node_with_cut_separation(
                instance,
                node,
                cut_sets,
                cut_epsilon,
                probability_relaxation=probability_relaxation,
            )
            nodes_processed += 1
        else:
            relaxation = node.relaxation
        if not relaxation.success:
            relaxation_failures += 1
            pruned_nodes += 1
            finite_active_bounds = [item[0] for item in active if np.isfinite(item[0])]
            global_lower_bound = min(finite_active_bounds, default=global_lower_bound)
            emit(
                "node_infeasible",
                node_depth=node.depth,
                node_lower_bound=np.inf,
                active_nodes=len(active),
                message=relaxation.message,
            )
            continue
        node.lower_bound = valid_objective_lower_bound(relaxation.lower_bound)
        valid_lower_bound_seen = True
        if node.lower_bound >= upper_bound - epsilon:
            pruned_nodes += 1
            global_lower_bound = min((item[0] for item in active), default=upper_bound)
            emit(
                "node_bound_pruned",
                node_depth=node.depth,
                node_lower_bound=float(node.lower_bound),
                active_nodes=len(active),
            )
            if upper_bound - global_lower_bound <= epsilon:
                converged = True
                termination_reason = "gap_tolerance"
                break
            continue
        candidate_y = np.clip(relaxation.y, 0.0, 1.0)
        if float(np.dot(instance.retrofit_costs, candidate_y)) <= instance.budget_retrofit + 1e-7:
            candidate = evaluate_fixed_y(instance, candidate_y, epsilon=fixed_y_epsilon)
            fixed_y_oracle_calls += 1
            if candidate.objective < upper_bound:
                incumbent = candidate
                upper_bound = candidate.objective
                emit(
                    "incumbent",
                    node_depth=node.depth,
                    node_lower_bound=float(node.lower_bound),
                    active_nodes=len(active),
                )
        if node.lower_bound >= upper_bound - epsilon:
            pruned_nodes += 1
            global_lower_bound = min((item[0] for item in active), default=upper_bound)
            if upper_bound - global_lower_bound <= epsilon:
                converged = True
                termination_reason = "gap_tolerance"
                break
            continue
        branch = select_branch_variable(instance, node, probability_relaxation=probability_relaxation)
        if branch is None:
            global_lower_bound = min(global_lower_bound, node.lower_bound)
            if upper_bound - global_lower_bound <= epsilon:
                converged = True
                termination_reason = "gap_tolerance"
                break
            continue
        child_a, child_b = bisect_node(node, branch)
        for child in (child_a, child_b):
            child.lower_bound = node.lower_bound
            max_tree_depth = max(max_tree_depth, child.depth)
            can_bound_child = nodes_processed < max_nodes and (
                time_limit_sec is None or time.time() - start_time < time_limit_sec
            )
            if can_bound_child:
                child_relaxation = solve_node_with_cut_separation(
                    instance,
                    child,
                    cut_sets,
                    cut_epsilon,
                    probability_relaxation=probability_relaxation,
                )
                nodes_processed += 1
                if not child_relaxation.success:
                    relaxation_failures += 1
                    pruned_nodes += 1
                    continue
                child.lower_bound = valid_objective_lower_bound(child_relaxation.lower_bound)
                child.relaxation = child_relaxation
                valid_lower_bound_seen = True
                if child.lower_bound >= upper_bound - epsilon:
                    pruned_nodes += 1
                    continue
            counter += 1
            heappush(active, (child.lower_bound, counter, child))
        global_lower_bound = min((item[0] for item in active), default=upper_bound)
        emit(
            "node_processed",
            node_depth=node.depth,
            node_lower_bound=float(node.lower_bound),
            active_nodes=len(active),
            branch_class=branch.variable_class,
            branch_index=",".join(map(str, branch.index)),
        )
        if upper_bound - global_lower_bound <= epsilon:
            converged = True
            termination_reason = "gap_tolerance"
            break
    if active:
        finite_bounds = [item[0] for item in active if np.isfinite(item[0])]
        global_lower_bound = min(finite_bounds, default=global_lower_bound)
    else:
        if valid_lower_bound_seen:
            global_lower_bound = upper_bound
            converged = True
            termination_reason = "queue_empty" if termination_reason == "node_limit" else termination_reason
        else:
            global_lower_bound = -np.inf
            converged = False
            termination_reason = "relaxation_infeasible"
    runtime_sec = time.time() - start_time
    recourse_cuts_generated = sum(len(cuts) for cuts in cut_sets.values())
    emit(
        "finish",
        active_nodes=len(active),
        converged=converged,
        termination_reason=termination_reason,
        relaxation_failures=relaxation_failures,
        recourse_cuts_generated=recourse_cuts_generated,
    )
    return SBBResult(
        objective=upper_bound,
        lower_bound=float(global_lower_bound),
        gap=float(upper_bound - global_lower_bound),
        z=incumbent.z,
        w=incumbent.w,
        y=incumbent.y,
        fixed_y_result=incumbent,
        nodes_processed=nodes_processed,
        nodes_remaining=len(active),
        converged=converged,
        pruned_nodes=pruned_nodes,
        max_tree_depth=max_tree_depth,
        fixed_y_oracle_calls=fixed_y_oracle_calls,
        recourse_cuts_generated=recourse_cuts_generated,
        runtime_sec=runtime_sec,
        termination_reason=termination_reason,
    )


def root_node(
    instance: DADInstance,
    probability_relaxation: ProbabilityRelaxation = "product_tree",
) -> SBBNode:
    validate_probability_relaxation(probability_relaxation)
    if instance.hazard_regimes is not None:
        raise NotImplementedError("Continuous SBB does not yet include spatial hazard-regime mixtures; use exact fixed-y enumeration.")
    y_bounds = np.column_stack([np.zeros(len(instance.links)), np.ones(len(instance.links))])
    pi_bounds = np.column_stack([np.zeros(len(instance.states)), np.ones(len(instance.states))])
    state_loss_upper, _, _ = evaluate_plan_losses(
        instance,
        np.zeros(len(instance.zones), dtype=float),
        np.zeros(len(instance.centers), dtype=float),
    )
    state_loss_upper = np.maximum(0.0, np.asarray(state_loss_upper, dtype=float))
    max_loss_upper = float(max(1.0, state_loss_upper.max()))
    theta_lower = (
        np.zeros(len(instance.states), dtype=float)
        if density_cap_active(instance) or instance.ambiguity_radius <= 1e-12
        else -max_loss_upper * np.ones(len(instance.states), dtype=float)
    )
    theta_bounds = np.column_stack([theta_lower, state_loss_upper])
    density_bounds = (
        np.column_stack([np.zeros(len(instance.states)), state_loss_upper])
        if density_cap_active(instance)
        else np.empty((0, 2), dtype=float)
    )
    removal_bounds = (
        np.column_stack(
            [np.zeros(len(instance.states)), max_loss_upper * np.ones(len(instance.states))]
        )
        if density_cap_active(instance)
        else np.empty((0, 2), dtype=float)
    )
    omega_bounds = (
        (0.0, max_loss_upper)
        if density_cap_active(instance)
        else ((0.0, 0.0) if instance.ambiguity_radius <= 1e-12 else (0.0, max_loss_upper))
    )
    v_bounds: dict[tuple[int, int], tuple[float, float]] = {}
    if probability_relaxation == "product_tree" and len(instance.links) >= 2:
        for state_index, state in enumerate(instance.states):
            if state.is_tail:
                continue
            for product_index in range(1, len(instance.links)):
                v_bounds[(state_index, product_index)] = (0.0, 1.0)
    return tighten_probability_bounds(
        instance,
        SBBNode(
            y_bounds=y_bounds,
            v_bounds=v_bounds,
            pi_bounds=pi_bounds,
            theta_bounds=theta_bounds,
            density_bounds=density_bounds,
            removal_bounds=removal_bounds,
            omega_bounds=omega_bounds,
        ),
    )


def tighten_probability_bounds(instance: DADInstance, node: SBBNode) -> SBBNode:
    """Intersect node probability/product bounds with interval products implied by y-bounds."""
    tightened = clone_node(node)
    if not instance.links:
        return tightened
    for state_index, state in enumerate(instance.states):
        if state.is_tail:
            continue
        lower_product = 1.0
        upper_product = 1.0
        for link_index in range(len(instance.links)):
            factor_lower, factor_upper = link_factor_bounds(instance, tightened, state, link_index)
            lower_product *= factor_lower
            upper_product *= factor_upper
            if link_index >= 1 and (state_index, link_index) in tightened.v_bounds:
                old_lower, old_upper = tightened.v_bounds[(state_index, link_index)]
                tightened.v_bounds[(state_index, link_index)] = (
                    max(float(old_lower), float(lower_product)),
                    min(float(old_upper), float(upper_product)),
                )
        old_lower, old_upper = tightened.pi_bounds[state_index]
        tightened.pi_bounds[state_index, 0] = max(float(old_lower), float(lower_product))
        tightened.pi_bounds[state_index, 1] = min(float(old_upper), float(upper_product))
    for _ in range(len(instance.states) + 1):
        previous = tightened.pi_bounds.copy()
        lower_sum = float(tightened.pi_bounds[:, 0].sum())
        upper_sum = float(tightened.pi_bounds[:, 1].sum())
        for state_index in range(len(instance.states)):
            lower, upper = tightened.pi_bounds[state_index]
            tightened.pi_bounds[state_index, 0] = max(lower, 1.0 - (upper_sum - upper))
            tightened.pi_bounds[state_index, 1] = min(upper, 1.0 - (lower_sum - lower))
        if np.allclose(previous, tightened.pi_bounds, atol=1e-12, rtol=0.0):
            break
    if density_cap_active(instance):
        theta_lower = tightened.theta_bounds[:, 0]
        theta_upper = tightened.theta_bounds[:, 1]
        omega_lower = max(float(tightened.omega_bounds[0]), float(theta_lower.min()))
        omega_upper = min(float(tightened.omega_bounds[1]), float(theta_upper.max()))
        tightened.omega_bounds = (omega_lower, omega_upper)
        for state_index in range(len(instance.states)):
            density_lower, density_upper = tightened.density_bounds[state_index]
            removal_lower, removal_upper = tightened.removal_bounds[state_index]
            tightened.density_bounds[state_index, 0] = max(0.0, density_lower)
            tightened.density_bounds[state_index, 1] = min(
                density_upper,
                max(0.0, float(theta_upper[state_index]) - omega_lower),
            )
            tightened.removal_bounds[state_index, 0] = max(0.0, removal_lower)
            tightened.removal_bounds[state_index, 1] = min(
                removal_upper,
                max(0.0, omega_upper - float(theta_lower[state_index])),
            )
    return tightened


def solve_node_with_cut_separation(
    instance: DADInstance,
    node: SBBNode,
    cut_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    epsilon_cut: float,
    max_cut_iterations: int = 100,
    probability_relaxation: ProbabilityRelaxation = "product_tree",
) -> NodeRelaxationResult:
    latest: NodeRelaxationResult | None = None
    for iteration in range(1, max_cut_iterations + 1):
        relaxation = solve_node_relaxation(
            instance,
            node,
            cut_sets,
            probability_relaxation=probability_relaxation,
        )
        latest = relaxation
        if not relaxation.success:
            return relaxation
        violated = False
        for state_index, state in enumerate(instance.states):
            recourse = solve_recourse(instance, state, relaxation.z, relaxation.w)
            h_value = h_from_dual(instance, recourse.alpha, recourse.beta, relaxation.z, relaxation.w)
            if density_cap_active(instance):
                cut_is_violated = relaxation.theta[state_index] < h_value - epsilon_cut
            else:
                cut_is_violated = relaxation.theta[state_index] < h_value - relaxation.omega - epsilon_cut
            if cut_is_violated:
                candidate = (recourse.alpha, recourse.beta)
                if not has_cut(cut_sets[state_index], candidate):
                    cut_sets[state_index].append(candidate)
                    violated = True
        if not violated:
            return NodeRelaxationResult(**{**relaxation.__dict__, "cut_iterations": iteration})
    if latest is None:
        raise RuntimeError("Node cut separation did not start.")
    return latest


def initialize_recourse_cuts(
    instance: DADInstance,
    cut_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    incumbent: FixedYResult,
) -> None:
    trial_points = [
        (
            np.zeros(len(instance.zones), dtype=float),
            np.zeros(len(instance.centers), dtype=float),
        ),
        (np.asarray(incumbent.z, dtype=float), np.asarray(incumbent.w, dtype=float)),
    ]
    for z, w in trial_points:
        for state_index, state in enumerate(instance.states):
            recourse = solve_recourse(instance, state, z, w)
            candidate = (recourse.alpha, recourse.beta)
            if not has_cut(cut_sets[state_index], candidate):
                cut_sets[state_index].append(candidate)


def valid_objective_lower_bound(raw_lower_bound: float) -> float:
    """Apply the trivial valid bound 0 <= robust expected loss.

    The LP relaxation can be negative because McCormick products involving
    adversarial dual variables are loose. The true node objective is a
    worst-case expected loss and is therefore nonnegative, so max(LB, 0) is a
    valid strengthened lower bound for reporting and pruning.
    """
    if not np.isfinite(raw_lower_bound):
        return float(raw_lower_bound)
    return max(0.0, float(raw_lower_bound))


def solve_node_relaxation(
    instance: DADInstance,
    node: SBBNode,
    cut_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    probability_relaxation: ProbabilityRelaxation = "product_tree",
) -> NodeRelaxationResult:
    validate_probability_relaxation(probability_relaxation)
    if instance.hazard_regimes is not None:
        raise NotImplementedError("Continuous SBB does not yet include spatial hazard-regime mixtures; use exact fixed-y enumeration.")
    node = tighten_probability_bounds(instance, node)
    if not node_bounds_consistent(node):
        return NodeRelaxationResult(
            lower_bound=np.inf,
            z=np.array([]),
            w=np.array([]),
            y=np.array([]),
            pi=np.array([]),
            theta=np.array([]),
            density_dual=np.array([]),
            removal_dual=np.array([]),
            omega=np.nan,
            kappa=np.nan,
            success=False,
            message="Node bounds are inconsistent after probability-bound tightening.",
        )
    index = build_variable_index(instance, probability_relaxation=probability_relaxation)
    c = np.zeros(index.size)
    c[index.omega] = 0.0 if density_cap_active(instance) else 1.0
    c[index.kappa] = instance.ambiguity_radius
    for chi_index in index.chi:
        c[chi_index] = 1.0
    if density_cap_active(instance):
        for density_chi_index in index.density_chi:
            c[density_chi_index] = instance.ambiguity_density_cap - 1.0
        for removal_chi_index in index.removal_chi:
            c[removal_chi_index] = 1.0
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    equalities: list[np.ndarray] = []
    equality_rhs: list[float] = []
    add_budget_constraints(instance, index, rows, rhs)
    add_probability_lift(
        instance,
        node,
        index,
        rows,
        rhs,
        equalities,
        equality_rhs,
        probability_relaxation=probability_relaxation,
    )
    add_dual_epigraph(instance, index, cut_sets, rows, rhs)
    if density_cap_active(instance):
        add_capped_tv_transfer_constraints(instance, index, rows, rhs)
    else:
        add_theta_kappa_constraints(instance, index, rows, rhs)
    add_pi_theta_products(instance, node, index, rows, rhs)
    add_pi_density_products(instance, node, index, rows, rhs)
    add_pi_removal_products(instance, node, index, rows, rhs)
    bounds = build_bounds(instance, node, index)
    solution = linprog(
        c=c,
        A_ub=np.vstack(rows) if rows else None,
        b_ub=np.asarray(rhs, dtype=float) if rhs else None,
        A_eq=np.vstack(equalities) if equalities else None,
        b_eq=np.asarray(equality_rhs, dtype=float) if equality_rhs else None,
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        return NodeRelaxationResult(
            lower_bound=np.inf,
            z=np.array([]),
            w=np.array([]),
            y=np.array([]),
            pi=np.array([]),
            theta=np.array([]),
            density_dual=np.array([]),
            removal_dual=np.array([]),
            omega=np.nan,
            kappa=np.nan,
            success=False,
            message=solution.message,
        )
    return NodeRelaxationResult(
        lower_bound=float(solution.fun),
        z=solution.x[index.z],
        w=solution.x[index.w],
        y=solution.x[index.y],
        pi=solution.x[index.pi],
        theta=solution.x[index.theta],
        density_dual=solution.x[index.density_dual],
        removal_dual=solution.x[index.removal_dual],
        omega=float(solution.x[index.omega]),
        kappa=float(solution.x[index.kappa]),
        success=True,
    )


def node_bounds_consistent(node: SBBNode, atol: float = 1e-10) -> bool:
    arrays = [
        node.y_bounds,
        node.pi_bounds,
        node.theta_bounds,
        node.density_bounds,
        node.removal_bounds,
    ]
    for bounds in arrays:
        if np.any(bounds[:, 0] > bounds[:, 1] + atol):
            return False
    for lower, upper in node.v_bounds.values():
        if lower > upper + atol:
            return False
    if node.omega_bounds[0] > node.omega_bounds[1] + atol:
        return False
    return True


def build_variable_index(
    instance: DADInstance,
    probability_relaxation: ProbabilityRelaxation = "product_tree",
) -> VariableIndex:
    validate_probability_relaxation(probability_relaxation)
    if instance.hazard_regimes is not None:
        raise NotImplementedError("Continuous SBB does not yet include spatial hazard-regime mixtures; use exact fixed-y enumeration.")
    cursor = 0
    z = list(range(cursor, cursor + len(instance.zones)))
    cursor += len(instance.zones)
    w = list(range(cursor, cursor + len(instance.centers)))
    cursor += len(instance.centers)
    y = list(range(cursor, cursor + len(instance.links)))
    cursor += len(instance.links)
    pi = list(range(cursor, cursor + len(instance.states)))
    cursor += len(instance.states)
    v: dict[tuple[int, int], int] = {}
    if probability_relaxation == "product_tree" and len(instance.links) >= 2:
        for state_index, state in enumerate(instance.states):
            if state.is_tail:
                continue
            for product_index in range(1, len(instance.links)):
                v[(state_index, product_index)] = cursor
                cursor += 1
    theta = list(range(cursor, cursor + len(instance.states)))
    cursor += len(instance.states)
    chi = list(range(cursor, cursor + len(instance.states)))
    cursor += len(instance.states)
    density_dual: list[int] = []
    density_chi: list[int] = []
    removal_dual: list[int] = []
    removal_chi: list[int] = []
    if density_cap_active(instance):
        density_dual = list(range(cursor, cursor + len(instance.states)))
        cursor += len(instance.states)
        density_chi = list(range(cursor, cursor + len(instance.states)))
        cursor += len(instance.states)
        removal_dual = list(range(cursor, cursor + len(instance.states)))
        cursor += len(instance.states)
        removal_chi = list(range(cursor, cursor + len(instance.states)))
        cursor += len(instance.states)
    omega = cursor
    cursor += 1
    kappa = cursor
    cursor += 1
    return VariableIndex(
        z=z,
        w=w,
        y=y,
        pi=pi,
        v=v,
        theta=theta,
        chi=chi,
        density_dual=density_dual,
        density_chi=density_chi,
        removal_dual=removal_dual,
        removal_chi=removal_chi,
        omega=omega,
        kappa=kappa,
        size=cursor,
    )


def add_budget_constraints(
    instance: DADInstance,
    index: VariableIndex,
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    row = np.zeros(index.size)
    row[index.z] = instance.renovation_costs
    rows.append(row)
    rhs.append(instance.budget_renovation)
    row = np.zeros(index.size)
    row[index.w] = instance.capacity_costs
    rows.append(row)
    rhs.append(instance.budget_capacity)
    row = np.zeros(index.size)
    row[index.y] = instance.retrofit_costs
    rows.append(row)
    rhs.append(instance.budget_retrofit)


def add_probability_lift(
    instance: DADInstance,
    node: SBBNode,
    index: VariableIndex,
    rows: list[np.ndarray],
    rhs: list[float],
    equalities: list[np.ndarray],
    equality_rhs: list[float],
    probability_relaxation: ProbabilityRelaxation = "product_tree",
) -> None:
    validate_probability_relaxation(probability_relaxation)
    if instance.hazard_regimes is not None:
        raise NotImplementedError("Continuous SBB does not yet include spatial hazard-regime mixtures; use exact fixed-y enumeration.")
    non_tail_indices = [i for i, state in enumerate(instance.states) if not state.is_tail]
    for state_index in non_tail_indices:
        state = instance.states[state_index]
        if len(instance.links) == 0:
            row = np.zeros(index.size)
            row[index.pi[state_index]] = 1.0
            equalities.append(row)
            equality_rhs.append(1.0)
            continue
        if len(instance.links) == 1:
            factor, _, _ = link_factor_expr(instance, node, index, state, 0)
            row = np.zeros(index.size)
            row[index.pi[state_index]] = 1.0
            add_expr_to_row(row, factor, -1.0)
            equalities.append(row)
            equality_rhs.append(factor.constant)
            continue
        if probability_relaxation in {"corner_boxes", "corner_link_cuts"}:
            if probability_relaxation == "corner_link_cuts":
                add_corner_link_cuts(instance, node, index, state_index, state, rows, rhs)
            continue
        factor_0, lower_0, upper_0 = link_factor_expr(instance, node, index, state, 0)
        factor_1, lower_1, upper_1 = link_factor_expr(instance, node, index, state, 1)
        first_product = index.v[(state_index, 1)]
        add_mccormick(rows, rhs, index.size, first_product, factor_0, lower_0, upper_0, factor_1, lower_1, upper_1)
        for product_index in range(2, len(instance.links)):
            previous_product = LinearExpr.variable(index.v[(state_index, product_index - 1)])
            previous_lower, previous_upper = node.v_bounds[(state_index, product_index - 1)]
            factor, lower, upper = link_factor_expr(instance, node, index, state, product_index)
            product_variable = index.v[(state_index, product_index)]
            add_mccormick(
                rows,
                rhs,
                index.size,
                product_variable,
                previous_product,
                previous_lower,
                previous_upper,
                factor,
                lower,
                upper,
            )
        row = np.zeros(index.size)
        row[index.pi[state_index]] = 1.0
        row[index.v[(state_index, len(instance.links) - 1)]] = -1.0
        equalities.append(row)
        equality_rhs.append(0.0)
    row = np.zeros(index.size)
    for state_index in range(len(instance.states)):
        row[index.pi[state_index]] = 1.0
    equalities.append(row)
    equality_rhs.append(1.0)


def add_corner_link_cuts(
    instance: DADInstance,
    node: SBBNode,
    index: VariableIndex,
    state_index: int,
    state: State,
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    """Add valid affine probability cuts from the remaining link-factor bounds."""
    for link_index in range(len(instance.links)):
        other_lower = 1.0
        other_upper = 1.0
        for other_index in range(len(instance.links)):
            if other_index == link_index:
                continue
            factor_lower, factor_upper = link_factor_bounds(instance, node, state, other_index)
            other_lower *= factor_lower
            other_upper *= factor_upper
        factor, _, _ = link_factor_expr(instance, node, index, state, link_index)
        pi = LinearExpr.variable(index.pi[state_index])
        add_linear_inequality(
            rows,
            rhs,
            index.size,
            factor.scaled(other_lower).plus(pi.scaled(-1.0)),
            0.0,
        )
        add_linear_inequality(
            rows,
            rhs,
            index.size,
            pi.plus(factor.scaled(-other_upper)),
            0.0,
        )


def add_dual_epigraph(
    instance: DADInstance,
    index: VariableIndex,
    cut_sets: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    for state_index in range(len(instance.states)):
        row = np.zeros(index.size)
        row[index.theta[state_index]] = -1.0
        if not density_cap_active(instance):
            row[index.omega] = -1.0
        rows.append(row)
        rhs.append(0.0)
        for alpha, beta in cut_sets[state_index]:
            constant, z_coefficients, w_coefficients = h_coefficients(instance, alpha, beta)
            row = np.zeros(index.size)
            row[index.z] = z_coefficients
            row[index.w] = w_coefficients
            row[index.theta[state_index]] = -1.0
            if not density_cap_active(instance):
                row[index.omega] = -1.0
            rows.append(row)
            rhs.append(-constant)
    survival_matrices = [instance.survival_matrix(state) for state in instance.states]
    for lower_state_index, lower_state in enumerate(instance.states):
        lower_failures = set(lower_state.failed_links)
        lower_survival = survival_matrices[lower_state_index]
        for upper_state_index, upper_state in enumerate(instance.states):
            if lower_state_index == upper_state_index:
                continue
            failure_set_dominates = upper_state.is_tail or (
                not lower_state.is_tail
                and lower_failures < set(upper_state.failed_links)
            )
            if not failure_set_dominates:
                continue
            upper_survival = survival_matrices[upper_state_index]
            if np.any(upper_survival > lower_survival + 1e-12):
                continue
            row = np.zeros(index.size)
            row[index.theta[lower_state_index]] = 1.0
            row[index.theta[upper_state_index]] = -1.0
            rows.append(row)
            rhs.append(0.0)


def add_theta_kappa_constraints(
    instance: DADInstance,
    index: VariableIndex,
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    for theta_index in index.theta:
        row = np.zeros(index.size)
        row[theta_index] = 1.0
        row[index.kappa] = -0.5
        rows.append(row)
        rhs.append(0.0)
        row = np.zeros(index.size)
        row[theta_index] = -1.0
        row[index.kappa] = -0.5
        rows.append(row)
        rhs.append(0.0)


def add_capped_tv_transfer_constraints(
    instance: DADInstance,
    index: VariableIndex,
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    for state_index in range(len(instance.states)):
        row = np.zeros(index.size)
        row[index.theta[state_index]] = 1.0
        row[index.omega] = -1.0
        row[index.kappa] = -1.0
        row[index.density_dual[state_index]] = -1.0
        rows.append(row)
        rhs.append(0.0)

        row = np.zeros(index.size)
        row[index.omega] = 1.0
        row[index.theta[state_index]] = -1.0
        row[index.removal_dual[state_index]] = -1.0
        rows.append(row)
        rhs.append(0.0)


def add_pi_theta_products(
    instance: DADInstance,
    node: SBBNode,
    index: VariableIndex,
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    for state_index in range(len(instance.states)):
        pi_expr = LinearExpr.variable(index.pi[state_index])
        theta_expr = LinearExpr.variable(index.theta[state_index])
        pi_lower, pi_upper = node.pi_bounds[state_index]
        theta_lower, theta_upper = node.theta_bounds[state_index]
        add_mccormick(
            rows,
            rhs,
            index.size,
            index.chi[state_index],
            pi_expr,
            float(pi_lower),
            float(pi_upper),
            theta_expr,
            float(theta_lower),
            float(theta_upper),
        )


def add_pi_density_products(
    instance: DADInstance,
    node: SBBNode,
    index: VariableIndex,
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    if not density_cap_active(instance):
        return
    for state_index in range(len(instance.states)):
        pi_expr = LinearExpr.variable(index.pi[state_index])
        density_expr = LinearExpr.variable(index.density_dual[state_index])
        pi_lower, pi_upper = node.pi_bounds[state_index]
        density_lower, density_upper = node.density_bounds[state_index]
        add_mccormick(
            rows,
            rhs,
            index.size,
            index.density_chi[state_index],
            pi_expr,
            float(pi_lower),
            float(pi_upper),
            density_expr,
            float(density_lower),
            float(density_upper),
        )


def add_pi_removal_products(
    instance: DADInstance,
    node: SBBNode,
    index: VariableIndex,
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    if not density_cap_active(instance):
        return
    for state_index in range(len(instance.states)):
        pi_expr = LinearExpr.variable(index.pi[state_index])
        removal_expr = LinearExpr.variable(index.removal_dual[state_index])
        pi_lower, pi_upper = node.pi_bounds[state_index]
        removal_lower, removal_upper = node.removal_bounds[state_index]
        add_mccormick(
            rows,
            rhs,
            index.size,
            index.removal_chi[state_index],
            pi_expr,
            float(pi_lower),
            float(pi_upper),
            removal_expr,
            float(removal_lower),
            float(removal_upper),
        )


def build_bounds(instance: DADInstance, node: SBBNode, index: VariableIndex) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * index.size
    for variable in index.z:
        bounds[variable] = (0.0, 1.0)
    for center_index, variable in enumerate(index.w):
        cost = instance.capacity_costs[center_index]
        upper = instance.budget_capacity / cost if cost > 0 else None
        bounds[variable] = (0.0, upper)
    for link_index, variable in enumerate(index.y):
        bounds[variable] = tuple(map(float, node.y_bounds[link_index]))
    for state_index, variable in enumerate(index.pi):
        bounds[variable] = tuple(map(float, node.pi_bounds[state_index]))
    for key, variable in index.v.items():
        bounds[variable] = node.v_bounds[key]
    for state_index, variable in enumerate(index.theta):
        bounds[variable] = tuple(map(float, node.theta_bounds[state_index]))
    d_max = instance.d_max
    for state_index, variable in enumerate(index.chi):
        pi_lower, pi_upper = node.pi_bounds[state_index]
        theta_lower, theta_upper = node.theta_bounds[state_index]
        products = [
            pi_lower * theta_lower,
            pi_lower * theta_upper,
            pi_upper * theta_lower,
            pi_upper * theta_upper,
        ]
        bounds[variable] = (float(min(products)), float(max(products)))
    for state_index, variable in enumerate(index.density_dual):
        bounds[variable] = tuple(map(float, node.density_bounds[state_index]))
    for state_index, variable in enumerate(index.density_chi):
        pi_lower, pi_upper = node.pi_bounds[state_index]
        density_lower, density_upper = node.density_bounds[state_index]
        products = [
            pi_lower * density_lower,
            pi_lower * density_upper,
            pi_upper * density_lower,
            pi_upper * density_upper,
        ]
        bounds[variable] = (float(min(products)), float(max(products)))
    for state_index, variable in enumerate(index.removal_dual):
        bounds[variable] = tuple(map(float, node.removal_bounds[state_index]))
    for state_index, variable in enumerate(index.removal_chi):
        pi_lower, pi_upper = node.pi_bounds[state_index]
        removal_lower, removal_upper = node.removal_bounds[state_index]
        products = [
            pi_lower * removal_lower,
            pi_lower * removal_upper,
            pi_upper * removal_lower,
            pi_upper * removal_upper,
        ]
        bounds[variable] = (float(min(products)), float(max(products)))
    bounds[index.omega] = tuple(map(float, node.omega_bounds))
    if density_cap_active(instance):
        gamma_upper = max(0.0, float(node.theta_bounds[:, 1].max()) - float(node.omega_bounds[0]))
        bounds[index.kappa] = (0.0, gamma_upper)
    else:
        bounds[index.kappa] = (0.0, 2.0 * d_max)
    return bounds


def density_cap_active(instance: DADInstance) -> bool:
    return instance.ambiguity_density_cap is not None


def link_factor_expr(
    instance: DADInstance,
    node: SBBNode,
    index: VariableIndex,
    state: State,
    link_index: int,
) -> tuple[LinearExpr, float, float]:
    link = instance.links[link_index]
    baseline = link.baseline_failure_probability
    residual = link.residual_failure_probability
    reduction = baseline - residual
    y_lower, y_upper = node.y_bounds[link_index]
    y_variable = index.y[link_index]
    if link.id in state.failed_links:
        expr = LinearExpr(baseline, {y_variable: -reduction})
        lower = baseline - reduction * y_upper
        upper = baseline - reduction * y_lower
    else:
        expr = LinearExpr(1.0 - baseline, {y_variable: reduction})
        lower = 1.0 - baseline + reduction * y_lower
        upper = 1.0 - baseline + reduction * y_upper
    return expr, float(lower), float(upper)


def link_factor_bounds(
    instance: DADInstance,
    node: SBBNode,
    state: State,
    link_index: int,
) -> tuple[float, float]:
    link = instance.links[link_index]
    baseline = link.baseline_failure_probability
    residual = link.residual_failure_probability
    reduction = baseline - residual
    y_lower, y_upper = node.y_bounds[link_index]
    if link.id in state.failed_links:
        lower = baseline - reduction * y_upper
        upper = baseline - reduction * y_lower
    else:
        lower = 1.0 - baseline + reduction * y_lower
        upper = 1.0 - baseline + reduction * y_upper
    return float(lower), float(upper)


def add_mccormick(
    rows: list[np.ndarray],
    rhs: list[float],
    size: int,
    result_index: int,
    u_expr: LinearExpr,
    u_lower: float,
    u_upper: float,
    v_expr: LinearExpr,
    v_lower: float,
    v_upper: float,
) -> None:
    add_linear_inequality(
        rows,
        rhs,
        size,
        v_expr.scaled(u_lower).plus(u_expr.scaled(v_lower)).plus(LinearExpr.variable(result_index, -1.0)),
        u_lower * v_lower,
    )
    add_linear_inequality(
        rows,
        rhs,
        size,
        v_expr.scaled(u_upper).plus(u_expr.scaled(v_upper)).plus(LinearExpr.variable(result_index, -1.0)),
        u_upper * v_upper,
    )
    add_linear_inequality(
        rows,
        rhs,
        size,
        LinearExpr.variable(result_index).plus(u_expr.scaled(-v_lower)).plus(v_expr.scaled(-u_upper)),
        -u_upper * v_lower,
    )
    add_linear_inequality(
        rows,
        rhs,
        size,
        LinearExpr.variable(result_index).plus(u_expr.scaled(-v_upper)).plus(v_expr.scaled(-u_lower)),
        -u_lower * v_upper,
    )


def add_linear_inequality(
    rows: list[np.ndarray],
    rhs: list[float],
    size: int,
    expr: LinearExpr,
    upper_bound: float,
) -> None:
    row = np.zeros(size)
    for index, value in expr.coefficients.items():
        row[index] += value
    rows.append(row)
    rhs.append(float(upper_bound - expr.constant))


def add_expr_to_row(row: np.ndarray, expr: LinearExpr, scale: float = 1.0) -> None:
    for index, value in expr.coefficients.items():
        row[index] += scale * value


def h_coefficients(instance: DADInstance, alpha: np.ndarray, beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    cut = type("Cut", (), {})()
    cut.distribution = np.array([1.0])
    cut.alphas = np.asarray([alpha], dtype=float)
    cut.betas = np.asarray([beta], dtype=float)
    return aggregate_cut_for_single_state(instance, alpha, beta)


def aggregate_cut_for_single_state(
    instance: DADInstance,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    base = instance.base_demands
    immediate = instance.base_immediate_losses
    one_minus_beta = 1.0 - np.asarray(beta, dtype=float)
    loss_exposure_coefficients = immediate + one_minus_beta * base
    constant = float(np.sum(loss_exposure_coefficients) - np.dot(alpha, instance.existing_capacities))
    z_coefficients = -loss_exposure_coefficients
    w_coefficients = -np.asarray(alpha, dtype=float)
    return constant, z_coefficients, w_coefficients


def h_from_dual(
    instance: DADInstance,
    alpha: np.ndarray,
    beta: np.ndarray,
    z: Sequence[float],
    w: Sequence[float],
) -> float:
    constant, z_coefficients, w_coefficients = aggregate_cut_for_single_state(instance, alpha, beta)
    return float(constant + np.dot(z_coefficients, z) + np.dot(w_coefficients, w))


def has_cut(existing: list[tuple[np.ndarray, np.ndarray]], candidate: tuple[np.ndarray, np.ndarray], atol: float = 1e-8) -> bool:
    alpha, beta = candidate
    return any(np.allclose(alpha, old_alpha, atol=atol) and np.allclose(beta, old_beta, atol=atol) for old_alpha, old_beta in existing)


def select_branch_variable(
    instance: DADInstance,
    node: SBBNode,
    min_width: float = 1e-6,
    probability_relaxation: ProbabilityRelaxation = "product_tree",
) -> BranchKey | None:
    validate_probability_relaxation(probability_relaxation)
    if instance.hazard_regimes is not None:
        raise NotImplementedError("Continuous SBB does not yet include spatial hazard-regime mixtures; use exact fixed-y enumeration.")
    node = tighten_probability_bounds(instance, node)
    candidates: list[tuple[float, int, BranchKey]] = []
    priority = {"omega": 7, "theta": 6, "density": 5, "removal": 4, "pi": 3, "v": 2, "y": 1}
    for index, (lower, upper) in enumerate(node.y_bounds):
        width = normalized_branch_width(lower, upper, "y", instance)
        if width > min_width:
            candidates.append((width, priority["y"], BranchKey("y", (index,))))
    if probability_relaxation == "product_tree":
        for key, (lower, upper) in node.v_bounds.items():
            width = normalized_branch_width(lower, upper, "v", instance)
            if width > min_width:
                candidates.append((width, priority["v"], BranchKey("v", key)))
    for index, (lower, upper) in enumerate(node.pi_bounds):
        width = normalized_branch_width(lower, upper, "pi", instance)
        if width > min_width:
            candidates.append((width, priority["pi"], BranchKey("pi", (index,))))
    for index, (lower, upper) in enumerate(node.theta_bounds):
        width = normalized_branch_width(lower, upper, "theta", instance)
        if width > min_width:
            candidates.append((width, priority["theta"], BranchKey("theta", (index,))))
    for index, (lower, upper) in enumerate(node.density_bounds):
        width = normalized_branch_width(lower, upper, "density", instance)
        if width > min_width:
            candidates.append((width, priority["density"], BranchKey("density", (index,))))
    for index, (lower, upper) in enumerate(node.removal_bounds):
        width = normalized_branch_width(lower, upper, "removal", instance)
        if width > min_width:
            candidates.append((width, priority["removal"], BranchKey("removal", (index,))))
    omega_width = normalized_branch_width(*node.omega_bounds, "omega", instance)
    if omega_width > min_width:
        candidates.append((omega_width, priority["omega"], BranchKey("omega", (0,))))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def relative_width(lower: float, upper: float) -> float:
    return float((upper - lower) / (1.0 + abs(lower)))


def normalized_branch_width(lower: float, upper: float, variable_class: BranchClass, instance: DADInstance) -> float:
    width = max(0.0, float(upper - lower))
    if variable_class in {"y", "v", "pi"}:
        return width
    if variable_class == "theta":
        return width / max(1.0, 2.0 * instance.d_max)
    if variable_class in {"density", "removal", "omega"}:
        return width / max(1.0, 2.0 * instance.d_max)
    return relative_width(lower, upper)


def bisect_node(node: SBBNode, branch: BranchKey) -> tuple[SBBNode, SBBNode]:
    left = clone_node(node)
    right = clone_node(node)
    if branch.variable_class == "y":
        index = branch.index[0]
        lower, upper = node.y_bounds[index]
        midpoint = 0.5 * (lower + upper)
        left.y_bounds[index, 1] = midpoint
        right.y_bounds[index, 0] = midpoint
    elif branch.variable_class == "v":
        lower, upper = node.v_bounds[branch.index]
        midpoint = 0.5 * (lower + upper)
        left.v_bounds[branch.index] = (lower, midpoint)
        right.v_bounds[branch.index] = (midpoint, upper)
    elif branch.variable_class == "pi":
        index = branch.index[0]
        lower, upper = node.pi_bounds[index]
        midpoint = 0.5 * (lower + upper)
        left.pi_bounds[index, 1] = midpoint
        right.pi_bounds[index, 0] = midpoint
    elif branch.variable_class == "theta":
        index = branch.index[0]
        lower, upper = node.theta_bounds[index]
        midpoint = 0.5 * (lower + upper)
        left.theta_bounds[index, 1] = midpoint
        right.theta_bounds[index, 0] = midpoint
    elif branch.variable_class == "density":
        index = branch.index[0]
        lower, upper = node.density_bounds[index]
        midpoint = 0.5 * (lower + upper)
        left.density_bounds[index, 1] = midpoint
        right.density_bounds[index, 0] = midpoint
    elif branch.variable_class == "removal":
        index = branch.index[0]
        lower, upper = node.removal_bounds[index]
        midpoint = 0.5 * (lower + upper)
        left.removal_bounds[index, 1] = midpoint
        right.removal_bounds[index, 0] = midpoint
    elif branch.variable_class == "omega":
        lower, upper = node.omega_bounds
        midpoint = 0.5 * (lower + upper)
        left.omega_bounds = (lower, midpoint)
        right.omega_bounds = (midpoint, upper)
    left.depth = node.depth + 1
    right.depth = node.depth + 1
    left.relaxation = None
    right.relaxation = None
    return left, right


def clone_node(node: SBBNode) -> SBBNode:
    return SBBNode(
        y_bounds=node.y_bounds.copy(),
        v_bounds=dict(node.v_bounds),
        pi_bounds=node.pi_bounds.copy(),
        theta_bounds=node.theta_bounds.copy(),
        density_bounds=node.density_bounds.copy(),
        removal_bounds=node.removal_bounds.copy(),
        omega_bounds=tuple(node.omega_bounds),
        lower_bound=node.lower_bound,
        depth=node.depth,
        relaxation=node.relaxation,
    )

