from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from math import isfinite
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OracleEvaluation:
    status: str
    objective: float | None = None
    lower_bound: float | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    w: np.ndarray | None = None
    iterations: int = 0
    oracle_gap: float = 0.0
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class MonotoneBox:
    node_id: int
    lower: np.ndarray
    upper: np.ndarray
    depth: int
    lower_bound: float
    upper_evaluation: OracleEvaluation

    @property
    def width(self) -> float:
        return float(np.max(self.upper - self.lower))


@dataclass(frozen=True)
class MonotoneBBResult:
    incumbent: OracleEvaluation
    global_lower_bound: float
    absolute_gap: float
    relative_gap_percent: float
    converged: bool
    termination_reason: str
    leaves: tuple[MonotoneBox, ...]
    nodes_created: int
    nodes_processed: int
    nodes_budget_pruned: int
    nodes_capability_pruned: int
    nodes_bound_pruned: int
    nodes_epsilon_pruned: int
    nodes_closed_feasible_upper: int
    nodes_active_at_end: int
    maximum_depth: int
    unique_oracle_calls: int
    oracle_cache_hits: int
    initial_corner_evaluations: int
    total_fixed_y_iterations: int
    maximum_oracle_gap: float
    convergence_events: tuple[Mapping[str, Any], ...]
    incumbent_history: tuple[Mapping[str, Any], ...]


class OracleIncomplete(RuntimeError):
    pass


def _as_evaluation(value: OracleEvaluation | Mapping[str, Any], y: np.ndarray) -> OracleEvaluation:
    if isinstance(value, OracleEvaluation):
        return value
    status = str(value.get("status", "feasible"))
    objective = value.get("objective")
    lower_bound = value.get("lower_bound")
    if objective is not None:
        objective = float(objective)
    if lower_bound is not None:
        lower_bound = float(lower_bound)
    return OracleEvaluation(
        status=status,
        objective=objective,
        lower_bound=lower_bound,
        y=np.asarray(value.get("y", y), dtype=float),
        z=None if value.get("z") is None else np.asarray(value["z"], dtype=float),
        w=None if value.get("w") is None else np.asarray(value["w"], dtype=float),
        iterations=int(value.get("iterations", 0)),
        oracle_gap=float(value.get("oracle_gap", 0.0)),
        payload=dict(value),
        error=None if value.get("error") is None else str(value.get("error")),
    )


def _validate_evaluation(evaluation: OracleEvaluation) -> None:
    if evaluation.status in {"infeasible", "capability_infeasible", "pruned"}:
        return
    if evaluation.status not in {"feasible", "optimal"}:
        raise OracleIncomplete(evaluation.error or f"Oracle returned status {evaluation.status!r}.")
    if evaluation.objective is None or evaluation.lower_bound is None:
        raise OracleIncomplete("A feasible oracle result must provide objective and lower_bound.")
    if not isfinite(evaluation.objective) or not isfinite(evaluation.lower_bound):
        raise OracleIncomplete("A feasible oracle result must provide finite objective and lower_bound.")
    if evaluation.lower_bound > evaluation.objective + 1e-8:
        raise OracleIncomplete("Oracle lower_bound exceeds its feasible objective.")


def run_monotone_box_bb(
    *,
    initial_boxes: Sequence[tuple[Sequence[float], Sequence[float]]],
    costs: Sequence[float],
    budget: float,
    oracle: Callable[[np.ndarray], OracleEvaluation | Mapping[str, Any]],
    incumbent: OracleEvaluation | Mapping[str, Any],
    budget_tolerance: float = 1e-10,
    box_width_tolerance: float = 1e-8,
    abs_gap_target: float = 0.0,
    rel_gap_target: float = 0.001,
    class_tolerance: float = 1e-6,
    branching_rule: str = "width",
    evaluate_boundary: bool = False,
    cache: dict[tuple[float, ...], OracleEvaluation] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> MonotoneBBResult:
    costs_array = np.asarray(costs, dtype=float)
    if costs_array.ndim != 1 or costs_array.size == 0 or np.any(costs_array < 0.0):
        raise ValueError("costs must be a nonnegative one-dimensional vector.")
    if not isfinite(float(budget)) or budget < 0.0:
        raise ValueError("budget must be finite and nonnegative.")
    if branching_rule not in {"width", "budget_weighted_width"}:
        raise ValueError("branching_rule must be 'width' or 'budget_weighted_width'.")
    dimension = costs_array.size
    cache = {} if cache is None else cache
    oracle_calls = 0
    cache_hits = 0
    initial_corner_evaluations = 0
    total_iterations = 0
    maximum_oracle_gap = 0.0
    next_node_id = 1
    nodes_created = 0
    nodes_processed = 0
    nodes_budget_pruned = 0
    nodes_capability_pruned = 0
    nodes_bound_pruned = 0
    nodes_epsilon_pruned = 0
    nodes_closed_feasible_upper = 0
    maximum_depth = 0
    active: dict[int, MonotoneBox] = {}
    leaves: dict[int, MonotoneBox] = {}
    events: list[Mapping[str, Any]] = []
    incumbent_history: list[Mapping[str, Any]] = []

    best = _as_evaluation(incumbent, np.asarray(incumbent.y if isinstance(incumbent, OracleEvaluation) and incumbent.y is not None else [], dtype=float))
    _validate_evaluation(best)
    if best.objective is None or best.y is None:
        raise ValueError("The incumbent must contain a feasible objective and y vector.")
    best_y = np.asarray(best.y, dtype=float)
    if best_y.shape != (dimension,):
        raise ValueError("The incumbent dimension does not match costs.")
    if float(costs_array @ best_y) > budget + budget_tolerance:
        raise ValueError("The supplied incumbent violates the retrofit budget.")

    def cache_key(vector: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(vector, dtype=float), 12).tolist())

    def evaluate_corner(vector: np.ndarray) -> OracleEvaluation:
        nonlocal oracle_calls, cache_hits, total_iterations, maximum_oracle_gap
        key = cache_key(vector)
        if key in cache:
            cache_hits += 1
            return cache[key]
        evaluation = _as_evaluation(oracle(np.asarray(vector, dtype=float).copy()), vector)
        _validate_evaluation(evaluation)
        cache[key] = evaluation
        oracle_calls += 1
        total_iterations += evaluation.iterations
        maximum_oracle_gap = max(maximum_oracle_gap, float(evaluation.oracle_gap))
        return evaluation

    def emit(event: str, *, node: MonotoneBox | None = None) -> None:
        if best.objective is None:
            return
        lower = min((box.lower_bound for box in leaves.values()), default=float("nan"))
        gap = float(best.objective - lower) if isfinite(lower) else float("nan")
        record = {
            "event_index": len(events) + 1,
            "oracle_calls": oracle_calls,
            "nodes_processed": nodes_processed,
            "active_nodes": len(active),
            "incumbent_UB": float(best.objective),
            "global_LB": lower,
            "absolute_gap": gap,
            "relative_gap_percent": 100.0 * gap / max(1.0, abs(float(best.objective))) if isfinite(gap) else float("nan"),
            "event": event,
        }
        if node is not None:
            record.update({"node_id": node.node_id, "depth": node.depth, "node_LB": node.lower_bound})
        events.append(record)
        if progress_callback is not None:
            progress_callback(record)

    def absolute_target() -> float:
        return max(float(abs_gap_target), float(rel_gap_target) * max(1.0, abs(float(best.objective))))

    def add_box(lower: Sequence[float], upper: Sequence[float], depth: int, *, reused: OracleEvaluation | None = None) -> None:
        nonlocal next_node_id, nodes_created, maximum_depth, best, best_y, nodes_budget_pruned, nodes_capability_pruned, cache_hits
        lower_array = np.asarray(lower, dtype=float)
        upper_array = np.asarray(upper, dtype=float)
        if lower_array.shape != (dimension,) or upper_array.shape != (dimension,):
            raise ValueError("Box bounds have the wrong dimension.")
        if np.any(lower_array < -budget_tolerance) or np.any(upper_array > 1.0 + budget_tolerance) or np.any(lower_array > upper_array + budget_tolerance):
            raise ValueError("Box bounds must satisfy 0 <= lower <= upper <= 1.")
        lower_array = np.clip(lower_array, 0.0, 1.0)
        upper_array = np.clip(upper_array, 0.0, 1.0)
        box_id = next_node_id
        next_node_id += 1
        nodes_created += 1
        maximum_depth = max(maximum_depth, depth)
        lower_budget = float(costs_array @ lower_array)
        if lower_budget > budget + budget_tolerance:
            nodes_budget_pruned += 1
            emit("budget_pruned")
            return
        if reused is not None:
            cache_hits += 1
        evaluation = reused if reused is not None else evaluate_corner(upper_array)
        if evaluation.status in {"infeasible", "capability_infeasible", "pruned"}:
            nodes_capability_pruned += 1
            emit("capability_pruned")
            return
        if evaluation.lower_bound is None or evaluation.objective is None:
            raise OracleIncomplete("Feasible upper-corner evaluation lacks bounds.")
        box = MonotoneBox(box_id, lower_array, upper_array, depth, float(evaluation.lower_bound), evaluation)
        active[box_id] = box
        leaves[box_id] = box
        upper_budget = float(costs_array @ upper_array)
        if upper_budget <= budget + budget_tolerance and float(evaluation.objective) < float(best.objective) - 1e-12:
            best = evaluation
            best_y = np.asarray(evaluation.y, dtype=float)
            incumbent_history.append({
                "event": "incumbent_improvement",
                "node_id": box_id,
                "objective": float(evaluation.objective),
                "lower_bound": float(evaluation.lower_bound),
                "y_json": np.asarray(evaluation.y, dtype=float).tolist(),
                "z_json": None if evaluation.z is None else np.asarray(evaluation.z, dtype=float).tolist(),
                "w_json": None if evaluation.w is None else np.asarray(evaluation.w, dtype=float).tolist(),
            })
            emit("incumbent_improvement", node=box)

    for lower, upper in initial_boxes:
        initial_corner_evaluations += 1
        add_box(lower, upper, 0)

    while active:
        target = absolute_target()
        candidate = min(active.values(), key=lambda box: (box.lower_bound, box.depth, box.node_id))
        if candidate.lower_bound >= float(best.objective) - 1e-12:
            active.pop(candidate.node_id)
            leaves[candidate.node_id] = candidate
            nodes_bound_pruned += 1
            nodes_processed += 1
            emit("bound_pruned", node=candidate)
            continue
        if candidate.lower_bound >= float(best.objective) - target:
            active.pop(candidate.node_id)
            leaves[candidate.node_id] = candidate
            nodes_epsilon_pruned += 1
            nodes_processed += 1
            emit("epsilon_pruned", node=candidate)
            continue
        if float(costs_array @ candidate.upper) <= budget + budget_tolerance:
            active.pop(candidate.node_id)
            leaves[candidate.node_id] = candidate
            nodes_closed_feasible_upper += 1
            nodes_processed += 1
            emit("closed_feasible_upper", node=candidate)
            continue
        if candidate.width <= box_width_tolerance:
            nodes_processed += 1
            emit("width_tolerance_active", node=candidate)
            break
        split_index = int(np.argmax(candidate.upper - candidate.lower))
        if branching_rule == "budget_weighted_width":
            widths = candidate.upper - candidate.lower
            scores = np.divide(widths, np.maximum(costs_array, 1e-12))
            split_index = int(np.argmax(scores))
        midpoint = 0.5 * (candidate.lower[split_index] + candidate.upper[split_index])
        child_lower_a = candidate.lower.copy()
        child_upper_a = candidate.upper.copy()
        child_upper_a[split_index] = midpoint
        child_lower_b = candidate.lower.copy()
        child_lower_b[split_index] = midpoint
        child_upper_b = candidate.upper.copy()
        active.pop(candidate.node_id)
        leaves.pop(candidate.node_id, None)
        nodes_processed += 1
        add_box(child_lower_a, child_upper_a, candidate.depth + 1)
        add_box(child_lower_b, child_upper_b, candidate.depth + 1, reused=candidate.upper_evaluation)
        emit("branched", node=candidate)

    lower = min((box.lower_bound for box in leaves.values()), default=float("nan"))
    gap = float(best.objective - lower)
    relative_gap = 100.0 * gap / max(1.0, abs(float(best.objective)))
    final_budget_slack = budget - float(costs_array @ np.asarray(best.y, dtype=float))
    if final_budget_slack < -budget_tolerance:
        raise RuntimeError("The B&B incumbent violates the retrofit budget.")
    converged = bool(isfinite(gap) and gap <= absolute_target() + 1e-10 and not active)
    termination_reason = "epsilon_gap_closed" if converged else ("box_width_tolerance" if active else "queue_exhausted")
    return MonotoneBBResult(
        incumbent=best,
        global_lower_bound=float(lower),
        absolute_gap=gap,
        relative_gap_percent=relative_gap,
        converged=converged,
        termination_reason=termination_reason,
        leaves=tuple(sorted(leaves.values(), key=lambda box: box.node_id)),
        nodes_created=nodes_created,
        nodes_processed=nodes_processed,
        nodes_budget_pruned=nodes_budget_pruned,
        nodes_capability_pruned=nodes_capability_pruned,
        nodes_bound_pruned=nodes_bound_pruned,
        nodes_epsilon_pruned=nodes_epsilon_pruned,
        nodes_closed_feasible_upper=nodes_closed_feasible_upper,
        nodes_active_at_end=len(active),
        maximum_depth=maximum_depth,
        unique_oracle_calls=oracle_calls,
        oracle_cache_hits=cache_hits,
        initial_corner_evaluations=initial_corner_evaluations,
        total_fixed_y_iterations=total_iterations,
        maximum_oracle_gap=maximum_oracle_gap,
        convergence_events=tuple(events),
        incumbent_history=tuple(incumbent_history),
    )


def optimizer_cover(result: MonotoneBBResult, incumbent_objective: float, target: float) -> tuple[MonotoneBox, ...]:
    threshold = float(incumbent_objective) + float(target)
    return tuple(box for box in result.leaves if box.lower_bound <= threshold + 1e-12)


def classify_cover(boxes: Sequence[MonotoneBox], class_tolerance: float = 1e-6) -> dict[str, Any]:
    if not boxes:
        return {
            "potential_box_count": 0,
            "A_like_box_count": 0,
            "B_like_box_count": 0,
            "mixed_box_count": 0,
            "policy_class_status": "unresolved",
            "y_min": [],
            "y_max": [],
            "corridor_2_full_certified": False,
            "corridor_3_full_certified": False,
        }
    lower = np.vstack([box.lower for box in boxes])
    upper = np.vstack([box.upper for box in boxes])
    if lower.shape[1] >= 5:
        a_like = sum(float(box.lower[0]) > float(box.upper[4]) + class_tolerance for box in boxes)
        b_like = sum(float(box.lower[4]) > float(box.upper[0]) + class_tolerance for box in boxes)
        mixed = len(boxes) - a_like - b_like
        status = "resolved_A_like" if mixed == 0 and a_like == len(boxes) else "resolved_B_like" if mixed == 0 and b_like == len(boxes) else "unresolved"
    else:
        a_like = 0
        b_like = 0
        mixed = len(boxes)
        status = "unresolved"
    return {
        "potential_box_count": len(boxes),
        "A_like_box_count": a_like,
        "B_like_box_count": b_like,
        "mixed_box_count": mixed,
        "policy_class_status": status,
        "y_min": lower.min(axis=0).tolist(),
        "y_max": upper.max(axis=0).tolist(),
        "corridor_2_full_certified": bool(lower.shape[1] >= 3 and np.all(lower[:, 1] >= 1.0 - class_tolerance)),
        "corridor_3_full_certified": bool(lower.shape[1] >= 3 and np.all(lower[:, 2] >= 1.0 - class_tolerance)),
    }






