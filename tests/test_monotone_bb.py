from __future__ import annotations

import numpy as np
import pytest

from ejor_dad.monotone_bb import OracleEvaluation, classify_cover, run_monotone_box_bb


def _linear_oracle(costs, values, budget):
    calls = []

    def oracle(y):
        y = np.asarray(y, dtype=float)
        calls.append(tuple(y))
        objective = float(10.0 - np.dot(values, y))
        return OracleEvaluation("feasible", objective, objective, y=y, z=np.zeros(1), w=np.zeros(1))

    return oracle, calls


def test_fractional_knapsack_certificate_and_determinism():
    costs = np.array([2.0, 3.0])
    values = np.array([4.0, 5.0])
    oracle, calls = _linear_oracle(costs, values, 3.0)
    incumbent = OracleEvaluation("feasible", 10.0, 10.0, y=np.zeros(2), z=np.zeros(1), w=np.zeros(1))
    result = run_monotone_box_bb(
        initial_boxes=[(np.zeros(2), np.ones(2))],
        costs=costs,
        budget=3.0,
        oracle=oracle,
        incumbent=incumbent,
        rel_gap_target=1e-10,
    )
    assert result.global_lower_bound <= result.incumbent.objective + 1e-10
    assert result.incumbent.objective <= 5.0 + 1e-6
    assert result.absolute_gap <= 1e-6
    assert len(calls) == result.unique_oracle_calls
    assert result.initial_corner_evaluations == 1
    oracle2, calls2 = _linear_oracle(costs, values, 3.0)
    result2 = run_monotone_box_bb(
        initial_boxes=[(np.zeros(2), np.ones(2))],
        costs=costs,
        budget=3.0,
        oracle=oracle2,
        incumbent=incumbent,
        rel_gap_target=1e-10,
    )
    assert result2.incumbent.objective == result.incumbent.objective
    assert result2.global_lower_bound == result.global_lower_bound
    assert calls2 == calls


def test_oracle_lower_bound_drives_pruning():
    def oracle(y):
        return OracleEvaluation("feasible", 100.0, 0.0, y=np.asarray(y), z=np.zeros(1), w=np.zeros(1))

    incumbent = OracleEvaluation("feasible", 100.0, 100.0, y=np.zeros(1), z=np.zeros(1), w=np.zeros(1))
    result = run_monotone_box_bb(
        initial_boxes=[(np.zeros(1), np.ones(1))],
        costs=[1.0],
        budget=0.5,
        oracle=oracle,
        incumbent=incumbent,
        rel_gap_target=0.0,
    )
    assert result.global_lower_bound == 0.0
    assert result.incumbent.objective == 100.0


def test_budget_and_capability_pruning_and_cover_classification():
    calls = []

    def oracle(y):
        calls.append(tuple(y))
        if y[0] > 0.75:
            return OracleEvaluation("infeasible", y=np.asarray(y))
        return OracleEvaluation("feasible", 1.0, 0.0, y=np.asarray(y), z=np.zeros(1), w=np.zeros(1))

    incumbent = OracleEvaluation("feasible", 1.0, 1.0, y=np.array([0.0]), z=np.zeros(1), w=np.zeros(1))
    result = run_monotone_box_bb(
        initial_boxes=[(np.array([0.4]), np.array([1.0])), (np.array([0.0]), np.array([0.5]))],
        costs=[1.0],
        budget=0.6,
        oracle=oracle,
        incumbent=incumbent,
        rel_gap_target=0.0,
    )
    assert result.nodes_capability_pruned == 1
    assert result.incumbent.objective == 1.0
    assert classify_cover(result.leaves)["potential_box_count"] >= 1


def test_parent_upper_corner_is_cached_for_children():
    calls = []
    def oracle(y):
        calls.append(tuple(y))
        return OracleEvaluation("feasible", 9.0, 0.0, y=np.asarray(y), z=np.zeros(1), w=np.zeros(1))
    incumbent = OracleEvaluation("feasible", 10.0, 10.0, y=np.zeros(1), z=np.zeros(1), w=np.zeros(1))
    result = run_monotone_box_bb(
        initial_boxes=[(np.zeros(1), np.ones(1))],
        costs=[1.0],
        budget=0.5,
        oracle=oracle,
        incumbent=incumbent,
        rel_gap_target=0.0,
    )
    assert result.oracle_cache_hits >= 1
    assert len(calls) == result.unique_oracle_calls
    assert result.initial_corner_evaluations == 1





def test_upper_corner_feasible_box_closes_exactly():
    def oracle(y):
        y = np.asarray(y, dtype=float)
        value = 3.0 - float(y[0])
        return OracleEvaluation("feasible", value, 0.0, y=y, z=np.zeros(1), w=np.zeros(1))
    incumbent = OracleEvaluation("feasible", 3.0, 3.0, y=np.zeros(1), z=np.zeros(1), w=np.zeros(1))
    result = run_monotone_box_bb(
        initial_boxes=[(np.zeros(1), np.array([0.5]))],
        costs=[1.0], budget=0.75, oracle=oracle, incumbent=incumbent, rel_gap_target=0.0,
    )
    assert result.nodes_closed_feasible_upper == 1
    assert result.global_lower_bound <= result.incumbent.objective + 1e-12


def test_budget_infeasible_box_is_pruned_without_oracle_call():
    calls = []
    def oracle(y):
        calls.append(tuple(y))
        return OracleEvaluation("feasible", 1.0, 1.0, y=np.asarray(y), z=np.zeros(1), w=np.zeros(1))
    incumbent = OracleEvaluation("feasible", 1.0, 1.0, y=np.zeros(1), z=np.zeros(1), w=np.zeros(1))
    result = run_monotone_box_bb(
        initial_boxes=[(np.array([0.8]), np.ones(1))],
        costs=[1.0], budget=0.5, oracle=oracle, incumbent=incumbent, rel_gap_target=0.0,
    )
    assert result.nodes_budget_pruned == 1
    assert calls == []


def test_small_dense_grid_is_contained_by_certificate():
    costs = np.array([1.0, 1.0])
    values = np.array([2.0, 1.0])
    def oracle(y):
        y = np.asarray(y, dtype=float)
        value = 8.0 - float(values @ y)
        return OracleEvaluation("feasible", value, value, y=y, z=np.zeros(1), w=np.zeros(1))
    incumbent = OracleEvaluation("feasible", 8.0, 8.0, y=np.zeros(2), z=np.zeros(1), w=np.zeros(1))
    result = run_monotone_box_bb(
        initial_boxes=[(np.zeros(2), np.ones(2))], costs=costs, budget=1.0,
        oracle=oracle, incumbent=incumbent, rel_gap_target=1e-8,
    )
    dense = min(8.0 - float(values @ np.asarray(y)) for y in (np.array([0.0, 0.0]), np.array([0.0, 0.25]), np.array([0.0, 0.5]), np.array([0.0, 0.75]), np.array([0.0, 1.0]), np.array([0.25, 0.0]), np.array([0.5, 0.0]), np.array([0.75, 0.0]), np.array([1.0, 0.0])))
    assert result.global_lower_bound <= dense + 1e-8
    assert float(costs @ result.incumbent.y) <= 1.0 + 1e-10
    assert result.incumbent.objective >= dense - 1e-8
def test_rejects_over_budget_supplied_incumbent():
    incumbent = OracleEvaluation(
        "feasible",
        1.0,
        1.0,
        y=np.array([1.0, 1.0]),
        z=np.zeros(1),
        w=np.zeros(1),
    )
    with pytest.raises(ValueError, match="incumbent violates"):
        run_monotone_box_bb(
            initial_boxes=[(np.zeros(2), np.ones(2))],
            costs=[1.0, 1.0],
            budget=1.0,
            oracle=lambda y: incumbent,
            incumbent=incumbent,
            rel_gap_target=0.0,
        )


def test_budget_feasible_incumbent_matches_small_enumeration():
    costs = np.array([1.0, 2.0])
    values = np.array([3.0, 1.0])
    budget = 2.0

    def oracle(y):
        y = np.asarray(y, dtype=float)
        objective = 10.0 - float(values @ y)
        return OracleEvaluation("feasible", objective, objective, y=y, z=np.zeros(1), w=np.zeros(1))

    incumbent = OracleEvaluation(
        "feasible",
        10.0,
        10.0,
        y=np.zeros(2),
        z=np.zeros(1),
        w=np.zeros(1),
    )
    result = run_monotone_box_bb(
        initial_boxes=[(np.zeros(2), np.ones(2))],
        costs=costs,
        budget=budget,
        oracle=oracle,
        incumbent=incumbent,
        rel_gap_target=1e-8,
    )
    grid = np.array(
        [
            [first, second]
            for first in (0.0, 0.25, 0.5, 0.75, 1.0)
            for second in (0.0, 0.25, 0.5, 0.75, 1.0)
            if float(costs @ np.array([first, second])) <= budget + 1e-12
        ]
    )
    enumeration_objective = min(10.0 - float(values @ y) for y in grid)
    assert float(costs @ result.incumbent.y) <= budget + 1e-10
    assert result.incumbent.objective <= enumeration_objective + 1e-8
