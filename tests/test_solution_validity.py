from __future__ import annotations

from src.feasibility import is_feasible, validate_solution
from src.instance_generator import InstanceGenerator
from src.parameters import HeuristicConfig, SearchConfig, GenerationSettings
from src.heuristics import run_nils


def _config() -> SearchConfig:
    return SearchConfig(
        seed=33,
        generation=GenerationSettings(
            num_customers=10,
            num_trucks=2,
            num_drones=2,
            region="dense_urban",
            priority_share=(0.2, 0.3, 0.5),
        ),
        heuristics=HeuristicConfig(
            max_outer_iter=4,
            max_no_improve=2,
            time_limit_seconds=60,
            random_seed=33,
        ),
    )


def test_nils_solution_is_structurally_valid() -> None:
    config = _config()
    instance = InstanceGenerator.from_search_config(config).generate_single(11, "validity_case")
    sol = run_nils(
        instance,
        seed=config.heuristics.random_seed,
        max_iter=config.heuristics.max_outer_iter,
        max_no_improve=config.heuristics.max_no_improve,
        time_limit=60,
    )
    findings = validate_solution(instance, sol)
    assert is_feasible(instance, sol)
    assert all(item.severity != "ERROR" for item in findings)
    assert sol.components["truck_cost"] >= 0.0
    assert sol.components["drone_cost"] >= 0.0
    assert sol.components["tardiness_cost"] >= 0.0

