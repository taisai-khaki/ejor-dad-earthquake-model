from __future__ import annotations

from src.heuristics import build_initial_solution
from src.heuristics import run_nils
from src.instance_generator import InstanceGenerator
from src.parameters import HeuristicConfig, SearchConfig, GenerationSettings


def _config() -> SearchConfig:
    return SearchConfig(
        seed=11,
        generation=GenerationSettings(
            num_customers=12,
            num_trucks=2,
            num_drones=2,
            region="dense_urban",
            priority_share=(0.3, 0.3, 0.4),
        ),
        heuristics=HeuristicConfig(max_outer_iter=3, max_no_improve=2, time_limit_seconds=30, random_seed=11),
    )


def test_constructive_solution_valid_structure() -> None:
    instance = InstanceGenerator.from_search_config(_config()).generate_single(21, "constr_case")
    sol = build_initial_solution(instance, seed=99)
    assert sol.status == "heuristic_feasible"
    assert len(sol.truck_routes) == instance.num_trucks
    for route in sol.truck_routes.values():
        assert route[0] == 0 and route[-1] == 0
        assert len(route) >= 2


def test_nils_improves_or_matches_constructive() -> None:
    instance = InstanceGenerator.from_search_config(_config()).generate_single(22, "nils_case")
    config = _config()
    initial = build_initial_solution(instance, seed=22)
    from src.heuristics.nils import evaluate_solution

    initial_obj, _ = evaluate_solution(instance, initial)
    solved = run_nils(
        instance,
        seed=config.heuristics.random_seed,
        max_iter=config.heuristics.max_outer_iter,
        max_no_improve=config.heuristics.max_no_improve,
        time_limit=30,
    )
    assert solved.objective <= initial_obj + 1e-9 or solved.status != "nils_complete"

