from __future__ import annotations

from src.feasibility import validate_solution, is_feasible
from src.instance_generator import InstanceGenerator
from src.parameters import SearchConfig, GenerationSettings
from src.heuristics import build_initial_solution
from src.data_models import SolutionData


def _config() -> SearchConfig:
    return SearchConfig(
        seed=7,
        generation=GenerationSettings(
            num_customers=6,
            num_trucks=1,
            num_drones=1,
            region="dense_urban",
            priority_share=(0.3, 0.3, 0.4),
        ),
    )


def _build_base_instance():
    generator = InstanceGenerator.from_search_config(_config())
    return generator.generate_single(99, name="feasibility_case")


def test_feasible_initial_solution_passes_checker() -> None:
    instance = _build_base_instance()
    sol = build_initial_solution(instance, seed=123)
    findings = validate_solution(instance, sol)
    assert is_feasible(instance, sol)
    assert all(item.severity != "ERROR" for item in findings)


def test_unserved_customer_detected() -> None:
    instance = _build_base_instance()
    empty = SolutionData(
        instance_name=instance.name,
        status="empty",
        objective=0.0,
        components={},
    )
    findings = validate_solution(instance, empty)
    assert any(item.code == "UNSERVED" for item in findings)
    assert not is_feasible(instance, empty)

