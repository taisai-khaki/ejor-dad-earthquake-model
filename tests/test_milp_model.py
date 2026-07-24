from __future__ import annotations

import pytest

from src.parameters import MILPConfig
from src.instance_generator import InstanceGenerator
from src.parameters import SearchConfig, GenerationSettings
from src.milp import solve_instance


def _config() -> SearchConfig:
    return SearchConfig(
        seed=101,
        generation=GenerationSettings(
            num_customers=4,
            num_trucks=1,
            num_drones=1,
            region="dense_urban",
            priority_share=(0.33, 0.33, 0.34),
            instance_reps_per_size=1,
        ),
        milp=MILPConfig(enabled=True, time_limit_seconds=20, mip_gap=0.05, threads=2),
    )


def _gurobi_available() -> bool:
    try:
        import gurobipy  # type: ignore
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gurobi_available(), reason="gurobipy is unavailable in this environment.")
def test_milp_solver_returns_solution() -> None:
    config = _config()
    generator = InstanceGenerator.from_search_config(config)
    instance = generator.generate_single(3, "milp_smoke")
    sol = solve_instance(instance, config)
    assert isinstance(sol.status, str)
    assert sol.status != "no_solution"
    assert sol.objective >= 0.0
