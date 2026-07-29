from __future__ import annotations

from ejor_dad import (
    AidCenter,
    DADInstance,
    Link,
    SurvivalParams,
    Zone,
    evaluate_fixed_y,
    generate_failure_states,
    solve_global_sbb,
)


def build_demo_instance() -> DADInstance:
    links = [
        Link("l1", "depot", "a", baseline_failure_probability=0.10, retrofit_cost=50, travel_time=10),
        Link("l2", "a", "ward_1", baseline_failure_probability=0.18, retrofit_cost=60, travel_time=15),
        Link("l3", "depot", "ward_2", baseline_failure_probability=0.08, retrofit_cost=40, travel_time=30),
    ]
    states = generate_failure_states(links, max_failures=2, include_tail=True)
    return DADInstance(
        zones=[
            Zone("ward_1", population=1000, collapse_fraction=0.30, renovation_cost=300, node="ward_1"),
            Zone("ward_2", population=700, collapse_fraction=0.20, renovation_cost=180, node="ward_2"),
        ],
        links=links,
        centers=[
            AidCenter("hospital", node="depot", existing_capacity=120, capacity_unit_cost=1.0),
        ],
        budget_renovation=180,
        budget_retrofit=80,
        budget_capacity=80,
        ambiguity_radius=0.10,
        states=states,
        survival=SurvivalParams(a=0.95, b=-0.025, c=1.0, d=0.0),
    )


if __name__ == "__main__":
    instance = build_demo_instance()
    fixed = evaluate_fixed_y(instance, y=[0.0, 0.0, 0.0], epsilon=1e-6)
    print("Fixed y=0")
    print("  objective:", round(fixed.objective, 4))
    print("  z:", fixed.z.round(4).tolist())
    print("  w:", fixed.w.round(4).tolist())
    print("  worst-case p:", fixed.worst_case_distribution.round(4).tolist())
    sbb = solve_global_sbb(instance, epsilon=1e-2, max_nodes=30)
    print("SBB")
    print("  objective:", round(sbb.objective, 4))
    print("  lower bound:", round(sbb.lower_bound, 4))
    print("  gap:", round(sbb.gap, 4))
    print("  y:", sbb.y.round(4).tolist())
    print("  z:", sbb.z.round(4).tolist())
    print("  w:", sbb.w.round(4).tolist())
