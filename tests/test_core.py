from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import linprog

from ejor_dad.certification import (
    budget_intersecting_grid_cells,
    continuous_grid_certificate,
    validate_upper_corner_certificate_instance,
)
from ejor_dad.channels import decompose_road_retrofit_channels
from ejor_dad.fixed_y import evaluate_fixed_plan, evaluate_fixed_y
from ejor_dad.model import (
    AidCenter,
    DADInstance,
    FailureMomentEnvelope,
    Link,
    PiecewiseLinearResponseParams,
    SurvivalParams,
    State,
    ThresholdResponseParams,
    Zone,
)
from ejor_dad.moments import build_failure_moment_system, moment_bound_diagnostics
from ejor_dad.recourse import solve_capability, solve_recourse
from ejor_dad.sbb import (
    initialize_recourse_cuts,
    root_node,
    solve_global_sbb,
    solve_node_with_cut_separation,
    tighten_probability_bounds,
    valid_objective_lower_bound,
)
from ejor_dad.states import generate_failure_states, nominal_probabilities
from ejor_dad.tv import capped_tv_profile, worst_case_tv_distribution


def small_instance() -> DADInstance:
    links = [
        Link("l1", "c", "z", 0.2, 10, travel_time=10),
    ]
    states = generate_failure_states(links, max_failures=None, include_tail=False)
    return DADInstance(
        zones=[Zone("zone", 100, 0.5, 20, node="z")],
        links=links,
        centers=[AidCenter("center", "c", existing_capacity=10, capacity_unit_cost=1)],
        budget_renovation=10,
        budget_retrofit=10,
        budget_capacity=5,
        ambiguity_radius=0.1,
        states=states,
        survival=SurvivalParams(a=1.0, b=0.0, c=1.0, d=0.0),
    )



def performance_adjusted_instance() -> DADInstance:
    links = [
        Link(
            "l1",
            "c",
            "z",
            baseline_failure_probability=0.20,
            retrofit_cost=10.0,
            travel_time=10.0,
            residual_failure_probability=0.05,
            failure_delay_reduction=0.50,
        )
    ]
    return DADInstance(
        zones=[Zone("zone", 100.0, 0.5, 20.0, node="z")],
        links=links,
        centers=[AidCenter("center", "c", existing_capacity=50.0, capacity_unit_cost=1.0)],
        budget_renovation=0.0,
        budget_retrofit=10.0,
        budget_capacity=0.0,
        ambiguity_radius=0.0,
        states=generate_failure_states(links, max_failures=None, include_tail=False),
        survival=SurvivalParams(a=1.0, b=-0.02, c=1.0, d=0.0),
        intact_travel_times=[[10.0]],
        failure_penalty_matrices={"l1": [[20.0]]},
    )


def test_residual_failure_floor_remains_after_full_retrofit() -> None:
    instance = performance_adjusted_instance()
    probabilities = nominal_probabilities(instance.links, instance.states, [1.0])
    failed_state_index = next(index for index, state in enumerate(instance.states) if state.failed_links)

    assert np.isclose(probabilities[failed_state_index], 0.05)
    assert np.allclose(instance.effective_failure_probabilities([1.0]), [0.05])


def test_retrofit_reduces_conditional_delay_only_in_failed_state() -> None:
    instance = performance_adjusted_instance()
    intact = next(state for state in instance.states if not state.failed_links)
    failed = next(state for state in instance.states if state.failed_links)

    assert np.allclose(instance.travel_times(intact, y=[0.0]), [[10.0]])
    assert np.allclose(instance.travel_times(intact, y=[1.0]), [[10.0]])
    assert np.allclose(instance.travel_times(failed, y=[0.0]), [[30.0]])
    assert np.allclose(instance.travel_times(failed, y=[1.0]), [[20.0]])


def test_fixed_y_uses_performance_adjusted_recourse() -> None:
    instance = performance_adjusted_instance()
    unprotected = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)
    hardened = evaluate_fixed_y(instance, [1.0], epsilon=1e-8)

    assert hardened.objective < unprotected.objective



def test_fixed_complete_plan_matches_fixed_y_solution() -> None:
    instance = performance_adjusted_instance()
    optimized = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)
    complete = evaluate_fixed_plan(instance, optimized.z, optimized.w, optimized.y)

    assert np.isclose(complete.objective, optimized.objective, atol=1e-8)
    assert np.isclose(complete.nominal_objective, optimized.nominal_distribution @ optimized.state_losses)
    assert np.allclose(complete.state_losses, optimized.state_losses)
    assert np.allclose(complete.worst_case_distribution, optimized.worst_case_distribution)

def test_sbb_rejects_conditional_delay_extension() -> None:
    with np.testing.assert_raises(NotImplementedError):
        solve_global_sbb(performance_adjusted_instance(), epsilon=1.0, max_nodes=5)
def solve_tv_direct_lp(
    nominal: np.ndarray,
    losses: np.ndarray,
    rho: float,
    density_cap: float,
    maximize: bool = True,
) -> tuple[float, np.ndarray]:
    num_states = len(nominal)
    objective = np.concatenate([(-losses if maximize else losses), np.zeros(num_states)])
    rows = []
    rhs = []
    for state_index in range(num_states):
        row = np.zeros(2 * num_states)
        row[state_index] = 1.0
        row[num_states + state_index] = -1.0
        rows.append(row)
        rhs.append(nominal[state_index])
        row = np.zeros(2 * num_states)
        row[state_index] = -1.0
        row[num_states + state_index] = -1.0
        rows.append(row)
        rhs.append(-nominal[state_index])
        row = np.zeros(2 * num_states)
        row[state_index] = 1.0
        rows.append(row)
        rhs.append(density_cap * nominal[state_index])
    row = np.zeros(2 * num_states)
    row[num_states:] = 1.0
    rows.append(row)
    rhs.append(2.0 * rho)
    equality = np.zeros((1, 2 * num_states))
    equality[0, :num_states] = 1.0
    direct = linprog(
        c=objective,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        A_eq=equality,
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, None)] * (2 * num_states),
        method="highs",
    )
    assert direct.success
    value = -direct.fun if maximize else direct.fun
    return float(value), direct.x[:num_states]



def test_time_sensitive_outcome_separates_access_and_immediate_loss() -> None:
    links = [Link("l1", "center", "zone", 0.2, 1.0, travel_time=10.0)]
    instance = DADInstance(
        zones=[
            Zone(
                "zone",
                population=100.0,
                collapse_fraction=0.5,
                renovation_cost=1.0,
                node="zone",
                time_sensitive_fraction=0.4,
                immediate_loss_fraction=0.2,
            )
        ],
        links=links,
        centers=[AidCenter("center", "center", existing_capacity=10.0, capacity_unit_cost=1.0)],
        budget_renovation=0.0,
        budget_retrofit=1.0,
        budget_capacity=0.0,
        ambiguity_radius=0.0,
        states=generate_failure_states(links, max_failures=None, include_tail=False),
        survival=SurvivalParams(a=1.0, b=0.0, c=1.0, d=0.0),
    )

    result = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)

    assert np.allclose(instance.base_demands, [20.0])
    assert np.allclose(instance.base_immediate_losses, [10.0])
    assert np.isclose(instance.at_risk_population([0.0]), 50.0)
    assert np.isclose(instance.modelled_loss_exposure_after_renovation([0.0]), 30.0)
    assert np.allclose(result.state_losses, [20.0, 30.0])


def test_threshold_response_is_a_timely_access_indicator() -> None:
    response = ThresholdResponseParams(threshold_minutes=60.0)

    assert response.fraction(60.0) == 1.0
    assert response.fraction(60.1) == 0.0
    assert response.fraction(float("inf")) == 0.0


def test_sbb_matches_fixed_y_with_immediate_losses() -> None:
    instance = small_instance()
    instance.zones[0] = Zone(
        "zone",
        population=100.0,
        collapse_fraction=0.5,
        renovation_cost=20.0,
        node="z",
        time_sensitive_fraction=0.4,
        immediate_loss_fraction=0.2,
    )
    instance.ambiguity_density_cap = 2.0
    fixed_y = 0.3
    fixed = evaluate_fixed_y(instance, [fixed_y], epsilon=1e-8)
    node = root_node(instance)
    node.y_bounds[0] = [fixed_y, fixed_y]
    cut_sets = {state_index: [] for state_index in range(len(instance.states))}
    initialize_recourse_cuts(instance, cut_sets, fixed)
    relaxation = solve_node_with_cut_separation(instance, node, cut_sets, epsilon_cut=1e-9)

    assert relaxation.success
    assert np.isclose(relaxation.lower_bound, fixed.objective, atol=1e-6)

def test_nominal_probabilities_sum_to_one() -> None:
    instance = small_instance()
    probabilities = nominal_probabilities(instance.links, instance.states, [0.5])
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.0)


def test_tv_worst_case_moves_to_high_loss() -> None:
    result = worst_case_tv_distribution([0.5, 0.5], [0.0, 10.0], rho=0.2)
    assert np.allclose(result.distribution, [0.3, 0.7])
    assert np.isclose(result.value, 7.0)


def test_support_preserving_tv_blocks_zero_nominal_states() -> None:
    nominal = [0.5, 0.5, 0.0]
    losses = [0.0, 1.0, 100.0]
    unrestricted = worst_case_tv_distribution(nominal, losses, rho=0.25)
    capped = worst_case_tv_distribution(nominal, losses, rho=0.25, density_cap=2.0)

    assert np.isclose(unrestricted.distribution[2], 0.25)
    assert np.isclose(capped.distribution[2], 0.0)
    assert np.all(capped.distribution <= 2.0 * np.asarray(nominal) + 1e-12)
    assert np.allclose(capped.distribution, [0.25, 0.75, 0.0])


def test_support_preserving_tv_matches_direct_lp() -> None:
    nominal = np.asarray([0.55, 0.30, 0.15, 0.0])
    losses = np.asarray([2.0, 8.0, 5.0, 100.0])
    rho = 0.20
    density_cap = 1.7
    result = worst_case_tv_distribution(nominal, losses, rho=rho, density_cap=density_cap)
    direct_value, direct_distribution = solve_tv_direct_lp(nominal, losses, rho, density_cap)

    assert np.isclose(result.value, direct_value)
    assert np.allclose(result.distribution, direct_distribution)


def test_support_preserving_tv_matches_random_direct_lps() -> None:
    generator = np.random.default_rng(20260712)
    for _ in range(20):
        nominal = generator.dirichlet(np.ones(6))
        nominal[generator.integers(0, len(nominal))] = 0.0
        nominal /= nominal.sum()
        losses = generator.uniform(0.0, 100.0, len(nominal))
        rho = float(generator.uniform(0.0, 0.5))
        density_cap = float(generator.uniform(1.0, 4.0))
        for maximize in [True, False]:
            result = worst_case_tv_distribution(
                nominal,
                losses,
                rho=rho,
                maximize=maximize,
                density_cap=density_cap,
            )
            direct_value, _ = solve_tv_direct_lp(
                nominal,
                losses,
                rho,
                density_cap,
                maximize=maximize,
            )

            assert np.isclose(result.value, direct_value, atol=1e-8)
            assert np.all(result.distribution <= density_cap * nominal + 1e-10)


def test_density_cap_one_returns_nominal_distribution() -> None:
    nominal = np.asarray([0.2, 0.3, 0.5, 0.0])
    result = worst_case_tv_distribution(
        nominal,
        [1.0, 50.0, 10.0, 1000.0],
        rho=0.75,
        density_cap=1.0,
    )

    assert np.allclose(result.distribution, nominal)
    assert np.isclose(result.moved_mass, 0.0)


def test_tv_respects_additional_linear_moment_bound() -> None:
    result = worst_case_tv_distribution(
        [0.5, 0.5],
        [0.0, 10.0],
        rho=0.4,
        density_cap=2.0,
        inequality_matrix=[[0.0, 1.0]],
        inequality_rhs=[0.6],
    )

    assert np.allclose(result.distribution, [0.4, 0.6])
    assert np.isclose(result.value, 6.0)


def test_fixed_y_respects_failure_marginal_envelope() -> None:
    instance = small_instance()
    instance.ambiguity_density_cap = 2.0
    instance.failure_moment_envelope = FailureMomentEnvelope(
        marginal_relative_tolerance=0.0,
        marginal_absolute_tolerance=0.02,
    )
    result = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)
    failed_state_index = next(index for index, state in enumerate(instance.states) if state.failed_links)

    assert result.worst_case_distribution[failed_state_index] <= 0.22 + 1e-9
    system = build_failure_moment_system(instance, result.nominal_distribution)
    diagnostics = moment_bound_diagnostics(system, result.worst_case_distribution)
    assert all(float(row["lower_slack"]) >= -1e-8 for row in diagnostics)
    assert all(float(row["upper_slack"]) >= -1e-8 for row in diagnostics)


def test_failure_moment_system_bounds_count_mean_and_second_moment() -> None:
    links = [
        Link("l1", "c", "a", 0.25, 1.0),
        Link("l2", "a", "z", 0.40, 1.0),
    ]
    instance = DADInstance(
        zones=[Zone("zone", 100, 0.5, 1.0, node="z")],
        links=links,
        centers=[AidCenter("center", "c", 10.0, 1.0)],
        budget_renovation=0.0,
        budget_retrofit=2.0,
        budget_capacity=0.0,
        ambiguity_radius=0.2,
        ambiguity_density_cap=2.0,
        failure_moment_envelope=FailureMomentEnvelope(
            count_mean_absolute_tolerance=0.1,
            count_second_moment_relative_tolerance=0.1,
        ),
        states=generate_failure_states(links, max_failures=None, include_tail=False),
    )
    nominal = nominal_probabilities(instance.links, instance.states, [0.0, 0.0])
    system = build_failure_moment_system(instance, nominal)
    diagnostics = moment_bound_diagnostics(system, nominal)

    assert {row["name"] for row in diagnostics} == {
        "failure_count_mean",
        "failure_count_fixed_center_second_moment",
    }
    assert all(not bool(row["active"]) for row in diagnostics)


def test_sbb_rejects_unimplemented_failure_moment_relaxation() -> None:
    instance = small_instance()
    instance.failure_moment_envelope = FailureMomentEnvelope(
        count_mean_absolute_tolerance=0.1,
    )

    with np.testing.assert_raises(NotImplementedError):
        solve_global_sbb(instance, epsilon=1.0, max_nodes=5)


def test_recourse_primal_dual_match() -> None:
    instance = small_instance()
    state = next(state for state in instance.states if not state.failed_links)
    recourse = solve_recourse(instance, state, z=[0.0], w=[0.0])
    assert np.isclose(recourse.survivors, 10.0)
    dual_value = (instance.existing_capacities @ recourse.alpha) + (instance.base_demands @ recourse.beta)
    assert np.isclose(dual_value, recourse.survivors)


def test_fixed_y_runs_and_respects_budgets() -> None:
    instance = small_instance()
    result = evaluate_fixed_y(instance, [0.0], epsilon=1e-6)
    assert result.objective >= 0.0
    assert instance.renovation_costs @ result.z <= instance.budget_renovation + 1e-7
    assert instance.capacity_costs @ result.w <= instance.budget_capacity + 1e-7


def test_fixed_y_accepts_ascending_radius_warm_start_cuts() -> None:
    instance = small_instance()
    instance.ambiguity_density_cap = 2.0
    instance.ambiguity_radius = 0.0
    nominal = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)
    instance.ambiguity_radius = 0.2
    cold = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)
    warm = evaluate_fixed_y(
        instance,
        [0.0],
        epsilon=1e-8,
        initial_cuts=nominal.cuts,
    )

    assert warm.objective == pytest.approx(cold.objective, abs=1e-8)
    assert np.allclose(warm.z, cold.z, atol=1e-8)
    assert np.allclose(warm.w, cold.w, atol=1e-8)


def test_fixed_y_raises_when_cut_generation_does_not_close() -> None:
    with np.testing.assert_raises(RuntimeError):
        evaluate_fixed_y(small_instance(), [0.0], epsilon=0.0, max_iterations=1)


def test_aid_center_and_budget_inputs_are_validated() -> None:
    with np.testing.assert_raises(ValueError):
        AidCenter("center", "node", existing_capacity=-1.0, capacity_unit_cost=1.0)
    with np.testing.assert_raises(ValueError):
        AidCenter("center", "node", existing_capacity=1.0, capacity_unit_cost=0.0)
    with np.testing.assert_raises(ValueError):
        DADInstance(
            zones=[Zone("zone", 100.0, 0.5, 1.0, node="z")],
            links=[],
            centers=[AidCenter("center", "c", 1.0, 1.0)],
            budget_renovation=-1.0,
            budget_retrofit=0.0,
            budget_capacity=0.0,
            ambiguity_radius=0.0,
            states=[State(id="intact")],
        )


def test_sbb_runs_on_tiny_instance() -> None:
    instance = small_instance()
    result = solve_global_sbb(instance, epsilon=1.0, max_nodes=5)
    assert result.objective >= 0.0
    assert instance.retrofit_costs @ result.y <= instance.budget_retrofit + 1e-7


def test_sbb_probability_bounds_tighten_from_y_intervals() -> None:
    instance = small_instance()
    node = root_node(instance)
    node.y_bounds[0] = [0.25, 0.75]

    tightened = tighten_probability_bounds(instance, node)
    failed_state_index = next(index for index, state in enumerate(instance.states) if state.failed_links)
    intact_state_index = next(index for index, state in enumerate(instance.states) if not state.failed_links)

    assert np.allclose(tightened.pi_bounds[failed_state_index], [0.05, 0.15])
    assert np.allclose(tightened.pi_bounds[intact_state_index], [0.85, 0.95])



def test_sbb_probability_bounds_respect_residual_failure_floor() -> None:
    instance = small_instance()
    instance.links[0] = Link(
        "l1",
        "c",
        "z",
        baseline_failure_probability=0.20,
        retrofit_cost=10.0,
        travel_time=10.0,
        residual_failure_probability=0.04,
    )
    node = root_node(instance)
    node.y_bounds[0] = [0.25, 0.75]

    tightened = tighten_probability_bounds(instance, node)
    failed_state_index = next(index for index, state in enumerate(instance.states) if state.failed_links)
    intact_state_index = next(index for index, state in enumerate(instance.states) if not state.failed_links)

    assert np.allclose(tightened.pi_bounds[failed_state_index], [0.08, 0.16])
    assert np.allclose(tightened.pi_bounds[intact_state_index], [0.84, 0.92])
def test_sbb_reported_lower_bound_uses_nonnegative_objective_floor() -> None:
    assert valid_objective_lower_bound(-123.0) == 0.0
    assert valid_objective_lower_bound(12.5) == 12.5
    assert valid_objective_lower_bound(np.inf) == np.inf


def test_capped_tv_sbb_dual_matches_fixed_y_oracle() -> None:
    instance = small_instance()
    instance.ambiguity_density_cap = 2.0
    instance.survival = SurvivalParams(a=1.0, b=-0.1, c=1.0, d=0.0)
    for fixed_y in [0.3, 1.0]:
        fixed = evaluate_fixed_y(instance, [fixed_y], epsilon=1e-8)
        node = root_node(instance)
        node.y_bounds[0] = [fixed_y, fixed_y]
        cut_sets = {state_index: [] for state_index in range(len(instance.states))}
        initialize_recourse_cuts(instance, cut_sets, fixed)
        relaxation = solve_node_with_cut_separation(instance, node, cut_sets, epsilon_cut=1e-9)

        assert relaxation.success
        assert np.isclose(relaxation.lower_bound, fixed.objective, atol=1e-6)


def test_capped_tv_sbb_dual_matches_four_state_oracle() -> None:
    links = [
        Link("l1", "c", "a", 0.35, 1.0, travel_time=5.0),
        Link("l2", "a", "z", 0.45, 1.0, travel_time=5.0),
    ]
    instance = DADInstance(
        zones=[Zone("zone", 120, 0.5, 20, node="z")],
        links=links,
        centers=[AidCenter("center", "c", existing_capacity=30, capacity_unit_cost=1)],
        budget_renovation=5,
        budget_retrofit=2,
        budget_capacity=5,
        ambiguity_radius=0.15,
        ambiguity_density_cap=1.8,
        states=generate_failure_states(links, max_failures=None, include_tail=False),
        survival=SurvivalParams(a=1.0, b=-0.08, c=1.0, d=0.0),
    )
    fixed_y = np.asarray([0.2, 0.8])
    fixed = evaluate_fixed_y(instance, fixed_y, epsilon=1e-8)
    for probability_relaxation in ("product_tree", "corner_boxes", "corner_link_cuts"):
        node = root_node(instance, probability_relaxation=probability_relaxation)
        node.y_bounds[:, 0] = fixed_y
        node.y_bounds[:, 1] = fixed_y
        cut_sets = {state_index: [] for state_index in range(len(instance.states))}
        initialize_recourse_cuts(instance, cut_sets, fixed)
        relaxation = solve_node_with_cut_separation(
            instance,
            node,
            cut_sets,
            epsilon_cut=1e-9,
            probability_relaxation=probability_relaxation,
        )

        assert relaxation.success
        assert np.isclose(relaxation.lower_bound, fixed.objective, atol=1e-6)


def test_corner_link_cuts_strengthen_plain_corner_boxes() -> None:
    links = [
        Link("l1", "c", "a", 0.35, 1.0, travel_time=5.0),
        Link("l2", "a", "z", 0.45, 1.0, travel_time=5.0),
    ]
    instance = DADInstance(
        zones=[Zone("zone", 120, 0.5, 20, node="z")],
        links=links,
        centers=[AidCenter("center", "c", existing_capacity=30, capacity_unit_cost=1)],
        budget_renovation=5,
        budget_retrofit=2,
        budget_capacity=5,
        ambiguity_radius=0.15,
        ambiguity_density_cap=1.8,
        states=generate_failure_states(links, max_failures=None, include_tail=False),
        survival=SurvivalParams(a=1.0, b=-0.08, c=1.0, d=0.0),
    )
    box_cuts = {state_index: [] for state_index in range(len(instance.states))}
    box_relaxation = solve_node_with_cut_separation(
        instance,
        root_node(instance, probability_relaxation="corner_boxes"),
        box_cuts,
        epsilon_cut=1e-9,
        probability_relaxation="corner_boxes",
    )
    link_cuts = {state_index: list(cuts) for state_index, cuts in box_cuts.items()}
    link_relaxation = solve_node_with_cut_separation(
        instance,
        root_node(instance, probability_relaxation="corner_link_cuts"),
        link_cuts,
        epsilon_cut=1e-9,
        probability_relaxation="corner_link_cuts",
    )

    assert box_relaxation.success
    assert link_relaxation.success
    assert link_relaxation.lower_bound >= box_relaxation.lower_bound - 1e-8


def threshold_certificate_instance(retrofit_budget: float = 0.5) -> DADInstance:
    links = [
        Link(
            "l1",
            "center",
            "zone",
            baseline_failure_probability=0.20,
            retrofit_cost=1.0,
            travel_time=10.0,
            residual_failure_probability=0.05,
            failure_delay_reduction=0.50,
        )
    ]
    return DADInstance(
        zones=[
            Zone(
                "zone",
                population=100.0,
                collapse_fraction=0.5,
                renovation_cost=1.0,
                node="zone",
                time_sensitive_fraction=0.25,
            )
        ],
        links=links,
        centers=[AidCenter("center", "center", existing_capacity=50.0, capacity_unit_cost=1.0)],
        budget_renovation=0.0,
        budget_retrofit=retrofit_budget,
        budget_capacity=0.0,
        ambiguity_radius=0.20,
        ambiguity_density_cap=2.0,
        states=generate_failure_states(links, max_failures=None, include_tail=False),
        survival=ThresholdResponseParams(threshold_minutes=20.0),
        intact_travel_times=[[10.0]],
        failure_penalty_matrices={"l1": [[20.0]]},
    )


def test_capped_tv_profile_matches_direct_lp_across_radii() -> None:
    generator = np.random.default_rng(482)
    for _ in range(12):
        nominal = generator.dirichlet(np.ones(5))
        losses = generator.uniform(0.0, 30.0, size=5)
        profile = capped_tv_profile(nominal, losses, density_cap=1.8)
        for rho in [0.0, 0.03, 0.11, 0.25, 0.60]:
            direct_value, direct_distribution = solve_tv_direct_lp(
                nominal,
                losses,
                rho,
                density_cap=1.8,
            )
            profiled = profile.evaluate(rho)
            assert np.isclose(profiled.value, direct_value, atol=1e-9)
            assert np.allclose(profiled.distribution, direct_distribution, atol=1e-9)


def test_certificate_only_evaluation_allows_over_budget_upper_corner() -> None:
    instance = performance_adjusted_instance()
    instance.budget_retrofit = 5.0

    with np.testing.assert_raises(ValueError):
        evaluate_fixed_y(instance, [1.0], epsilon=1e-8)
    result = evaluate_fixed_y(
        instance,
        [1.0],
        epsilon=1e-8,
        enforce_retrofit_budget=False,
    )

    assert np.isfinite(result.objective)
    assert instance.renovation_costs @ result.z <= instance.budget_renovation + 1e-8
    assert instance.capacity_costs @ result.w <= instance.budget_capacity + 1e-8


def test_upper_corner_certificate_bounds_continuous_feasible_policies() -> None:
    instance = threshold_certificate_instance(retrofit_budget=0.5)
    validate_upper_corner_certificate_instance(instance)
    cells = budget_intersecting_grid_cells(instance.retrofit_costs, instance.budget_retrofit, [0.0, 0.5, 1.0])
    feasible_grid_values = [
        evaluate_fixed_y(instance, [0.0], epsilon=1e-8).objective,
        evaluate_fixed_y(instance, [0.5], epsilon=1e-8).objective,
    ]
    upper_corner_values = [
        evaluate_fixed_y(instance, cell.upper, epsilon=1e-8, enforce_retrofit_budget=False).objective
        for cell in cells
    ]
    certificate = continuous_grid_certificate(cells, upper_corner_values, min(feasible_grid_values))

    assert len(cells) == 2
    assert certificate.continuous_lower_bound <= min(feasible_grid_values) + 1e-8
    assert certificate.grid_upper_bound == min(feasible_grid_values)
    assert certificate.lower_bound_cell.upper_budget_used > instance.budget_retrofit


def test_road_channel_decomposition_sums_to_total_effect() -> None:
    instance = threshold_certificate_instance(retrofit_budget=1.0)
    result = evaluate_fixed_y(instance, [1.0], epsilon=1e-8)
    channels = decompose_road_retrofit_channels(instance, result.z, result.w, result.y)

    assert np.isclose(
        channels.total_road_improvement,
        channels.conditional_consequence_improvement + channels.probability_channel_improvement,
        atol=1e-8,
    )
    assert channels.total_road_improvement >= -1e-8


def test_certificate_validator_accepts_sorted_multi_link_states() -> None:
    links = [
        Link("z_link", "center", "middle", 0.2, 1.0, travel_time=5.0),
        Link("a_link", "middle", "zone", 0.3, 1.0, travel_time=5.0),
    ]
    instance = DADInstance(
        zones=[Zone("zone", 100.0, 0.5, 1.0, node="zone", time_sensitive_fraction=0.25)],
        links=links,
        centers=[AidCenter("center", "center", existing_capacity=10.0, capacity_unit_cost=1.0)],
        budget_renovation=0.0,
        budget_retrofit=1.0,
        budget_capacity=0.0,
        ambiguity_radius=0.1,
        ambiguity_density_cap=2.0,
        states=generate_failure_states(links, max_failures=None, include_tail=False),
        survival=ThresholdResponseParams(threshold_minutes=30.0),
    )

    validate_upper_corner_certificate_instance(instance)


def test_piecewise_linear_response_interpolates_monotonically() -> None:
    response = PiecewiseLinearResponseParams(
        knots=((0.0, 1.0), (30.0, 1.0), (60.0, 0.75), (120.0, 0.25), (180.0, 0.0))
    )

    assert np.isclose(response.fraction(30.0), 1.0)
    assert np.isclose(response.fraction(45.0), 0.875)
    assert np.isclose(response.fraction(90.0), 0.5)
    assert np.isclose(response.fraction(240.0), 0.0)
    with np.testing.assert_raises(ValueError):
        PiecewiseLinearResponseParams(knots=((0.0, 0.5), (60.0, 0.75)))


def test_graded_response_activates_conditional_delay_channel() -> None:
    instance = threshold_certificate_instance(retrofit_budget=1.0)
    instance.survival = PiecewiseLinearResponseParams(
        knots=((0.0, 1.0), (10.0, 1.0), (30.0, 0.0), (60.0, 0.0))
    )
    validate_upper_corner_certificate_instance(instance)
    result = evaluate_fixed_y(instance, [1.0], epsilon=1e-8)
    channels = decompose_road_retrofit_channels(instance, result.z, result.w, result.y)

    assert channels.conditional_consequence_improvement > 0.0
    assert channels.total_road_improvement >= channels.conditional_consequence_improvement - 1e-8


def test_optimized_road_channel_decomposition_sums_to_total_effect() -> None:
    from ejor_dad.channels import decompose_optimized_road_channels

    instance = performance_adjusted_instance()
    result = evaluate_fixed_y(instance, [0.5])
    channels = decompose_optimized_road_channels(instance, result.y)
    assert channels.total_road_improvement == pytest.approx(
        channels.conditional_consequence_improvement + channels.probability_channel_improvement,
        abs=1e-7,
    )


def test_minimum_saturation_radius_attains_cap_only_value() -> None:
    from ejor_dad.tv import minimum_saturation_radius

    nominal = np.array([0.7, 0.2, 0.1])
    losses = np.array([1.0, 4.0, 10.0])
    saturation = minimum_saturation_radius(nominal, losses, density_cap=2.0)
    attained = worst_case_tv_distribution(
        nominal, losses, saturation.minimum_radius + 1e-8, density_cap=2.0
    )
    assert attained.value == pytest.approx(saturation.cap_only_value, abs=2e-7)


def test_lexicographic_selection_resolves_numerical_ties() -> None:
    from types import SimpleNamespace
    from ejor_dad.selection import select_lexicographic_best

    results = [
        SimpleNamespace(objective=10.0, y=np.array([1.0, 0.0])),
        SimpleNamespace(objective=10.0 + 1e-10, y=np.array([0.0, 1.0])),
    ]
    selection = select_lexicographic_best(results, objective_tolerance=1e-8)
    assert selection.tied_policy_count == 2
    assert np.array_equal(selection.selected.y, np.array([0.0, 1.0]))




def test_hard_protected_population_requirement_enters_fixed_y_master() -> None:
    instance = small_instance()
    instance.minimum_protected_population = 25.0
    result = evaluate_fixed_y(instance, [0.0])
    protected = instance.protected_population_coefficients @ result.z
    assert protected >= 25.0 - 1e-7


def test_loss_recourse_is_floor_free_and_capability_is_separate() -> None:
    instance = small_instance()
    instance.minimum_zone_service_fraction = np.array([0.10])
    state = instance.states[0]
    result = solve_recourse(instance, state, [0.0], [0.0], y=[0.0])
    demand = float(instance.demand_after_renovation([0.0])[0])
    dual_value = (
        result.alpha @ instance.capacity_after_investment([0.0])
        + result.beta @ instance.demand_after_renovation([0.0])
    )
    assert result.survivors == pytest.approx(dual_value, abs=1e-7)
    capability = solve_capability(instance, state, [0.0], [0.0], y=[0.0])
    assert capability.feasible
    timely_service = float((capability.survival * capability.dispatch).sum(axis=0)[0])
    assert timely_service >= 0.10 * demand - 1e-7


def test_capability_master_uses_post_renovation_demand() -> None:
    instance = small_instance()
    instance.states = [instance.states[0]]
    instance.critical_service_state_ids = {instance.states[0].id}
    instance.minimum_zone_service_fraction = np.array([0.50])
    instance.minimum_protected_population = 25.0
    instance.budget_renovation = instance.renovation_costs[0]

    result = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)

    assert result.z[0] >= 0.50 - 1e-7
    capability = solve_capability(
        instance,
        instance.states[0],
        result.z,
        result.w,
        y=result.y,
    )
    assert capability.feasible


def test_initial_cuts_validate_radius_road_and_state_support() -> None:
    instance = small_instance()
    baseline = evaluate_fixed_y(instance, [0.0], epsilon=1e-8)
    larger_radius = replace(instance, ambiguity_radius=0.2)
    reused = evaluate_fixed_y(
        larger_radius,
        [0.0],
        epsilon=1e-8,
        initial_cuts=baseline.cuts,
    )
    assert reused.objective >= 0.0

    with pytest.raises(ValueError, match="different road vector"):
        evaluate_fixed_y(
            instance,
            [0.5],
            epsilon=1e-8,
            initial_cuts=baseline.cuts,
        )

    reordered = replace(instance, states=list(reversed(instance.states)))
    with pytest.raises(ValueError, match="different state support"):
        evaluate_fixed_y(
            reordered,
            [0.0],
            epsilon=1e-8,
            initial_cuts=baseline.cuts,
        )


def test_hazard_regime_mixture_creates_positive_road_correlation() -> None:
    from ejor_dad.model import HazardRegime
    from ejor_dad.states import generate_regime_failure_states

    links = [
        Link("a", "u", "v", 0.2, 1.0, residual_failure_probability=0.02),
        Link("b", "v", "w", 0.2, 1.0, residual_failure_probability=0.02),
    ]
    regimes = [
        HazardRegime("low", 0.5, link_failure_multipliers={"a": 0.25, "b": 0.25}),
        HazardRegime("high", 0.5, link_failure_multipliers={"a": 2.0, "b": 2.0}),
    ]
    states = generate_regime_failure_states(links, regimes)
    probabilities = nominal_probabilities(links, states, [0.0, 0.0], regimes)
    failure_a = np.array(["a" in state.failed_links for state in states], dtype=float)
    failure_b = np.array(["b" in state.failed_links for state in states], dtype=float)
    covariance = probabilities @ (failure_a * failure_b) - (probabilities @ failure_a) * (probabilities @ failure_b)
    assert covariance > 0.0
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities @ failure_a == pytest.approx(links[0].failure_probability(0.0), abs=1e-10)
    assert probabilities @ failure_b == pytest.approx(links[1].failure_probability(0.0), abs=1e-10)
    full_retrofit = nominal_probabilities(links, states, [1.0, 1.0], regimes)
    low_mass = full_retrofit[
        np.array([state.hazard_regime_id == "low" for state in states])
    ].sum()
    high_mass = full_retrofit[
        np.array([state.hazard_regime_id == "high" for state in states])
    ].sum()
    low_failure_a = full_retrofit[
        np.array(
            [
                state.hazard_regime_id == "low" and "a" in state.failed_links
                for state in states
            ]
        )
    ].sum() / low_mass
    high_failure_a = full_retrofit[
        np.array(
            [
                state.hazard_regime_id == "high" and "a" in state.failed_links
                for state in states
            ]
        )
    ].sum() / high_mass
    assert low_failure_a != pytest.approx(high_failure_a)
    assert full_retrofit @ failure_a == pytest.approx(
        links[0].failure_probability(1.0),
        abs=1e-10,
    )


def test_failed_facility_removes_existing_and_added_capacity() -> None:
    instance = small_instance()
    failed_state = State("facility_failed", failed_centers=(instance.centers[0].id,))
    instance.states = [failed_state]
    result = solve_recourse(instance, failed_state, [0.0], [100.0], y=[0.0])
    assert result.survivors == pytest.approx(0.0)
    assert result.dispatch.sum() == pytest.approx(0.0)
