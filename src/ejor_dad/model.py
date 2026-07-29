from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, inf, isfinite
from typing import Callable, Mapping, Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Zone:
    id: str
    population: float
    collapse_fraction: float
    renovation_cost: float
    node: str | None = None
    region: str | None = None
    time_sensitive_fraction: float = 1.0
    immediate_loss_fraction: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "population": self.population,
            "collapse_fraction": self.collapse_fraction,
            "renovation_cost": self.renovation_cost,
            "time_sensitive_fraction": self.time_sensitive_fraction,
            "immediate_loss_fraction": self.immediate_loss_fraction,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.population < 0.0:
            raise ValueError("population must be nonnegative.")
        if not 0.0 <= self.collapse_fraction <= 1.0:
            raise ValueError("collapse_fraction must lie in [0, 1].")
        if self.renovation_cost < 0.0:
            raise ValueError("renovation_cost must be nonnegative.")
        if not 0.0 <= self.time_sensitive_fraction <= 1.0:
            raise ValueError("time_sensitive_fraction must lie in [0, 1].")
        if not 0.0 <= self.immediate_loss_fraction <= 1.0:
            raise ValueError("immediate_loss_fraction must lie in [0, 1].")
        if self.time_sensitive_fraction + self.immediate_loss_fraction > 1.0 + 1e-12:
            raise ValueError(
                "time_sensitive_fraction and immediate_loss_fraction must sum to at most 1."
            )

    @property
    def at_risk(self) -> float:
        return self.population * self.collapse_fraction

    @property
    def time_sensitive_demand(self) -> float:
        """At-risk people whose outcome is modeled as access-responsive."""
        return self.at_risk * self.time_sensitive_fraction

    @property
    def immediate_loss(self) -> float:
        """At-risk people with a modeled loss that does not depend on emergency access."""
        return self.at_risk * self.immediate_loss_fraction


@dataclass(frozen=True)
class Link:
    id: str
    tail: str
    head: str
    baseline_failure_probability: float
    retrofit_cost: float
    travel_time: float = 1.0
    residual_failure_probability: float = 0.0
    failure_delay_reduction: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "baseline_failure_probability": self.baseline_failure_probability,
            "retrofit_cost": self.retrofit_cost,
            "travel_time": self.travel_time,
            "residual_failure_probability": self.residual_failure_probability,
            "failure_delay_reduction": self.failure_delay_reduction,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= self.baseline_failure_probability <= 1.0:
            raise ValueError("baseline_failure_probability must lie in [0, 1].")
        if not 0.0 <= self.residual_failure_probability <= self.baseline_failure_probability:
            raise ValueError(
                "residual_failure_probability must lie in [0, baseline_failure_probability]."
            )
        if self.retrofit_cost < 0.0:
            raise ValueError("retrofit_cost must be nonnegative.")
        if self.travel_time < 0.0:
            raise ValueError("travel_time must be nonnegative.")
        if not 0.0 <= self.failure_delay_reduction <= 1.0:
            raise ValueError("failure_delay_reduction must lie in [0, 1].")

    def failure_probability(self, retrofit_level: float) -> float:
        if not isfinite(retrofit_level) or not 0.0 <= retrofit_level <= 1.0:
            raise ValueError("retrofit_level must lie in [0, 1].")
        return float(
            self.residual_failure_probability
            + (self.baseline_failure_probability - self.residual_failure_probability)
            * (1.0 - retrofit_level)
        )

    def failure_delay_multiplier(self, retrofit_level: float) -> float:
        if not isfinite(retrofit_level) or not 0.0 <= retrofit_level <= 1.0:
            raise ValueError("retrofit_level must lie in [0, 1].")
        return float(1.0 - self.failure_delay_reduction * retrofit_level)


@dataclass(frozen=True)
class AidCenter:
    id: str
    node: str
    existing_capacity: float
    capacity_unit_cost: float

    def __post_init__(self) -> None:
        values = {
            "existing_capacity": self.existing_capacity,
            "capacity_unit_cost": self.capacity_unit_cost,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.existing_capacity < 0.0:
            raise ValueError("existing_capacity must be nonnegative.")
        if self.capacity_unit_cost <= 0.0:
            raise ValueError("capacity_unit_cost must be positive.")


@dataclass(frozen=True)
class SurvivalParams:
    a: float = 0.92
    b: float = -0.03
    c: float = 1.0
    d: float = 0.02

    def fraction(self, travel_time: float) -> float:
        if not isfinite(travel_time):
            return 0.0
        return float(max(0.0, min(1.0, self.a * exp(self.b * (travel_time**self.c)) + self.d)))


@dataclass(frozen=True)
class ThresholdResponseParams:
    """Binary timely-access response rule for a declared emergency response window.

    The returned value is an access indicator, not a clinical survival probability. It
    is useful when the model outcome is the number of time-sensitive patients who can
    receive care within a documented planning threshold.
    """

    threshold_minutes: float
    timely_access_value: float = 1.0
    late_access_value: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "threshold_minutes": self.threshold_minutes,
            "timely_access_value": self.timely_access_value,
            "late_access_value": self.late_access_value,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.threshold_minutes < 0.0:
            raise ValueError("threshold_minutes must be nonnegative.")
        if not 0.0 <= self.timely_access_value <= 1.0:
            raise ValueError("timely_access_value must lie in [0, 1].")
        if not 0.0 <= self.late_access_value <= 1.0:
            raise ValueError("late_access_value must lie in [0, 1].")

    def fraction(self, travel_time: float) -> float:
        if not isfinite(travel_time):
            return 0.0
        return float(
            self.timely_access_value
            if travel_time <= self.threshold_minutes
            else self.late_access_value
        )


@dataclass(frozen=True)
class PiecewiseLinearResponseParams:
    """Graded timely-access credit over declared travel-time breakpoints.

    Values are planning-service credits rather than clinical survival
    probabilities. Linear interpolation avoids the all-or-nothing behavior of
    a single response threshold while retaining a transparent monotone rule.
    """

    knots: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        normalized = tuple((float(minutes), float(value)) for minutes, value in self.knots)
        if len(normalized) < 2:
            raise ValueError("At least two response knots are required.")
        minutes = np.asarray([knot[0] for knot in normalized], dtype=float)
        values = np.asarray([knot[1] for knot in normalized], dtype=float)
        if not np.all(np.isfinite(minutes)) or not np.all(np.isfinite(values)):
            raise ValueError("Response knots must be finite.")
        if minutes[0] < 0.0 or np.any(np.diff(minutes) <= 0.0):
            raise ValueError("Response-knot times must be nonnegative and strictly increasing.")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("Response-knot values must lie in [0, 1].")
        if np.any(np.diff(values) > 1e-12):
            raise ValueError("Response-knot values must be nonincreasing in travel time.")
        object.__setattr__(self, "knots", normalized)

    def fraction(self, travel_time: float) -> float:
        if not isfinite(travel_time):
            return 0.0
        minutes = np.asarray([knot[0] for knot in self.knots], dtype=float)
        values = np.asarray([knot[1] for knot in self.knots], dtype=float)
        return float(np.interp(max(0.0, travel_time), minutes, values))


@dataclass(frozen=True)
class FailureMomentEnvelope:
    marginal_relative_tolerance: float | None = None
    marginal_absolute_tolerance: float = 0.0
    joint_relative_tolerance: float | None = None
    joint_absolute_tolerance: float = 0.0
    count_mean_absolute_tolerance: float | None = None
    count_second_moment_relative_tolerance: float | None = None
    count_second_moment_absolute_tolerance: float = 0.0

    def __post_init__(self) -> None:
        optional_values = {
            "marginal_relative_tolerance": self.marginal_relative_tolerance,
            "joint_relative_tolerance": self.joint_relative_tolerance,
            "count_mean_absolute_tolerance": self.count_mean_absolute_tolerance,
            "count_second_moment_relative_tolerance": self.count_second_moment_relative_tolerance,
        }
        for name, value in optional_values.items():
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative when provided.")
        required_values = {
            "marginal_absolute_tolerance": self.marginal_absolute_tolerance,
            "joint_absolute_tolerance": self.joint_absolute_tolerance,
            "count_second_moment_absolute_tolerance": self.count_second_moment_absolute_tolerance,
        }
        for name, value in required_values.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    @property
    def marginal_active(self) -> bool:
        return self.marginal_relative_tolerance is not None or self.marginal_absolute_tolerance > 0.0

    @property
    def joint_active(self) -> bool:
        return self.joint_relative_tolerance is not None or self.joint_absolute_tolerance > 0.0

    @property
    def count_second_moment_active(self) -> bool:
        return (
            self.count_second_moment_relative_tolerance is not None
            or self.count_second_moment_absolute_tolerance > 0.0
        )


@dataclass(frozen=True)
class HazardRegime:
    id: str
    probability: float
    failed_centers: tuple[str, ...] = ()
    link_failure_multipliers: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Hazard regime id must be nonempty.")
        if not isfinite(self.probability) or self.probability < 0.0:
            raise ValueError("Hazard regime probability must be finite and nonnegative.")
        object.__setattr__(self, "failed_centers", tuple(sorted(self.failed_centers)))
        if any(not isfinite(value) or value < 0.0 for value in self.link_failure_multipliers.values()):
            raise ValueError("Link failure multipliers must be finite and nonnegative.")


@dataclass(frozen=True)
class State:
    id: str
    failed_links: tuple[str, ...] = ()
    is_tail: bool = False
    failed_centers: tuple[str, ...] = ()
    hazard_regime_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "failed_links", tuple(sorted(self.failed_links)))
        object.__setattr__(self, "failed_centers", tuple(sorted(self.failed_centers)))


TravelTimeProvider = Callable[["DADInstance", State], np.ndarray]


@dataclass
class DADInstance:
    zones: list[Zone]
    links: list[Link]
    centers: list[AidCenter]
    budget_renovation: float
    budget_retrofit: float
    budget_capacity: float
    ambiguity_radius: float
    states: list[State]
    survival: SurvivalParams | ThresholdResponseParams | PiecewiseLinearResponseParams = field(
        default_factory=SurvivalParams
    )
    graph: nx.DiGraph | None = None
    travel_time_provider: TravelTimeProvider | None = None
    precomputed_travel_times: Mapping[str, Sequence[Sequence[float]]] | None = None
    intact_travel_times: Sequence[Sequence[float]] | None = None
    failure_penalty_matrices: Mapping[str, Sequence[Sequence[float]]] | None = None
    ambiguity_density_cap: float | None = None
    failure_moment_envelope: FailureMomentEnvelope | None = None
    minimum_protected_population: float = 0.0
    minimum_zone_service_fraction: float | Sequence[float] = 0.0
    hazard_regimes: list[HazardRegime] | None = None
    critical_service_state_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if not self.zones:
            raise ValueError("At least one zone is required.")
        if not self.centers:
            raise ValueError("At least one aid center is required.")
        budgets = {
            "budget_renovation": self.budget_renovation,
            "budget_retrofit": self.budget_retrofit,
            "budget_capacity": self.budget_capacity,
        }
        for name, value in budgets.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not self.links and any(not state.is_tail and state.failed_links for state in self.states):
            raise ValueError("Failure states reference links, but no links were provided.")
        if not 0 <= self.ambiguity_radius <= 1:
            raise ValueError("ambiguity_radius must lie in [0, 1].")
        if self.ambiguity_density_cap is not None and (
            not isfinite(self.ambiguity_density_cap) or self.ambiguity_density_cap < 1.0
        ):
            raise ValueError("ambiguity_density_cap must be finite and at least 1 when provided.")
        if not isfinite(self.minimum_protected_population) or self.minimum_protected_population < 0.0:
            raise ValueError("minimum_protected_population must be finite and nonnegative.")
        service_floor = np.asarray(self.minimum_zone_service_fraction, dtype=float)
        if service_floor.ndim == 0:
            service_floor = np.full(len(self.zones), float(service_floor))
        if service_floor.shape != (len(self.zones),):
            raise ValueError("minimum_zone_service_fraction must be scalar or one value per zone.")
        if np.any(~np.isfinite(service_floor)) or np.any(service_floor < 0.0) or np.any(service_floor > 1.0):
            raise ValueError("minimum_zone_service_fraction must lie in [0, 1].")
        self.minimum_zone_service_fraction = service_floor
        link_ids = {link.id for link in self.links}
        if len(link_ids) != len(self.links):
            raise ValueError("Link ids must be unique.")
        for state in self.states:
            unknown = set(state.failed_links) - link_ids
            if unknown:
                raise ValueError(f"State {state.id} references unknown links: {sorted(unknown)}")
        if (self.intact_travel_times is None) != (self.failure_penalty_matrices is None):
            raise ValueError(
                "intact_travel_times and failure_penalty_matrices must be provided together."
            )
        if self.intact_travel_times is not None and self.failure_penalty_matrices is not None:
            expected_shape = (len(self.centers), len(self.zones))
            intact = np.asarray(self.intact_travel_times, dtype=float)
            if intact.shape != expected_shape:
                raise ValueError(
                    f"intact_travel_times has shape {intact.shape}, expected {expected_shape}."
                )
            unknown_penalties = set(self.failure_penalty_matrices) - link_ids
            if unknown_penalties:
                raise ValueError(
                    f"failure_penalty_matrices references unknown links: {sorted(unknown_penalties)}"
                )
            missing_penalties = link_ids - set(self.failure_penalty_matrices)
            if missing_penalties:
                raise ValueError(
                    f"failure_penalty_matrices is missing links: {sorted(missing_penalties)}"
                )
            for link_id, matrix in self.failure_penalty_matrices.items():
                values = np.asarray(matrix, dtype=float)
                if values.shape != expected_shape:
                    raise ValueError(
                        f"Failure-penalty matrix for {link_id} has shape {values.shape}, "
                        f"expected {expected_shape}."
                    )
                if np.any(values < -1e-12):
                    raise ValueError(f"Failure-penalty matrix for {link_id} must be nonnegative.")
        state_ids = {state.id for state in self.states}
        if self.critical_service_state_ids is not None:
            unknown_critical_states = set(self.critical_service_state_ids) - state_ids
            if unknown_critical_states:
                raise ValueError(f"Unknown critical service states: {sorted(unknown_critical_states)}")
            self.critical_service_state_ids = set(self.critical_service_state_ids)
        center_ids = {center.id for center in self.centers}
        for state in self.states:
            unknown_centers = set(state.failed_centers) - center_ids
            if unknown_centers:
                raise ValueError(f"State {state.id} references unknown centers: {sorted(unknown_centers)}")
        if self.hazard_regimes is not None:
            regime_ids = {regime.id for regime in self.hazard_regimes}
            if len(regime_ids) != len(self.hazard_regimes):
                raise ValueError("Hazard regime ids must be unique.")
            if not np.isclose(sum(regime.probability for regime in self.hazard_regimes), 1.0, atol=1e-10):
                raise ValueError("Hazard regime probabilities must sum to one.")
            for regime in self.hazard_regimes:
                if set(regime.failed_centers) - center_ids:
                    raise ValueError(f"Hazard regime {regime.id} references unknown centers.")
                if set(regime.link_failure_multipliers) - link_ids:
                    raise ValueError(f"Hazard regime {regime.id} references unknown links.")
            if any(state.hazard_regime_id not in regime_ids for state in self.states):
                raise ValueError("Every state must reference a known hazard regime.")
        if self.graph is None:
            self.graph = build_graph(self.links)

    @property
    def zone_ids(self) -> list[str]:
        return [zone.id for zone in self.zones]

    @property
    def link_ids(self) -> list[str]:
        return [link.id for link in self.links]

    @property
    def center_ids(self) -> list[str]:
        return [center.id for center in self.centers]

    @property
    def base_demands(self) -> np.ndarray:
        return np.array([zone.time_sensitive_demand for zone in self.zones], dtype=float)

    @property
    def base_immediate_losses(self) -> np.ndarray:
        return np.array([zone.immediate_loss for zone in self.zones], dtype=float)

    @property
    def base_modelled_loss_exposures(self) -> np.ndarray:
        return self.base_demands + self.base_immediate_losses

    @property
    def protected_population_coefficients(self) -> np.ndarray:
        return np.array([zone.at_risk for zone in self.zones], dtype=float)

    @property
    def zone_service_fractions(self) -> np.ndarray:
        return np.asarray(self.minimum_zone_service_fraction, dtype=float)

    @property
    def renovation_costs(self) -> np.ndarray:
        return np.array([zone.renovation_cost for zone in self.zones], dtype=float)

    @property
    def retrofit_costs(self) -> np.ndarray:
        return np.array([link.retrofit_cost for link in self.links], dtype=float)

    @property
    def failure_probabilities(self) -> np.ndarray:
        return np.array([link.baseline_failure_probability for link in self.links], dtype=float)

    @property
    def residual_failure_probabilities(self) -> np.ndarray:
        return np.array([link.residual_failure_probability for link in self.links], dtype=float)

    @property
    def has_retrofit_performance_effects(self) -> bool:
        return any(link.failure_delay_reduction > 1e-12 for link in self.links)

    def effective_failure_probabilities(self, y: Sequence[float]) -> np.ndarray:
        y_vec = np.asarray(y, dtype=float)
        if y_vec.shape != (len(self.links),):
            raise ValueError(f"Expected {len(self.links)} retrofit decisions, received {len(y_vec)}.")
        return np.array(
            [link.failure_probability(float(retrofit_level)) for link, retrofit_level in zip(self.links, y_vec)],
            dtype=float,
        )

    @property
    def existing_capacities(self) -> np.ndarray:
        return np.array([center.existing_capacity for center in self.centers], dtype=float)

    @property
    def capacity_costs(self) -> np.ndarray:
        return np.array([center.capacity_unit_cost for center in self.centers], dtype=float)

    @property
    def d_max(self) -> float:
        return float(self.base_modelled_loss_exposures.sum())

    def demand_after_renovation(self, z: Sequence[float]) -> np.ndarray:
        z_vec = np.asarray(z, dtype=float)
        return self.base_demands * (1.0 - z_vec)

    def immediate_losses_after_renovation(self, z: Sequence[float]) -> np.ndarray:
        z_vec = np.asarray(z, dtype=float)
        return self.base_immediate_losses * (1.0 - z_vec)

    def at_risk_population(self, z: Sequence[float]) -> float:
        z_vec = np.asarray(z, dtype=float)
        return float(np.asarray([zone.at_risk for zone in self.zones], dtype=float) @ (1.0 - z_vec))

    def modelled_loss_exposure_after_renovation(self, z: Sequence[float]) -> float:
        return float(
            self.demand_after_renovation(z).sum()
            + self.immediate_losses_after_renovation(z).sum()
        )

    def loss_after_response(
        self,
        z: Sequence[float],
        survivors: float | Sequence[float] | np.ndarray,
    ) -> float | np.ndarray:
        """Return immediate loss plus time-sensitive demand not served by recourse."""
        exposure = self.modelled_loss_exposure_after_renovation(z)
        survivors_array = np.asarray(survivors, dtype=float)
        losses = exposure - survivors_array
        return float(losses) if losses.ndim == 0 else losses

    def service_fractions_for_state(self, state: State) -> np.ndarray:
        if self.critical_service_state_ids is not None and state.id not in self.critical_service_state_ids:
            return np.zeros(len(self.zones), dtype=float)
        return self.zone_service_fractions

    def facility_availability(self, state: State) -> np.ndarray:
        failed = set(state.failed_centers)
        return np.array([0.0 if center.id in failed else 1.0 for center in self.centers], dtype=float)

    def capacity_after_investment(self, w: Sequence[float]) -> np.ndarray:
        return self.existing_capacities + np.asarray(w, dtype=float)

    def travel_times(self, state: State, y: Sequence[float] | None = None) -> np.ndarray:
        if state.is_tail:
            return np.full((len(self.centers), len(self.zones)), inf, dtype=float)
        if self.has_retrofit_performance_effects:
            if y is None:
                raise ValueError("Retrofit decisions are required when failure-delay reduction is active.")
            return self.performance_adjusted_travel_times(state, y)
        if self.precomputed_travel_times is not None and state.id in self.precomputed_travel_times:
            matrix = np.asarray(self.precomputed_travel_times[state.id], dtype=float)
            expected = (len(self.centers), len(self.zones))
            if matrix.shape != expected:
                raise ValueError(f"Travel-time matrix for {state.id} has shape {matrix.shape}, expected {expected}.")
            return matrix
        if self.travel_time_provider is not None:
            return np.asarray(self.travel_time_provider(self, state), dtype=float)
        return network_travel_times(self, state)

    def performance_adjusted_travel_times(self, state: State, y: Sequence[float]) -> np.ndarray:
        if self.intact_travel_times is None or self.failure_penalty_matrices is None:
            raise ValueError(
                "Retrofit-dependent failure delays require intact_travel_times and failure_penalty_matrices."
            )
        y_vec = np.asarray(y, dtype=float)
        if y_vec.shape != (len(self.links),) or np.any(y_vec < -1e-12) or np.any(y_vec > 1.0 + 1e-12):
            raise ValueError("Retrofit decisions must have one value in [0, 1] for every link.")
        link_lookup = {link.id: link for link in self.links}
        retrofit_lookup = {link.id: float(y_vec[index]) for index, link in enumerate(self.links)}
        matrix = np.asarray(self.intact_travel_times, dtype=float).copy()
        for link_id in state.failed_links:
            link = link_lookup[link_id]
            penalty = np.asarray(self.failure_penalty_matrices[link_id], dtype=float)
            matrix += penalty * link.failure_delay_multiplier(retrofit_lookup[link_id])
        return matrix

    def survival_matrix(self, state: State, y: Sequence[float] | None = None) -> np.ndarray:
        travel_times = self.travel_times(state, y=y)
        vectorized = np.vectorize(self.survival.fraction)
        return vectorized(travel_times)


def build_graph(links: Sequence[Link]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for link in links:
        graph.add_edge(link.tail, link.head, link_id=link.id, travel_time=link.travel_time)
    return graph


def network_travel_times(instance: DADInstance, state: State) -> np.ndarray:
    if instance.graph is None:
        raise ValueError("A graph or precomputed travel times are required.")
    graph = instance.graph.copy()
    failed = set(state.failed_links)
    edges_to_remove = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("link_id") in failed
    ]
    graph.remove_edges_from(edges_to_remove)
    matrix = np.full((len(instance.centers), len(instance.zones)), inf, dtype=float)
    for center_index, center in enumerate(instance.centers):
        lengths = nx.single_source_dijkstra_path_length(graph, center.node, weight="travel_time")
        for zone_index, zone in enumerate(instance.zones):
            if zone.node is None:
                raise ValueError(f"Zone {zone.id} needs a network node or precomputed travel times.")
            matrix[center_index, zone_index] = lengths.get(zone.node, inf)
    return matrix




