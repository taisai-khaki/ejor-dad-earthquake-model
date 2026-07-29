from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ejor_dad.model import AidCenter, DADInstance, Link, SurvivalParams, Zone
from ejor_dad.states import generate_failure_states


def zone_exposure_from_survey(
    survey: pd.DataFrame,
    zone_col: str,
    damage_col: str,
    population_col: str | None = None,
    severe_values: Sequence[object] | None = None,
    severe_min_grade: float | None = None,
    renovation_cost_per_person: float = 1.0,
    min_records: int = 1,
) -> pd.DataFrame:
    """Aggregate Nepal building/household survey rows into EJOR zone P_rl and q_rl.

    If population_col is supplied, q_rl is occupant-weighted. Otherwise it is
    the severe/collapsed building share. This intentionally uses
    severe_count / surveyed_count for building-weighted q_rl.
    """
    if severe_values is None and severe_min_grade is None:
        raise ValueError("Provide severe_values or severe_min_grade.")
    data = survey.copy()
    if severe_values is not None:
        severe_set = set(severe_values)
        data["_severe_damage"] = data[damage_col].isin(severe_set)
    else:
        data["_severe_damage"] = pd.to_numeric(data[damage_col], errors="coerce") >= float(severe_min_grade)
    if population_col is not None:
        data["_population"] = pd.to_numeric(data[population_col], errors="coerce").fillna(0.0)
        grouped = data.groupby(zone_col, dropna=False)
        output = grouped.apply(
            lambda frame: pd.Series(
                {
                    "population": frame["_population"].sum(),
                    "survey_records": len(frame),
                    "severe_records": int(frame["_severe_damage"].sum()),
                    "collapse_fraction": weighted_fraction(frame["_severe_damage"], frame["_population"]),
                }
            )
        ).reset_index().rename(columns={zone_col: "zone_id"})
    else:
        grouped = data.groupby(zone_col, dropna=False)
        output = grouped["_severe_damage"].agg(
            population="size",
            survey_records="size",
            severe_records="sum",
            collapse_fraction="mean",
        ).reset_index().rename(columns={zone_col: "zone_id"})
    output = output[output["survey_records"] >= min_records].copy()
    output["collapse_fraction"] = output["collapse_fraction"].fillna(0.0).clip(0.0, 1.0)
    output["renovation_cost"] = output["population"] * renovation_cost_per_person
    return output


def weighted_fraction(indicator: pd.Series, weights: pd.Series) -> float:
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return float(indicator.mean()) if len(indicator) else 0.0
    return float((indicator.astype(float) * weights).sum() / weight_sum)


def road_failure_probability_score(
    roads: pd.DataFrame,
    shaking_col: str | None = None,
    landslide_buffer_col: str | None = None,
    steep_slope_col: str | None = None,
    bridge_col: str | None = None,
    vulnerable_class_col: str | None = None,
    base: float = 0.02,
    shaking_weight: float = 0.10,
    landslide_weight: float = 0.15,
    slope_weight: float = 0.08,
    bridge_or_class_weight: float = 0.05,
    cap: float = 0.40,
) -> pd.Series:
    """Scenario-calibrated Phi_ij hazard score for road links."""
    score = pd.Series(base, index=roads.index, dtype=float)
    if shaking_col:
        score += shaking_weight * roads[shaking_col].astype(bool).astype(float)
    if landslide_buffer_col:
        score += landslide_weight * roads[landslide_buffer_col].astype(bool).astype(float)
    if steep_slope_col:
        score += slope_weight * roads[steep_slope_col].astype(bool).astype(float)
    bridge_or_class = pd.Series(False, index=roads.index)
    if bridge_col:
        bridge_or_class = bridge_or_class | roads[bridge_col].astype(bool)
    if vulnerable_class_col:
        bridge_or_class = bridge_or_class | roads[vulnerable_class_col].astype(bool)
    score += bridge_or_class_weight * bridge_or_class.astype(float)
    return score.clip(0.0, cap)


def capacity_from_facility_type(
    facilities: pd.DataFrame,
    type_col: str,
    capacity_map: Mapping[object, float] | None = None,
    default_capacity: float = 50.0,
) -> pd.Series:
    if capacity_map is None:
        capacity_map = {
            "hospital": 500.0,
            "clinic": 150.0,
            "health_post": 80.0,
            "police": 100.0,
            "army": 250.0,
            "red_cross": 200.0,
            "relief_center": 150.0,
        }
    return facilities[type_col].map(capacity_map).fillna(default_capacity).astype(float)


def build_instance_from_tables(
    zones: pd.DataFrame,
    links: pd.DataFrame,
    centers: pd.DataFrame,
    budgets: Mapping[str, float],
    ambiguity_radius: float,
    max_failures: int = 2,
    include_tail: bool = True,
    survival: SurvivalParams | None = None,
    travel_times: pd.DataFrame | None = None,
) -> DADInstance:
    zone_objects = [
        Zone(
            id=str(row.zone_id),
            population=float(row.population),
            collapse_fraction=float(row.collapse_fraction),
            renovation_cost=float(row.renovation_cost),
            node=str(row.node) if "node" in zones.columns and pd.notna(row.node) else None,
            region=str(row.region) if "region" in zones.columns and pd.notna(row.region) else None,
        )
        for row in zones.itertuples(index=False)
    ]
    link_objects = [
        Link(
            id=str(row.link_id),
            tail=str(row.tail),
            head=str(row.head),
            baseline_failure_probability=float(row.failure_probability),
            retrofit_cost=float(row.retrofit_cost),
            travel_time=float(row.travel_time) if "travel_time" in links.columns else 1.0,
        )
        for row in links.itertuples(index=False)
    ]
    center_objects = [
        AidCenter(
            id=str(row.center_id),
            node=str(row.node),
            existing_capacity=float(row.existing_capacity),
            capacity_unit_cost=float(row.capacity_unit_cost),
        )
        for row in centers.itertuples(index=False)
    ]
    states = generate_failure_states(link_objects, max_failures=max_failures, include_tail=include_tail)
    precomputed = None
    if travel_times is not None:
        precomputed = long_travel_times_to_matrices(travel_times, states, center_objects, zone_objects)
    return DADInstance(
        zones=zone_objects,
        links=link_objects,
        centers=center_objects,
        budget_renovation=float(budgets["B_Z"]),
        budget_retrofit=float(budgets["B_Y"]),
        budget_capacity=float(budgets["B_X"]),
        ambiguity_radius=ambiguity_radius,
        states=states,
        survival=survival or SurvivalParams(),
        precomputed_travel_times=precomputed,
    )


def long_travel_times_to_matrices(
    travel_times: pd.DataFrame,
    states,
    centers: Sequence[AidCenter],
    zones: Sequence[Zone],
) -> dict[str, np.ndarray]:
    required = {"state_id", "center_id", "zone_id", "travel_time"}
    missing = required - set(travel_times.columns)
    if missing:
        raise ValueError(f"travel_times is missing columns: {sorted(missing)}")
    center_index = {center.id: index for index, center in enumerate(centers)}
    zone_index = {zone.id: index for index, zone in enumerate(zones)}
    matrices = {
        state.id: np.full((len(centers), len(zones)), np.inf, dtype=float)
        for state in states
        if not state.is_tail
    }
    for row in travel_times.itertuples(index=False):
        state_id = str(row.state_id)
        if state_id not in matrices:
            continue
        matrices[state_id][center_index[str(row.center_id)], zone_index[str(row.zone_id)]] = float(row.travel_time)
    return matrices


def read_csv_tables(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    folder = Path(folder)
    zones = pd.read_csv(folder / "zones.csv")
    links = pd.read_csv(folder / "links.csv")
    centers = pd.read_csv(folder / "centers.csv")
    travel_path = folder / "travel_times.csv"
    travel_times = pd.read_csv(travel_path) if travel_path.exists() else None
    return zones, links, centers, travel_times
