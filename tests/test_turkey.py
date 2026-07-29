from __future__ import annotations

import numpy as np
import pandas as pd

from ejor_dad.turkey import (
    aggregate_destroyed_counts,
    aggregate_total_building_counts,
    build_regular_grid_from_points,
    estimate_q_from_building_counts,
    score_turkey_link_failure_probability,
)


def test_grid_and_destroyed_counts() -> None:
    destroyed = pd.DataFrame(
        {
            "osm_id": [1, 2, 3],
            "longitude": [36.0, 36.001, 36.2],
            "latitude": [37.0, 37.001, 37.2],
            "destroyed_indicator": [1, 1, 1],
        }
    )
    assigned, zones = build_regular_grid_from_points(destroyed, cell_size_km=5)
    counts = aggregate_destroyed_counts(assigned)
    assert len(zones) >= 2
    assert counts["destroyed_buildings"].sum() == 3


def test_q_from_destroyed_and_total_counts() -> None:
    destroyed = pd.DataFrame({"zone_id": ["a", "b"], "destroyed_buildings": [2, 1]})
    all_buildings = pd.DataFrame({"zone_id": ["a", "b", "c"], "total_buildings": [10, 2, 4]})
    result = estimate_q_from_building_counts(destroyed, all_buildings)
    q = dict(zip(result["zone_id"], result["collapse_fraction"]))
    assert np.isclose(q["a"], 0.2)
    assert np.isclose(q["b"], 0.5)
    assert np.isclose(q["c"], 0.0)


def test_total_counts_and_link_scoring() -> None:
    buildings = pd.DataFrame({"osm_id": [1, 2, 3], "zone_id": ["a", "a", "b"]})
    totals = aggregate_total_building_counts(buildings)
    assert dict(zip(totals["zone_id"], totals["total_buildings"])) == {"a": 2, "b": 1}
    roads = pd.DataFrame(
        {
            "highway": ["primary", "residential"],
            "pgv": [80.0, 20.0],
            "near_destroyed": [True, False],
            "slope": [30.0, 5.0],
            "bridge": [True, False],
        }
    )
    phi = score_turkey_link_failure_probability(
        roads,
        pgv_col="pgv",
        near_destroyed_col="near_destroyed",
        slope_col="slope",
        bridge_col="bridge",
    )
    assert phi.iloc[0] > phi.iloc[1]
    assert (phi <= 0.45).all()
