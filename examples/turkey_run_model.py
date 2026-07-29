from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

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
from ejor_dad.turkey import (
    download_zenodo_influencing_factors,
    haversine_km,
    normalize_01,
    overpass_roads_bbox,
    sample_geotiff_to_dataframe,
)


def main() -> None:
    base = Path("data_work/turkey")
    zones = pd.read_csv(base / "turkey_zones_model_input.csv")
    centers = pd.read_csv(base / "turkey_centers_model_input.csv")
    destroyed = pd.read_csv(base / "hotosm_destroyed_buildings_with_grid.csv")

    zones = zones[zones["population"] > 0].copy()
    zones["at_risk"] = zones["population"] * zones["collapse_fraction"]
    zones = zones.sort_values("at_risk", ascending=False).head(12).reset_index(drop=True)
    centers = centers.head(6).copy()

    nodes_path = base / "osm_road_nodes_bbox.csv"
    edges_path = base / "osm_road_edges_bbox.csv"
    if not nodes_path.exists() or not edges_path.exists():
        nodes, edges = download_osm_roads(base, zones, centers)
    else:
        nodes = pd.read_csv(nodes_path, dtype={"node": str})
        edges = pd.read_csv(edges_path, dtype={"tail": str, "head": str, "link_id": str, "osm_way_id": str})

    graph = road_graph_from_edges(edges)
    zone_nodes = nearest_road_nodes(zones, nodes, "centroid_lon", "centroid_lat")
    center_nodes = nearest_road_nodes(centers, nodes, "longitude", "latitude")
    zones["node"] = zone_nodes
    centers["node"] = center_nodes

    route_edge_counts, base_travel_times, path_link_ids = collect_shortest_path_data(graph, centers["node"], zones["node"])
    hazard_paths = ensure_hazard_rasters(base)
    candidate_links = candidate_links_from_routes(edges, nodes, route_edge_counts, destroyed, hazard_paths, max_links=5)
    states = generate_failure_states(candidate_links, max_failures=2, include_tail=True)

    zone_objects = [
        Zone(
            id=str(row.zone_id),
            population=float(row.population),
            collapse_fraction=float(row.collapse_fraction),
            renovation_cost=float(max(row.renovation_cost, 1.0)),
            node=str(row.node),
            region=str(row.region),
        )
        for row in zones.itertuples(index=False)
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

    budget_renovation = 0.20 * sum(zone.renovation_cost for zone in zone_objects)
    budget_retrofit = 0.25 * sum(link.retrofit_cost for link in candidate_links)
    budget_capacity = 0.20 * sum(center.existing_capacity * center.capacity_unit_cost for center in center_objects)

    instance = DADInstance(
        zones=zone_objects,
        links=candidate_links,
        centers=center_objects,
        budget_renovation=budget_renovation,
        budget_retrofit=budget_retrofit,
        budget_capacity=budget_capacity,
        ambiguity_radius=0.10,
        states=states,
        survival=SurvivalParams(a=0.95, b=-0.025, c=1.0, d=0.0),
        graph=graph,
    )
    precomputed = scenario_travel_times(states, base_travel_times, path_link_ids)
    instance.precomputed_travel_times = precomputed

    fixed = evaluate_fixed_y(instance, np.zeros(len(candidate_links)), epsilon=1e-5, max_iterations=100)
    heuristic = heuristic_retrofit_search(instance, max_random=30)
    sbb = solve_global_sbb(instance, epsilon=1e-2, fixed_y_epsilon=1e-5, max_nodes=30)

    best = min([fixed, heuristic, sbb.fixed_y_result], key=lambda result: result.objective)
    write_outputs(base, instance, fixed, heuristic, sbb, best, zones, centers, candidate_links)
    print("Turkey first-pass DR-DAD run complete")
    print("zones:", len(zones), "centers:", len(centers), "candidate links:", len(candidate_links), "states:", len(states))
    print("no-retrofit objective:", round(float(fixed.objective), 3))
    print("heuristic objective:", round(float(heuristic.objective), 3))
    print("sbb objective:", round(float(sbb.objective), 3))
    print("best objective:", round(float(best.objective), 3))
    print("best reduction:", round(float(fixed.objective - best.objective), 3))
    print("outputs:", (base / "results").resolve())


def download_osm_roads(base: Path, zones: pd.DataFrame, centers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    west = min(float(zones["centroid_lon"].min()), float(centers["longitude"].min())) - 0.03
    east = max(float(zones["centroid_lon"].max()), float(centers["longitude"].max())) + 0.03
    south = min(float(zones["centroid_lat"].min()), float(centers["latitude"].min())) - 0.03
    north = max(float(zones["centroid_lat"].max()), float(centers["latitude"].max())) + 0.03
    nodes, edges = overpass_roads_bbox(
        south,
        west,
        north,
        east,
        highway_types=("motorway", "trunk", "primary", "secondary", "tertiary", "residential"),
        timeout_seconds=180,
    )
    nodes.to_csv(base / "osm_road_nodes_bbox.csv", index=False)
    edges.to_csv(base / "osm_road_edges_bbox.csv", index=False)
    return nodes, edges


def ensure_hazard_rasters(base: Path) -> dict[str, Path]:
    hazard_dir = base / "zenodo_hazards"
    requested = ("PGV", "Fault", "Epicenter")
    existing = {name: next(hazard_dir.rglob(f"{name}.tif"), None) for name in requested}
    if all(path is not None and path.exists() for path in existing.values()):
        return {name: path for name, path in existing.items() if path is not None}
    return download_zenodo_influencing_factors(hazard_dir, factor_names=requested)


def road_graph_from_edges(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        tail = str(row.tail)
        head = str(row.head)
        travel_time = float(row.travel_time)
        existing = graph.get_edge_data(tail, head)
        if existing is None or travel_time < existing["travel_time"]:
            graph.add_edge(
                tail,
                head,
                link_id=str(row.link_id),
                travel_time=travel_time,
                highway=str(row.highway),
                bridge=bool(row.bridge),
                distance_km=float(row.distance_km),
                retrofit_cost=float(row.retrofit_cost),
            )
    return graph


def nearest_road_nodes(points: pd.DataFrame, nodes: pd.DataFrame, lon_col: str, lat_col: str) -> list[str]:
    node_lons = nodes["lon"].to_numpy(dtype=float)
    node_lats = nodes["lat"].to_numpy(dtype=float)
    node_ids = nodes["node"].astype(str).to_numpy()
    nearest = []
    for row in points.itertuples(index=False):
        lon = float(getattr(row, lon_col))
        lat = float(getattr(row, lat_col))
        x = (node_lons - lon) * np.cos(np.radians(lat))
        y = node_lats - lat
        index = int(np.argmin(x * x + y * y))
        nearest.append(str(node_ids[index]))
    return nearest


def collect_shortest_path_data(
    graph: nx.DiGraph,
    center_nodes: pd.Series,
    zone_nodes: pd.Series,
) -> tuple[Counter[tuple[str, str]], np.ndarray, list[list[set[str]]]]:
    counts: Counter[tuple[str, str]] = Counter()
    center_list = list(center_nodes.astype(str))
    zone_list = list(zone_nodes.astype(str))
    base_times = np.full((len(center_list), len(zone_list)), np.inf, dtype=float)
    path_link_ids: list[list[set[str]]] = [[set() for _ in zone_list] for _ in center_list]
    for center_index, center_node in enumerate(center_list):
        lengths, paths = nx.single_source_dijkstra(graph, center_node, weight="travel_time")
        for zone_index, zone_node in enumerate(zone_list):
            if zone_node not in paths:
                continue
            base_times[center_index, zone_index] = float(lengths[zone_node])
            path = paths[zone_node]
            edge_pairs = list(zip(path[:-1], path[1:]))
            counts.update(edge_pairs)
            for tail, head in edge_pairs:
                data = graph.get_edge_data(tail, head)
                if data and "link_id" in data:
                    path_link_ids[center_index][zone_index].add(str(data["link_id"]))
    return counts, base_times, path_link_ids


def scenario_travel_times(
    states,
    base_travel_times: np.ndarray,
    path_link_ids: list[list[set[str]]],
    failure_penalty: float = 0.75,
) -> dict[str, np.ndarray]:
    matrices = {}
    for state in states:
        if state.is_tail:
            continue
        failed = set(state.failed_links)
        matrix = base_travel_times.copy()
        for center_index, row in enumerate(path_link_ids):
            for zone_index, links in enumerate(row):
                failures_on_path = len(failed & links)
                if failures_on_path and np.isfinite(matrix[center_index, zone_index]):
                    matrix[center_index, zone_index] *= 1.0 + failure_penalty * failures_on_path
        matrices[state.id] = matrix
    return matrices


def candidate_links_from_routes(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    route_edge_counts: Counter[tuple[str, str]],
    destroyed: pd.DataFrame,
    hazard_paths: dict[str, Path] | None,
    max_links: int,
) -> list[Link]:
    edge_lookup = edges.set_index(["tail", "head"], drop=False)
    candidate_rows = []
    for (tail, head), count in route_edge_counts.most_common(max_links * 10):
        if (tail, head) not in edge_lookup.index:
            continue
        row = edge_lookup.loc[(tail, head)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        candidate_rows.append((count, row))
    hazard_frame = candidate_hazard_frame(candidate_rows, nodes, destroyed, hazard_paths)
    scored = []
    for hazard_row in hazard_frame.itertuples(index=False):
        count = int(hazard_row.route_count)
        row = edge_lookup.loc[(str(hazard_row.tail), str(hazard_row.head))]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        highway = str(row["highway"]).lower()
        critical = highway in {"motorway", "trunk", "primary", "secondary", "tertiary"}
        bridge = str(row["bridge"]).lower() in {"true", "yes", "1"}
        probability = min(
            0.55,
            0.02
            + 0.20 * float(hazard_row.pgv_norm)
            + 0.10 * float(hazard_row.fault_proximity_norm)
            + 0.05 * float(hazard_row.epicenter_proximity_norm)
            + 0.10 * float(hazard_row.near_destroyed)
            + 0.05 * critical
            + 0.06 * bridge,
        )
        class_multiplier = {"motorway": 3.0, "trunk": 2.5, "primary": 2.0, "secondary": 1.5, "tertiary": 1.2}.get(highway, 1.0)
        terrain_multiplier = 1.0 + 0.5 * float(hazard_row.pgv_norm)
        retrofit_cost = max(float(row["distance_km"]), 0.05) * class_multiplier * terrain_multiplier
        scored.append(
            (
                count,
                Link(
                    id=str(row["link_id"]),
                    tail=str(row["tail"]),
                    head=str(row["head"]),
                    baseline_failure_probability=probability,
                    retrofit_cost=retrofit_cost,
                    travel_time=float(row["travel_time"]),
                ),
            )
        )
    selected = []
    seen = set()
    for _, link in sorted(scored, key=lambda item: (-item[0], -item[1].retrofit_cost)):
        if link.id in seen:
            continue
        selected.append(link)
        seen.add(link.id)
        if len(selected) >= max_links:
            break
    return selected


def candidate_hazard_frame(
    candidate_rows: list[tuple[int, pd.Series]],
    nodes: pd.DataFrame,
    destroyed: pd.DataFrame,
    hazard_paths: dict[str, Path] | None,
) -> pd.DataFrame:
    node_lookup = nodes.set_index("node")[["lon", "lat"]].to_dict("index")
    rows = []
    for count, row in candidate_rows:
        tail = str(row["tail"])
        head = str(row["head"])
        if tail not in node_lookup or head not in node_lookup:
            continue
        midpoint_lon = 0.5 * (float(node_lookup[tail]["lon"]) + float(node_lookup[head]["lon"]))
        midpoint_lat = 0.5 * (float(node_lookup[tail]["lat"]) + float(node_lookup[head]["lat"]))
        rows.append(
            {
                "tail": tail,
                "head": head,
                "route_count": count,
                "longitude": midpoint_lon,
                "latitude": midpoint_lat,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if hazard_paths:
        available = {name: path for name, path in hazard_paths.items() if Path(path).exists()}
        frame = sample_geotiff_to_dataframe(frame, available, prefix="hazard")
    for column in ("hazard_pgv", "hazard_fault", "hazard_epicenter"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["pgv_norm"] = normalize_01(frame["hazard_pgv"]).to_numpy()
    frame["fault_proximity_norm"] = inverse_normalize(frame["hazard_fault"])
    frame["epicenter_proximity_norm"] = inverse_normalize(frame["hazard_epicenter"])
    frame["near_destroyed"] = destroyed_proximity_indicator(frame, destroyed, threshold_km=0.75)
    return frame


def inverse_normalize(values) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if series.notna().sum() == 0:
        return np.zeros(len(series), dtype=float)
    normalized = normalize_01(values).to_numpy(dtype=float)
    return 1.0 - normalized


def destroyed_proximity_indicator(frame: pd.DataFrame, destroyed: pd.DataFrame, threshold_km: float) -> np.ndarray:
    if destroyed.empty or not {"longitude", "latitude"}.issubset(destroyed.columns):
        return np.zeros(len(frame), dtype=float)
    coords = destroyed[["longitude", "latitude"]].dropna().to_numpy(dtype=float)
    if len(coords) == 0:
        return np.zeros(len(frame), dtype=float)
    indicators = []
    for row in frame.itertuples(index=False):
        lon = float(row.longitude)
        lat = float(row.latitude)
        close = False
        for destroyed_lon, destroyed_lat in coords:
            if haversine_km(lon, lat, destroyed_lon, destroyed_lat) <= threshold_km:
                close = True
                break
        indicators.append(float(close))
    return np.array(indicators, dtype=float)


def heuristic_retrofit_search(instance: DADInstance, max_random: int = 30):
    rng = np.random.default_rng(42)
    num_links = len(instance.links)
    costs = instance.retrofit_costs
    budget = instance.budget_retrofit
    plans = [np.zeros(num_links)]
    for link_index in range(num_links):
        y = np.zeros(num_links)
        y[link_index] = min(1.0, budget / costs[link_index]) if costs[link_index] > 0 else 0.0
        plans.append(y)
    order = np.argsort(costs)
    greedy = np.zeros(num_links)
    remaining = budget
    for link_index in order:
        amount = min(1.0, remaining / costs[link_index]) if costs[link_index] > 0 else 0.0
        greedy[link_index] = amount
        remaining -= amount * costs[link_index]
        if remaining <= 1e-9:
            break
    plans.append(greedy)
    for _ in range(max_random):
        weights = rng.random(num_links)
        weights = weights / weights.sum()
        y = np.minimum(1.0, (budget * weights) / costs)
        plans.append(y)
    best = None
    seen = set()
    for plan in plans:
        key = tuple(np.round(plan, 6))
        if key in seen:
            continue
        seen.add(key)
        if np.dot(costs, plan) > budget + 1e-8:
            continue
        result = evaluate_fixed_y(instance, plan, epsilon=1e-5, max_iterations=100)
        if best is None or result.objective < best.objective:
            best = result
    if best is None:
        raise RuntimeError("No feasible retrofit plan was evaluated.")
    return best


def write_outputs(
    base: Path,
    instance: DADInstance,
    fixed,
    heuristic,
    sbb,
    best,
    zones: pd.DataFrame,
    centers: pd.DataFrame,
    candidate_links: list[Link],
) -> None:
    out = base / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "metric": [
                "fixed_y_objective",
                "best_incumbent_objective",
                "heuristic_objective",
                "sbb_incumbent_objective",
                "best_incumbent_reduction",
                "sbb_certificate_valid",
                "sbb_reported_lower_bound",
                "sbb_reported_gap",
                "nodes_processed",
                "zones",
                "centers",
                "candidate_links",
                "states",
                "budget_renovation",
                "budget_retrofit",
                "budget_capacity",
            ],
            "value": [
                fixed.objective,
                best.objective,
                heuristic.objective,
                sbb.objective,
                fixed.objective - best.objective,
                float(sbb.lower_bound <= best.objective + 1e-6),
                sbb.lower_bound,
                sbb.gap,
                sbb.nodes_processed,
                len(instance.zones),
                len(instance.centers),
                len(instance.links),
                len(instance.states),
                instance.budget_renovation,
                instance.budget_retrofit,
                instance.budget_capacity,
            ],
        }
    ).to_csv(out / "summary.csv", index=False)
    zones.to_csv(out / "zones_used.csv", index=False)
    centers.to_csv(out / "centers_used.csv", index=False)
    pd.DataFrame(
        [
            {
                "link_id": link.id,
                "tail": link.tail,
                "head": link.head,
                "failure_probability": link.baseline_failure_probability,
                "retrofit_cost": link.retrofit_cost,
                "travel_time": link.travel_time,
                "y": sbb.y[index],
                "y_heuristic": heuristic.y[index],
                "y_best": best.y[index],
            }
            for index, link in enumerate(candidate_links)
        ]
    ).to_csv(out / "candidate_links.csv", index=False)
    pd.DataFrame({"zone_id": instance.zone_ids, "z": best.z}).to_csv(out / "renovation_decisions.csv", index=False)
    pd.DataFrame({"center_id": instance.center_ids, "w": best.w}).to_csv(out / "capacity_decisions.csv", index=False)
    pd.DataFrame(
        {
            "state_id": [state.id for state in instance.states],
            "nominal_probability": best.nominal_distribution,
            "worst_case_probability": best.worst_case_distribution,
            "loss": best.state_losses,
            "survivors": best.state_survivors,
        }
    ).to_csv(out / "state_outcomes.csv", index=False)


if __name__ == "__main__":
    main()
