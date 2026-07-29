from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ejor_dad import AidCenter, DADInstance, Link, SurvivalParams, Zone, evaluate_fixed_y, generate_failure_states
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text
from ejor_dad.fixed_y import evaluate_plan_losses
from ejor_dad.states import nominal_probabilities
from ejor_dad.tv import worst_case_tv_distribution

import turkey_run_model as run


BASE_RENOVATION_FRACTION = 0.20
BASE_RETROFIT_FRACTION = 0.25
BASE_CAPACITY_FRACTION = 0.20
BASE_RHO = 0.10
CACHE_VERSION = "turkey-paper-results-v3-rich-experiments"


def main() -> None:
    args = parse_args()
    base = Path("data_work/turkey")
    out = base / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)
    cache = CheckpointStore(out / "checkpoints")

    bundle = build_bundle(base)
    base_instance = bundle["instance"]
    fixed_y0 = cached_fixed_y(cache, "base_fixed_y0", base_instance, np.zeros(len(base_instance.links)), args.force)
    incumbent = cached_heuristic(cache, "base_heuristic_incumbent", base_instance, max_random=12, force=args.force)
    no_investment = cache.get_or_compute(
        versioned_key("base_no_investment"),
        lambda: {
            "value": evaluate_given_plan(
                base_instance,
                np.zeros(len(base_instance.links)),
                np.zeros(len(base_instance.zones)),
                np.zeros(len(base_instance.centers)),
            )
        },
        force=args.force,
    )["value"]

    write_table(source_table(), out, "table_01_data_sources")

    write_table(instance_summary_table(base_instance, bundle), out, "table_02_instance_summary")

    main_results = main_results_table(base_instance, no_investment, fixed_y0, incumbent)
    write_table(main_results, out, "table_03_main_results", float_format="%.3f")

    sector = sector_ablation_table(bundle, no_investment, cache, args.force)
    write_table(sector, out, "table_04_sector_ablation", float_format="%.3f")

    ambiguity = ambiguity_sensitivity_table(bundle, cache, args.force)
    write_table(ambiguity, out, "table_05_ambiguity_sensitivity", float_format="%.3f")

    budgets = budget_sensitivity_table(bundle, cache, args.force)
    write_table(budgets, out, "table_06_budget_sensitivity", float_format="%.3f")

    policies = policy_comparison_table(bundle, no_investment, cache, args.force)
    write_table(policies, out, "table_11_policy_comparison", float_format="%.3f")

    budget_design = budget_design_table(bundle, no_investment, cache, args.force)
    write_table(budget_design, out, "table_12_budget_design", float_format="%.3f")

    link_sensitivity = candidate_link_sensitivity_table(base, cache, args.force)
    write_table(link_sensitivity, out, "table_13_candidate_link_sensitivity", float_format="%.3f")

    write_table(zone_table(bundle["zones"], incumbent), out, "table_07_zone_decisions", float_format="%.3f")

    write_table(link_table(base_instance, incumbent), out, "table_08_link_decisions", float_format="%.3f")

    write_table(center_table(bundle["centers"], incumbent), out, "table_09_capacity_decisions", float_format="%.3f")

    write_table(state_table(base_instance, incumbent), out, "table_10_worst_case_states", float_format="%.6f")

    method_note(out)
    print("Paper tables written to:", out.resolve())
    print(main_results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable Turkey DR-DAD paper tables.")
    parser.add_argument("--force", action="store_true", help="Recompute all checkpoints instead of resuming.")
    return parser.parse_args()


def write_table(dataframe: pd.DataFrame, output_dir: Path, stem: str, float_format: str | None = None) -> None:
    atomic_write_dataframe(dataframe, output_dir / f"{stem}.csv", kind="csv")
    latex_kwargs = {"escape": True}
    if float_format is not None:
        latex_kwargs["float_format"] = float_format
    atomic_write_dataframe(dataframe, output_dir / f"{stem}.tex", kind="latex", **latex_kwargs)


def versioned_key(key: str) -> str:
    return f"{CACHE_VERSION}__{key}"


def cached_fixed_y(
    cache: CheckpointStore,
    key: str,
    instance: DADInstance,
    y: np.ndarray,
    force: bool,
):
    payload = cache.get_or_compute(
        versioned_key(key),
        lambda: fixed_result_payload(evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=100)),
        force=force,
    )
    return payload_to_result(payload)


def cached_heuristic(
    cache: CheckpointStore,
    key: str,
    instance: DADInstance,
    max_random: int,
    force: bool,
):
    payload = cache.get_or_compute(
        versioned_key(key),
        lambda: fixed_result_payload(run.heuristic_retrofit_search(instance, max_random=max_random)),
        force=force,
    )
    return payload_to_result(payload)


def fixed_result_payload(result) -> dict:
    return {
        "objective": float(result.objective),
        "lower_bound": float(result.lower_bound),
        "z": result.z.tolist(),
        "w": result.w.tolist(),
        "y": result.y.tolist(),
        "nominal_distribution": result.nominal_distribution.tolist(),
        "worst_case_distribution": result.worst_case_distribution.tolist(),
        "state_losses": result.state_losses.tolist(),
        "state_survivors": result.state_survivors.tolist(),
        "iterations": int(result.iterations),
    }


def payload_to_result(payload: dict):
    return SimpleNamespace(
        objective=float(payload["objective"]),
        lower_bound=float(payload.get("lower_bound", payload["objective"])),
        z=np.array(payload["z"], dtype=float),
        w=np.array(payload["w"], dtype=float),
        y=np.array(payload["y"], dtype=float),
        nominal_distribution=np.array(payload["nominal_distribution"], dtype=float),
        worst_case_distribution=np.array(payload["worst_case_distribution"], dtype=float),
        state_losses=np.array(payload["state_losses"], dtype=float),
        state_survivors=np.array(payload["state_survivors"], dtype=float),
        iterations=int(payload.get("iterations", 0)),
    )


def build_bundle(
    base: Path,
    zone_count: int = 12,
    center_count: int = 6,
    link_count: int = 5,
    rho: float = BASE_RHO,
    renovation_fraction: float = BASE_RENOVATION_FRACTION,
    retrofit_fraction: float = BASE_RETROFIT_FRACTION,
    capacity_fraction: float = BASE_CAPACITY_FRACTION,
) -> dict:
    zones = pd.read_csv(base / "turkey_zones_model_input.csv")
    centers = pd.read_csv(base / "turkey_centers_model_input.csv")
    destroyed = pd.read_csv(base / "hotosm_destroyed_buildings_with_grid.csv")
    zones = zones[zones["population"] > 0].copy()
    zones["at_risk"] = zones["population"] * zones["collapse_fraction"]
    zones = zones.sort_values("at_risk", ascending=False).head(zone_count).reset_index(drop=True)
    centers = centers.head(center_count).copy()
    nodes = pd.read_csv(base / "osm_road_nodes_bbox.csv", dtype={"node": str})
    edges = pd.read_csv(base / "osm_road_edges_bbox.csv", dtype={"tail": str, "head": str, "link_id": str, "osm_way_id": str})
    graph = run.road_graph_from_edges(edges)
    zones["node"] = run.nearest_road_nodes(zones, nodes, "centroid_lon", "centroid_lat")
    centers["node"] = run.nearest_road_nodes(centers, nodes, "longitude", "latitude")
    route_edge_counts, base_travel_times, path_link_ids = run.collect_shortest_path_data(graph, centers["node"], zones["node"])
    hazard_paths = run.ensure_hazard_rasters(base)
    links = run.candidate_links_from_routes(edges, nodes, route_edge_counts, destroyed, hazard_paths, max_links=link_count)
    states = generate_failure_states(links, max_failures=2, include_tail=True)

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
    budget_renovation = renovation_fraction * sum(zone.renovation_cost for zone in zone_objects)
    budget_retrofit = retrofit_fraction * sum(link.retrofit_cost for link in links)
    budget_capacity = capacity_fraction * sum(center.existing_capacity * center.capacity_unit_cost for center in center_objects)
    instance = DADInstance(
        zones=zone_objects,
        links=links,
        centers=center_objects,
        budget_renovation=budget_renovation,
        budget_retrofit=budget_retrofit,
        budget_capacity=budget_capacity,
        ambiguity_radius=rho,
        states=states,
        survival=SurvivalParams(a=0.95, b=-0.025, c=1.0, d=0.0),
        graph=graph,
    )
    instance.precomputed_travel_times = run.scenario_travel_times(states, base_travel_times, path_link_ids)
    return {"instance": instance, "zones": zones, "centers": centers, "links": links, "states": states}


def rebuild_with_budget(bundle: dict, renovation_fraction: float, retrofit_fraction: float, capacity_fraction: float, rho: float | None = None) -> DADInstance:
    instance: DADInstance = bundle["instance"]
    new_instance = DADInstance(
        zones=instance.zones,
        links=instance.links,
        centers=instance.centers,
        budget_renovation=renovation_fraction * sum(zone.renovation_cost for zone in instance.zones),
        budget_retrofit=retrofit_fraction * sum(link.retrofit_cost for link in instance.links),
        budget_capacity=capacity_fraction * sum(center.existing_capacity * center.capacity_unit_cost for center in instance.centers),
        ambiguity_radius=instance.ambiguity_radius if rho is None else rho,
        states=instance.states,
        survival=instance.survival,
        graph=instance.graph,
        precomputed_travel_times=instance.precomputed_travel_times,
    )
    return new_instance


def evaluate_given_plan(instance: DADInstance, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> float:
    nominal = nominal_probabilities(instance.links, instance.states, y)
    losses, _, _ = evaluate_plan_losses(instance, z, w)
    return worst_case_tv_distribution(
        nominal,
        losses,
        instance.ambiguity_radius,
        maximize=True,
        density_cap=instance.ambiguity_density_cap,
    ).value


def source_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Building damage labels", "Zenodo 18437501, This_study shapefiles", "Observed / derived", "Defines severe/destroyed buildings for q_rl"],
            ["Destroyed buildings", "HDX HOTOSM Turkey destroyed buildings", "Observed post-event", "Used as GIS-ready validation numerator"],
            ["Population", "WorldPop Turkey 2020 constrained UN-adjusted raster", "Estimated gridded population", "Aggregated to grid zones for P_rl"],
            ["Health facilities", "HDX HOTOSM Turkey health facilities", "Observed locations / proxy capacity", "Facility type maps to w_k^0 and lambda_k"],
            ["Road network", "OpenStreetMap via Overpass/Geofabrik-compatible schema", "Observed geospatial", "Defines L and baseline travel times"],
            ["Link failure probabilities", "Road class and damaged-corridor exposure", "Scenario-calibrated", "Constructed Phi_ij"],
            ["Costs and budgets", "Official unit-cost anchor plus scenario fractions", "Scenario-calibrated", "Used for C_rl, C_ij, lambda_k and budgets"],
        ],
        columns=["Model input", "Source", "Status", "Use"],
    )


def instance_summary_table(instance: DADInstance, bundle: dict) -> pd.DataFrame:
    zones = bundle["zones"]
    return pd.DataFrame(
        [
            ["Zones", len(instance.zones)],
            ["Emergency centers", len(instance.centers)],
            ["Candidate road links", len(instance.links)],
            ["Failure states", len(instance.states)],
            ["Total population in modeled zones", zones["population"].sum()],
            ["Baseline at-risk population D(0)", sum(zone.at_risk for zone in instance.zones)],
            ["Mean collapse fraction q_rl", zones["collapse_fraction"].mean()],
            ["Renovation budget B_Z", instance.budget_renovation],
            ["Road retrofit budget B_Y", instance.budget_retrofit],
            ["Capacity budget B_X", instance.budget_capacity],
            ["Ambiguity radius rho", instance.ambiguity_radius],
        ],
        columns=["Quantity", "Value"],
    )


def main_results_table(instance: DADInstance, no_investment: float, fixed_y0, incumbent) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["No investment", no_investment, 0.0, "z=w=y=0"],
            ["Renovation/capacity only", fixed_y0.objective, no_investment - fixed_y0.objective, "Exact fixed-y evaluation with y=0"],
            ["All sectors incumbent", incumbent.objective, no_investment - incumbent.objective, "Exact fixed-y evaluation of heuristic y"],
        ],
        columns=["Plan", "Worst-case expected deaths", "Reduction vs no investment", "Method"],
    )


def sector_ablation_table(bundle: dict, no_investment: float, cache: CheckpointStore, force: bool) -> pd.DataFrame:
    scenarios = [
        ("Building only", BASE_RENOVATION_FRACTION, 0.0, 0.0, False),
        ("Capacity only", 0.0, 0.0, BASE_CAPACITY_FRACTION, False),
        ("Building + capacity", BASE_RENOVATION_FRACTION, 0.0, BASE_CAPACITY_FRACTION, False),
        ("All sectors incumbent", BASE_RENOVATION_FRACTION, BASE_RETROFIT_FRACTION, BASE_CAPACITY_FRACTION, True),
    ]
    rows = []
    for name, beta_z, beta_y, beta_x, use_heuristic in scenarios:
        instance = rebuild_with_budget(bundle, beta_z, beta_y, beta_x)
        if use_heuristic:
            result = cached_heuristic(
                cache,
                f"sector_{name}_bz_{beta_z}_by_{beta_y}_bx_{beta_x}",
                instance,
                max_random=8,
                force=force,
            )
        else:
            result = cached_fixed_y(
                cache,
                f"sector_{name}_bz_{beta_z}_by_{beta_y}_bx_{beta_x}",
                instance,
                np.zeros(len(instance.links)),
                force=force,
            )
        rows.append([name, beta_z, beta_y, beta_x, result.objective, no_investment - result.objective])
    return pd.DataFrame(rows, columns=["Scenario", "beta_Z", "beta_Y", "beta_X", "Worst-case expected deaths", "Reduction vs no investment"])


def ambiguity_sensitivity_table(bundle: dict, cache: CheckpointStore, force: bool) -> pd.DataFrame:
    rows = []
    for rho in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        instance = rebuild_with_budget(bundle, BASE_RENOVATION_FRACTION, BASE_RETROFIT_FRACTION, BASE_CAPACITY_FRACTION, rho=rho)
        result = cached_heuristic(cache, f"ambiguity_rho_{rho}", instance, max_random=8, force=force)
        rows.append([rho, result.objective, np.dot(instance.retrofit_costs, result.y), result.y.sum()])
    return pd.DataFrame(rows, columns=["rho", "Worst-case expected deaths", "Retrofit budget used", "Sum retrofit ratios"])


def budget_sensitivity_table(bundle: dict, cache: CheckpointStore, force: bool) -> pd.DataFrame:
    rows = []
    for beta in [0.10, 0.20, 0.30]:
        instance = rebuild_with_budget(bundle, beta, beta, beta)
        result = cached_heuristic(cache, f"budget_beta_{beta}", instance, max_random=8, force=force)
        rows.append([beta, result.objective, instance.budget_renovation, instance.budget_retrofit, instance.budget_capacity])
    return pd.DataFrame(rows, columns=["Common budget fraction", "Worst-case expected deaths", "B_Z", "B_Y", "B_X"])


def policy_comparison_table(bundle: dict, no_investment: float, cache: CheckpointStore, force: bool) -> pd.DataFrame:
    policies = [
        ("No investment", 0.0, 0.0, 0.0),
        ("Building only", BASE_RENOVATION_FRACTION, 0.0, 0.0),
        ("Road only", 0.0, BASE_RETROFIT_FRACTION, 0.0),
        ("Capacity only", 0.0, 0.0, BASE_CAPACITY_FRACTION),
        ("Building + road", BASE_RENOVATION_FRACTION, BASE_RETROFIT_FRACTION, 0.0),
        ("Building + capacity", BASE_RENOVATION_FRACTION, 0.0, BASE_CAPACITY_FRACTION),
        ("Road + capacity", 0.0, BASE_RETROFIT_FRACTION, BASE_CAPACITY_FRACTION),
        ("All sectors", BASE_RENOVATION_FRACTION, BASE_RETROFIT_FRACTION, BASE_CAPACITY_FRACTION),
    ]
    rows = []
    for name, beta_z, beta_y, beta_x in policies:
        instance = rebuild_with_budget(bundle, beta_z, beta_y, beta_x)
        result = cached_policy_result(cache, f"policy_{name}_bz_{beta_z}_by_{beta_y}_bx_{beta_x}", instance, beta_y, force)
        rows.append([name, beta_z, beta_y, beta_x, result.objective, no_investment - result.objective])
    return pd.DataFrame(
        rows,
        columns=["Policy", "beta_Z", "beta_Y", "beta_X", "Worst-case expected deaths", "Reduction vs no investment"],
    )


def budget_design_table(bundle: dict, no_investment: float, cache: CheckpointStore, force: bool) -> pd.DataFrame:
    designs = [
        ("Balanced low", 0.10, 0.10, 0.10),
        ("Balanced medium", 0.20, 0.20, 0.20),
        ("Balanced high", 0.30, 0.30, 0.30),
        ("Building-heavy", 0.30, 0.10, 0.10),
        ("Road-heavy", 0.10, 0.30, 0.10),
        ("Capacity-heavy", 0.10, 0.10, 0.30),
        ("No road", 0.20, 0.00, 0.20),
        ("No building", 0.00, 0.20, 0.20),
        ("No capacity", 0.20, 0.20, 0.00),
        ("Building/capacity-heavy", 0.30, 0.10, 0.30),
        ("Road/capacity-heavy", 0.10, 0.30, 0.30),
        ("Building/road-heavy", 0.30, 0.30, 0.10),
    ]
    rows = []
    for name, beta_z, beta_y, beta_x in designs:
        instance = rebuild_with_budget(bundle, beta_z, beta_y, beta_x)
        result = cached_policy_result(cache, f"budget_design_{name}_bz_{beta_z}_by_{beta_y}_bx_{beta_x}", instance, beta_y, force)
        rows.append([name, beta_z, beta_y, beta_x, result.objective, no_investment - result.objective])
    return pd.DataFrame(
        rows,
        columns=["Design", "beta_Z", "beta_Y", "beta_X", "Worst-case expected deaths", "Reduction vs no investment"],
    )


def candidate_link_sensitivity_table(base: Path, cache: CheckpointStore, force: bool) -> pd.DataFrame:
    rows = []
    for link_count in [3, 5, 8]:
        bundle = build_bundle(base, link_count=link_count)
        instance = bundle["instance"]
        result = cached_heuristic(cache, f"candidate_links_{link_count}", instance, max_random=8, force=force)
        fixed_y0 = cached_fixed_y(cache, f"candidate_links_{link_count}_fixed_y0", instance, np.zeros(len(instance.links)), force=force)
        rows.append(
            [
                link_count,
                len(instance.states),
                fixed_y0.objective,
                result.objective,
                fixed_y0.objective - result.objective,
                np.dot(instance.retrofit_costs, result.y),
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "candidate_links",
            "states",
            "No-retrofit objective",
            "Best incumbent objective",
            "Road-retrofit gain",
            "Retrofit budget used",
        ],
    )


def cached_policy_result(cache: CheckpointStore, key: str, instance: DADInstance, beta_y: float, force: bool):
    if beta_y > 0:
        return cached_heuristic(cache, key, instance, max_random=8, force=force)
    return cached_fixed_y(cache, key, instance, np.zeros(len(instance.links)), force=force)


def zone_table(zones: pd.DataFrame, incumbent) -> pd.DataFrame:
    output = zones.copy()
    output["renovation_ratio_z"] = incumbent.z
    output["at_risk_population"] = output["population"] * output["collapse_fraction"]
    return output[
        [
            "zone_id",
            "population",
            "collapse_fraction",
            "at_risk_population",
            "total_buildings",
            "severe_buildings",
            "destroyed_buildings",
            "renovation_ratio_z",
        ]
    ].sort_values("at_risk_population", ascending=False)


def link_table(instance: DADInstance, incumbent) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                link.id,
                link.baseline_failure_probability,
                link.retrofit_cost,
                link.travel_time,
                incumbent.y[index],
            ]
            for index, link in enumerate(instance.links)
        ],
        columns=["link_id", "Phi_ij", "C_ij", "travel_time_min", "retrofit_ratio_y"],
    )


def center_table(centers: pd.DataFrame, incumbent) -> pd.DataFrame:
    output = centers.copy()
    output["capacity_added_w"] = incumbent.w
    return output[["center_id", "name", "facility_type", "existing_capacity", "capacity_unit_cost", "capacity_added_w"]]


def state_table(instance: DADInstance, incumbent) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "state_id": [state.id for state in instance.states],
            "failed_links": [len(state.failed_links) if not state.is_tail else "tail" for state in instance.states],
            "nominal_probability": incumbent.nominal_distribution,
            "worst_case_probability": incumbent.worst_case_distribution,
            "loss": incumbent.state_losses,
            "survivors": incumbent.state_survivors,
        }
    )
    output["probability_shift"] = output["worst_case_probability"] - output["nominal_probability"]
    return output.sort_values(["worst_case_probability", "loss"], ascending=[False, False]).head(12)


def method_note(out: Path) -> None:
    note = (
        "The tables report feasible incumbent solutions. For each tested retrofit plan y, "
        "the renovation/capacity subproblem is solved by the exact fixed-y cutting-plane "
        "evaluation. The continuous global SBB certificate is not reported for the Turkey "
        "instance because continuous certification is not part of the Turkey empirical design; therefore results "
        "should be described as scenario-calibrated incumbent solutions, not proven global optima."
    )
    atomic_write_text(out / "method_note.txt", note)


if __name__ == "__main__":
    main()
