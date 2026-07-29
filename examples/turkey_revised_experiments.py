from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ejor_dad import AidCenter, DADInstance, Link, SurvivalParams, Zone, evaluate_fixed_y, generate_failure_states
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text
from ejor_dad.fixed_y import evaluate_plan_losses
from ejor_dad.sbb import root_node, solve_global_sbb, solve_node_with_cut_separation, valid_objective_lower_bound
from ejor_dad.states import nominal_probabilities
from ejor_dad.tv import worst_case_tv_distribution

import turkey_run_model as run


SEED = 20260707
RHO_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
CACHE_VERSION = "revised-v1"


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    link_count: int
    renovation_fraction: float
    retrofit_fraction: float
    capacity_fraction: float
    q_scale: float
    q_floor: float
    capacity_scale: float
    phi_scale: float
    phi_add: float
    phi_cap: float
    failure_penalty: float
    survival_b: float
    notes: str


REGIMES = {
    "exposure_dominant": RegimeConfig(
        name="exposure_dominant",
        link_count=5,
        renovation_fraction=0.20,
        retrofit_fraction=0.25,
        capacity_fraction=0.20,
        q_scale=1.00,
        q_floor=0.0,
        capacity_scale=1.00,
        phi_scale=1.00,
        phi_add=0.00,
        phi_cap=0.55,
        failure_penalty=0.75,
        survival_b=-0.025,
        notes="Current empirical calibration; exposure is expected to dominate.",
    ),
    "access_constrained": RegimeConfig(
        name="access_constrained",
        link_count=6,
        renovation_fraction=0.03,
        retrofit_fraction=0.60,
        capacity_fraction=0.00,
        q_scale=0.70,
        q_floor=0.10,
        capacity_scale=100.00,
        phi_scale=1.35,
        phi_add=0.10,
        phi_cap=0.75,
        failure_penalty=12.00,
        survival_b=-0.140,
        notes="Hazard-informed access stress test; road disruption should materially affect survival.",
    ),
    "capacity_constrained": RegimeConfig(
        name="capacity_constrained",
        link_count=5,
        renovation_fraction=0.03,
        retrofit_fraction=0.05,
        capacity_fraction=8.00,
        q_scale=0.55,
        q_floor=0.08,
        capacity_scale=0.03,
        phi_scale=0.45,
        phi_add=0.00,
        phi_cap=0.30,
        failure_penalty=0.20,
        survival_b=-0.012,
        notes="Capacity stress test; initial emergency capacity is deliberately tight.",
    ),
    "mechanism_active": RegimeConfig(
        name="mechanism_active",
        link_count=5,
        renovation_fraction=0.06,
        retrofit_fraction=0.25,
        capacity_fraction=0.80,
        q_scale=0.72,
        q_floor=0.08,
        capacity_scale=3.00,
        phi_scale=1.00,
        phi_add=0.00,
        phi_cap=0.70,
        failure_penalty=1.00,
        survival_b=-0.110,
        notes="Heterogeneous no-tail stress test designed to separate nominal-frequency and worst-state hedging incentives.",
    ),
}

POLICY_REGIMES = ["exposure_dominant", "access_constrained", "capacity_constrained"]


@dataclass
class DataContext:
    base: Path
    output: Path
    zones: pd.DataFrame
    centers: pd.DataFrame
    destroyed: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    route_edge_counts: Counter[tuple[str, str]]
    base_travel_times: np.ndarray
    path_link_ids: list[list[set[str]]]
    hazard_paths: dict[str, Path]


@dataclass
class InstanceBundle:
    experiment_id: str
    regime: str
    state_mode: str
    instance: DADInstance
    zones: pd.DataFrame
    centers: pd.DataFrame
    link_meta: pd.DataFrame
    config: dict[str, Any]


def main() -> None:
    args = parse_args()
    base = Path("data_work/turkey")
    out = base / "revised_experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    log_path = out / "logs" / f"revised_experiments_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_path)
    command = " ".join([sys.executable, *sys.argv])
    atomic_write_text(out / "command.txt", command)
    write_status(out, "running", "startup", "Revised experiments started.", log_path)

    started = time.time()
    cache = CheckpointStore(out / "checkpoints")
    runtime_rows: list[dict[str, Any]] = []
    manifest_notes: list[str] = []

    try:
        write_code_audit(out)
        write_master_config(out, args)
        context = load_context(base, out, force_routes=args.force_routes)
        blocks = set(args.blocks or ["all"])

        if "all" in blocks or "mechanism" in blocks:
            block_start = time.time()
            write_status(out, "running", "mechanism", "Running full-state no-tail mechanism experiments.", log_path)
            mechanism_tables(context, out, cache, force=args.force)
            runtime_rows.append(runtime_row("mechanism", block_start))

        if "all" in blocks or "dda" in blocks:
            block_start = time.time()
            write_status(out, "running", "dda", "Running fixed-center vs decision-dependent DRO comparison.", log_path)
            dda_value_comparison(context, out, cache, force=args.force)
            runtime_rows.append(runtime_row("dda", block_start))

        if "all" in blocks or "sbb" in blocks:
            block_start = time.time()
            write_status(out, "running", "sbb", "Running small SBB certification attempt.", log_path)
            sbb_certificate(
                context,
                out,
                cache,
                force=args.force,
                time_limit_sec=args.sbb_time_limit_sec,
                m2_max_nodes=args.sbb_m2_max_nodes,
                m5_max_nodes=args.sbb_m5_max_nodes,
            )
            runtime_rows.append(runtime_row("sbb", block_start))

        if "all" in blocks or "sbbdiag" in blocks:
            block_start = time.time()
            write_status(out, "running", "sbbdiag", "Running five-link SBB root-bound diagnostics.", log_path)
            sbb_root_bound_diagnostics(context, out)
            runtime_rows.append(runtime_row("sbbdiag", block_start))

        if "all" in blocks or "ddacert" in blocks:
            block_start = time.time()
            write_status(out, "running", "ddacert", "Running certified small-scale DDA rho sweep.", log_path)
            certified_dda_sweep(
                context,
                out,
                cache,
                force=args.force,
                time_limit_sec=args.sbb_time_limit_sec,
                link_counts=args.ddacert_link_counts,
                m2_max_nodes=args.ddacert_m2_max_nodes,
                m3_max_nodes=args.ddacert_m3_max_nodes,
            )
            runtime_rows.append(runtime_row("ddacert", block_start))

        if "all" in blocks or "discrete" in blocks:
            block_start = time.time()
            write_status(out, "running", "discrete", "Running m=5 discretized enumeration and DDA sweep.", log_path)
            discretized_m5_enumeration(context, out, cache, force=args.force, log_path=log_path)
            runtime_rows.append(runtime_row("discrete", block_start))

        if "all" in blocks or "budgetsens" in blocks:
            block_start = time.time()
            write_status(out, "running", "budgetsens", "Running discretized m=5 road-budget sensitivity.", log_path)
            discretized_m5_budget_sensitivity(context, out, cache, force=args.force, log_path=log_path)
            runtime_rows.append(runtime_row("budgetsens", block_start))

        if "all" in blocks or "regimes" in blocks:
            block_start = time.time()
            write_status(out, "running", "regimes", "Running bottleneck-regime policy comparisons.", log_path)
            regime_tables(context, out, cache, force=args.force)
            runtime_rows.append(runtime_row("regimes", block_start))

        if "all" in blocks or "candidate" in blocks:
            block_start = time.time()
            write_status(out, "running", "candidate", "Running revised candidate-link sensitivity.", log_path)
            candidate_link_sensitivity_revised(context, out, cache, force=args.force)
            runtime_rows.append(runtime_row("candidate", block_start))

        if "all" in blocks or "figures" in blocks:
            block_start = time.time()
            write_status(out, "running", "figures", "Generating revised figures.", log_path)
            make_figures(out)
            runtime_rows.append(runtime_row("figures", block_start))

        write_readme(out)
        write_manifest(out, started, log_path, manifest_notes)
        atomic_write_dataframe(pd.DataFrame(runtime_rows), out / "runtime_summary.csv")
        write_status(out, "completed", "complete", "Revised experiments completed.", log_path, exit_code=0)
        logging.info("Revised experiments completed in %.2f seconds", time.time() - started)
    except Exception as exc:
        logging.exception("Revised experiments failed.")
        write_status(out, "failed", "error", str(exc), log_path, exit_code=1)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised Turkey DR-DAD computational experiments.")
    parser.add_argument("--force", action="store_true", help="Recompute experiment checkpoints.")
    parser.add_argument("--force-routes", action="store_true", help="Recompute OSM route cache.")
    parser.add_argument("--blocks", nargs="*", help="Subset of blocks: mechanism dda sbb sbbdiag ddacert discrete budgetsens regimes candidate figures.")
    parser.add_argument("--sbb-time-limit-sec", type=float, default=900.0)
    parser.add_argument("--sbb-m2-max-nodes", type=int, default=1000)
    parser.add_argument("--sbb-m5-max-nodes", type=int, default=120)
    parser.add_argument("--ddacert-link-counts", type=int, nargs="*", default=[2, 3])
    parser.add_argument("--ddacert-m2-max-nodes", type=int, default=1500)
    parser.add_argument("--ddacert-m3-max-nodes", type=int, default=5000)
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def runtime_row(block: str, started: float) -> dict[str, Any]:
    return {"block": block, "runtime_sec": time.time() - started, "completed_at_epoch": time.time()}


def write_status(
    out: Path,
    status: str,
    block: str,
    message: str,
    log_path: Path,
    exit_code: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "block": block,
        "message": message,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
        "log_path": str(log_path.resolve()),
        "output_dir": str(out.resolve()),
        "checkpoint_dir": str((out / "checkpoints").resolve()),
    }
    if exit_code is not None:
        payload["exit_code"] = exit_code
    if extra:
        payload.update(extra)
    atomic_write_text(out / "run_status.json", json.dumps(payload, indent=2))


def write_master_config(out: Path, args: argparse.Namespace) -> None:
    config = {
        "seed": SEED,
        "rho_values": RHO_VALUES,
        "cache_version": CACHE_VERSION,
        "args": vars(args),
        "regimes": {name: config.__dict__ for name, config in REGIMES.items()},
    }
    atomic_write_text(out / "configs" / "revised_experiments_config.json", json.dumps(config, indent=2))


def write_experiment_config(out: Path, experiment_id: str, config: dict[str, Any]) -> None:
    config = dict(config)
    config.setdefault("seed", SEED)
    atomic_write_text(out / "configs" / f"{experiment_id}.json", json.dumps(config, indent=2, default=str))


def load_context(base: Path, out: Path, force_routes: bool = False) -> DataContext:
    logging.info("Loading Turkey data context.")
    zones = pd.read_csv(base / "turkey_zones_model_input.csv")
    centers = pd.read_csv(base / "turkey_centers_model_input.csv")
    destroyed = pd.read_csv(base / "hotosm_destroyed_buildings_with_grid.csv")
    nodes = pd.read_csv(base / "osm_road_nodes_bbox.csv", dtype={"node": str})
    edges = pd.read_csv(
        base / "osm_road_edges_bbox.csv",
        dtype={"tail": str, "head": str, "link_id": str, "osm_way_id": str},
    )
    hazard_paths = run.ensure_hazard_rasters(base)

    zones = zones[zones["population"] > 0].copy()
    zones["at_risk"] = zones["population"] * zones["collapse_fraction"]
    zones = zones.sort_values("at_risk", ascending=False).head(12).reset_index(drop=True)
    centers = centers.head(6).copy()

    route_cache = out / "route_cache_z12_c6.pkl"
    if route_cache.exists() and not force_routes:
        logging.info("Loading cached OSM routes from %s", route_cache)
        with route_cache.open("rb") as handle:
            cached = pickle.load(handle)
        zones = cached["zones"]
        centers = cached["centers"]
        route_edge_counts = Counter({tuple(key): value for key, value in cached["route_edge_counts"]})
        base_travel_times = cached["base_travel_times"]
        path_link_ids = cached["path_link_ids"]
    else:
        logging.info("Building OSM graph and center-zone route cache.")
        graph = run.road_graph_from_edges(edges)
        zones["node"] = run.nearest_road_nodes(zones, nodes, "centroid_lon", "centroid_lat")
        centers["node"] = run.nearest_road_nodes(centers, nodes, "longitude", "latitude")
        route_edge_counts, base_travel_times, path_link_ids = run.collect_shortest_path_data(
            graph,
            centers["node"],
            zones["node"],
        )
        payload = {
            "zones": zones,
            "centers": centers,
            "route_edge_counts": list(route_edge_counts.items()),
            "base_travel_times": base_travel_times,
            "path_link_ids": path_link_ids,
        }
        temp_path = route_cache.with_suffix(".tmp")
        with temp_path.open("wb") as handle:
            pickle.dump(payload, handle)
        os.replace(temp_path, route_cache)

    logging.info("Loaded context: %d zones, %d centers, %d OSM route edges.", len(zones), len(centers), len(route_edge_counts))
    return DataContext(base, out, zones, centers, destroyed, nodes, edges, route_edge_counts, base_travel_times, path_link_ids, hazard_paths)


def build_bundle(
    context: DataContext,
    regime_name: str,
    state_mode: str,
    rho: float,
    renovation_fraction: float | None = None,
    retrofit_fraction: float | None = None,
    capacity_fraction: float | None = None,
    link_count: int | None = None,
    experiment_id: str | None = None,
) -> InstanceBundle:
    regime = REGIMES[regime_name]
    link_count = regime.link_count if link_count is None else link_count
    renovation_fraction = regime.renovation_fraction if renovation_fraction is None else renovation_fraction
    retrofit_fraction = regime.retrofit_fraction if retrofit_fraction is None else retrofit_fraction
    capacity_fraction = regime.capacity_fraction if capacity_fraction is None else capacity_fraction

    zones = context.zones.copy()
    centers = context.centers.copy()
    zones["collapse_fraction"] = np.clip(zones["collapse_fraction"] * regime.q_scale, regime.q_floor, 1.0)
    centers["existing_capacity"] = centers["existing_capacity"] * regime.capacity_scale

    raw_links, link_meta = fast_candidate_links(context, max_links=link_count)
    links = transform_links_for_regime(raw_links, regime_name, regime)
    link_meta = link_meta[link_meta["link_id"].isin([link.id for link in links])].copy()
    link_meta["regime_phi"] = [next(link.baseline_failure_probability for link in links if link.id == link_id) for link_id in link_meta["link_id"]]

    if state_mode == "full_state_no_tail":
        states = generate_failure_states(links, max_failures=None, include_tail=False)
    elif state_mode == "reduced_tail":
        states = generate_failure_states(links, max_failures=2, include_tail=True)
    else:
        raise ValueError(f"Unknown state_mode: {state_mode}")

    if regime_name == "mechanism_active":
        link_penalties = mechanism_link_penalties(links)
        precomputed = weighted_scenario_travel_times(states, context.base_travel_times, context.path_link_ids, link_penalties)
        config_link_penalties = link_penalties
    else:
        precomputed = run.scenario_travel_times(
            states,
            context.base_travel_times,
            context.path_link_ids,
            failure_penalty=regime.failure_penalty,
        )
        config_link_penalties = None
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
    instance = DADInstance(
        zones=zone_objects,
        links=links,
        centers=center_objects,
        budget_renovation=renovation_fraction * sum(zone.renovation_cost for zone in zone_objects),
        budget_retrofit=retrofit_fraction * sum(link.retrofit_cost for link in links),
        budget_capacity=capacity_fraction * sum(center.existing_capacity * center.capacity_unit_cost for center in center_objects),
        ambiguity_radius=rho,
        states=states,
        survival=SurvivalParams(a=0.95, b=regime.survival_b, c=1.0, d=0.0),
        precomputed_travel_times=precomputed,
    )
    config = {
        "experiment_id": experiment_id,
        "regime": regime_name,
        "state_mode": state_mode,
        "rho": rho,
        "link_count": link_count,
        "renovation_fraction": renovation_fraction,
        "retrofit_fraction": retrofit_fraction,
        "capacity_fraction": capacity_fraction,
        "regime_config": regime.__dict__,
        "link_penalties": config_link_penalties,
    }
    return InstanceBundle(experiment_id or f"{regime_name}_{state_mode}_m{link_count}_rho{rho}", regime_name, state_mode, instance, zones, centers, link_meta, config)


def transform_links_for_regime(raw_links: list[Link], regime_name: str, regime: RegimeConfig) -> list[Link]:
    links = []
    for index, link in enumerate(raw_links):
        phi = float(np.clip(link.baseline_failure_probability * regime.phi_scale + regime.phi_add, 0.005, regime.phi_cap))
        cost = link.retrofit_cost
        if regime_name == "mechanism_active":
            phi_targets = [0.66, 0.18, 0.48, 0.28, 0.38, 0.32]
            phi = phi_targets[index] if index < len(phi_targets) else phi
            if index == 0:
                cost = max(cost, 0.80)
            elif index in {1, 3, 4}:
                cost = max(0.12, min(cost, 0.22))
        links.append(
            Link(
                id=link.id,
                tail=link.tail,
                head=link.head,
                baseline_failure_probability=phi,
                retrofit_cost=cost,
                travel_time=link.travel_time,
            )
        )
    return links


def mechanism_link_penalties(links: list[Link]) -> dict[str, float]:
    penalties = {}
    for index, link in enumerate(links):
        if index == 0:
            penalties[link.id] = 2.0
        elif index == 1:
            penalties[link.id] = 20.0
        elif index == 2:
            penalties[link.id] = 3.0
        else:
            penalties[link.id] = 1.8
    return penalties


def weighted_scenario_travel_times(
    states: list[Any],
    base_travel_times: np.ndarray,
    path_link_ids: list[list[set[str]]],
    link_penalties: dict[str, float],
) -> dict[str, np.ndarray]:
    matrices = {}
    for state in states:
        if state.is_tail:
            continue
        failed = set(state.failed_links)
        matrix = base_travel_times.copy()
        for center_index, row in enumerate(path_link_ids):
            for zone_index, links_on_path in enumerate(row):
                if not np.isfinite(matrix[center_index, zone_index]):
                    continue
                multiplier = 1.0
                for failed_link in failed & links_on_path:
                    multiplier *= 1.0 + link_penalties.get(failed_link, 1.0)
                matrix[center_index, zone_index] *= multiplier
        matrices[state.id] = matrix
    return matrices


def fast_candidate_links(context: DataContext, max_links: int) -> tuple[list[Link], pd.DataFrame]:
    edge_lookup = context.edges.set_index(["tail", "head"], drop=False)
    candidate_records: list[dict[str, Any]] = []
    node_lookup = context.nodes.set_index("node")[["lon", "lat"]].to_dict("index")

    for (tail, head), count in context.route_edge_counts.most_common(max_links * 30):
        if (tail, head) not in edge_lookup.index or tail not in node_lookup or head not in node_lookup:
            continue
        row = edge_lookup.loc[(tail, head)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        midpoint_lon = 0.5 * (float(node_lookup[tail]["lon"]) + float(node_lookup[head]["lon"]))
        midpoint_lat = 0.5 * (float(node_lookup[tail]["lat"]) + float(node_lookup[head]["lat"]))
        candidate_records.append(
            {
                "link_id": str(row["link_id"]),
                "tail": str(row["tail"]),
                "head": str(row["head"]),
                "route_count": int(count),
                "highway": str(row["highway"]).lower(),
                "bridge": str(row["bridge"]).lower() in {"true", "yes", "1"},
                "distance_km": float(row["distance_km"]),
                "travel_time": float(row["travel_time"]),
                "longitude": midpoint_lon,
                "latitude": midpoint_lat,
            }
        )
    frame = pd.DataFrame(candidate_records).drop_duplicates("link_id")
    if frame.empty:
        raise RuntimeError("No candidate route links were found.")

    available_hazards = {name: path for name, path in context.hazard_paths.items() if Path(path).exists()}
    if available_hazards:
        frame = run.sample_geotiff_to_dataframe(frame, available_hazards, prefix="hazard")
    for column in ("hazard_pgv", "hazard_fault", "hazard_epicenter"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["pgv_norm"] = normalize_series(frame["hazard_pgv"])
    frame["fault_proximity_norm"] = 1.0 - normalize_series(frame["hazard_fault"])
    frame["epicenter_proximity_norm"] = 1.0 - normalize_series(frame["hazard_epicenter"])
    frame["near_destroyed"] = fast_destroyed_proximity(frame, context.destroyed, threshold_km=0.75)
    frame["critical_road"] = frame["highway"].isin(["motorway", "trunk", "primary", "secondary", "tertiary"]).astype(float)
    frame["phi"] = np.minimum(
        0.55,
        0.02
        + 0.20 * frame["pgv_norm"]
        + 0.10 * frame["fault_proximity_norm"]
        + 0.05 * frame["epicenter_proximity_norm"]
        + 0.10 * frame["near_destroyed"]
        + 0.05 * frame["critical_road"]
        + 0.06 * frame["bridge"].astype(float),
    )
    class_multiplier = frame["highway"].map({"motorway": 3.0, "trunk": 2.5, "primary": 2.0, "secondary": 1.5, "tertiary": 1.2}).fillna(1.0)
    frame["retrofit_cost"] = np.maximum(frame["distance_km"], 0.05) * class_multiplier * (1.0 + 0.5 * frame["pgv_norm"])
    frame["selection_score"] = frame["route_count"] * 100.0 + frame["phi"] * 10.0 + frame["near_destroyed"] * 3.0
    frame = frame.sort_values(["route_count", "selection_score", "phi"], ascending=[False, False, False]).head(max_links).copy()
    links = [
        Link(
            id=str(row.link_id),
            tail=str(row.tail),
            head=str(row.head),
            baseline_failure_probability=float(row.phi),
            retrofit_cost=float(max(row.retrofit_cost, 1e-4)),
            travel_time=float(row.travel_time),
        )
        for row in frame.itertuples(index=False)
    ]
    return links, frame.reset_index(drop=True)


def normalize_series(values: Iterable[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if series.notna().sum() == 0:
        return np.zeros(len(series), dtype=float)
    low = float(series.min())
    high = float(series.max())
    if math.isclose(low, high):
        return np.zeros(len(series), dtype=float)
    return ((series.fillna(low) - low) / (high - low)).to_numpy(dtype=float)


def fast_destroyed_proximity(frame: pd.DataFrame, destroyed: pd.DataFrame, threshold_km: float) -> np.ndarray:
    coords = destroyed[["longitude", "latitude"]].dropna().to_numpy(dtype=float)
    if len(coords) == 0:
        return np.zeros(len(frame), dtype=float)
    threshold_deg = threshold_km / 111.0
    output = []
    destroyed_lon = coords[:, 0]
    destroyed_lat = coords[:, 1]
    for row in frame.itertuples(index=False):
        lon = float(row.longitude)
        lat = float(row.latitude)
        mask = (np.abs(destroyed_lon - lon) <= threshold_deg) & (np.abs(destroyed_lat - lat) <= threshold_deg)
        if not mask.any():
            output.append(0.0)
            continue
        local = coords[mask]
        x = (local[:, 0] - lon) * np.cos(np.radians(lat))
        y = local[:, 1] - lat
        output.append(float(np.any(np.sqrt(x * x + y * y) * 111.0 <= threshold_km)))
    return np.array(output, dtype=float)


def mechanism_tables(context: DataContext, out: Path, cache: CheckpointStore, force: bool) -> None:
    summary_rows = []
    for link_count in [5, 6]:
        experiment_id = f"mechanism_full_no_tail_m{link_count}"
        bundle = build_bundle(context, "exposure_dominant", "full_state_no_tail", rho=0.10, link_count=link_count, experiment_id=experiment_id)
        write_experiment_config(out, experiment_id, bundle.config)
        instance = bundle.instance
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "state_mode": "full_state_no_tail",
                "num_candidate_links": len(instance.links),
                "num_states": len(instance.states),
                "num_zones": len(instance.zones),
                "num_centers": len(instance.centers),
                "no_tail_used": not any(state.is_tail for state in instance.states),
                "total_population": float(sum(zone.population for zone in instance.zones)),
                "D0": float(instance.d_max),
                "mean_q": float(np.mean([zone.collapse_fraction for zone in instance.zones])),
                "max_q": float(np.max([zone.collapse_fraction for zone in instance.zones])),
                "min_q": float(np.min([zone.collapse_fraction for zone in instance.zones])),
                "notes": "All failure states are explicitly enumerated; no artificial tail state is used.",
            }
        )
    write_table(pd.DataFrame(summary_rows), out, "table_mechanism_full_state_summary")

    rho_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    inert_by_regime: dict[str, bool] = {}
    for mechanism_regime in ["exposure_dominant", "mechanism_active"]:
        base_bundle = build_bundle(
            context,
            mechanism_regime,
            "full_state_no_tail",
            rho=0.10,
            link_count=5,
            experiment_id=f"rho_reopt_base_{mechanism_regime}",
        )
        rows, shifts, inert = rho_reoptimized_tables(base_bundle, out, cache, force=force, write=False)
        rho_rows.extend(rows)
        shift_rows.extend(shifts)
        inert_by_regime[mechanism_regime] = inert
    write_table(pd.DataFrame(rho_rows), out, "table_rho_reoptimized_y")
    write_table(pd.DataFrame(shift_rows), out, "table_rho_probability_shift")
    atomic_write_text(
        out / "mechanism_flag.json",
        json.dumps(
            {
                "mechanism_inert": not any(not value for value in inert_by_regime.values()),
                "mechanism_inert_by_regime": inert_by_regime,
            },
            indent=2,
        ),
    )
    base_bundle = build_bundle(context, "exposure_dominant", "full_state_no_tail", rho=0.10, link_count=5, experiment_id="rho_reopt_base")
    reachability_override_check(base_bundle, out)


def rho_reoptimized_tables(
    base_bundle: InstanceBundle,
    out: Path,
    cache: CheckpointStore,
    force: bool,
    write: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    rows = []
    shift_rows = []
    previous_y: np.ndarray | None = None
    y_changed = False
    for rho in RHO_VALUES:
        experiment_id = f"rho_reoptimized_no_tail_{base_bundle.regime}_rho{rho:.2f}"
        bundle = build_bundle_from_existing(base_bundle, rho=rho, experiment_id=experiment_id)
        write_experiment_config(out, experiment_id, bundle.config)
        start = time.time()
        result = optimize_y_by_search(cache, experiment_id, bundle.instance, force=force, max_random=8)
        runtime = time.time() - start
        if previous_y is not None and not np.allclose(result.y, previous_y, atol=1e-5):
            y_changed = True
        previous_y = result.y.copy()
        nominal_objective = float(np.dot(result.nominal_distribution, result.state_losses))
        targets = target_states(bundle.instance, result)
        rows.append(
            {
                "experiment_id": experiment_id,
                "mechanism_regime": base_bundle.regime,
                "rho": rho,
                "objective": float(result.objective),
                "nominal_objective": nominal_objective,
                "y_budget_used": float(np.dot(bundle.instance.retrofit_costs, result.y)),
                "sum_y": float(result.y.sum()),
                "selected_y_json": json.dumps(result.y.tolist()),
                "selected_links_json": json.dumps(selected_link_payload(bundle.instance, result.y)),
                "worst_case_target_states_json": json.dumps(targets["target_states"]),
                "target_is_tail": bool(targets["target_is_tail"]),
                "runtime_sec": runtime,
                "solver_status": "ok",
            }
        )
        shift_rows.extend(probability_shift_rows(bundle.instance, result, experiment_id, rho))

    if not y_changed:
        logging.warning("Mechanism check for %s: selected y did not change across rho values; mechanism_inert=true.", base_bundle.regime)
    if write:
        write_table(pd.DataFrame(rows), out, "table_rho_reoptimized_y")
        write_table(pd.DataFrame(shift_rows), out, "table_rho_probability_shift")
        atomic_write_text(out / "mechanism_flag.json", json.dumps({"mechanism_inert": not y_changed}, indent=2))
    return rows, shift_rows, not y_changed


def build_bundle_from_existing(base_bundle: InstanceBundle, rho: float, experiment_id: str) -> InstanceBundle:
    old = base_bundle.instance
    instance = DADInstance(
        zones=old.zones,
        links=old.links,
        centers=old.centers,
        budget_renovation=old.budget_renovation,
        budget_retrofit=old.budget_retrofit,
        budget_capacity=old.budget_capacity,
        ambiguity_radius=rho,
        states=old.states,
        survival=old.survival,
        precomputed_travel_times=old.precomputed_travel_times,
    )
    config = dict(base_bundle.config)
    config["rho"] = rho
    config["experiment_id"] = experiment_id
    return InstanceBundle(experiment_id, base_bundle.regime, base_bundle.state_mode, instance, base_bundle.zones, base_bundle.centers, base_bundle.link_meta, config)


def optimize_y_by_search(
    cache: CheckpointStore,
    key_prefix: str,
    instance: DADInstance,
    force: bool,
    max_random: int = 8,
    nominal_override: np.ndarray | None = None,
) -> SimpleNamespace:
    rng = np.random.default_rng(SEED + stable_int(key_prefix))
    plans = candidate_y_plans(instance, rng, max_random=max_random)
    best = None
    for plan_index, plan in enumerate(plans):
        key = versioned_key(f"{key_prefix}__plan_{plan_index}_{hash_array(plan)}")
        payload = cache.get_or_compute(
            key,
            lambda plan=plan: fixed_result_payload(
                evaluate_fixed_y(
                    instance,
                    plan,
                    epsilon=1e-5,
                    max_iterations=120,
                    nominal_distribution_override=nominal_override,
                )
            ),
            force=force,
        )
        result = payload_to_result(payload)
        if best is None or result.objective < best.objective - 1e-7:
            best = result
    if best is None:
        raise RuntimeError("No feasible y plan was evaluated.")
    return best


def candidate_y_plans(instance: DADInstance, rng: np.random.Generator, max_random: int) -> list[np.ndarray]:
    costs = instance.retrofit_costs
    phi = instance.failure_probabilities
    budget = instance.budget_retrofit
    num_links = len(costs)
    plans: list[np.ndarray] = [np.zeros(num_links, dtype=float)]

    for link_index in range(num_links):
        y = np.zeros(num_links, dtype=float)
        y[link_index] = min(1.0, budget / costs[link_index]) if costs[link_index] > 0 else 0.0
        plans.append(y)

    orders = [
        np.argsort(costs),
        np.argsort(-phi),
        np.argsort(-(phi / np.maximum(costs, 1e-9))),
        np.arange(num_links),
        np.arange(num_links)[::-1],
    ]
    for order in orders:
        plans.append(fill_budget(order, costs, budget))

    for size in [2, 3]:
        for combo in combinations_limited(range(num_links), size, limit=12):
            plans.append(fill_budget(np.array(combo), costs, budget))

    for _ in range(max_random):
        weights = rng.dirichlet(np.ones(num_links))
        y = np.minimum(1.0, (budget * weights) / np.maximum(costs, 1e-9))
        plans.append(y)

    unique: list[np.ndarray] = []
    seen = set()
    for plan in plans:
        plan = np.clip(plan, 0.0, 1.0)
        if float(np.dot(costs, plan)) > budget + 1e-7:
            scale = budget / float(np.dot(costs, plan))
            plan = plan * max(0.0, min(1.0, scale))
        key = tuple(np.round(plan, 6))
        if key not in seen:
            seen.add(key)
            unique.append(plan)
    return unique


def fill_budget(order: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    y = np.zeros(len(costs), dtype=float)
    remaining = budget
    for index in order:
        if index < 0 or index >= len(costs) or costs[index] <= 0:
            continue
        amount = min(1.0, remaining / costs[index])
        y[index] = amount
        remaining -= amount * costs[index]
        if remaining <= 1e-9:
            break
    return y


def combinations_limited(items: Iterable[int], size: int, limit: int) -> list[tuple[int, ...]]:
    from itertools import combinations

    output = []
    for combo in combinations(items, size):
        output.append(combo)
        if len(output) >= limit:
            break
    return output


def fixed_result_payload(result: Any) -> dict[str, Any]:
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


def payload_to_result(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        objective=float(payload["objective"]),
        lower_bound=float(payload.get("lower_bound", payload["objective"])),
        z=np.asarray(payload["z"], dtype=float),
        w=np.asarray(payload["w"], dtype=float),
        y=np.asarray(payload["y"], dtype=float),
        nominal_distribution=np.asarray(payload["nominal_distribution"], dtype=float),
        worst_case_distribution=np.asarray(payload["worst_case_distribution"], dtype=float),
        state_losses=np.asarray(payload["state_losses"], dtype=float),
        state_survivors=np.asarray(payload["state_survivors"], dtype=float),
        iterations=int(payload.get("iterations", 0)),
    )


def versioned_key(key: str) -> str:
    return f"{CACHE_VERSION}__{key}"


def hash_array(values: np.ndarray) -> str:
    rounded = np.round(values.astype(float), 8)
    payload = json.dumps(rounded.tolist(), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def stable_int(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 100000


def selected_link_payload(instance: DADInstance, y: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "link_id": link.id,
            "y": float(y[index]),
            "phi": float(link.baseline_failure_probability),
            "cost": float(link.retrofit_cost),
        }
        for index, link in enumerate(instance.links)
        if y[index] > 1e-7
    ]


def target_states(instance: DADInstance, result: SimpleNamespace) -> dict[str, Any]:
    shift = result.worst_case_distribution - result.nominal_distribution
    target_indices = np.where(shift > 1e-8)[0].tolist()
    if not target_indices:
        max_loss = float(np.max(result.state_losses))
        target_indices = np.where(result.state_losses >= max_loss - 1e-7)[0].tolist()
    target_ids = [instance.states[index].id for index in target_indices]
    return {
        "target_states": target_ids,
        "target_is_tail": any(instance.states[index].is_tail for index in target_indices),
    }


def probability_shift_rows(instance: DADInstance, result: SimpleNamespace, experiment_id: str, rho: float) -> list[dict[str, Any]]:
    shift = result.worst_case_distribution - result.nominal_distribution
    target_indices = set(np.where(shift > 1e-8)[0].tolist())
    source_indices = set(np.where(shift < -1e-8)[0].tolist())
    rows = []
    for index, state in enumerate(instance.states):
        rows.append(
            {
                "experiment_id": experiment_id,
                "rho": rho,
                "state_id": state.id,
                "failed_links": "tail" if state.is_tail else len(state.failed_links),
                "Q_value": float(result.state_survivors[index]),
                "gamma_value": float(result.state_losses[index]),
                "nominal_prob": float(result.nominal_distribution[index]),
                "worst_case_prob": float(result.worst_case_distribution[index]),
                "prob_shift": float(shift[index]),
                "is_target_state": index in target_indices,
                "is_source_state": index in source_indices,
                "is_tail": bool(state.is_tail),
            }
        )
    return rows


def reachability_override_check(bundle: InstanceBundle, out: Path) -> None:
    rows = []
    instance = bundle.instance
    for state in instance.states[: min(8, len(instance.states))]:
        travel_times = instance.travel_times(state)
        survival = instance.survival_matrix(state)
        for center_index, center in enumerate(instance.centers):
            for zone_index, zone in enumerate(instance.zones):
                tau = float(travel_times[center_index, zone_index])
                reachable = bool(np.isfinite(tau))
                rows.append(
                    {
                        "experiment_id": bundle.experiment_id,
                        "state_id": state.id,
                        "center_id": center.id,
                        "zone_id": zone.id,
                        "reachable": reachable,
                        "tau": tau if reachable else "inf",
                        "xi_value": float(survival[center_index, zone_index]),
                    }
                )
    write_table(pd.DataFrame(rows), out, "table_reachability_override_check")


def dda_value_comparison(context: DataContext, out: Path, cache: CheckpointStore, force: bool) -> None:
    rows = []
    base_bundle = build_bundle(context, "exposure_dominant", "full_state_no_tail", rho=0.10, link_count=5, experiment_id="dda_base")
    fixed_center = nominal_probabilities(base_bundle.instance.links, base_bundle.instance.states, np.zeros(len(base_bundle.instance.links)))
    for rho in RHO_VALUES:
        bundle = build_bundle_from_existing(base_bundle, rho=rho, experiment_id=f"dda_rho{rho:.2f}")
        instance = bundle.instance
        policies = [
            ("nominal_stochastic", "decision_dependent_rho0", None, 0.0),
            ("fixed_center_DRO", "fixed_center", fixed_center, rho),
            ("decision_dependent_DRO", "decision_dependent", None, rho),
        ]
        for policy_type, optimized_under, nominal_override, optimization_rho in policies:
            opt_instance = build_bundle_from_existing(bundle, rho=optimization_rho, experiment_id=f"{bundle.experiment_id}_{policy_type}").instance
            start = time.time()
            result = optimize_y_by_search(
                cache,
                f"dda_{policy_type}_rho{rho:.2f}",
                opt_instance,
                force=force,
                max_random=6,
                nominal_override=nominal_override,
            )
            objective_optimized = float(result.objective)
            common_eval = evaluate_fixed_y(instance, result.y, epsilon=1e-5, max_iterations=120)
            runtime = time.time() - start
            rows.append(
                {
                    "experiment_id": bundle.experiment_id,
                    "rho": rho,
                    "policy_type": policy_type,
                    "optimized_under": optimized_under,
                    "evaluated_under": "decision_dependent_DRO",
                    "objective_optimized": objective_optimized,
                    "objective_common_evaluation": float(common_eval.objective),
                    "y_budget_used": float(np.dot(instance.retrofit_costs, result.y)),
                    "selected_y_json": json.dumps(result.y.tolist()),
                    "worst_case_target_states_json": json.dumps(target_states(instance, common_eval)["target_states"]),
                    "runtime_sec": runtime,
                }
            )
    write_table(pd.DataFrame(rows), out, "table_dda_value_comparison")


def sbb_certificate(
    context: DataContext,
    out: Path,
    cache: CheckpointStore,
    force: bool,
    time_limit_sec: float,
    m2_max_nodes: int = 1000,
    m5_max_nodes: int = 120,
) -> None:
    specs = [
        {"link_count": 2, "max_nodes": m2_max_nodes, "time_limit_sec": min(time_limit_sec, 600.0), "role": "certified_small"},
        {"link_count": 5, "max_nodes": m5_max_nodes, "time_limit_sec": time_limit_sec, "role": "m5_diagnostic"},
    ]
    table_rows = []
    all_trace_rows = []
    best_vectors = {}
    prior_certified_links: list[str] | None = None
    prior_certified_y: list[float] | None = None
    for spec in specs:
        link_count = int(spec["link_count"])
        experiment_id = f"sbb_full_no_tail_m{link_count}_rho0.10"
        bundle = build_bundle(context, "exposure_dominant", "full_state_no_tail", rho=0.10, link_count=link_count, experiment_id=experiment_id)
        write_experiment_config(out, experiment_id, bundle.config | spec)
        key = versioned_key(f"{experiment_id}__sbb_fixed")
        link_ids = [link.id for link in bundle.instance.links]
        initial_y = None
        warm_start_objective = None
        warm_start_source = ""
        nested_from_m2 = False
        if link_count > 2 and prior_certified_links is not None and prior_certified_y is not None:
            nested_from_m2 = all(link_id in link_ids for link_id in prior_certified_links)
            if nested_from_m2:
                initial_y = np.zeros(len(link_ids), dtype=float)
                for source_index, link_id in enumerate(prior_certified_links):
                    initial_y[link_ids.index(link_id)] = prior_certified_y[source_index]
                warm_result = evaluate_fixed_y(bundle.instance, initial_y, epsilon=1e-5, max_iterations=120)
                warm_start_objective = float(warm_result.objective)
                warm_start_source = "padded_m2_certified_y"

        def run_sbb(
            bundle: InstanceBundle = bundle,
            spec: dict[str, Any] = spec,
            initial_y: np.ndarray | None = initial_y,
            warm_start_objective: float | None = warm_start_objective,
            warm_start_source: str = warm_start_source,
            nested_from_m2: bool = nested_from_m2,
        ) -> dict[str, Any]:
            trace_records: list[dict[str, Any]] = []
            result = solve_global_sbb(
                bundle.instance,
                epsilon=1e-2,
                fixed_y_epsilon=1e-5,
                max_nodes=int(spec["max_nodes"]),
                time_limit_sec=float(spec["time_limit_sec"]),
                initial_y=initial_y,
                trace_callback=trace_records.append,
            )
            return {
                "objective": float(result.objective),
                "lower_bound": float(result.lower_bound),
                "gap": float(result.gap),
                "y": result.y.tolist(),
                "z": result.z.tolist(),
                "w": result.w.tolist(),
                "nodes_processed": int(result.nodes_processed),
                "nodes_remaining": int(result.nodes_remaining),
                "converged": bool(result.converged),
                "pruned_nodes": int(result.pruned_nodes),
                "max_tree_depth": int(result.max_tree_depth),
                "fixed_y_oracle_calls": int(result.fixed_y_oracle_calls),
                "recourse_cuts_generated": int(result.recourse_cuts_generated),
                "relaxation_failures": int(result.relaxation_failures),
                "runtime_sec": float(result.runtime_sec),
                "termination_reason": result.termination_reason,
                "warm_start_source": warm_start_source,
                "warm_start_objective": warm_start_objective,
                "nested_from_m2": nested_from_m2,
                "trace_records": trace_records,
            }

        payload = cache.get_or_compute(key, run_sbb, force=force)
        trace = pd.DataFrame(payload.get("trace_records", []))
        if trace.empty:
            trace = pd.DataFrame(
                [
                    {
                        "time_sec": 0,
                        "processed_nodes": 0,
                        "incumbent_UB": payload["objective"],
                        "global_LB": payload["lower_bound"],
                    }
                ]
            )
        trace["experiment_id"] = experiment_id
        trace["rel_gap_percent"] = [
            rel_gap(ub, lb) for ub, lb in zip(trace.get("incumbent_UB", pd.Series(dtype=float)), trace.get("global_LB", pd.Series(dtype=float)))
        ]
        all_trace_rows.extend(trace.to_dict("records"))
        best_vectors[experiment_id] = payload["y"]
        if link_count == 2 and payload.get("converged"):
            prior_certified_links = link_ids
            prior_certified_y = payload["y"]
        table_rows.append(
            {
                "experiment_id": experiment_id,
                "run_role": spec["role"],
                "state_mode": "full_state_no_tail",
                "num_candidate_links": len(bundle.instance.links),
                "num_states": len(bundle.instance.states),
                "rho": bundle.instance.ambiguity_radius,
                "UB": payload["objective"],
                "LB": payload["lower_bound"],
                "abs_gap": payload["gap"],
                "rel_gap_percent": rel_gap(payload["objective"], payload["lower_bound"]),
                "processed_nodes": payload["nodes_processed"],
                "pruned_nodes": payload["pruned_nodes"],
                "max_tree_depth": payload["max_tree_depth"],
                "fixed_y_oracle_calls": payload["fixed_y_oracle_calls"],
                "recourse_cuts_generated": payload["recourse_cuts_generated"],
                "relaxation_failures": payload["relaxation_failures"],
                "runtime_sec": payload["runtime_sec"],
                "nested_from_m2": payload.get("nested_from_m2", False),
                "warm_start_source": payload.get("warm_start_source", ""),
                "warm_start_objective": payload.get("warm_start_objective", ""),
                "solver_status": "converged" if payload["converged"] else "time_or_node_limited",
                "termination_reason": payload["termination_reason"],
            }
        )
    trace = pd.DataFrame(all_trace_rows)
    atomic_write_dataframe(trace, out / "sbb_node_log.csv")
    gap_trace = trace[["experiment_id", "time_sec", "processed_nodes", "incumbent_UB", "global_LB", "rel_gap_percent"]].copy()
    atomic_write_dataframe(gap_trace, out / "sbb_gap_trace.csv")
    atomic_write_text(out / "best_y_vector.json", json.dumps(best_vectors, indent=2))
    table = pd.DataFrame(table_rows)
    write_table(table, out, "table_sbb_certificate")


def sbb_root_bound_diagnostics(context: DataContext, out: Path) -> None:
    experiment_id = "sbb_root_diag_full_no_tail_m5_rho0.10"
    bundle = build_bundle(context, "exposure_dominant", "full_state_no_tail", rho=0.10, link_count=5, experiment_id=experiment_id)
    write_experiment_config(out, experiment_id, bundle.config | {"diagnostic": "root_bound"})
    instance = bundle.instance

    rows = []
    current_node = root_node(instance)
    current = solve_node_with_cut_separation(
        instance,
        current_node,
        {i: [] for i in range(len(instance.states))},
        epsilon_cut=1e-7,
        max_cut_iterations=40,
    )
    rows.append(
        root_diag_row(
            experiment_id,
            "current_raw_relaxation",
            instance,
            current,
            reported_lower_bound=current.lower_bound if current.success else np.nan,
            notes="Original SBB root relaxation before applying the known nonnegative objective floor.",
        )
    )
    rows.append(
        root_diag_row(
            experiment_id,
            "add_nonnegative_lb_floor",
            instance,
            current,
            reported_lower_bound=valid_objective_lower_bound(current.lower_bound) if current.success else np.nan,
            notes="Same relaxation with the valid bound robust expected loss >= 0 applied to the reported LB.",
        )
    )

    tightened_node, theta_notes = statewise_theta_bound_node(instance)
    tightened = solve_node_with_cut_separation(
        instance,
        tightened_node,
        {i: [] for i in range(len(instance.states))},
        epsilon_cut=1e-7,
        max_cut_iterations=40,
    )
    rows.append(
        root_diag_row(
            experiment_id,
            "statewise_theta_bounds",
            instance,
            tightened,
            reported_lower_bound=valid_objective_lower_bound(tightened.lower_bound) if tightened.success else np.nan,
            notes=theta_notes,
        )
    )
    write_table(pd.DataFrame(rows), out, "table_sbb_root_bound_diagnostics")


def root_diag_row(
    experiment_id: str,
    test: str,
    instance: DADInstance,
    relaxation: Any,
    reported_lower_bound: float,
    notes: str,
) -> dict[str, Any]:
    raw_lb = float(relaxation.lower_bound) if relaxation.success else np.nan
    return {
        "experiment_id": experiment_id,
        "test": test,
        "num_candidate_links": len(instance.links),
        "num_states": len(instance.states),
        "raw_root_LB": raw_lb,
        "reported_root_LB": reported_lower_bound,
        "root_LB_materially_positive": bool(np.isfinite(reported_lower_bound) and reported_lower_bound > 0.01 * instance.d_max),
        "success": bool(relaxation.success),
        "cut_iterations": int(getattr(relaxation, "cut_iterations", 0)),
        "omega": float(relaxation.omega) if relaxation.success else np.nan,
        "kappa": float(relaxation.kappa) if relaxation.success else np.nan,
        "message": getattr(relaxation, "message", ""),
        "notes": notes,
    }


def statewise_theta_bound_node(instance: DADInstance):
    node = root_node(instance)
    z0 = np.zeros(len(instance.zones), dtype=float)
    w0 = np.zeros(len(instance.centers), dtype=float)
    state_loss_upper, _, _ = evaluate_plan_losses(instance, z0, w0)
    state_loss_upper = np.maximum(0.0, np.asarray(state_loss_upper, dtype=float))
    max_loss_upper = float(max(1.0, state_loss_upper.max()))
    node.theta_bounds[:, 0] = np.maximum(node.theta_bounds[:, 0], -max_loss_upper)
    node.theta_bounds[:, 1] = np.minimum(node.theta_bounds[:, 1], state_loss_upper)
    notes = (
        "Theta bounds tightened using statewise loss upper bounds computed at z=0,w=0; "
        f"max state loss upper bound={max_loss_upper:.3f}."
    )
    return node, notes


def certified_dda_sweep(
    context: DataContext,
    out: Path,
    cache: CheckpointStore,
    force: bool,
    time_limit_sec: float,
    link_counts: list[int] | None = None,
    m2_max_nodes: int = 1500,
    m3_max_nodes: int = 5000,
) -> None:
    rows: list[dict[str, Any]] = []
    result_by_link_count: dict[int, dict[float, dict[str, Any]]] = {}
    link_counts = [2, 3] if link_counts is None else link_counts
    for link_count in link_counts:
        result_by_link_count[link_count] = {}
        for rho in RHO_VALUES:
            experiment_id = f"certified_dda_mechanism_active_m{link_count}_rho{rho:.2f}"
            bundle = build_bundle(
                context,
                "mechanism_active",
                "full_state_no_tail",
                rho=rho,
                link_count=link_count,
                experiment_id=experiment_id,
            )
            write_experiment_config(out, experiment_id, bundle.config | {"diagnostic": "certified_dda_sweep"})
            max_nodes = m2_max_nodes if link_count == 2 else m3_max_nodes
            key = versioned_key(f"{experiment_id}__sbb_certified_dda")

            def run_certified_sbb(bundle: InstanceBundle = bundle, max_nodes: int = max_nodes) -> dict[str, Any]:
                trace_records: list[dict[str, Any]] = []
                result = solve_global_sbb(
                    bundle.instance,
                    epsilon=1e-2,
                    fixed_y_epsilon=1e-5,
                    max_nodes=max_nodes,
                    time_limit_sec=time_limit_sec,
                    trace_callback=trace_records.append,
                )
                fixed = result.fixed_y_result
                return {
                    "objective": float(result.objective),
                    "lower_bound": float(result.lower_bound),
                    "gap": float(result.gap),
                    "rel_gap_percent": rel_gap(result.objective, result.lower_bound),
                    "y": result.y.tolist(),
                    "z": result.z.tolist(),
                    "w": result.w.tolist(),
                    "nominal_objective": float(np.dot(fixed.nominal_distribution, fixed.state_losses)),
                    "nodes_processed": int(result.nodes_processed),
                    "pruned_nodes": int(result.pruned_nodes),
                    "max_tree_depth": int(result.max_tree_depth),
                    "fixed_y_oracle_calls": int(result.fixed_y_oracle_calls),
                    "recourse_cuts_generated": int(result.recourse_cuts_generated),
                    "relaxation_failures": int(result.relaxation_failures),
                    "runtime_sec": float(result.runtime_sec),
                    "solver_status": "converged" if result.converged else "time_or_node_limited",
                    "termination_reason": result.termination_reason,
                    "target_states": target_states(bundle.instance, fixed)["target_states"],
                    "target_is_tail": target_states(bundle.instance, fixed)["target_is_tail"],
                    "trace_records": trace_records,
                }

            payload = cache.get_or_compute(key, run_certified_sbb, force=force)
            result_by_link_count[link_count][rho] = {"bundle": bundle, "payload": payload}

        y0 = np.asarray(result_by_link_count[link_count][0.0]["payload"]["y"], dtype=float)
        for rho in RHO_VALUES:
            bundle = result_by_link_count[link_count][rho]["bundle"]
            payload = result_by_link_count[link_count][rho]["payload"]
            y_rho = np.asarray(payload["y"], dtype=float)
            y0_eval = evaluate_fixed_y(bundle.instance, y0, epsilon=1e-5, max_iterations=120)
            delta_rho = float(y0_eval.objective - payload["objective"])
            rows.append(
                {
                    "experiment_id": f"certified_dda_mechanism_active_m{link_count}_rho{rho:.2f}",
                    "mechanism_regime": "mechanism_active",
                    "state_mode": "full_state_no_tail",
                    "num_candidate_links": link_count,
                    "num_states": len(bundle.instance.states),
                    "rho": rho,
                    "certified_objective": payload["objective"],
                    "certified_LB": payload["lower_bound"],
                    "abs_gap": payload["gap"],
                    "rel_gap_percent": payload["rel_gap_percent"],
                    "selected_y_json": json.dumps(payload["y"]),
                    "sum_y": float(y_rho.sum()),
                    "y_budget_used": float(np.dot(bundle.instance.retrofit_costs, y_rho)),
                    "nominal_objective_selected_y": payload["nominal_objective"],
                    "objective_using_rho0_policy": float(y0_eval.objective),
                    "delta_rho_value": delta_rho,
                    "y_diff_norm_from_rho0": float(np.linalg.norm(y_rho - y0, ord=1)),
                    "worst_case_target_states_json": json.dumps(payload["target_states"]),
                    "target_is_tail": bool(payload["target_is_tail"]),
                    "runtime_sec": payload["runtime_sec"],
                    "processed_nodes": payload["nodes_processed"],
                    "solver_status": payload["solver_status"],
                    "termination_reason": payload["termination_reason"],
                }
            )
    write_table(pd.DataFrame(rows), out, "table_certified_dda_sweep")


def rel_gap(ub: float, lb: float) -> float:
    ub = float(ub)
    lb = float(lb)
    if not np.isfinite(ub) or not np.isfinite(lb):
        return float("inf")
    return 100.0 * max(0.0, ub - lb) / max(1.0, abs(ub))


def regime_tables(context: DataContext, out: Path, cache: CheckpointStore, force: bool) -> None:
    policy_rows = []
    exposure_rows = []
    diagnostic_rows = []
    policies = [
        ("no investment", 0.0, 0.0, 0.0, "fixed"),
        ("exposure-only heuristic", None, 0.0, 0.0, "exposure"),
        ("building-only optimization", None, 0.0, 0.0, "fixed"),
        ("capacity-only optimization", 0.0, 0.0, None, "fixed"),
        ("road-only incumbent", 0.0, None, 0.0, "search"),
        ("building + capacity", None, 0.0, None, "fixed"),
        ("building + road", None, None, 0.0, "search"),
        ("road + capacity", 0.0, None, None, "search"),
        ("all-sector incumbent", None, None, None, "search"),
    ]
    for regime_name in POLICY_REGIMES:
        regime = REGIMES[regime_name]
        no_investment_obj = None
        policy_results: dict[str, dict[str, Any]] = {}
        for policy_name, beta_z, beta_y, beta_x, method in policies:
            experiment_id = f"regime_{regime_name}_{safe_name(policy_name)}"
            beta_z = regime.renovation_fraction if beta_z is None else beta_z
            beta_y = regime.retrofit_fraction if beta_y is None else beta_y
            beta_x = regime.capacity_fraction if beta_x is None else beta_x
            bundle = build_bundle(
                context,
                regime_name,
                "full_state_no_tail",
                rho=0.10,
                renovation_fraction=beta_z,
                retrofit_fraction=beta_y,
                capacity_fraction=beta_x,
                link_count=regime.link_count,
                experiment_id=experiment_id,
            )
            write_experiment_config(out, experiment_id, bundle.config | {"policy": policy_name})
            start = time.time()
            if method == "exposure":
                exact = exposure_only_solution(bundle.instance)
                objective = evaluate_given_plan(bundle.instance, np.zeros(len(bundle.instance.links)), exact["z"], np.zeros(len(bundle.instance.centers)))
                result = SimpleNamespace(
                    objective=objective,
                    y=np.zeros(len(bundle.instance.links)),
                    z=exact["z"],
                    w=np.zeros(len(bundle.instance.centers)),
                )
                exposure_rows.append(exposure_benchmark_row(regime_name, "exact LP", bundle.instance, result, exact["z"], start))
            elif method == "search":
                result = optimize_y_by_search(cache, experiment_id, bundle.instance, force=force, max_random=6)
            else:
                result = cached_fixed_y(cache, experiment_id, bundle.instance, np.zeros(len(bundle.instance.links)), force=force)
            runtime = time.time() - start
            if policy_name == "no investment":
                objective = evaluate_given_plan(bundle.instance, np.zeros(len(bundle.instance.links)), np.zeros(len(bundle.instance.zones)), np.zeros(len(bundle.instance.centers)))
            else:
                objective = float(result.objective)
            if no_investment_obj is None:
                no_investment_obj = objective
            reduction = float(no_investment_obj - objective)
            policy_results[policy_name] = {"objective": objective, "result": result}
            policy_rows.append(
                {
                    "regime": regime_name,
                    "policy": policy_name,
                    "objective": objective,
                    "reduction_vs_no_investment": reduction,
                    "reduction_percent": 100.0 * reduction / max(1.0, no_investment_obj),
                    "y_budget_used": float(np.dot(bundle.instance.retrofit_costs, result.y)),
                    "z_budget_used": float(np.dot(bundle.instance.renovation_costs, result.z)),
                    "x_budget_used": float(np.dot(bundle.instance.capacity_costs, result.w)),
                    "selected_y_json": json.dumps(result.y.tolist()),
                    "total_renovation": float(np.sum(result.z)),
                    "total_capacity_added": float(np.sum(result.w)),
                    "runtime_sec": runtime,
                    "certified": False,
                    "rel_gap_percent_if_certified": "",
                }
            )

        no_obj = float(policy_results["no investment"]["objective"])
        all_obj = float(policy_results["all-sector incumbent"]["objective"])
        building_obj = float(policy_results["building-only optimization"]["objective"])
        capacity_obj = float(policy_results["capacity-only optimization"]["objective"])
        road_obj = float(policy_results["road-only incumbent"]["objective"])
        bc_obj = float(policy_results["building + capacity"]["objective"])
        br_obj = float(policy_results["building + road"]["objective"])
        diagnostic_rows.append(
            {
                "regime": regime_name,
                "D0": float(build_bundle(context, regime_name, "full_state_no_tail", rho=0.10, link_count=regime.link_count).instance.d_max),
                "objective_no_investment": no_obj,
                "objective_building_only": building_obj,
                "objective_capacity_only": capacity_obj,
                "objective_road_only": road_obj,
                "objective_all_sector": all_obj,
                "building_share_of_total_reduction": (no_obj - building_obj) / max(1e-9, no_obj - all_obj),
                "road_marginal_gain_after_building_capacity": bc_obj - all_obj,
                "capacity_marginal_gain_after_building_road": br_obj - all_obj,
                "access_binding_indicator": (bc_obj - all_obj) > 0.01 * max(1.0, no_obj - all_obj),
                "capacity_binding_indicator": (br_obj - all_obj) > 0.01 * max(1.0, no_obj - all_obj),
                "exposure_binding_indicator": (no_obj - building_obj) > 0.50 * max(1.0, no_obj - all_obj),
            }
        )

        building_only = policy_results["building-only optimization"]["objective"]
        all_sector = policy_results["all-sector incumbent"]["objective"]
        for row in exposure_rows:
            if row["regime"] == regime_name:
                row["comparison_to_building_only"] = row["objective"] - building_only
                row["comparison_to_all_sector"] = row["objective"] - all_sector

    write_table(pd.DataFrame(policy_rows), out, "table_regime_policy_comparison")
    write_table(pd.DataFrame(diagnostic_rows), out, "table_regime_bottleneck_diagnostics")
    write_table(pd.DataFrame(exposure_rows), out, "table_exposure_heuristic_benchmark")


def cached_fixed_y(cache: CheckpointStore, key_prefix: str, instance: DADInstance, y: np.ndarray, force: bool) -> SimpleNamespace:
    payload = cache.get_or_compute(
        versioned_key(f"{key_prefix}__fixed_y_{hash_array(y)}"),
        lambda: fixed_result_payload(evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=120)),
        force=force,
    )
    return payload_to_result(payload)


def exposure_only_solution(instance: DADInstance) -> dict[str, Any]:
    c = -instance.base_demands
    rows = np.asarray([instance.renovation_costs], dtype=float)
    rhs = np.asarray([instance.budget_renovation], dtype=float)
    result = linprog(c=c, A_ub=rows, b_ub=rhs, bounds=[(0.0, 1.0)] * len(instance.zones), method="highs")
    if not result.success:
        raise RuntimeError(f"Exposure-only LP failed: {result.message}")
    return {"z": result.x, "objective_saved_exposure": float(-result.fun)}


def exposure_benchmark_row(
    regime: str,
    benchmark_type: str,
    instance: DADInstance,
    result: SimpleNamespace,
    z: np.ndarray,
    started: float,
) -> dict[str, Any]:
    no_investment = evaluate_given_plan(instance, np.zeros(len(instance.links)), np.zeros(len(instance.zones)), np.zeros(len(instance.centers)))
    reduction = no_investment - float(result.objective)
    return {
        "regime": regime,
        "benchmark_type": benchmark_type,
        "objective": float(result.objective),
        "reduction_vs_no_investment": reduction,
        "reduction_percent": 100.0 * reduction / max(1.0, no_investment),
        "z_budget_used": float(np.dot(instance.renovation_costs, z)),
        "selected_z_json": json.dumps(z.tolist()),
        "comparison_to_building_only": "",
        "comparison_to_all_sector": "",
        "runtime_sec": time.time() - started,
    }


def evaluate_given_plan(instance: DADInstance, y: np.ndarray, z: np.ndarray, w: np.ndarray, nominal_override: np.ndarray | None = None) -> float:
    nominal = nominal_override if nominal_override is not None else nominal_probabilities(instance.links, instance.states, y)
    losses, _, _ = evaluate_plan_losses(instance, z, w)
    return worst_case_tv_distribution(
        nominal,
        losses,
        instance.ambiguity_radius,
        maximize=True,
        density_cap=instance.ambiguity_density_cap,
    ).value


def discretized_m5_enumeration(
    context: DataContext,
    out: Path,
    cache: CheckpointStore,
    force: bool,
    log_path: Path,
) -> None:
    from itertools import product

    levels = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
    total_candidates = int(len(levels) ** 5)
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    y0: np.ndarray | None = None

    for rho in RHO_VALUES:
        started = time.time()
        experiment_id = f"discretized_m5_no_tail_rho{rho:.2f}"
        bundle = build_bundle(context, "exposure_dominant", "full_state_no_tail", rho=rho, link_count=5, experiment_id=experiment_id)
        instance = bundle.instance
        write_experiment_config(
            out,
            experiment_id,
            bundle.config
            | {
                "diagnostic": "m5_discretized_enumeration",
                "grid_levels": levels.tolist(),
                "total_grid_candidates": total_candidates,
            },
        )

        results: list[SimpleNamespace] = []
        infeasible_budget = 0
        for candidate_index, values in enumerate(product(levels, repeat=5), start=1):
            y = np.asarray(values, dtype=float)
            budget_used = float(np.dot(instance.retrofit_costs, y))
            if budget_used > instance.budget_retrofit + 1e-7:
                infeasible_budget += 1
                continue
            key = versioned_key(f"{experiment_id}__grid_{candidate_index:04d}_{hash_array(y)}")
            existed = cache.exists(key)

            def compute(y: np.ndarray = y) -> dict[str, Any]:
                eval_started = time.time()
                payload = fixed_result_payload(evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=120))
                payload["eval_runtime_sec"] = time.time() - eval_started
                payload["candidate_index"] = candidate_index
                payload["budget_used"] = budget_used
                return payload

            payload = cache.get_or_compute(key, compute, force=force)
            result = payload_to_result(payload)
            result.candidate_index = int(payload.get("candidate_index", candidate_index))
            result.eval_runtime_sec = float(payload.get("eval_runtime_sec", 0.0))
            result.budget_used = float(payload.get("budget_used", budget_used))
            result.loaded_from_cache = bool(existed and not force)
            results.append(result)

            if len(results) % 100 == 0:
                write_status(
                    out,
                    "running",
                    "discrete",
                    f"rho={rho:.2f}: evaluated {len(results)} feasible grid policies.",
                    log_path,
                    extra={
                        "rho": rho,
                        "feasible_evaluated": len(results),
                        "total_grid_candidates": total_candidates,
                        "infeasible_budget": infeasible_budget,
                    },
                )
                logging.info(
                    "Discretized m5 rho=%.2f: evaluated %d feasible policies; skipped %d infeasible.",
                    rho,
                    len(results),
                    infeasible_budget,
                )

        if not results:
            raise RuntimeError(f"No feasible discretized policies were evaluated for rho={rho:.2f}.")

        results.sort(key=lambda result: result.objective)
        best = results[0]
        if rho == 0.0:
            y0 = np.asarray(best.y, dtype=float)
        if y0 is None:
            raise RuntimeError("rho=0 policy must be evaluated before positive rho values.")
        y0_eval = cached_fixed_y(cache, f"{experiment_id}_rho0_policy_eval", instance, y0, force=force)
        delta_rho = float(y0_eval.objective - best.objective)

        no_investment = evaluate_given_plan(instance, np.zeros(5), np.zeros(len(instance.zones)), np.zeros(len(instance.centers)))
        no_retrofit = cached_fixed_y(cache, f"{experiment_id}_no_retrofit", instance, np.zeros(5), force=force)
        exposure = exposure_only_solution(instance)
        exposure_objective = evaluate_given_plan(instance, np.zeros(5), exposure["z"], np.zeros(len(instance.centers)))

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "rho": rho,
                "num_links": len(instance.links),
                "num_states": len(instance.states),
                "grid_levels_json": json.dumps(levels.tolist()),
                "total_grid_candidates": total_candidates,
                "feasible_evaluated_candidates": len(results),
                "infeasible_budget_candidates": infeasible_budget,
                "best_discretized_objective": float(best.objective),
                "best_y_json": json.dumps(best.y.tolist()),
                "best_y_budget_used": float(best.budget_used),
                "objective_using_rho0_policy": float(y0_eval.objective),
                "delta_rho_value": delta_rho,
                "y_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(best.y) - y0, ord=1)),
                "runtime_sec": time.time() - started,
            }
        )

        for rank, result in enumerate(results[:10], start=1):
            gap = float(result.objective - best.objective)
            row = {
                "experiment_id": experiment_id,
                "rho": rho,
                "rank": rank,
                "candidate_index": int(result.candidate_index),
                "objective": float(result.objective),
                "gap_to_best": gap,
                "gap_percent": 100.0 * gap / max(1.0, abs(best.objective)),
                "y_budget_used": float(result.budget_used),
                "selected_y_json": json.dumps(result.y.tolist()),
                "iterations": int(result.iterations),
                "loaded_from_cache": bool(result.loaded_from_cache),
            }
            for link_index, value in enumerate(result.y, start=1):
                row[f"y{link_index}"] = float(value)
            top_rows.append(row)

        for threshold in [0.01, 0.05, 0.10, 0.50]:
            count = sum(
                1
                for result in results
                if 100.0 * float(result.objective - best.objective) / max(1.0, abs(best.objective)) <= threshold + 1e-12
            )
            near_rows.append(
                {
                    "experiment_id": experiment_id,
                    "rho": rho,
                    "threshold_percent": threshold,
                    "near_optimal_policy_count": count,
                    "share_of_feasible_candidates": count / len(results),
                    "feasible_evaluated_candidates": len(results),
                }
            )

        sector_values = [
            ("no investment", no_investment),
            ("no retrofit; z,w optimized", no_retrofit.objective),
            ("exposure-only exact LP", exposure_objective),
            ("all-sector discretized", best.objective),
        ]
        for label, objective in sector_values:
            sector_rows.append(
                {
                    "experiment_id": experiment_id,
                    "rho": rho,
                    "comparison": label,
                    "objective": float(objective),
                    "gap_to_all_sector_discretized": float(objective - best.objective),
                    "reduction_vs_no_investment": float(no_investment - objective),
                }
            )

    write_table(pd.DataFrame(summary_rows), out, "table_discretized_m5_summary")
    write_table(pd.DataFrame(top_rows), out, "table_discretized_m5_top10")
    write_table(pd.DataFrame(near_rows), out, "table_discretized_m5_near_optimal")
    write_table(pd.DataFrame(sector_rows), out, "table_discretized_m5_sector_comparison")


def discretized_m5_budget_sensitivity(
    context: DataContext,
    out: Path,
    cache: CheckpointStore,
    force: bool,
    log_path: Path,
) -> None:
    from itertools import product

    levels = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
    total_candidates = int(len(levels) ** 5)
    base_fraction = REGIMES["exposure_dominant"].retrofit_fraction
    budget_multipliers = [0.50, 1.00, 1.50]
    summary_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    for multiplier in budget_multipliers:
        y0: np.ndarray | None = None
        retrofit_fraction = base_fraction * multiplier
        for rho in RHO_VALUES:
            started = time.time()
            experiment_id = f"budget_sensitivity_m5_mult{multiplier:.2f}_rho{rho:.2f}"
            bundle = build_bundle(
                context,
                "exposure_dominant",
                "full_state_no_tail",
                rho=rho,
                retrofit_fraction=retrofit_fraction,
                link_count=5,
                experiment_id=experiment_id,
            )
            instance = bundle.instance
            checkpoint_experiment_id = (
                f"discretized_m5_no_tail_rho{rho:.2f}" if abs(multiplier - 1.0) <= 1e-12 else experiment_id
            )
            write_experiment_config(
                out,
                experiment_id,
                bundle.config
                | {
                    "diagnostic": "m5_discretized_budget_sensitivity",
                    "budget_multiplier": multiplier,
                    "base_retrofit_fraction": base_fraction,
                    "grid_levels": levels.tolist(),
                    "total_grid_candidates": total_candidates,
                    "checkpoint_experiment_id": checkpoint_experiment_id,
                },
            )

            results: list[SimpleNamespace] = []
            infeasible_budget = 0
            for candidate_index, values in enumerate(product(levels, repeat=5), start=1):
                y = np.asarray(values, dtype=float)
                budget_used = float(np.dot(instance.retrofit_costs, y))
                if budget_used > instance.budget_retrofit + 1e-7:
                    infeasible_budget += 1
                    continue
                key = versioned_key(f"{checkpoint_experiment_id}__grid_{candidate_index:04d}_{hash_array(y)}")
                existed = cache.exists(key)

                def compute(y: np.ndarray = y) -> dict[str, Any]:
                    eval_started = time.time()
                    payload = fixed_result_payload(evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=120))
                    payload["eval_runtime_sec"] = time.time() - eval_started
                    payload["candidate_index"] = candidate_index
                    payload["budget_used"] = budget_used
                    return payload

                payload = cache.get_or_compute(key, compute, force=force)
                result = payload_to_result(payload)
                result.candidate_index = int(payload.get("candidate_index", candidate_index))
                result.eval_runtime_sec = float(payload.get("eval_runtime_sec", 0.0))
                result.budget_used = float(payload.get("budget_used", budget_used))
                result.loaded_from_cache = bool(existed and not force)
                results.append(result)

                if len(results) % 100 == 0:
                    write_status(
                        out,
                        "running",
                        "budgetsens",
                        f"budget={multiplier:.2f}x rho={rho:.2f}: evaluated {len(results)} feasible policies.",
                        log_path,
                        extra={
                            "budget_multiplier": multiplier,
                            "rho": rho,
                            "feasible_evaluated": len(results),
                            "total_grid_candidates": total_candidates,
                            "infeasible_budget": infeasible_budget,
                        },
                    )
                    logging.info(
                        "Budget sensitivity m5 multiplier=%.2f rho=%.2f: evaluated %d feasible policies; skipped %d infeasible.",
                        multiplier,
                        rho,
                        len(results),
                        infeasible_budget,
                    )

            if not results:
                raise RuntimeError(f"No feasible policies for budget multiplier={multiplier:.2f}, rho={rho:.2f}.")

            results.sort(key=lambda result: result.objective)
            best = results[0]
            if rho == 0.0:
                y0 = np.asarray(best.y, dtype=float)
            if y0 is None:
                raise RuntimeError("rho=0 budget-sensitivity policy must be evaluated before positive rho values.")
            y0_eval = cached_fixed_y(cache, f"{experiment_id}_rho0_policy_eval", instance, y0, force=force)
            no_investment = evaluate_given_plan(instance, np.zeros(5), np.zeros(len(instance.zones)), np.zeros(len(instance.centers)))
            no_retrofit = cached_fixed_y(cache, f"{experiment_id}_no_retrofit", instance, np.zeros(5), force=force)

            summary_rows.append(
                {
                    "experiment_id": experiment_id,
                    "budget_multiplier": multiplier,
                    "rho": rho,
                    "num_links": len(instance.links),
                    "num_states": len(instance.states),
                    "retrofit_fraction": retrofit_fraction,
                    "retrofit_budget": float(instance.budget_retrofit),
                    "total_grid_candidates": total_candidates,
                    "feasible_evaluated_candidates": len(results),
                    "infeasible_budget_candidates": infeasible_budget,
                    "best_discretized_objective": float(best.objective),
                    "best_y_json": json.dumps(best.y.tolist()),
                    "best_y_budget_used": float(best.budget_used),
                    "objective_using_rho0_policy": float(y0_eval.objective),
                    "delta_rho_value": float(y0_eval.objective - best.objective),
                    "y_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(best.y) - y0, ord=1)),
                    "no_investment_objective": float(no_investment),
                    "no_retrofit_zw_objective": float(no_retrofit.objective),
                    "road_value_over_no_retrofit": float(no_retrofit.objective - best.objective),
                    "runtime_sec": time.time() - started,
                }
            )

            for threshold in [0.01, 0.05, 0.10, 0.50]:
                count = sum(
                    1
                    for result in results
                    if 100.0 * float(result.objective - best.objective) / max(1.0, abs(best.objective)) <= threshold + 1e-12
                )
                near_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "budget_multiplier": multiplier,
                        "rho": rho,
                        "threshold_percent": threshold,
                        "near_optimal_policy_count": count,
                        "share_of_feasible_candidates": count / len(results),
                        "feasible_evaluated_candidates": len(results),
                    }
                )

            for rank, result in enumerate(results[:5], start=1):
                top_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "budget_multiplier": multiplier,
                        "rho": rho,
                        "rank": rank,
                        "objective": float(result.objective),
                        "gap_to_best": float(result.objective - best.objective),
                        "gap_percent": 100.0 * float(result.objective - best.objective) / max(1.0, abs(best.objective)),
                        "selected_y_json": json.dumps(result.y.tolist()),
                        "y_budget_used": float(result.budget_used),
                    }
                )

    write_table(pd.DataFrame(summary_rows), out, "table_discretized_m5_budget_sensitivity")
    write_table(pd.DataFrame(near_rows), out, "table_discretized_m5_budget_sensitivity_near_optimal")
    write_table(pd.DataFrame(top_rows), out, "table_discretized_m5_budget_sensitivity_top5")


def candidate_link_sensitivity_revised(context: DataContext, out: Path, cache: CheckpointStore, force: bool) -> None:
    rows = []
    for link_count in [3, 5, 6]:
        experiment_id = f"candidate_revised_full_no_tail_m{link_count}"
        bundle = build_bundle(context, "exposure_dominant", "full_state_no_tail", rho=0.10, link_count=link_count, experiment_id=experiment_id)
        write_experiment_config(out, experiment_id, bundle.config)
        start = time.time()
        fixed_y0 = cached_fixed_y(cache, f"{experiment_id}_y0", bundle.instance, np.zeros(len(bundle.instance.links)), force=force)
        best = optimize_y_by_search(cache, experiment_id, bundle.instance, force=force, max_random=6)
        runtime = time.time() - start
        targets = target_states(bundle.instance, best)
        rows.append(
            {
                "state_mode": "full_state_no_tail",
                "num_candidate_links": link_count,
                "num_states": len(bundle.instance.states),
                "no_retrofit_objective": float(fixed_y0.objective),
                "best_road_objective": float(best.objective),
                "road_gain": float(fixed_y0.objective - best.objective),
                "selected_y_json": json.dumps(best.y.tolist()),
                "worst_case_target_states_json": json.dumps(targets["target_states"]),
                "target_is_tail": targets["target_is_tail"],
                "runtime_sec": runtime,
                "certified": False,
                "rel_gap_percent_if_certified": "",
            }
        )
    write_table(pd.DataFrame(rows), out, "table_candidate_link_sensitivity_revised")


def make_figures(out: Path) -> None:
    figure_dir = out / "figures"
    table_dir = out / "tables"
    figure_dir.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    rho = pd.read_csv(table_dir / "table_rho_reoptimized_y.csv")
    if "mechanism_regime" in rho.columns and (rho["mechanism_regime"] == "mechanism_active").any():
        rho = rho[rho["mechanism_regime"] == "mechanism_active"].copy()
    y_rows = []
    for row in rho.itertuples(index=False):
        y = json.loads(row.selected_y_json)
        for index, value in enumerate(y):
            y_rows.append({"rho": row.rho, "link": f"link {index + 1}", "y": value})
    plot_line(pd.DataFrame(y_rows), "rho", "y", "link", "Selected retrofit ratios by rho", "fig_rho_y_change_mechanism", figure_dir)

    shifts = pd.read_csv(table_dir / "table_rho_probability_shift.csv")
    selected_rho = 0.10 if (shifts["rho"] == 0.10).any() else shifts["rho"].iloc[0]
    shift_subset = shifts[shifts["rho"] == selected_rho].copy()
    shift_subset["state_short"] = shift_subset["state_id"].str.replace("fail__", "fail ", regex=False).str.slice(0, 45)
    plot_probability_shift(shift_subset, figure_dir)

    gap_path = out / "sbb_gap_trace.csv"
    if gap_path.exists():
        gap = pd.read_csv(gap_path)
        plot_simple(gap, "processed_nodes", "rel_gap_percent", "SBB relative gap trace", "fig_sbb_gap_trace", figure_dir)

    regimes = pd.read_csv(table_dir / "table_regime_policy_comparison.csv")
    contribution = regimes[regimes["policy"].isin(["building-only optimization", "capacity-only optimization", "road-only incumbent", "all-sector incumbent"])].copy()
    plot_grouped_bars(contribution, figure_dir)

    exposure = pd.read_csv(table_dir / "table_exposure_heuristic_benchmark.csv")
    comparison = regimes[regimes["policy"].isin(["building-only optimization", "building + capacity", "all-sector incumbent"])][["regime", "policy", "objective"]]
    exposure_plot = exposure[["regime", "benchmark_type", "objective"]].rename(columns={"benchmark_type": "policy"})
    plot_exposure_vs_drdad(pd.concat([comparison, exposure_plot], ignore_index=True), figure_dir)


def plot_line(data: pd.DataFrame, x: str, y: str, hue: str, title: str, name: str, figure_dir: Path) -> None:
    plt.figure(figsize=(7.2, 4.6))
    for label, group in data.groupby(hue):
        plt.plot(group[x], group[y], marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(fontsize=8)
    savefig(figure_dir, name)


def plot_probability_shift(data: pd.DataFrame, figure_dir: Path) -> None:
    data = data.sort_values("prob_shift")
    plt.figure(figsize=(9.5, 6.0))
    positions = np.arange(len(data))
    plt.barh(positions - 0.18, data["nominal_prob"], height=0.35, label="nominal")
    plt.barh(positions + 0.18, data["worst_case_prob"], height=0.35, label="worst case")
    plt.yticks(positions, data["state_short"], fontsize=7)
    plt.xlabel("Probability")
    plt.title("No-tail nominal vs worst-case probability by modeled state")
    plt.legend()
    savefig(figure_dir, "fig_worst_case_shift_no_tail")


def plot_simple(data: pd.DataFrame, x: str, y: str, title: str, name: str, figure_dir: Path) -> None:
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(data[x], data[y], marker="o", linewidth=2)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    savefig(figure_dir, name)


def plot_grouped_bars(data: pd.DataFrame, figure_dir: Path) -> None:
    pivot = data.pivot(index="regime", columns="policy", values="reduction_vs_no_investment")
    pivot.plot(kind="bar", figsize=(9.0, 5.2))
    plt.ylabel("Reduction vs no investment")
    plt.title("Mitigation contribution by bottleneck regime")
    plt.xticks(rotation=20, ha="right")
    savefig(figure_dir, "fig_regime_contribution")


def plot_exposure_vs_drdad(data: pd.DataFrame, figure_dir: Path) -> None:
    pivot = data.pivot_table(index="regime", columns="policy", values="objective", aggfunc="min")
    pivot.plot(kind="bar", figsize=(9.0, 5.2))
    plt.ylabel("Worst-case expected deaths")
    plt.title("Exposure-only benchmark versus DR-DAD policies")
    plt.xticks(rotation=20, ha="right")
    savefig(figure_dir, "fig_exposure_heuristic_vs_drdad")


def savefig(figure_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(figure_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(figure_dir / f"{name}.svg", bbox_inches="tight")
    plt.close()


def write_table(dataframe: pd.DataFrame, out: Path, stem: str) -> None:
    atomic_write_dataframe(dataframe, out / "tables" / f"{stem}.csv")
    atomic_write_dataframe(dataframe, out / "tables" / f"{stem}.tex", kind="latex", escape=True)


def safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def write_code_audit(out: Path) -> None:
    text = """# Revised Computation Code Audit

- Türkiye instance construction: `examples/turkey_paper_results.py::build_bundle` and `examples/turkey_run_model.py`.
- State generation: `src/ejor_dad/states.py::generate_failure_states`.
- Conservative tail state: `generate_failure_states(..., include_tail=True)` creates `tail_ge_*`; `src/ejor_dad/recourse.py::solve_recourse` assigns zero survivors for `state.is_tail`.
- Fixed-y evaluation: `src/ejor_dad/fixed_y.py::evaluate_fixed_y`.
- Road-retrofit search: previous incumbent search is `examples/turkey_run_model.py::heuristic_retrofit_search`; revised search is in `examples/turkey_revised_experiments.py::optimize_y_by_search`.
- SBB / Algorithm 3 code: `src/ejor_dad/sbb.py::solve_global_sbb`; revised runs record trace and certificate diagnostics.
"""
    atomic_write_text(out / "code_audit.md", text)


def write_readme(out: Path) -> None:
    text = """# Revised Turkey DR-DAD Experiments

Run all revised experiments:

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\run_turkey_revised_experiments_resumable.ps1
```

Run detached so computation continues if Codex/Desktop closes:

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\start_turkey_revised_experiments_detached.ps1
```

Monitor:

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\status_turkey_revised_experiments.ps1
```

Important outputs are in `tables/`, `figures/`, `configs/`, `logs/`, and `checkpoints/`.
The full-state no-tail mechanism tables should be used for decision-dependent ambiguity claims.
Reduced-tail results from the old run should be treated only as conservative diagnostics.
"""
    atomic_write_text(out / "README.md", text)


def write_manifest(out: Path, started: float, log_path: Path, notes: list[str]) -> None:
    git_hash = None
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=out.parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_hash = "unavailable"
    manifest = {
        "generated_at_epoch": time.time(),
        "runtime_sec": time.time() - started,
        "tables": sorted(path.name for path in (out / "tables").glob("*.csv")),
        "figures": sorted(path.name for path in (out / "figures").glob("*.*")),
        "configs": sorted(path.name for path in (out / "configs").glob("*.json")),
        "seed": SEED,
        "rho_values": RHO_VALUES,
        "code_commit_hash": git_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "log_path": str(log_path.resolve()),
        "notes": notes,
    }
    atomic_write_text(out / "revised_experiments_manifest.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
