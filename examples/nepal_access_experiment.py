from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad import AidCenter, DADInstance, Link, SurvivalParams, Zone, evaluate_fixed_y, generate_failure_states
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text


RHO_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
CACHE_VERSION = "nepal-access-v1"
GRID_LEVELS = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)


@dataclass(frozen=True)
class Corridor:
    id: str
    label: str
    affected_districts: tuple[str, ...]
    phi: float
    cost: float
    penalty_minutes: float


CORRIDORS = [
    Corridor("corr_gorkha_dhading", "Gorkha-Dhading access corridor", ("Gorkha", "Dhading"), 0.34, 1.00, 95.0),
    Corridor("corr_rasuwa_nuwakot", "Rasuwa-Nuwakot mountain corridor", ("Rasuwa", "Nuwakot"), 0.42, 0.90, 125.0),
    Corridor("corr_sindhu_kavre", "Sindhupalchok-Kavre corridor", ("Sindhupalchok", "Kavrepalanchok"), 0.38, 1.00, 110.0),
    Corridor("corr_dolakha_ramechhap", "Dolakha-Ramechhap corridor", ("Dolakha", "Ramechhap"), 0.33, 0.85, 105.0),
    Corridor("corr_okhal_sindhuli", "Okhaldhunga-Sindhuli corridor", ("Okhaldhunga", "Sindhuli"), 0.30, 0.80, 90.0),
]


def main() -> None:
    args = parse_args()
    out = Path("data_work/nepal/access_experiment")
    table_dir = out / "tables"
    log_dir = out / "logs"
    checkpoint_dir = out / "checkpoints"
    for path in [table_dir, log_dir, checkpoint_dir, out / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"nepal_access_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cache = CheckpointStore(checkpoint_dir)
    write_status(out, "running", "startup", "Nepal access experiment started.", log_path)

    try:
        started = time.time()
        run_discretized_dda(out, cache, force=args.force, log_path=log_path)
        write_status(out, "completed", "complete", "Nepal access experiment completed.", log_path, exit_code=0)
        atomic_write_text(out / "runtime_summary.json", json.dumps({"runtime_sec": time.time() - started}, indent=2))
    except Exception as exc:
        write_status(out, "failed", "error", str(exc), log_path, exit_code=1)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nepal access-stress DR-DAD discretized experiment.")
    parser.add_argument("--force", action="store_true", help="Recompute all fixed-y checkpoints.")
    return parser.parse_args()


def run_discretized_dda(out: Path, cache: CheckpointStore, force: bool, log_path: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    y0: np.ndarray | None = None

    total_grid = int(len(GRID_LEVELS) ** len(CORRIDORS))
    for rho in RHO_VALUES:
        started = time.time()
        instance, metadata = build_nepal_access_instance(rho)
        experiment_id = f"nepal_access_m5_rho{rho:.2f}"
        write_config(out, experiment_id, metadata | {"rho": rho, "grid_levels": GRID_LEVELS.tolist(), "total_grid_candidates": total_grid})
        if rho == 0.0:
            for corridor, link in zip(CORRIDORS, instance.links):
                link_rows.append(
                    {
                        "link_id": link.id,
                        "label": corridor.label,
                        "affected_districts": "|".join(corridor.affected_districts),
                        "baseline_failure_probability": link.baseline_failure_probability,
                        "retrofit_cost": link.retrofit_cost,
                        "penalty_minutes": corridor.penalty_minutes,
                    }
                )

        results: list[SimpleNamespace] = []
        infeasible = 0
        for candidate_index, values in enumerate(product(GRID_LEVELS, repeat=len(CORRIDORS)), start=1):
            y = np.asarray(values, dtype=float)
            budget_used = float(np.dot(instance.retrofit_costs, y))
            if budget_used > instance.budget_retrofit + 1e-9:
                infeasible += 1
                continue
            key = versioned_key(f"{experiment_id}__grid_{candidate_index:04d}_{hash_array(y)}")
            existed = cache.exists(key)

            def compute(y: np.ndarray = y) -> dict[str, Any]:
                eval_started = time.time()
                result = evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=160)
                return fixed_result_payload(result) | {
                    "candidate_index": candidate_index,
                    "budget_used": budget_used,
                    "eval_runtime_sec": time.time() - eval_started,
                }

            payload = cache.get_or_compute(key, compute, force=force)
            result = payload_to_result(payload)
            result.candidate_index = int(payload.get("candidate_index", candidate_index))
            result.budget_used = float(payload.get("budget_used", budget_used))
            result.loaded_from_cache = bool(existed and not force)
            results.append(result)
            if len(results) % 100 == 0:
                write_status(
                    out,
                    "running",
                    "dda_sweep",
                    f"rho={rho:.2f}: evaluated {len(results)} feasible Nepal access policies.",
                    log_path,
                    extra={"rho": rho, "evaluated": len(results), "infeasible": infeasible, "total_grid": total_grid},
                )

        results.sort(key=lambda item: item.objective)
        best = results[0]
        if rho == 0.0:
            y0 = np.asarray(best.y, dtype=float)
        if y0 is None:
            raise RuntimeError("rho=0 must be run before positive rho values.")
        y0_eval = cached_fixed_y(cache, f"{experiment_id}_rho0_policy", instance, y0, force)
        no_retrofit = cached_fixed_y(cache, f"{experiment_id}_no_retrofit", instance, np.zeros(len(CORRIDORS)), force)
        no_investment_objective = evaluate_no_investment(instance)

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "rho": rho,
                "num_zones": len(instance.zones),
                "num_centers": len(instance.centers),
                "num_links": len(instance.links),
                "num_states": len(instance.states),
                "total_grid_candidates": total_grid,
                "feasible_evaluated_candidates": len(results),
                "infeasible_budget_candidates": infeasible,
                "best_discretized_objective": float(best.objective),
                "best_y_json": json.dumps(best.y.tolist()),
                "best_y_budget_used": float(best.budget_used),
                "objective_using_rho0_policy": float(y0_eval.objective),
                "delta_rho_value": float(y0_eval.objective - best.objective),
                "y_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(best.y) - y0, ord=1)),
                "no_retrofit_zw_objective": float(no_retrofit.objective),
                "road_value_over_no_retrofit": float(no_retrofit.objective - best.objective),
                "no_investment_objective": float(no_investment_objective),
                "runtime_sec": time.time() - started,
                "data_status": "access-stress case: real P_rl/centroids, global q proxy, constructed corridor Phi_ij",
            }
        )

        for rank, result in enumerate(results[:10], start=1):
            top_rows.append(
                {
                    "rho": rho,
                    "rank": rank,
                    "objective": float(result.objective),
                    "gap_to_best": float(result.objective - best.objective),
                    "gap_percent": 100.0 * float(result.objective - best.objective) / max(1.0, abs(best.objective)),
                    "selected_y_json": json.dumps(result.y.tolist()),
                    "y_budget_used": float(result.budget_used),
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
                    "rho": rho,
                    "threshold_percent": threshold,
                    "near_optimal_policy_count": count,
                    "share_of_feasible_candidates": count / len(results),
                    "feasible_evaluated_candidates": len(results),
                }
            )

        sector_rows.extend(
            [
                {
                    "rho": rho,
                    "comparison": "no investment",
                    "objective": float(no_investment_objective),
                    "gap_to_best": float(no_investment_objective - best.objective),
                },
                {
                    "rho": rho,
                    "comparison": "no retrofit; z,w optimized",
                    "objective": float(no_retrofit.objective),
                    "gap_to_best": float(no_retrofit.objective - best.objective),
                },
                {
                    "rho": rho,
                    "comparison": "all-sector discretized",
                    "objective": float(best.objective),
                    "gap_to_best": 0.0,
                },
            ]
        )

    write_table(pd.DataFrame(summary_rows), out, "table_nepal_access_dda_summary")
    write_table(pd.DataFrame(top_rows), out, "table_nepal_access_top10")
    write_table(pd.DataFrame(near_rows), out, "table_nepal_access_near_optimal")
    write_table(pd.DataFrame(sector_rows), out, "table_nepal_access_sector_comparison")
    write_table(pd.DataFrame(link_rows), out, "table_nepal_access_corridors")


def build_nepal_access_instance(rho: float) -> tuple[DADInstance, dict[str, Any]]:
    vdc = pd.read_csv("data_work/nepal/recovered_geography/tables/nepal_recovered_vdcmun_population.csv")
    vdc = vdc[vdc["match_status"] == "matched_admin3"].copy()
    damage = pd.read_csv("data_work/nepal/analysis/tables/nepal_damage_overview.csv")
    global_q = float(damage.loc[damage["metric"] == "severe_family_proxy_fraction", "value"].iloc[0])

    selected_districts = {
        "Gorkha",
        "Dhading",
        "Rasuwa",
        "Nuwakot",
        "Sindhupalchok",
        "Kavrepalanchok",
        "Dolakha",
        "Ramechhap",
        "Okhaldhunga",
        "Sindhuli",
    }
    zones_df = (
        vdc[vdc["district_name"].isin(selected_districts)]
        .sort_values("population", ascending=False)
        .head(14)
        .copy()
    )
    center_specs = [
        ("center_hetauda", "Hetauda Sub-Metropolitian City", "Makwanpur", 22000.0, 1.0),
        ("center_banepa", "Banepa Municipality", "Kavrepalanchok", 26000.0, 1.0),
        ("center_bidur", "Bidur Municipality", "Nuwakot", 21000.0, 1.0),
        ("center_gorkha", "Gorkha Municipality", "Gorkha", 18000.0, 1.0),
    ]
    centers_df = []
    for center_id, vdc_name, district, capacity, cost in center_specs:
        row = vdc[(vdc["district_name"] == district) & (vdc["vdcmun_name"] == vdc_name)].iloc[0]
        centers_df.append(
            {
                "center_id": center_id,
                "node": center_id,
                "district_name": district,
                "vdcmun_name": vdc_name,
                "lat": float(row["center_lat"]),
                "lon": float(row["center_lon"]),
                "capacity": capacity,
                "cost": cost,
            }
        )

    zones = [
        Zone(
            id=f"zone_{int(row.vdcmun_id)}",
            population=float(row.population),
            collapse_fraction=global_q,
            renovation_cost=max(1.0, float(row.population) * global_q / 10000.0),
            node=f"zone_{int(row.vdcmun_id)}",
            region=str(row.district_name),
        )
        for row in zones_df.itertuples(index=False)
    ]
    centers = [
        AidCenter(
            id=str(row["center_id"]),
            node=str(row["node"]),
            existing_capacity=float(row["capacity"]),
            capacity_unit_cost=float(row["cost"]),
        )
        for row in centers_df
    ]
    links = [
        Link(
            id=corridor.id,
            tail=f"{corridor.id}_tail",
            head=f"{corridor.id}_head",
            baseline_failure_probability=corridor.phi,
            retrofit_cost=corridor.cost,
            travel_time=1.0,
        )
        for corridor in CORRIDORS
    ]
    states = generate_failure_states(links, max_failures=None, include_tail=False)
    travel_times = precompute_travel_times(states, zones_df, centers_df)
    instance = DADInstance(
        zones=zones,
        links=links,
        centers=centers,
        budget_renovation=0.04 * sum(zone.renovation_cost for zone in zones),
        budget_retrofit=0.42 * sum(link.retrofit_cost for link in links),
        budget_capacity=0.45 * sum(center.existing_capacity * center.capacity_unit_cost for center in centers),
        ambiguity_radius=rho,
        states=states,
        survival=SurvivalParams(a=0.96, b=-0.012, c=1.0, d=0.0),
        precomputed_travel_times=travel_times,
    )
    metadata = {
        "source_population": "Recovered KLL/NPC household geography with HDX Admin-3 centroids.",
        "source_damage": "DrivenData severe family-proxy fraction used as global q because damage geo IDs are anonymized.",
        "global_q": global_q,
        "selected_zone_count": len(zones),
        "selected_zones": zones_df[["vdcmun_id", "vdcmun_name", "district_name", "population", "center_lat", "center_lon"]].to_dict(orient="records"),
        "centers": centers_df,
        "corridors": [corridor.__dict__ for corridor in CORRIDORS],
        "caveat": "This is a mechanism-active access stress test, not a full observed Nepal validation.",
    }
    return instance, metadata


def precompute_travel_times(states: list[Any], zones_df: pd.DataFrame, centers_df: list[dict[str, Any]]) -> dict[str, list[list[float]]]:
    base = np.zeros((len(centers_df), len(zones_df)), dtype=float)
    zone_rows = list(zones_df.itertuples(index=False))
    for center_index, center in enumerate(centers_df):
        for zone_index, zone in enumerate(zone_rows):
            km = haversine_km(float(center["lat"]), float(center["lon"]), float(zone.center_lat), float(zone.center_lon))
            mountain_factor = 1.45 if zone.district_name in {"Gorkha", "Rasuwa", "Dolakha", "Sindhupalchok"} else 1.20
            same_district_discount = 0.55 if zone.district_name == center["district_name"] else 1.0
            base[center_index, zone_index] = 15.0 + 1.8 * km * mountain_factor * same_district_discount
    matrices: dict[str, list[list[float]]] = {}
    for state in states:
        matrix = base.copy()
        for failed_link_id in state.failed_links:
            corridor = next(corridor for corridor in CORRIDORS if corridor.id == failed_link_id)
            for center_index, center in enumerate(centers_df):
                for zone_index, zone in enumerate(zone_rows):
                    if zone.district_name not in corridor.affected_districts:
                        continue
                    if center["district_name"] == zone.district_name:
                        matrix[center_index, zone_index] += 0.25 * corridor.penalty_minutes
                    else:
                        matrix[center_index, zone_index] += corridor.penalty_minutes
        matrices[state.id] = matrix.tolist()
    return matrices


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))


def cached_fixed_y(cache: CheckpointStore, key_prefix: str, instance: DADInstance, y: np.ndarray, force: bool) -> SimpleNamespace:
    payload = cache.get_or_compute(
        versioned_key(f"{key_prefix}__fixed_y_{hash_array(y)}"),
        lambda: fixed_result_payload(evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=160)),
        force=force,
    )
    return payload_to_result(payload)


def evaluate_no_investment(instance: DADInstance) -> float:
    from ejor_dad.fixed_y import evaluate_plan_losses
    from ejor_dad.states import nominal_probabilities
    from ejor_dad.tv import worst_case_tv_distribution

    y = np.zeros(len(instance.links), dtype=float)
    z = np.zeros(len(instance.zones), dtype=float)
    w = np.zeros(len(instance.centers), dtype=float)
    losses, _, _ = evaluate_plan_losses(instance, z, w)
    nominal = nominal_probabilities(instance.links, instance.states, y)
    return float(
        worst_case_tv_distribution(
            nominal,
            losses,
            instance.ambiguity_radius,
            maximize=True,
            density_cap=instance.ambiguity_density_cap,
        ).value
    )


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


def hash_array(values: np.ndarray) -> str:
    import hashlib

    rounded = np.round(np.asarray(values, dtype=float), 8)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def versioned_key(key: str) -> str:
    return f"{CACHE_VERSION}__{key}"


def write_config(out: Path, experiment_id: str, payload: dict[str, Any]) -> None:
    atomic_write_text(out / "configs" / f"{experiment_id}.json", json.dumps(payload, indent=2, default=str))


def write_table(dataframe: pd.DataFrame, out: Path, stem: str) -> None:
    atomic_write_dataframe(dataframe, out / "tables" / f"{stem}.csv")
    atomic_write_dataframe(dataframe, out / "tables" / f"{stem}.tex", kind="latex", escape=True)


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


if __name__ == "__main__":
    main()
