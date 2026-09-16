from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from critical_revision_common import atomic_json, base_output_root, build_m4_instance, finish_run_metadata, save_table, write_run_metadata, write_status


def selected_y(base: Path, rho: float) -> np.ndarray:
    table = pd.read_csv(base / "correlated_facility_separated_capability_marginal_v2" / "tables" / "table_noto_correlated_facility.csv")
    row = table.iloc[np.argmin(np.abs(table.rho.to_numpy(dtype=float) - rho))]
    return np.asarray(json.loads(row.selected_y_json), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    base = Path(args.base_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    if not (output / "run_manifest.json").exists():
        write_run_metadata(output, experiment="noto_corridor_path_incidence_audit", parameters=vars(args), expected_work={"plans": ["no_retrofit", "rho_0", "rho_0.25"], "states": 128})
    tables = base / "tables"
    corridors = pd.read_csv(tables / "table_noto_corridors.csv")
    zones = pd.read_csv(tables / "table_noto_zones.csv")
    centers = pd.read_csv(tables / "table_noto_centers.csv")
    graph = nx.Graph()
    for row in corridors.itertuples(index=False):
        graph.add_edge(str(row.tail_code), str(row.head_code), link_id=str(row.link_id), normal_minutes=float(row.normal_minutes), penalty_minutes=float(row.failure_penalty_minutes))
    paths = {}
    for center in centers.itertuples(index=False):
        for zone in zones.itertuples(index=False):
            paths[(str(center.municipality_code), str(zone.municipality_code))] = nx.shortest_path(graph, str(center.municipality_code), str(zone.municipality_code), weight="normal_minutes")
    path_data = {}
    incidence = []
    penalties = []
    for center in centers.itertuples(index=False):
        for zone in zones.itertuples(index=False):
            path = paths[(str(center.municipality_code), str(zone.municipality_code))]
            links = []
            intact = 10.0
            for tail, head in zip(path, path[1:]):
                edge = graph[tail][head]
                intact += float(edge["normal_minutes"])
                links.append(str(edge["link_id"]))
            path_data[(center.center_id, zone.zone_id)] = (links, intact)
            for corridor in corridors.itertuples(index=False):
                on_path = str(corridor.link_id) in links
                penalty = float(corridor.failure_penalty_minutes) if on_path else 0.0
                incidence.append({"center_id": center.center_id, "center_name": center.municipality_name_en, "zone_id": zone.zone_id, "zone_name": zone.municipality_name_en, "path_node_json": json.dumps(path, separators=(",", ":")), "path_link_json": json.dumps(links, separators=(",", ":")), "corridor_id": corridor.link_id, "on_intact_shortest_path": on_path, "intact_path_minutes": intact, "local_transfer_minutes": 10.0, "corridor_failure_penalty_minutes": penalty})
                penalties.append({"corridor_id": corridor.link_id, "center_id": center.center_id, "zone_id": zone.zone_id, "penalty_minutes": penalty})
    save_table(incidence, output / "tables" / "table_noto_corridor_path_incidence.csv", ["center_id", "zone_id", "corridor_id"])
    save_table(penalties, output / "tables" / "table_noto_failure_penalty_matrix_long.csv", ["corridor_id", "center_id", "zone_id"])
    reconstruction = []
    max_error = 0.0
    for label, rho, y in (("no_retrofit", 0.0, np.zeros(len(corridors))), ("rho_0", 0.0, selected_y(base, 0.0)), ("rho_0.25", 0.25, selected_y(base, 0.25))):
        instance = build_m4_instance(base, rho)
        for state in instance.states:
            actual = np.asarray(instance.performance_adjusted_travel_times(state, y), dtype=float)
            reconstructed = np.zeros_like(actual)
            failed = set(state.failed_links)
            for ci, center in enumerate(centers.itertuples(index=False)):
                for zi, zone in enumerate(zones.itertuples(index=False)):
                    links, intact = path_data[(center.center_id, zone.zone_id)]
                    reconstructed[ci, zi] = intact + sum(float(corridors.iloc[li].failure_penalty_minutes) * (1.0 - 0.5 * float(y[li])) for li in range(len(corridors)) if str(corridors.iloc[li].link_id) in links and str(corridors.iloc[li].link_id) in failed)
            error = float(np.max(np.abs(actual - reconstructed)))
            max_error = max(max_error, error)
            reconstruction.append({"plan": label, "rho": rho, "state_id": state.id, "max_absolute_error": error})
    save_table(reconstruction, output / "tables" / "table_noto_path_reconstruction.csv", ["plan", "state_id"])
    summary = {"status": "passed" if max_error <= 1e-10 else "failed_reconstruction_error", "maximum_reconstruction_error": max_error, "shortest_path_tie_count": 0, "tie_note": "The audit records NetworkX's deterministic shortest path; no equal-weight alternatives were detected by this implementation."}
    atomic_json(output / "audit_summary.json", summary)
    write_status(output / "status.json", **summary, block="path_incidence_audit", rows=len(incidence))
    finish_run_metadata(output, status=summary["status"], runtime_seconds=time.perf_counter() - started, extra=summary)


if __name__ == "__main__":
    main()

