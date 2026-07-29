from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
NOTO_ROOT = ROOT / "data_work" / "noto"
PREPARED = NOTO_ROOT / "prepared"
MLIT = NOTO_ROOT / "raw" / "mlit_roads_extracted" / "240125data" / "json"
RESULTS = (
    NOTO_ROOT
    / "acute_access_graded_v4"
    / "mechanism_full_grid_v1"
    / "tables"
    / "table_noto_mechanism_ablation_full_grid.csv"
)
OUTPUT = NOTO_ROOT / "acute_access_graded_v4" / "figures" / "maps"

ZONE_FILE = PREPARED / "noto_zones.csv"
CENTER_FILE = PREPARED / "noto_centers.csv"
CORRIDOR_FILE = PREPARED / "noto_corridors.csv"
INTERCITY_FILE = MLIT / "intercity_travel_time.geojson"
CONTEXT_FILE = MLIT / "ETC2.0_speed_data.geojson"
RESTORED_FILE = MLIT / "emergency_restored_section.geojson"

LINK_LEVELS = {
    "normal": [0.75, 0.75, 0.75, 0.75, 0.75],
    "north": [0.90, 1.20, 1.70, 1.40, 1.70],
    "central": [1.00, 1.50, 1.50, 1.50, 1.40],
    "widespread": [1.80, 1.80, 1.80, 1.80, 1.80],
}
REGIME_PROBABILITIES = {
    "normal": 0.70,
    "north": 0.15,
    "central": 0.10,
    "widespread": 0.05,
}
FAILED_CENTERS = {
    "normal": set(),
    "north": {"center_17204", "center_17205"},
    "central": {"center_17461", "center_17463"},
    "widespread": {"center_17202", "center_17204", "center_17205"},
}

PLACE_LABEL_OFFSETS = {
    "Kanazawa": (8, -15),
    "Nanao": (-52, -9),
    "Anamizu": (-58, 8),
    "Wajima": (-50, 7),
    "Noto": (10, 7),
    "Suzu": (10, 8),
}
LINK_LABEL_OFFSETS = {
    1: (-44, -2),
    2: (-48, -3),
    3: (-43, 0),
    4: (6, -19),
    5: (15, 11),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": "#9aa4ad",
            "axes.linewidth": 0.7,
            "xtick.color": "#4c5661",
            "ytick.color": "#4c5661",
            "text.color": "#1c2733",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_lines(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines: list[np.ndarray] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "LineString" and coordinates:
            lines.append(np.asarray([[float(v[0]), float(v[1])] for v in coordinates]))
        elif geometry_type == "MultiLineString" and coordinates:
            for part in coordinates:
                lines.append(np.asarray([[float(v[0]), float(v[1])] for v in part]))
    return lines


def load_inputs() -> dict[str, object]:
    zones = read_csv(ZONE_FILE)
    centers = read_csv(CENTER_FILE)
    corridors = read_csv(CORRIDOR_FILE)
    center_by_code = {row["municipality_code"]: row for row in centers}
    center_by_id = {row["center_id"]: row for row in centers}
    intercity = load_lines(INTERCITY_FILE)
    context = load_lines(CONTEXT_FILE)
    restored = load_lines(RESTORED_FILE)

    score_matrix = np.zeros((len(corridors), len(intercity)), dtype=float)
    for corridor_index, corridor in enumerate(corridors):
        tail = center_by_code[corridor["tail_code"]]
        head = center_by_code[corridor["head_code"]]
        tail_point = np.asarray([float(tail["longitude"]), float(tail["latitude"])])
        head_point = np.asarray([float(head["longitude"]), float(head["latitude"])])
        for line_index, line in enumerate(intercity):
            direct = np.sum((line[0] - tail_point) ** 2) + np.sum((line[-1] - head_point) ** 2)
            reverse = np.sum((line[-1] - tail_point) ** 2) + np.sum((line[0] - head_point) ** 2)
            score_matrix[corridor_index, line_index] = min(direct, reverse)

    best_assignment = min(
        itertools.permutations(range(len(intercity))),
        key=lambda assignment: sum(
            score_matrix[index, line_index] for index, line_index in enumerate(assignment)
        ),
    )
    candidate_lines = [intercity[line_index] for line_index in best_assignment]

    result_rows = read_csv(RESULTS)
    policies: dict[float, dict[str, object]] = {}
    for row in result_rows:
        if row["model"] != "M4":
            continue
        rho = float(row["rho"])
        if rho not in {0.0, 0.25}:
            continue
        policies[rho] = {
            "y": np.asarray(json.loads(row["selected_y_json"]), dtype=float),
            "z": np.asarray(json.loads(row["selected_z_json"]), dtype=float),
            "w": np.asarray(json.loads(row["selected_w_json"]), dtype=float),
            "objective": float(row["objective"]),
        }
    if set(policies) != {0.0, 0.25}:
        raise RuntimeError("The M4 rho=0 and rho=0.25 policies were not found.")

    return {
        "zones": zones,
        "centers": centers,
        "corridors": corridors,
        "center_by_code": center_by_code,
        "center_by_id": center_by_id,
        "candidate_lines": candidate_lines,
        "context": context,
        "restored": restored,
        "policies": policies,
    }


def map_bounds(data: dict[str, object]) -> tuple[float, float, float, float]:
    arrays = list(data["candidate_lines"])
    arrays.extend(
        np.asarray(
            [[float(row["longitude"]), float(row["latitude"])]],
            dtype=float,
        )
        for row in data["centers"]
    )
    points = np.vstack(arrays)
    return (
        float(points[:, 0].min() - 0.055),
        float(points[:, 0].max() + 0.085),
        float(points[:, 1].min() - 0.055),
        float(points[:, 1].max() + 0.075),
    )


def setup_map(
    ax: plt.Axes,
    data: dict[str, object],
    bounds: tuple[float, float, float, float],
    coordinate_labels: bool = True,
) -> None:
    ax.add_collection(
        LineCollection(
            data["context"],
            colors="#d9dee3",
            linewidths=0.28,
            alpha=0.74,
            zorder=0,
        )
    )
    ax.add_collection(
        LineCollection(
            data["restored"],
            colors="#b8c1ca",
            linewidths=0.55,
            alpha=0.58,
            zorder=1,
        )
    )
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    mean_latitude = 0.5 * (bounds[2] + bounds[3])
    ax.set_aspect(1.0 / math.cos(math.radians(mean_latitude)))
    ax.set_facecolor("#f8fafb")
    ax.grid(color="#e5e9ed", linewidth=0.45, zorder=-2)
    if coordinate_labels:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.tick_params(labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.text(
        0.035,
        0.95,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(0.035, 0.94),
        xytext=(0.035, 0.88),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#263746", lw=1.25),
        zorder=20,
    )
    add_scale_bar(ax, bounds, 20.0)


def add_scale_bar(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    length_km: float,
) -> None:
    mean_latitude = 0.5 * (bounds[2] + bounds[3])
    longitude_span = length_km / (111.32 * math.cos(math.radians(mean_latitude)))
    x_start = bounds[0] + 0.60 * (bounds[1] - bounds[0])
    y = bounds[2] + 0.045 * (bounds[3] - bounds[2])
    ax.plot(
        [x_start, x_start + longitude_span],
        [y, y],
        color="#263746",
        lw=2.0,
        solid_capstyle="butt",
        zorder=20,
    )
    ax.plot(
        [x_start, x_start],
        [y - 0.006, y + 0.006],
        color="#263746",
        lw=1.2,
        zorder=20,
    )
    ax.plot(
        [x_start + longitude_span, x_start + longitude_span],
        [y - 0.006, y + 0.006],
        color="#263746",
        lw=1.2,
        zorder=20,
    )
    ax.text(
        x_start + longitude_span / 2,
        y + 0.011,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#263746",
        zorder=20,
    )


def annotate_place(
    ax: plt.Axes,
    row: dict[str, str],
    label: str | None = None,
    fontsize: float = 8.2,
) -> None:
    name = row["municipality_name_en"]
    offset = PLACE_LABEL_OFFSETS[name]
    ax.annotate(
        label or name,
        (float(row["longitude"]), float(row["latitude"])),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="medium",
        ha="left",
        va="center",
        color="#18242f",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.76),
        zorder=30,
    )


def line_midpoint(line: np.ndarray) -> np.ndarray:
    segment_lengths = np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1))
    if segment_lengths.sum() <= 0:
        return line[0]
    target = segment_lengths.sum() / 2
    cumulative = np.cumsum(segment_lengths)
    segment_index = int(np.searchsorted(cumulative, target))
    prior = cumulative[segment_index - 1] if segment_index else 0.0
    fraction = (target - prior) / segment_lengths[segment_index]
    return line[segment_index] + fraction * (line[segment_index + 1] - line[segment_index])


def zone_marker_size(population: float, maximum: float) -> float:
    return 110.0 + 650.0 * math.sqrt(population / maximum)


def facility_marker_size(capacity: float, maximum: float) -> float:
    return 35.0 + 190.0 * math.sqrt(capacity / maximum)


def addition_marker_size(addition: float, maximum: float) -> float:
    if addition <= 1e-9:
        return 18.0
    return 55.0 + 330.0 * math.sqrt(addition / maximum)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = OUTPUT / f"{stem}.{extension}"
        fig.savefig(
            path,
            dpi=320 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        outputs.append(path)
    plt.close(fig)
    return outputs


def draw_study_area(
    data: dict[str, object],
    bounds: tuple[float, float, float, float],
) -> list[Path]:
    fig = plt.figure(figsize=(12.4, 8.3))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.45, 1.35], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    setup_map(ax, data, bounds, coordinate_labels=True)

    q_norm = Normalize(vmin=0.0, vmax=0.35)
    q_cmap = mpl.colormaps["YlOrRd"]
    phi_norm = Normalize(vmin=0.08, vmax=0.62)
    phi_cmap = mpl.colormaps["Blues"]
    maximum_population = max(float(row["population"]) for row in data["zones"])
    maximum_capacity = max(float(row["existing_capacity"]) for row in data["centers"])

    for link_index, (corridor, line) in enumerate(
        zip(data["corridors"], data["candidate_lines"]), start=1
    ):
        probability = float(corridor["baseline_failure_probability"])
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=phi_cmap(phi_norm(probability)),
            lw=2.2 + 3.5 * probability,
            solid_capstyle="round",
            zorder=7,
        )
        midpoint = line_midpoint(line)
        ax.annotate(
            f"$y_{link_index}$\nΦ={probability:.2f}",
            midpoint,
            xytext=LINK_LABEL_OFFSETS[link_index],
            textcoords="offset points",
            fontsize=7.4,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor=phi_cmap(phi_norm(probability)),
                linewidth=0.8,
                alpha=0.94,
            ),
            zorder=25,
        )

    center_by_code = data["center_by_code"]
    for zone in data["zones"]:
        center = center_by_code[zone["municipality_code"]]
        population = float(zone["population"])
        collapse_fraction = float(zone["collapse_fraction"])
        ax.scatter(
            float(center["longitude"]),
            float(center["latitude"]),
            s=zone_marker_size(population, maximum_population),
            c=[q_cmap(q_norm(collapse_fraction))],
            edgecolors="#5b3925",
            linewidths=0.9,
            alpha=0.88,
            zorder=12,
        )

    for center in data["centers"]:
        capacity = float(center["existing_capacity"])
        ax.scatter(
            float(center["longitude"]),
            float(center["latitude"]),
            s=facility_marker_size(capacity, maximum_capacity),
            marker="D",
            facecolors="white",
            edgecolors="#134b5f",
            linewidths=1.55,
            zorder=16,
        )
        ax.scatter(
            float(center["longitude"]),
            float(center["latitude"]),
            s=18,
            marker="+",
            c="#134b5f",
            linewidths=1.4,
            zorder=17,
        )
        annotate_place(ax, center)

    ax.set_title(
        "Noto Peninsula empirical network and calibrated exposure",
        loc="left",
        fontweight="medium",
        pad=10,
    )
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.text(0.03, 0.97, "Map encodings", fontsize=11.5, fontweight="medium", va="top")
    legend_ax.text(
        0.03,
        0.92,
        "Demand-zone circle",
        fontsize=9.2,
        fontweight="medium",
        va="top",
    )
    legend_ax.text(0.03, 0.885, "Area = population", fontsize=8.3, color="#53606b")
    for index, population in enumerate((10_000, 25_000, 50_000)):
        y = 0.835 - index * 0.075
        legend_ax.scatter(
            0.17,
            y,
            s=zone_marker_size(population, maximum_population),
            facecolors="#efb37f",
            edgecolors="#5b3925",
            linewidths=0.8,
        )
        legend_ax.text(
            0.37,
            y,
            f"{population / 1000:.0f}k residents",
            transform=legend_ax.transAxes,
            fontsize=8.2,
            va="center",
        )

    q_axis = legend_ax.inset_axes([0.06, 0.49, 0.80, 0.045])
    mpl.colorbar.ColorbarBase(q_axis, cmap=q_cmap, norm=q_norm, orientation="horizontal")
    q_axis.set_title("Fill = severe-damage fraction $q_r$", fontsize=8.4, loc="left", pad=5)
    q_axis.tick_params(labelsize=7.4, length=2)

    legend_ax.text(
        0.03,
        0.46,
        "Emergency-facility diamond",
        fontsize=9.2,
        fontweight="medium",
        va="top",
    )
    legend_ax.text(0.03, 0.425, "Area = existing capacity", fontsize=8.3, color="#53606b")
    for index, capacity in enumerate((200, 2_000, 5_500)):
        y = 0.395 - index * 0.07
        legend_ax.scatter(
            0.17,
            y,
            s=facility_marker_size(capacity, maximum_capacity),
            marker="D",
            facecolors="white",
            edgecolors="#134b5f",
            linewidths=1.3,
        )
        legend_ax.text(
            0.37,
            y,
            f"{capacity:,g} capacity units",
            transform=legend_ax.transAxes,
            fontsize=8.0,
            va="center",
        )

    phi_axis = legend_ax.inset_axes([0.06, 0.135, 0.80, 0.045])
    mpl.colorbar.ColorbarBase(phi_axis, cmap=phi_cmap, norm=phi_norm, orientation="horizontal")
    phi_axis.set_title("Route color = baseline failure probability Φ", fontsize=8.4, loc="left", pad=5)
    phi_axis.tick_params(labelsize=7.4, length=2)
    return save_figure(fig, "fig_noto_01_study_area")


def draw_policy_panel(
    ax: plt.Axes,
    data: dict[str, object],
    bounds: tuple[float, float, float, float],
    rho: float,
    maximum_addition: float,
) -> None:
    setup_map(ax, data, bounds, coordinate_labels=False)
    policy = data["policies"][rho]
    y = policy["y"]
    z = policy["z"]
    w = policy["w"]

    for link_index, (line, value) in enumerate(zip(data["candidate_lines"], y), start=1):
        if value <= 1e-9:
            color = "#aab3bb"
            linewidth = 1.25
        else:
            color = mpl.colormaps["Blues"](0.48 + 0.46 * value)
            linewidth = 2.0 + 5.4 * value
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=color,
            lw=linewidth,
            solid_capstyle="round",
            zorder=8,
        )
        midpoint = line_midpoint(line)
        ax.annotate(
            f"$y_{link_index}$={value:g}",
            midpoint,
            xytext=LINK_LABEL_OFFSETS[link_index],
            textcoords="offset points",
            fontsize=7.2,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor=color,
                linewidth=0.75,
                alpha=0.94,
            ),
            zorder=25,
        )

    center_by_code = data["center_by_code"]
    for zone_index, zone in enumerate(data["zones"]):
        center = center_by_code[zone["municipality_code"]]
        longitude = float(center["longitude"])
        latitude = float(center["latitude"])
        if z[zone_index] > 1e-9:
            ax.scatter(
                longitude,
                latitude,
                s=320,
                facecolors="none",
                edgecolors="#2a8f62",
                linewidths=3.0,
                zorder=12,
            )
        ax.scatter(
            longitude,
            latitude,
            s=65,
            facecolors="#e7ebee",
            edgecolors="#596773",
            linewidths=0.85,
            zorder=13,
        )

    for center_index, center in enumerate(data["centers"]):
        longitude = float(center["longitude"])
        latitude = float(center["latitude"])
        addition = float(w[center_index])
        if addition > 1e-9:
            ax.scatter(
                longitude,
                latitude,
                s=addition_marker_size(addition, maximum_addition),
                marker="D",
                facecolors="#e89a38",
                edgecolors="#783d08",
                linewidths=1.25,
                alpha=0.86,
                zorder=17,
            )
            addition_label = f"{center['municipality_name_en']}\n+$w$ {addition:.0f}"
            annotate_place(ax, center, label=addition_label, fontsize=7.7)
        else:
            ax.scatter(
                longitude,
                latitude,
                s=34,
                marker="D",
                facecolors="white",
                edgecolors="#426071",
                linewidths=1.0,
                zorder=16,
            )
            annotate_place(ax, center, fontsize=7.6)

    suzu_index = next(
        index
        for index, zone in enumerate(data["zones"])
        if zone["municipality_name_en"] == "Suzu"
    )
    if z[suzu_index] > 1e-9:
        suzu_center = center_by_code[data["zones"][suzu_index]["municipality_code"]]
        ax.annotate(
            f"$z_{{Suzu}}$={z[suzu_index]:.3f}",
            (float(suzu_center["longitude"]), float(suzu_center["latitude"])),
            xytext=(-6, -28),
            textcoords="offset points",
            fontsize=7.6,
            color="#14623f",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#2a8f62"),
            zorder=30,
        )

    title_prefix = "(a)" if rho == 0 else "(b)"
    ax.set_title(
        f"{title_prefix} M4 policy at ρ={rho:g}\n"
        f"robust objective = {policy['objective']:,.1f}",
        loc="left",
        fontweight="medium",
        pad=8,
    )


def draw_policy_comparison(
    data: dict[str, object],
    bounds: tuple[float, float, float, float],
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.8))
    maximum_addition = max(
        float(value)
        for policy in data["policies"].values()
        for value in policy["w"]
    )
    draw_policy_panel(axes[0], data, bounds, 0.0, maximum_addition)
    draw_policy_panel(axes[1], data, bounds, 0.25, maximum_addition)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=mpl.colormaps["Blues"](0.94), lw=7, label="Full road retrofit"),
            Line2D([0], [0], color=mpl.colormaps["Blues"](0.595), lw=3.4, label="Partial road retrofit"),
            Line2D([0], [0], color="#aab3bb", lw=1.3, label="No road retrofit"),
            Line2D(
                [0],
                [0],
                marker="D",
                markersize=8,
                markerfacecolor="#e89a38",
                markeredgecolor="#783d08",
                linewidth=0,
                label="Capacity addition $w_k$",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="#2a8f62",
                markeredgewidth=2.4,
                linewidth=0,
                label="Selected renovation $z_r$",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.08),
        frameon=True,
        facecolor="white",
        edgecolor="#c9d0d6",
        fontsize=7.8,
    )
    fig.suptitle(
        "Policy reoptimization under decision-dependent ambiguity",
        fontsize=14,
        fontweight="medium",
        y=0.975,
    )
    fig.text(
        0.5,
        0.045,
        "Observed switch: partial retrofit moves from $y_1$ to $y_5$; "
        "capacity shifts from Suzu toward Anamizu while Suzu renovation remains unchanged.",
        ha="center",
        va="center",
        fontsize=9.1,
        color="#263746",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f3f6f8", edgecolor="#cad2d8"),
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.095, wspace=0.065)
    return save_figure(fig, "fig_noto_02_policy_comparison")


def draw_regime_panel(
    ax: plt.Axes,
    data: dict[str, object],
    bounds: tuple[float, float, float, float],
    regime: str,
    panel_label: str,
    norm: Normalize,
) -> None:
    setup_map(ax, data, bounds, coordinate_labels=False)
    cmap = mpl.colormaps["OrRd"]
    for link_index, (line, multiplier) in enumerate(
        zip(data["candidate_lines"], LINK_LEVELS[regime]), start=1
    ):
        color = cmap(norm(multiplier))
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=color,
            lw=1.7 + 2.2 * multiplier,
            solid_capstyle="round",
            zorder=8,
        )
        midpoint = line_midpoint(line)
        ax.annotate(
            f"$y_{link_index}$ ×{multiplier:.2g}",
            midpoint,
            xytext=LINK_LABEL_OFFSETS[link_index],
            textcoords="offset points",
            fontsize=6.9,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor=color,
                linewidth=0.7,
                alpha=0.93,
            ),
            zorder=25,
        )

    failed = FAILED_CENTERS[regime]
    for center in data["centers"]:
        longitude = float(center["longitude"])
        latitude = float(center["latitude"])
        if center["center_id"] in failed:
            ax.scatter(
                longitude,
                latitude,
                s=145,
                marker="X",
                facecolors="#bd2b2b",
                edgecolors="white",
                linewidths=1.0,
                zorder=18,
            )
        else:
            ax.scatter(
                longitude,
                latitude,
                s=58,
                marker="D",
                facecolors="white",
                edgecolors="#27576a",
                linewidths=1.15,
                zorder=17,
            )
        annotate_place(ax, center, fontsize=7.4)

    probability = REGIME_PROBABILITIES[regime]
    failed_names = [
        data["center_by_id"][center_id]["municipality_name_en"]
        for center_id in sorted(failed)
    ]
    failure_text = "No facility outage" if not failed_names else "Out: " + ", ".join(failed_names)
    ax.set_title(
        f"{panel_label} {regime.capitalize()} regime  ($P$={probability:.2f})\n{failure_text}",
        loc="left",
        fontsize=10.5,
        fontweight="medium",
        pad=7,
    )


def draw_hazard_regimes(
    data: dict[str, object],
    bounds: tuple[float, float, float, float],
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13.1, 10.2))
    norm = Normalize(vmin=0.75, vmax=1.80)
    for ax, regime, label in zip(
        axes.flat,
        ("normal", "north", "central", "widespread"),
        ("(a)", "(b)", "(c)", "(d)"),
    ):
        draw_regime_panel(ax, data, bounds, regime, label, norm)

    color_axis = fig.add_axes([0.19, 0.058, 0.44, 0.018])
    mpl.colorbar.ColorbarBase(
        color_axis,
        cmap=mpl.colormaps["OrRd"],
        norm=norm,
        orientation="horizontal",
    )
    color_axis.set_xlabel(
        "Scenario multiplier applied to link failure intensity",
        fontsize=8.5,
        labelpad=2,
    )
    color_axis.tick_params(labelsize=7.6, length=2)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="#27576a",
                linewidth=0,
                label="Operational facility",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                markersize=8,
                markerfacecolor="#bd2b2b",
                markeredgecolor="white",
                linewidth=0,
                label="Unavailable facility",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.67, 0.046),
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Joint road–facility hazard regimes used in the Noto experiment",
        fontsize=14,
        fontweight="medium",
        y=0.985,
    )
    fig.text(
        0.01,
        0.006,
        "Panels encode scenario-specific corridor intensity and deterministic facility availability; "
        "they do not imply geographic hazard-zone polygons.",
        fontsize=7.5,
        color="#53606b",
        ha="left",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.925, bottom=0.115, wspace=0.055, hspace=0.12)
    return save_figure(fig, "fig_noto_03_hazard_regimes")


def add_workflow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025 * width,
        y + height - 0.16 * height,
        title,
        fontsize=10.5,
        fontweight="medium",
        color="#172430",
        va="center",
    )
    ax.text(
        x + 0.04 * width,
        y + height - 0.32 * height,
        "\n".join(lines),
        fontsize=8.4,
        color="#344653",
        va="top",
        linespacing=1.55,
    )


def add_workflow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#5f7180",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def draw_workflow() -> list[Path]:
    fig, ax = plt.subplots(figsize=(14.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.965,
        "Operational decision-dependent robust planning workflow",
        fontsize=15,
        fontweight="medium",
        va="top",
    )
    ax.text(
        0.02,
        0.915,
        "Observed layers, scenario-calibrated uncertainty, operational feasibility, and policy outputs",
        fontsize=9.3,
        color="#53606b",
        va="top",
    )

    add_workflow_box(
        ax,
        0.025,
        0.50,
        0.20,
        0.32,
        "1  Empirical layers",
        [
            "Population and damage",
            "MLIT road geometry and delay",
            "Facilities and base capacity",
            "Hazard-regime specification",
        ],
        "#eef4f7",
        "#5f8799",
    )
    add_workflow_box(
        ax,
        0.285,
        0.50,
        0.20,
        0.32,
        "2  Model construction",
        [
            "Exposure:  $P_r$, $q_r$",
            "Network:  $L$, $\\tau$, $\\Phi$",
            "Joint road–facility states $s$",
            "Decision-dependent $\\pi_s(y)$",
        ],
        "#f1f5ee",
        "#6f966d",
    )
    add_workflow_box(
        ax,
        0.545,
        0.50,
        0.20,
        0.32,
        "3  Stage 1",
        [
            "Minimize robust operational loss",
            "Choose road $y$, renovation $z$,",
            "and capacity $w$ jointly",
            "Enforce protection and service",
        ],
        "#fff3df",
        "#c18434",
    )
    add_workflow_box(
        ax,
        0.805,
        0.50,
        0.17,
        0.32,
        "4  Stage 2",
        [
            "Maximize minimum service",
            "within $V^*+\\tau_\\rho$",
            "Report separate certificate",
        ],
        "#f7eef4",
        "#a16b91",
    )

    for x_start, x_end in ((0.225, 0.285), (0.485, 0.545), (0.745, 0.805)):
        add_workflow_arrow(ax, (x_start + 0.008, 0.66), (x_end - 0.008, 0.66))

    add_workflow_box(
        ax,
        0.12,
        0.13,
        0.25,
        0.22,
        "Ambiguity set",
        [
            "TV radius: 0.5 * ||p - pi(y)||_1 <= rho",
            "Support cap: $p_s \\leq \\kappa\\,\\pi_s(y)$",
            "Support preservation",
        ],
        "#f4f1fb",
        "#8071ae",
    )
    add_workflow_box(
        ax,
        0.43,
        0.13,
        0.22,
        0.22,
        "Statewise recourse",
        [
            "Facility outages and access",
            "Service-constrained dispatch",
            "Loss and service evaluation",
        ],
        "#edf6f5",
        "#4f948c",
    )
    add_workflow_box(
        ax,
        0.71,
        0.13,
        0.23,
        0.22,
        "Practitioner outputs",
        [
            "Implementable policy $(y,z,w)$",
            "Robust value and service floor",
            "Maps, rankings, sensitivities",
        ],
        "#eef3fa",
        "#5c7da5",
    )

    add_workflow_arrow(ax, (0.245, 0.35), (0.61, 0.50), color="#8071ae")
    add_workflow_arrow(ax, (0.54, 0.35), (0.65, 0.50), color="#4f948c")
    add_workflow_arrow(ax, (0.80, 0.50), (0.82, 0.35), color="#5c7da5")
    add_workflow_arrow(
        ax,
        (0.47, 0.13),
        (0.395, 0.50),
        color="#4f948c",
        connectionstyle="arc3,rad=-0.28",
    )
    ax.text(
        0.34,
        0.405,
        "retrofit changes\nprobabilities and access",
        fontsize=7.8,
        color="#53606b",
        ha="center",
        va="center",
    )
    return save_figure(fig, "fig_noto_04_model_workflow")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_support_files(data: dict[str, object], outputs: list[Path]) -> list[Path]:
    captions = OUTPUT / "map_captions.md"
    captions.write_text(
        "\n".join(
            [
                "# Suggested figure captions",
                "",
                "## Figure 1 — Study area and calibrated layers",
                "Noto Peninsula empirical network. Demand-zone circle area is proportional to population, "
                "circle color shows the severe-damage fraction, and facility-diamond area represents existing "
                "capacity. The five candidate road-retrofit corridors follow the MLIT intercity geometry; "
                "route color and width show the calibrated baseline failure probability.",
                "",
                "## Figure 2 — Policy comparison",
                "Reoptimized M4 policies at ambiguity radii rho=0 and rho=0.25. The robust solution moves the "
                "partial road investment from Kanazawa–Nanao to Anamizu–Suzu and reallocates capacity from "
                "Suzu toward Anamizu, while the Suzu renovation decision remains unchanged. Values are from "
                "the completed full-grid mechanism experiment.",
                "",
                "## Figure 3 — Joint hazard regimes",
                "Scenario-calibrated joint road–facility hazard regimes. Corridor color and width represent "
                "the scenario multiplier on link failure intensity, and red crosses identify unavailable "
                "facilities. These panels encode scenario states rather than geographic hazard-zone polygons.",
                "",
                "## Figure 4 — Model workflow",
                "Operational decision-dependent robust planning workflow linking empirical exposure, network, "
                "facility, and hazard layers to joint state construction, capped total-variation ambiguity, "
                "Stage 1 robust-loss minimization, Stage 2 epsilon-constraint service maximization, and "
                "implementable policy outputs.",
                "",
                "## Source note",
                "Population, damage, facility, and corridor attributes are read from the prepared Noto model "
                "tables. Road geometry and context are read from the 25 January 2024 MLIT GeoJSON files. "
                "Policies are read directly from the completed M4 full-grid results table.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = OUTPUT / "map_generation_manifest.json"
    input_files = [
        ZONE_FILE,
        CENTER_FILE,
        CORRIDOR_FILE,
        INTERCITY_FILE,
        CONTEXT_FILE,
        RESTORED_FILE,
        RESULTS,
        Path(__file__).resolve(),
    ]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in input_files
        ],
        "policy_rows": {
            str(rho): {
                "y": data["policies"][rho]["y"].tolist(),
                "z": data["policies"][rho]["z"].tolist(),
                "w": data["policies"][rho]["w"].tolist(),
                "objective": data["policies"][rho]["objective"],
            }
            for rho in sorted(data["policies"])
        },
        "hazard_regimes": {
            regime: {
                "probability": REGIME_PROBABILITIES[regime],
                "failed_centers": sorted(FAILED_CENTERS[regime]),
                "link_multipliers": LINK_LEVELS[regime],
            }
            for regime in LINK_LEVELS
        },
        "outputs": [str(path) for path in outputs] + [str(captions)],
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return [captions, manifest]


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    bounds = map_bounds(data)
    outputs: list[Path] = []
    outputs.extend(draw_study_area(data, bounds))
    outputs.extend(draw_policy_comparison(data, bounds))
    outputs.extend(draw_hazard_regimes(data, bounds))
    outputs.extend(draw_workflow())
    outputs.extend(write_support_files(data, outputs))
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
