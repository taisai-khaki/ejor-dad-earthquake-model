from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_dataframe, atomic_write_text


REPO_DATA = Path("data_work/nepal/diversity_cooking_fuel_repo/data")
HDX_ADMIN3 = Path("data_work/nepal/admin_boundaries/npl_admin3.geojson")
OUT_DIR = Path("data_work/nepal/recovered_geography")
TABLE_DIR = OUT_DIR / "tables"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    demographics = pd.read_csv(REPO_DATA / "data_household/csv_household_demographics.csv.gz")
    location_map = pd.read_csv(REPO_DATA / "data_household/ward_vdcmun_district_name_mapping.csv.gz")
    district_centroids = district_centroids_from_geojson(REPO_DATA / "data_maps/nepal_district.geojson")

    district = aggregate_households(demographics, location_map, ["district_id", "district_name"])
    district = district.merge(district_centroids, on="district_name", how="left")
    write_table(district, "nepal_recovered_district_population")

    vdcmun = aggregate_households(demographics, location_map, ["district_id", "district_name", "vdcmun_id", "vdcmun_name"])
    if HDX_ADMIN3.exists():
        admin3_match = match_vdcmun_to_hdx_admin3(location_map, HDX_ADMIN3)
        write_table(admin3_match, "nepal_recovered_vdcmun_hdx_admin3_match")
        vdcmun = vdcmun.merge(
            admin3_match[
                [
                    "vdcmun_id",
                    "adm3_name",
                    "adm3_pcode",
                    "adm2_pcode",
                    "center_lat",
                    "center_lon",
                    "area_sqkm",
                    "match_status",
                ]
            ].drop_duplicates("vdcmun_id"),
            on="vdcmun_id",
            how="left",
        )
    write_table(vdcmun, "nepal_recovered_vdcmun_population")

    ward = aggregate_households(
        demographics,
        location_map,
        ["district_id", "district_name", "vdcmun_id", "vdcmun_name", "ward_id"],
    )
    write_table(ward, "nepal_recovered_ward_population")

    scale = pd.DataFrame(
        [
            scale_row("district", district),
            scale_row("vdcmun", vdcmun),
            scale_row("ward", ward),
        ]
    )
    write_table(scale, "nepal_recovered_geography_scale")

    damage_overview = pd.read_csv("data_work/nepal/analysis/tables/nepal_damage_overview.csv")
    severe_fraction = float(damage_overview.loc[damage_overview["metric"] == "severe_family_proxy_fraction", "value"].iloc[0])
    district_model = district.copy()
    district_model["P_rl"] = district_model["population"]
    district_model["q_rl_global_damage_proxy"] = severe_fraction
    district_model["D0_rl_global_damage_proxy"] = district_model["P_rl"] * district_model["q_rl_global_damage_proxy"]
    district_model["q_rl_status"] = "global DrivenData proxy; not zone-specific because damage geo IDs are anonymized"
    write_table(
        district_model[
            [
                "zone_id",
                "district_id",
                "district_name",
                "P_rl",
                "q_rl_global_damage_proxy",
                "D0_rl_global_damage_proxy",
                "households",
                "latitude",
                "longitude",
                "q_rl_status",
            ]
        ],
        "nepal_district_model_input_global_q_proxy",
    )

    write_linkage_assessment(demographics, location_map, district, vdcmun, ward, severe_fraction)


def aggregate_households(demographics: pd.DataFrame, location_map: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    merged = demographics.merge(location_map.drop_duplicates(), on=["district_id", "vdcmun_id", "ward_id"], how="left")
    grouped = (
        merged.groupby(keys, dropna=False)
        .agg(
            households=("household_id", "nunique"),
            population=("size_household", "sum"),
            mean_household_size=("size_household", "mean"),
            bank_account_share=("is_bank_account_present_in_household", "mean"),
            mean_household_head_age=("age_household_head", "mean"),
        )
        .reset_index()
    )
    level = keys[-1].replace("_id", "")
    grouped["zone_id"] = level + "_" + grouped[keys[-1]].astype(str)
    ordered = ["zone_id", *keys, "households", "population", "mean_household_size", "bank_account_share", "mean_household_head_age"]
    return grouped[ordered].sort_values("population", ascending=False)


def district_centroids_from_geojson(path: Path) -> pd.DataFrame:
    geo = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in geo.get("features", []):
        name = str(feature.get("properties", {}).get("DISTRICT", "")).strip()
        coordinates = list(iter_coordinates(feature.get("geometry", {}).get("coordinates", [])))
        if not name or not coordinates:
            continue
        array = np.asarray(coordinates, dtype=float)
        rows.append(
            {
                "district_name": name,
                "longitude": float(array[:, 0].mean()),
                "latitude": float(array[:, 1].mean()),
                "bbox_min_lon": float(array[:, 0].min()),
                "bbox_min_lat": float(array[:, 1].min()),
                "bbox_max_lon": float(array[:, 0].max()),
                "bbox_max_lat": float(array[:, 1].max()),
            }
        )
    return pd.DataFrame(rows)


def match_vdcmun_to_hdx_admin3(location_map: pd.DataFrame, admin3_path: Path) -> pd.DataFrame:
    vdcmun = location_map[["district_id", "district_name", "vdcmun_id", "vdcmun_name"]].drop_duplicates().copy()
    admin3 = pd.DataFrame([feature["properties"] for feature in json.loads(admin3_path.read_text(encoding="utf-8"))["features"]])
    vdcmun["norm_name"] = vdcmun["vdcmun_name"].map(normalize_admin_name)
    vdcmun["norm_district"] = vdcmun["district_name"].map(normalize_admin_name)
    admin3["norm_name"] = admin3["adm3_name"].map(normalize_admin_name)
    admin3["norm_district"] = admin3["adm2_name"].map(normalize_admin_name)
    matched = vdcmun.merge(
        admin3,
        on=["norm_name", "norm_district"],
        how="left",
        indicator=True,
    )
    matched["match_status"] = matched["_merge"].map({"both": "matched_admin3", "left_only": "unmatched_admin3"})
    return matched[
        [
            "district_id",
            "district_name",
            "vdcmun_id",
            "vdcmun_name",
            "adm3_name",
            "adm3_pcode",
            "adm2_name",
            "adm2_pcode",
            "center_lat",
            "center_lon",
            "area_sqkm",
            "match_status",
        ]
    ].drop_duplicates().sort_values(["district_id", "vdcmun_id", "adm3_pcode"])


def normalize_admin_name(value: Any) -> str:
    text = str(value).lower()
    replacements = {
        "sub-metropolitian city": "",
        "sub-metropolitan city": "",
        "metropolitan city": "",
        "rural municipality": "",
        "municipality": "",
        "gaunpalika": "",
        "nagarpalika": "",
        "neelakantha": "nilkhantha",
        "melanchi": "melamchi",
        "baitedhar": "baiteshwor",
        "netrawati dabajong": "netrawati dabjong",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "", text)


def iter_coordinates(value: Any):
    if isinstance(value, list) and len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
        yield value[:2]
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_coordinates(item)


def scale_row(name: str, table: pd.DataFrame) -> dict[str, Any]:
    return {
        "zone_scale": name,
        "zones": len(table),
        "households": int(table["households"].sum()),
        "population": float(table["population"].sum()),
        "min_population": float(table["population"].min()),
        "median_population": float(table["population"].median()),
        "max_population": float(table["population"].max()),
        "min_households": int(table["households"].min()),
        "median_households": float(table["households"].median()),
        "max_households": int(table["households"].max()),
    }


def write_linkage_assessment(
    demographics: pd.DataFrame,
    location_map: pd.DataFrame,
    district: pd.DataFrame,
    vdcmun: pd.DataFrame,
    ward: pd.DataFrame,
    severe_fraction: float,
) -> None:
    rows = [
        {
            "layer": "Recovered household geography",
            "status": "usable",
            "records": int(len(demographics)),
            "spatial_units": f"{len(district)} districts; {len(vdcmun)} municipalities/VDCs; {len(ward)} wards",
            "model_use": "Use for P_rl and interpretable Nepal zones.",
            "limitation": "No building-damage grade columns in the recovered household subset.",
        },
        {
            "layer": "DrivenData building damage",
            "status": "usable but anonymized",
            "records": 260601,
            "spatial_units": "31 geo_level_1; 1414 geo_level_2; 11595 geo_level_3 anonymized units",
            "model_use": "Use for q_rl distribution and building-damage/exposure calibration.",
            "limitation": "geo_level identifiers cannot currently be joined to real districts/VDCs/wards.",
        },
        {
            "layer": "District model input with global q",
            "status": "usable only as sensitivity baseline",
            "records": int(len(district)),
            "spatial_units": "11 real earthquake-affected districts",
            "model_use": f"P_rl from household population; q_rl set to global severe-family proxy {severe_fraction:.6f}.",
            "limitation": "This should not be presented as zone-specific observed damage.",
        },
    ]
    write_table(pd.DataFrame(rows), "nepal_data_linkage_assessment")
    atomic_write_text(
        OUT_DIR / "README.md",
        "\n".join(
            [
                "# Nepal Recovered Geography Layer",
                "",
                "Recovered from `raunakms/diversity_cooking_fuel`, which contains KLL/NPC household demographics, resources, ward/VDC/district mapping, and a Nepal district GeoJSON.",
                "",
                "Use `nepal_recovered_vdcmun_population.csv` or `nepal_recovered_ward_population.csv` for real population zones.",
                "Use `nepal_district_model_input_global_q_proxy.csv` only as a sensitivity baseline because the available DrivenData damage geography is anonymized.",
            ]
        ),
    )


def write_table(dataframe: pd.DataFrame, stem: str) -> None:
    atomic_write_dataframe(dataframe, TABLE_DIR / f"{stem}.csv")
    atomic_write_dataframe(dataframe, TABLE_DIR / f"{stem}.tex", kind="latex", escape=True)


if __name__ == "__main__":
    main()
