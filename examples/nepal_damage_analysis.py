from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_dataframe, atomic_write_text


RAW_DIR = Path("data_work/nepal/raw")
OUT_DIR = Path("data_work/nepal/analysis")
TABLE_DIR = OUT_DIR / "tables"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    values = pd.read_csv(RAW_DIR / "train_values.csv")
    labels = pd.read_csv(RAW_DIR / "train_labels.csv")
    data = values.merge(labels, on="building_id", validate="one_to_one")
    data["severe_damage"] = (data["damage_grade"] == 3).astype(int)
    data["moderate_or_severe_damage"] = (data["damage_grade"] >= 2).astype(int)
    data["family_proxy"] = data["count_families"].clip(lower=1)
    data["severe_family_proxy"] = data["family_proxy"] * data["severe_damage"]
    data["moderate_or_severe_family_proxy"] = data["family_proxy"] * data["moderate_or_severe_damage"]

    write_overview(data)
    for geo_col in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]:
        zone = zone_summary(data, geo_col)
        write_table(zone, f"nepal_zone_summary_{geo_col}")

    scale = scale_assessment(data)
    write_table(scale, "nepal_zone_scale_assessment")
    top_geo2 = zone_summary(data, "geo_level_2_id").sort_values("severe_family_proxy", ascending=False).head(50)
    write_table(top_geo2, "nepal_top50_geo2_exposure_zones")
    top_geo3 = zone_summary(data, "geo_level_3_id").sort_values("severe_family_proxy", ascending=False).head(100)
    write_table(top_geo3, "nepal_top100_geo3_exposure_zones")

    model_input = make_model_input_geo2(data)
    write_table(model_input, "nepal_geo2_model_input")
    write_metadata(data)


def write_overview(data: pd.DataFrame) -> None:
    rows = [
        {"metric": "buildings", "value": len(data)},
        {"metric": "geo_level_1_zones", "value": data["geo_level_1_id"].nunique()},
        {"metric": "geo_level_2_zones", "value": data["geo_level_2_id"].nunique()},
        {"metric": "geo_level_3_zones", "value": data["geo_level_3_id"].nunique()},
        {"metric": "severe_damage_buildings", "value": int(data["severe_damage"].sum())},
        {"metric": "severe_damage_fraction", "value": float(data["severe_damage"].mean())},
        {"metric": "moderate_or_severe_damage_fraction", "value": float(data["moderate_or_severe_damage"].mean())},
        {"metric": "family_proxy_total", "value": int(data["family_proxy"].sum())},
        {"metric": "severe_family_proxy_total", "value": int(data["severe_family_proxy"].sum())},
        {"metric": "severe_family_proxy_fraction", "value": float(data["severe_family_proxy"].sum() / data["family_proxy"].sum())},
    ]
    write_table(pd.DataFrame(rows), "nepal_damage_overview")


def zone_summary(data: pd.DataFrame, geo_col: str) -> pd.DataFrame:
    grouped = (
        data.groupby(geo_col)
        .agg(
            buildings=("building_id", "size"),
            family_proxy=("family_proxy", "sum"),
            severe_buildings=("severe_damage", "sum"),
            moderate_or_severe_buildings=("moderate_or_severe_damage", "sum"),
            severe_family_proxy=("severe_family_proxy", "sum"),
            moderate_or_severe_family_proxy=("moderate_or_severe_family_proxy", "sum"),
            mean_floors=("count_floors_pre_eq", "mean"),
            mean_age=("age", "mean"),
            mean_area_percentage=("area_percentage", "mean"),
            mean_height_percentage=("height_percentage", "mean"),
            share_mud_mortar_stone=("has_superstructure_mud_mortar_stone", "mean"),
            share_adobe_mud=("has_superstructure_adobe_mud", "mean"),
            share_rc_engineered=("has_superstructure_rc_engineered", "mean"),
        )
        .reset_index()
    )
    grouped["q_building_severe"] = grouped["severe_buildings"] / grouped["buildings"]
    grouped["q_family_proxy_severe"] = grouped["severe_family_proxy"] / grouped["family_proxy"]
    grouped["q_building_moderate_or_severe"] = grouped["moderate_or_severe_buildings"] / grouped["buildings"]
    grouped["q_family_proxy_moderate_or_severe"] = grouped["moderate_or_severe_family_proxy"] / grouped["family_proxy"]
    grouped["zone_id"] = geo_col.replace("_id", "") + "_" + grouped[geo_col].astype(str)
    ordered = [
        "zone_id",
        geo_col,
        "buildings",
        "family_proxy",
        "severe_buildings",
        "severe_family_proxy",
        "q_building_severe",
        "q_family_proxy_severe",
        "moderate_or_severe_buildings",
        "moderate_or_severe_family_proxy",
        "q_building_moderate_or_severe",
        "q_family_proxy_moderate_or_severe",
        "mean_floors",
        "mean_age",
        "mean_area_percentage",
        "mean_height_percentage",
        "share_mud_mortar_stone",
        "share_adobe_mud",
        "share_rc_engineered",
    ]
    return grouped[ordered].sort_values(["severe_family_proxy", "buildings"], ascending=[False, False])


def scale_assessment(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for geo_col in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]:
        zone = zone_summary(data, geo_col)
        for threshold in [1, 10, 25, 50, 100, 250, 500]:
            eligible = zone[zone["buildings"] >= threshold]
            rows.append(
                {
                    "geo_level": geo_col,
                    "min_buildings_threshold": threshold,
                    "zones_total": len(zone),
                    "zones_meeting_threshold": len(eligible),
                    "building_coverage": float(eligible["buildings"].sum() / zone["buildings"].sum()),
                    "family_proxy_coverage": float(eligible["family_proxy"].sum() / zone["family_proxy"].sum()),
                    "severe_family_proxy_coverage": float(
                        eligible["severe_family_proxy"].sum() / max(1, zone["severe_family_proxy"].sum())
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_model_input_geo2(data: pd.DataFrame) -> pd.DataFrame:
    zone = zone_summary(data, "geo_level_2_id").copy()
    zone = zone[zone["buildings"] >= 50].copy()
    zone["P_rl_family_proxy"] = zone["family_proxy"]
    zone["q_rl"] = zone["q_family_proxy_severe"]
    zone["D0_rl_family_proxy"] = zone["P_rl_family_proxy"] * zone["q_rl"]
    zone["recommended_use"] = np.where(
        zone["buildings"] >= 100,
        "primary_geo2_zone",
        "sensitivity_or_merge_small_zone",
    )
    return zone[
        [
            "zone_id",
            "geo_level_2_id",
            "buildings",
            "P_rl_family_proxy",
            "q_rl",
            "D0_rl_family_proxy",
            "q_building_severe",
            "recommended_use",
        ]
    ].sort_values("D0_rl_family_proxy", ascending=False)


def write_metadata(data: pd.DataFrame) -> None:
    metadata = {
        "source": "DrivenData Richter's Predictor: Modeling Earthquake Damage, public Nepal earthquake train_values/train_labels files.",
        "raw_files": [str((RAW_DIR / "train_values.csv").resolve()), str((RAW_DIR / "train_labels.csv").resolve())],
        "damage_definition": "severe_damage = 1{damage_grade == 3}; moderate_or_severe = 1{damage_grade >= 2}",
        "population_proxy": "family_proxy = max(count_families, 1), used only as a household/family exposure proxy.",
        "geography_limitation": "geo_level_* identifiers are anonymized and not directly linkable to OSM roads without an external geography crosswalk.",
        "rows": int(len(data)),
        "columns": list(data.columns),
    }
    atomic_write_text(OUT_DIR / "nepal_damage_analysis_metadata.json", json.dumps(metadata, indent=2))


def write_table(dataframe: pd.DataFrame, stem: str) -> None:
    atomic_write_dataframe(dataframe, TABLE_DIR / f"{stem}.csv")
    atomic_write_dataframe(dataframe, TABLE_DIR / f"{stem}.tex", kind="latex", escape=True)


if __name__ == "__main__":
    main()
