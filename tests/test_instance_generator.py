from __future__ import annotations

from pathlib import Path

from src.instance_generator import InstanceGenerator, load_instance_json, summary_to_row
from src.parameters import SearchConfig, GenerationSettings


def _small_config() -> SearchConfig:
    return SearchConfig(
        seed=2026,
        generation=GenerationSettings(
            num_customers=8,
            num_trucks=2,
            num_drones=2,
            region="dense_urban",
            coordinate_scale=10.0,
            priority_share=(0.2, 0.3, 0.5),
            demand_min=0.5,
            demand_max=1.2,
        ),
        experiment={"instance_reps_per_size": 1},
    )


def test_generate_and_load_instance_round_trip(tmp_path: Path) -> None:
    config = _small_config()
    generator = InstanceGenerator.from_search_config(config)

    instance = generator.generate_single(seed=42, name="round_trip")
    assert instance.num_customers == 8
    assert instance.num_trucks == 2
    assert instance.num_drones == 2
    assert instance.customer_ids[0] == 1
    assert instance.customer_ids[-1] == 8

    json_path = tmp_path / "round_trip.json"
    generator.save(instance, json_path)
    loaded = load_instance_json(json_path)
    assert loaded.name == "round_trip"
    assert loaded.num_customers == instance.num_customers
    assert loaded.truck_ids() == instance.truck_ids()
    assert loaded.drone_ids() == instance.drone_ids()

    row = summary_to_row(instance)
    assert row.num_customers == 8
    assert row.name == "round_trip"
    assert len(row.class_share) == 3

