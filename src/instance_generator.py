"""Synthetic benchmark-instance generator for UTDRP-DP."""
from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .data_models import Customer, Drone, InstanceData, InstanceSummary, ProblemConstants, PriorityClass, Truck
from .parameters import GenerationSettings, SearchConfig


@dataclass(frozen=True)
class _FleetSpec:
    """Internal immutable container for per-vehicle default specifications."""

    truck_speed_kmph: float
    truck_capacity_kg: float
    truck_cost_per_km: float
    truck_energy_cost_per_min: float
    drone_speed_kmph: float
    drone_capacity_kg: float
    drone_max_battery_wh: float
    drone_cost_per_km: float
    drone_energy_cost_per_min: float
    drone_energy_per_min_when_empty: float
    drone_energy_per_min_when_loaded: float
    drone_curb_weight_kg: float
    drone_parcel_weight_kg: float
    swap_time_s: float
    reload_time_s: float


def _to_float_list(raw: Dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return float(default)
    return float(value)


class InstanceGenerator:
    """Generate synthetic instances with validated metadata and persistence helpers."""

    def __init__(
        self,
        generation: GenerationSettings,
        priority_penalties: Dict[str, float] | None = None,
        base_class_windows: Dict[str, Sequence[float]] | None = None,
        *,
        fleet_spec: _FleetSpec | None = None,
        truck_spec: Dict | None = None,
        drone_spec: Dict | None = None,
        drone_energy: Dict | None = None,
    ) -> None:
        self.generation = generation
        self.priority_penalties = {
            "high": float((priority_penalties or {}).get("high", 100.0)),
            "medium": float((priority_penalties or {}).get("medium", 20.0)),
            "low": float((priority_penalties or {}).get("low", 1.0)),
        }
        base_class_windows = base_class_windows or {
            "high": (0.0, 60.0),
            "medium": (0.0, 120.0),
            "low": (0.0, 240.0),
        }
        self.base_class_windows = {
            "high": (float(base_class_windows["high"][0]), float(base_class_windows["high"][1])),
            "medium": (float(base_class_windows["medium"][0]), float(base_class_windows["medium"][1])),
            "low": (float(base_class_windows["low"][0]), float(base_class_windows["low"][1])),
        }
        self.fleet_spec = fleet_spec or self._build_legacy_fleet(
            truck_spec=truck_spec,
            drone_spec=drone_spec,
            drone_energy=drone_energy or {},
        )
        self._validate_inputs()

    @staticmethod
    def _build_legacy_fleet(
        truck_spec: Dict | None,
        drone_spec: Dict | None,
        drone_energy: Dict | None,
    ) -> _FleetSpec:
        if truck_spec is None or drone_spec is None:
            return _FleetSpec(
                truck_speed_kmph=35.0,
                truck_capacity_kg=500.0,
                truck_cost_per_km=0.8,
                truck_energy_cost_per_min=0.0,
                drone_speed_kmph=65.0,
                drone_capacity_kg=5.0,
                drone_max_battery_wh=1200.0,
                drone_cost_per_km=0.3,
                drone_energy_cost_per_min=0.0,
                drone_energy_per_min_when_empty=15.0,
                drone_energy_per_min_when_loaded=20.0,
                drone_curb_weight_kg=3.0,
                drone_parcel_weight_kg=2.0,
                swap_time_s=60.0,
                reload_time_s=60.0,
            )

        return _FleetSpec(
            truck_speed_kmph=float(truck_spec.get("speed_kmph", 35.0)),
            truck_capacity_kg=float(truck_spec.get("capacity_kg", 500.0)),
            truck_cost_per_km=float(truck_spec.get("cost_per_km", 0.8)),
            truck_energy_cost_per_min=float(truck_spec.get("energy_cost_per_min", 0.0)),
            drone_speed_kmph=float(drone_spec.get("speed_kmph", 65.0)),
            drone_capacity_kg=float(drone_spec.get("capacity_kg", 5.0)),
            drone_max_battery_wh=float(drone_spec.get("max_battery_wh", 1200.0)),
            drone_cost_per_km=float(drone_spec.get("cost_per_km", 0.3)),
            drone_energy_cost_per_min=float(drone_energy.get("energy_loaded_per_min", drone_spec.get("energy_cost_per_min", 0.0)))
            if drone_energy
            else float(drone_spec.get("energy_cost_per_min", 0.0)),
            drone_energy_per_min_when_empty=float(drone_energy.get("energy_empty_per_min", 15.0)) if drone_energy else 15.0,
            drone_energy_per_min_when_loaded=float(drone_energy.get("energy_loaded_per_min", 20.0)) if drone_energy else 20.0,
            drone_curb_weight_kg=float(drone_spec.get("curb_weight_kg", 3.0)),
            drone_parcel_weight_kg=float(drone_spec.get("parcel_weight_kg", 2.0)),
            swap_time_s=float(truck_spec.get("swap_time_s", 60.0)),
            reload_time_s=float(truck_spec.get("reload_time_s", 60.0)),
        )

    @classmethod
    def from_search_config(cls, config: SearchConfig) -> "InstanceGenerator":
        """Build a generator from a typed SearchConfig."""
        return cls(
            generation=config.generation,
            priority_penalties=config.priority_penalties,
            base_class_windows=config.base_windows,
            fleet_spec=_FleetSpec(
                truck_speed_kmph=config.vehicles.truck.speed_kmph,
                truck_capacity_kg=config.vehicles.truck.capacity_kg,
                truck_cost_per_km=config.vehicle_costs.truck_cost_per_km,
                truck_energy_cost_per_min=config.vehicle_costs.truck_energy_cost_per_min,
                drone_speed_kmph=config.vehicles.drone.speed_kmph,
                drone_capacity_kg=config.vehicles.drone.capacity_kg,
                drone_max_battery_wh=config.vehicles.drone.max_battery_wh,
                drone_cost_per_km=config.vehicle_costs.drone_cost_per_km,
                drone_energy_cost_per_min=config.vehicle_costs.drone_energy_cost_per_min,
                drone_energy_per_min_when_empty=float(config.experiment.get("drone_energy_empty_per_min", 15.0)),
                drone_energy_per_min_when_loaded=float(config.experiment.get("drone_energy_loaded_per_min", 20.0)),
                drone_curb_weight_kg=config.vehicles.drone.curb_weight_kg,
                drone_parcel_weight_kg=config.vehicles.drone.parcel_weight_kg,
                swap_time_s=config.vehicle_runtime.swap_time_s,
                reload_time_s=config.vehicle_runtime.reload_time_s,
            ),
        )

    def _validate_inputs(self) -> None:
        if self.generation.num_customers <= 0:
            raise ValueError("num_customers must be positive")
        if not (self.base_class_windows["high"][0] <= self.base_class_windows["high"][1]):
            raise ValueError("high-window lb must be <= ub")
        if not (self.base_class_windows["medium"][0] <= self.base_class_windows["medium"][1]):
            raise ValueError("medium-window lb must be <= ub")
        if not (self.base_class_windows["low"][0] <= self.base_class_windows["low"][1]):
            raise ValueError("low-window lb must be <= ub")
        for key in ("high", "medium", "low"):
            if key not in self.priority_penalties:
                raise ValueError(f"Missing penalty for priority '{key}'")

    def _coords_dispersed(self, rng: np.random.Generator, n: int) -> np.ndarray:
        scale = self.generation.coordinate_scale
        return rng.uniform(0.0, scale, size=(n, 2))

    def _coords_dense(self, rng: np.random.Generator, n: int) -> np.ndarray:
        scale = self.generation.coordinate_scale
        center = np.array([scale * 0.5, scale * 0.5])
        spread = max(0.1, scale * 0.08)
        return np.clip(
            rng.normal(loc=center, scale=spread, size=(n, 2)),
            0.0,
            scale,
        )

    def _coords_mixed(self, rng: np.random.Generator, n: int) -> np.ndarray:
        scale = self.generation.coordinate_scale
        mask = rng.uniform(0, 1, size=n) < 0.75
        out = np.empty((n, 2), dtype=float)
        out[mask] = np.clip(
            rng.normal(loc=scale * 0.5, scale=scale * 0.08, size=(mask.sum(), 2)),
            0.0,
            scale,
        )
        out[~mask] = rng.uniform(0.0, scale, size=((~mask).sum(), 2))
        return out

    def _sample_coordinates(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.generation.region == "dense_urban":
            return self._coords_dense(rng, n)
        if self.generation.region == "dispersed":
            return self._coords_dispersed(rng, n)
        if self.generation.region == "mixed":
            return self._coords_mixed(rng, n)
        raise ValueError(f"Unknown region '{self.generation.region}'")

    def _sample_priorities(self, rng: np.random.Generator, n: int) -> List[PriorityClass]:
        shares = np.cumsum(self.generation.priority_share)
        draws = rng.random(n)
        out: List[PriorityClass] = []
        for d in draws:
            if d < shares[0]:
                out.append("high")
            elif d < shares[1]:
                out.append("medium")
            else:
                out.append("low")
        return out

    def _window_for_priority(self, priority: PriorityClass, rng: np.random.Generator) -> tuple[float, float]:
        lb, ub = self.base_class_windows[priority]
        jitter = max(1e-6, 0.05 * self.generation.coordinate_scale)
        lb = max(0.0, lb + rng.normal(0.0, jitter / 8.0))
        ub = max(lb + 1.0, ub + rng.normal(0.0, jitter / 8.0))
        return float(lb), float(ub)

    def _build_fleet(self) -> tuple[list[Truck], list[Drone]]:
        trucks = [
            Truck(
                truck_id=t + 1,
                capacity_kg=self.fleet_spec.truck_capacity_kg,
                speed_kmph=self.fleet_spec.truck_speed_kmph,
                cost_per_km=self.fleet_spec.truck_cost_per_km,
                energy_cost_per_min=self.fleet_spec.truck_energy_cost_per_min,
            )
            for t in range(self.generation.num_trucks)
        ]

        drones = [
            Drone(
                drone_id=d + 1,
                capacity_kg=self.fleet_spec.drone_capacity_kg,
                speed_kmph=self.fleet_spec.drone_speed_kmph,
                max_battery_wh=self.fleet_spec.drone_max_battery_wh,
                energy_per_min_when_empty=float(self.fleet_spec.drone_energy_per_min_when_empty),
                energy_per_min_when_loaded=float(self.fleet_spec.drone_energy_per_min_when_loaded),
                cost_per_km=self.fleet_spec.drone_cost_per_km,
                energy_cost_per_min=self.fleet_spec.drone_energy_cost_per_min,
                curb_weight_kg=self.fleet_spec.drone_curb_weight_kg,
                parcel_weight_kg=self.fleet_spec.drone_parcel_weight_kg,
            )
            for d in range(self.generation.num_drones)
        ]
        return trucks, drones

    def generate_single(self, seed: int, name: str, overrides: Dict[str, int | float | str] | None = None) -> InstanceData:
        overrides = overrides or {}
        gen = deepcopy(self.generation)
        for key in ("num_customers", "num_trucks", "num_drones"):
            if key in overrides:
                gen = replace(gen, **{key: int(overrides[key])})

        rng = np.random.default_rng(int(seed))
        n = int(gen.num_customers)
        coords = self._sample_coordinates(rng, n)
        priorities = self._sample_priorities(rng, n)
        customers: list[Customer] = []
        for idx, ((x, y), p) in enumerate(zip(coords, priorities), start=1):
            lb, ub = self._window_for_priority(p, rng)
            customers.append(
                Customer(
                    node_id=idx,
                    x=float(np.clip(x, 0.0, gen.coordinate_scale)),
                    y=float(np.clip(y, 0.0, gen.coordinate_scale)),
                    demand_kg=float(rng.uniform(gen.demand_min, gen.demand_max)),
                    priority=p,
                    lb=lb,
                    ub=ub,
                    service_time_min=float(rng.uniform(0.5, 1.5)),
                )
            )

        trucks, drones = self._build_fleet()
        constants = ProblemConstants(
            swap_time_s=float(overrides.get("swap_time_s", self.fleet_spec.swap_time_s)),
            reload_time_s=float(overrides.get("reload_time_s", self.fleet_spec.reload_time_s)),
            max_shift_minutes=float(
                overrides.get("max_shift_minutes", overrides.get("t_max_minutes", 480.0))
            ),
            big_m=float(overrides.get("big_m", 10000.0)),
        )

        return InstanceData(
            name=name,
            seed=int(seed),
            region=gen.region,
            customers=customers[: int(gen.num_customers)],
            trucks=trucks[: int(gen.num_trucks)],
            drones=drones[: int(gen.num_drones)],
            priority_penalties=self.priority_penalties,
            constants=constants,
            metadata={
                "coordinate_scale": float(gen.coordinate_scale),
                "instance_class": gen.instance_class,
                "priority_share": list(gen.priority_share),
                "generation": {
                    "num_customers": int(gen.num_customers),
                    "num_trucks": int(gen.num_trucks),
                    "num_drones": int(gen.num_drones),
                    "region": gen.region,
                    "seed_offset": int(seed),
                },
            },
        )

    def generate_batch(
        self,
        seed: int,
        sizes: Sequence[int],
        reps_per_size: int,
        output_dir: str,
        tag: str,
        overrides: Dict[str, Dict[str, object]] | None = None,
    ) -> list[InstanceData]:
        """Create many instances and persist JSON + CSV copies."""
        overrides = overrides or {}
        output_dir = Path(output_dir)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        instances: list[InstanceData] = []
        for size in sizes:
            for rep in range(1, reps_per_size + 1):
                inst_seed = int(seed + 1000 * int(size) + rep)
                name = f"{tag}_{size}_{rep:02d}"
                inst = self.generate_single(
                    inst_seed,
                    name=name,
                    overrides={"num_customers": int(size), **(overrides.get(name, {}))},
                )
                self.save(inst, raw_dir / f"{name}.json")
                instances.append(inst)
        return instances

    def save(self, instance: InstanceData, path: Path) -> None:
        """Persist instance in JSON and customer table CSV."""
        payload = instance.as_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with (path.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "node_id",
                    "x",
                    "y",
                    "demand_kg",
                    "priority",
                    "lb",
                    "ub",
                    "service_time_min",
                ],
            )
            writer.writeheader()
            for c in instance.customers:
                writer.writerow(
                    {
                        "node_id": c.node_id,
                        "x": c.x,
                        "y": c.y,
                        "demand_kg": c.demand_kg,
                        "priority": c.priority,
                        "lb": c.lb,
                        "ub": c.ub,
                        "service_time_min": c.service_time_min,
                    }
                )


def summary_to_row(instance: InstanceData) -> InstanceSummary:
    shares = instance.class_share()
    return InstanceSummary(
        name=instance.name,
        seed=instance.seed,
        num_customers=instance.num_customers,
        num_trucks=instance.num_trucks,
        num_drones=instance.num_drones,
        region=instance.region,
        class_share=shares,
        coordinate_scale=float(instance.metadata.get("coordinate_scale", 20.0)),
    )


def load_instance_json(path: str | Path) -> InstanceData:
    """Load an InstanceData object from JSON generated by this module."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    customers = [
        Customer(
            node_id=int(item["node_id"]),
            x=float(item["x"]),
            y=float(item["y"]),
            demand_kg=float(item["demand_kg"]),
            priority=str(item["priority"]),
            lb=float(item["lb"]),
            ub=float(item["ub"]),
            service_time_min=float(item["service_time_min"]),
        )
        for item in raw["customers"]
    ]
    trucks = [
        Truck(
            truck_id=int(item["truck_id"]),
            capacity_kg=float(item["capacity_kg"]),
            speed_kmph=float(item["speed_kmph"]),
            cost_per_km=float(item["cost_per_km"]),
            energy_cost_per_min=float(item.get("energy_cost_per_min", 0.0)),
        )
        for item in raw["trucks"]
    ]
    drones = [
        Drone(
            drone_id=int(item["drone_id"]),
            capacity_kg=float(item["capacity_kg"]),
            speed_kmph=float(item["speed_kmph"]),
            max_battery_wh=float(item["max_battery_wh"]),
            energy_per_min_when_empty=float(item["energy_per_min_when_empty"]),
            energy_per_min_when_loaded=float(item["energy_per_min_when_loaded"]),
            cost_per_km=float(item["cost_per_km"]),
            energy_cost_per_min=float(item.get("energy_cost_per_min", 0.0)),
            curb_weight_kg=float(item.get("curb_weight_kg", 3.0)),
            parcel_weight_kg=float(item.get("parcel_weight_kg", 2.0)),
        )
        for item in raw["drones"]
    ]
    constants = ProblemConstants(
        swap_time_s=float(raw["constants"].get("swap_time_s", 60.0)),
        reload_time_s=float(raw["constants"].get("reload_time_s", 60.0)),
        max_shift_minutes=float(raw["constants"].get("max_shift_minutes", 480.0)),
        big_m=float(raw["constants"].get("big_m", 10000.0)),
    )
    return InstanceData(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        region=str(raw.get("region", "dense_urban")),
        customers=customers,
        trucks=trucks,
        drones=drones,
        priority_penalties={str(k): float(v) for k, v in raw["priority_penalties"].items()},
        constants=constants,
        metadata=raw.get("metadata", {}),
    )
