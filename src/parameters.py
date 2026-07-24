"""Configuration helpers and YAML loading for all experiment pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml


WindowConfig = Dict[str, Tuple[float, float]]
PriorityPenaltyConfig = Dict[str, float]


@dataclass(frozen=True)
class VehicleSpec:
    """Generic ground vehicle envelope."""

    speed_kmph: float = 35.0
    cost_per_km: float = 0.8
    capacity_kg: float = 500.0
    energy_cost_per_min: float = 0.0

    def __post_init__(self) -> None:
        if self.speed_kmph <= 0:
            raise ValueError("vehicle speed must be positive")
        if self.capacity_kg < 0:
            raise ValueError("capacity_kg must be non-negative")
        if self.cost_per_km < 0:
            raise ValueError("cost_per_km must be non-negative")


@dataclass(frozen=True)
class DroneSpec(VehicleSpec):
    """Drone-specific physical constants."""

    max_battery_wh: float = 1200.0
    curb_weight_kg: float = 3.0
    parcel_weight_kg: float = 2.0


@dataclass(frozen=True)
class AerodynamicModel:
    c1: float = 0.0549
    c2: float = 0.0024
    c3: float = 0.0022
    c4: float = 0.0089
    beta: float = 1.0
    rated_power_output: float = 1.0


@dataclass(frozen=True)
class VehicleCatalog:
    truck: VehicleSpec = field(default_factory=VehicleSpec)
    drone: DroneSpec = field(default_factory=DroneSpec)
    drone_aero: AerodynamicModel = field(default_factory=AerodynamicModel)


@dataclass(frozen=True)
class GenerationSettings:
    """Instance-generation controls."""

    num_customers: int
    num_trucks: int
    num_drones: int
    region: str = "dense_urban"
    instance_class: str = "medium"
    coordinate_scale: float = 20.0
    priority_share: Tuple[float, float, float] = (0.2, 0.3, 0.5)
    demand_min: float = 0.5
    demand_max: float = 4.5
    region_dispersion: Optional[float] = None
    instance_reps_per_size: int = 20
    sizes: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.num_customers <= 0 or self.num_trucks <= 0 or self.num_drones <= 0:
            raise ValueError("num_customers, num_trucks, num_drones must be positive")
        if self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be > 0")
        if self.demand_min <= 0 or self.demand_max <= self.demand_min:
            raise ValueError("demand_min, demand_max must satisfy 0 < min < max")
        if len(self.priority_share) != 3:
            raise ValueError("priority_share must contain 3 values")
        if abs(sum(self.priority_share) - 1.0) > 1e-8:
            raise ValueError("priority_share must sum to 1.0")
        if self.instance_reps_per_size <= 0:
            raise ValueError("instance_reps_per_size must be positive")


@dataclass(frozen=True)
class VehicleCostConfig:
    truck_cost_per_km: float = 0.8
    drone_cost_per_km: float = 0.3
    truck_energy_cost_per_min: float = 0.0
    drone_energy_cost_per_min: float = 0.0


@dataclass(frozen=True)
class MILPConfig:
    enabled: bool = True
    solver_backend: str = "pulp_cbc"
    time_limit_seconds: int = 3600
    mip_gap: float = 0.001
    threads: int = 4
    demand_mode: str = "clarified"


@dataclass(frozen=True)
class HeuristicConfig:
    max_outer_iter: int = 25
    max_no_improve: int = 5
    time_limit_seconds: int = 600
    random_seed: int = 2026


@dataclass(frozen=True)
class ConstraintConfig:
    t_max_minutes: float = 480.0
    big_m: float = 10000.0
    strict_time_windows: bool = True
    enforce_subtour_elim: bool = True


@dataclass(frozen=True)
class VehicleRuntimeConfig:
    fleet_class: str = "medium"
    region: str = "dense_urban"
    coordinate_scale: float = 20.0
    demand_min: float = 0.5
    demand_max: float = 4.5
    swap_time_s: float = 60.0
    reload_time_s: float = 60.0


@dataclass(frozen=True)
class SearchConfig:
    """Global study configuration used by CLI and experiment scripts."""

    seed: int = 2026
    generation: GenerationSettings = field(
        default_factory=lambda: GenerationSettings(
            num_customers=100,
            num_trucks=3,
            num_drones=5,
        )
    )
    priority_penalties: PriorityPenaltyConfig = field(
        default_factory=lambda: {
            "high": 100.0,
            "medium": 20.0,
            "low": 1.0,
        }
    )
    base_windows: WindowConfig = field(
        default_factory=lambda: {
            "high": (0.0, 240.0),
            "medium": (0.0, 360.0),
            "low": (0.0, 480.0),
        }
    )
    vehicle_costs: VehicleCostConfig = field(default_factory=VehicleCostConfig)
    milp: MILPConfig = field(default_factory=MILPConfig)
    heuristics: HeuristicConfig = field(default_factory=HeuristicConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    vehicles: VehicleCatalog = field(default_factory=VehicleCatalog)
    vehicle_runtime: VehicleRuntimeConfig = field(default_factory=VehicleRuntimeConfig)
    experiment: Dict[str, Any] = field(default_factory=dict)
    sensitivity: Dict[str, list[Any]] = field(default_factory=dict)


def _coerce_priority_share(raw: Sequence[float]) -> Tuple[float, float, float]:
    if len(raw) != 3:
        raise ValueError("priority_share must have exactly three elements")
    total = sum(float(v) for v in raw)
    if total <= 0:
        raise ValueError("priority_share must sum to a positive number")
    return tuple(float(v) / total for v in raw)


def build_class_window_map(base_windows: Dict[str, Sequence[float]]) -> WindowConfig:
    """Normalize and validate delivery windows keyed by priority class."""
    normalized: WindowConfig = {}
    for cls in ("high", "medium", "low"):
        if cls not in base_windows:
            raise ValueError(f"Missing window definition for class '{cls}'")
        low, high = base_windows[cls]
        low_f = float(low)
        high_f = float(high)
        if low_f < 0 or high_f < 0:
            raise ValueError(f"Window for {cls} must be non-negative")
        if low_f > high_f:
            raise ValueError(f"Invalid window ordering for class '{cls}': lb > ub")
        normalized[cls] = (low_f, high_f)
    return normalized


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML with clear errors for missing files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_search_config(data: Dict[str, Any]) -> SearchConfig:
    """Validate and map a config dictionary into typed objects."""
    gen_raw = data.get("generation", {})
    base_windows_raw = (
        data.get("base_class_windows", {})
        or data.get("window_by_priority", {})
        or data.get("base_windows", {})
        or gen_raw.get("base_class_windows", {})
        or gen_raw.get("window_by_priority", {})
        or gen_raw.get("base_windows", {})
    )
    priority_penalties = data.get("priority_penalties") or {
        "high": 100.0,
        "medium": 20.0,
        "low": 1.0,
    }

    generation = GenerationSettings(
        num_customers=int(gen_raw.get("num_customers", 100)),
        num_trucks=int(gen_raw.get("num_trucks", 3)),
        num_drones=int(gen_raw.get("num_drones", 5)),
        region=str(gen_raw.get("region", "dense_urban")),
        instance_class=str(gen_raw.get("instance_class", "medium")),
        coordinate_scale=float(gen_raw.get("coordinate_scale", 20.0)),
        priority_share=_coerce_priority_share(gen_raw.get("priority_share", (0.2, 0.3, 0.5))),
        demand_min=float(
            gen_raw.get("demand_kg", {}).get("min", gen_raw.get("demand_min", 0.5))
            if isinstance(gen_raw.get("demand_kg"), dict)
            else gen_raw.get("demand_min", 0.5)
        ),
        demand_max=float(
            gen_raw.get("demand_kg", {}).get("max", gen_raw.get("demand_max", 4.5))
            if isinstance(gen_raw.get("demand_kg"), dict)
            else gen_raw.get("demand_max", 4.5)
        ),
        region_dispersion=gen_raw.get("region_dispersion"),
        instance_reps_per_size=int(gen_raw.get("instance_reps_per_size", gen_raw.get("reps", 20))),
        sizes=tuple(gen_raw.get("sizes", ()) or ()),
    )

    milp_raw = data.get("milp", {})
    milp = MILPConfig(
        enabled=bool(milp_raw.get("enabled", True)),
        solver_backend=str(milp_raw.get("solver", milp_raw.get("solver_backend", "pulp_cbc"))),
        time_limit_seconds=int(milp_raw.get("time_limit_seconds", 3600)),
        mip_gap=float(milp_raw.get("mip_gap", 0.001)),
        threads=int(milp_raw.get("threads", 4)),
        demand_mode=str(milp_raw.get("demand_mode", "clarified")),
    )

    heur_raw = data.get("heuristics", {})
    if isinstance(heur_raw, dict) and "nils" in heur_raw:
        heur_raw = heur_raw["nils"]
    heuristics = HeuristicConfig(
        max_outer_iter=int((heur_raw or {}).get("max_outer_iter", 25)),
        max_no_improve=int((heur_raw or {}).get("max_no_improve", 5)),
        time_limit_seconds=int((heur_raw or {}).get("time_limit_seconds", 600)),
        random_seed=int((heur_raw or {}).get("random_seed", int(data.get("seed", 2026)))),
    )

    constraints_raw = data.get("constraints", {})
    constraints = ConstraintConfig(
        t_max_minutes=float(constraints_raw.get("t_max_minutes", 480.0)),
        big_m=float(constraints_raw.get("big_m", 10000.0)),
        strict_time_windows=bool(constraints_raw.get("strict_time_windows", True)),
        enforce_subtour_elim=bool(constraints_raw.get("enforce_subtour_elim", True)),
    )

    costs_raw = data.get("costs", {})
    vehicles_raw = data.get("vehicles", {})
    truck_raw = vehicles_raw.get("truck", {})
    drone_raw = vehicles_raw.get("drone", {})
    power_raw = drone_raw.get("power_coeff", {})

    vehicles = VehicleCatalog(
        truck=VehicleSpec(
            speed_kmph=float(truck_raw.get("speed_kmph", 35.0)),
            cost_per_km=float(costs_raw.get("truck_cost_per_km", truck_raw.get("cost_per_km", 0.8))),
            capacity_kg=float(truck_raw.get("capacity_kg", 500.0)),
            energy_cost_per_min=float(costs_raw.get("truck_energy_cost_per_min", truck_raw.get("energy_cost_per_min", 0.0))),
        ),
        drone=DroneSpec(
            speed_kmph=float(drone_raw.get("speed_kmph", 65.0)),
            cost_per_km=float(costs_raw.get("drone_cost_per_km", drone_raw.get("cost_per_km", 0.3))),
            capacity_kg=float(drone_raw.get("capacity_kg", 5.0)),
            max_battery_wh=float(drone_raw.get("max_battery_wh", 1200.0)),
            curb_weight_kg=float(drone_raw.get("curb_weight_kg", 3.0)),
            parcel_weight_kg=float(drone_raw.get("parcel_weight_kg", 2.0)),
            energy_cost_per_min=float(costs_raw.get("drone_energy_cost_per_min", drone_raw.get("energy_cost_per_min", 0.0))),
        ),
        drone_aero=AerodynamicModel(
            c1=float(power_raw.get("c1", 0.0549)),
            c2=float(power_raw.get("c2", 0.0024)),
            c3=float(power_raw.get("c3", 0.0022)),
            c4=float(power_raw.get("c4", 0.0089)),
            beta=float(power_raw.get("beta", 1.0)),
            rated_power_output=float(power_raw.get("rated_power_output", 1.0)),
        ),
    )

    vehicle_runtime = VehicleRuntimeConfig(
        fleet_class=str(generation.instance_class),
        region=generation.region,
        coordinate_scale=generation.coordinate_scale,
        demand_min=float(generation.demand_min),
        demand_max=float(generation.demand_max),
        swap_time_s=float(costs_raw.get("swap_time_s", 60.0)),
        reload_time_s=float(costs_raw.get("reload_time_s", 60.0)),
    )

    return SearchConfig(
        seed=int(data.get("seed", 2026)),
        generation=generation,
        priority_penalties={
            "high": float(priority_penalties.get("high", 100.0)),
            "medium": float(priority_penalties.get("medium", 20.0)),
            "low": float(priority_penalties.get("low", 1.0)),
        },
        base_windows=build_class_window_map(base_windows_raw),
        vehicle_costs=VehicleCostConfig(
            truck_cost_per_km=float(costs_raw.get("truck_cost_per_km", 0.8)),
            drone_cost_per_km=float(costs_raw.get("drone_cost_per_km", 0.3)),
            truck_energy_cost_per_min=float(costs_raw.get("truck_energy_cost_per_min", 0.0)),
            drone_energy_cost_per_min=float(costs_raw.get("drone_energy_cost_per_min", 0.0)),
        ),
        milp=milp,
        heuristics=heuristics,
        constraints=constraints,
        vehicles=vehicles,
        vehicle_runtime=vehicle_runtime,
        experiment=data.get("experiment", {}),
        sensitivity={
            k: [float(v) for v in value]
            for k, value in (data.get("sensitivity", {}) or {}).items()
            if isinstance(value, list)
        },
    )


def load_and_build_config(path: str | Path) -> SearchConfig:
    return build_search_config(load_yaml_config(path))
