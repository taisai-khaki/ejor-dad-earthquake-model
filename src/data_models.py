"""Typed domain objects used across the UTDRP-DP codebase."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pathlib import Path

import json
import numpy as np

PriorityClass = Literal["high", "medium", "low"]
PRIORITY_CLASSES: Tuple[PriorityClass, ...] = ("high", "medium", "low")


@dataclass(frozen=True)
class Customer:
    """Customer definition with spatial, demand, and service-tier attributes."""

    node_id: int
    x: float
    y: float
    demand_kg: float
    priority: PriorityClass
    lb: float
    ub: float
    service_time_min: float = 0.0

    def __post_init__(self) -> None:
        if self.node_id <= 0:
            raise ValueError("Customer node_id must be positive")
        if self.demand_kg < 0:
            raise ValueError("Customer demand must be non-negative")
        if self.lb < 0 or self.ub < 0:
            raise ValueError("Window bounds must be non-negative")
        if self.ub < self.lb:
            raise ValueError("Window upper bound must be >= lower bound")
        if self.service_time_min < 0:
            raise ValueError("service_time_min must be non-negative")
        if self.priority not in PRIORITY_CLASSES:
            raise ValueError(f"Invalid priority class: {self.priority}")

    @property
    def window(self) -> Tuple[float, float]:
        return self.lb, self.ub


@dataclass(frozen=True)
class Truck:
    """Truck physical and economic parameters."""

    truck_id: int
    capacity_kg: float
    speed_kmph: float
    cost_per_km: float
    energy_cost_per_min: float = 0.0

    def __post_init__(self) -> None:
        if self.truck_id <= 0:
            raise ValueError("Truck IDs must be positive")
        if self.capacity_kg < 0:
            raise ValueError("Truck capacity must be non-negative")
        if self.speed_kmph <= 0:
            raise ValueError("Truck speed must be positive")
        if self.cost_per_km < 0:
            raise ValueError("Truck cost per km must be non-negative")


@dataclass(frozen=True)
class Drone:
    """Drone physical, energy, and economic parameters."""

    drone_id: int
    capacity_kg: float
    speed_kmph: float
    max_battery_wh: float
    energy_per_min_when_empty: float
    energy_per_min_when_loaded: float
    cost_per_km: float
    energy_cost_per_min: float = 0.0
    curb_weight_kg: float = 0.0
    parcel_weight_kg: float = 2.0

    def __post_init__(self) -> None:
        if self.drone_id <= 0:
            raise ValueError("Drone IDs must be positive")
        if self.capacity_kg < 0:
            raise ValueError("Drone capacity must be non-negative")
        if self.speed_kmph <= 0:
            raise ValueError("Drone speed must be positive")
        if self.max_battery_wh <= 0:
            raise ValueError("Drone max battery must be positive")
        if self.cost_per_km < 0:
            raise ValueError("Drone cost per km must be non-negative")
        if self.energy_per_min_when_empty < 0 or self.energy_per_min_when_loaded < 0:
            raise ValueError("Energy rates must be non-negative")


@dataclass(frozen=True)
class ProblemConstants:
    """Static problem-level constants for time/cost and operational calibration."""

    swap_time_s: float = 60.0
    reload_time_s: float = 60.0
    max_shift_minutes: float = 480.0
    big_m: float = 10_000.0

    def __post_init__(self) -> None:
        if self.swap_time_s < 0 or self.reload_time_s < 0:
            raise ValueError("Swap and reload times must be non-negative")
        if self.max_shift_minutes <= 0:
            raise ValueError("max_shift_minutes must be positive")
        if self.big_m <= 0:
            raise ValueError("big_m must be positive")


@dataclass(frozen=True)
class InstanceSummary:
    """Compact high-level instance metadata used for persistence and reporting."""

    name: str
    seed: int
    num_customers: int
    num_trucks: int
    num_drones: int
    region: str
    class_share: Tuple[float, float, float]
    coordinate_scale: float


@dataclass
class InstanceData:
    """Full in-memory problem instance used by all modules."""

    name: str
    seed: int
    region: str
    customers: List[Customer]
    trucks: List[Truck]
    drones: List[Drone]
    priority_penalties: Dict[PriorityClass, float]
    constants: ProblemConstants
    metadata: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_customers != len(self.customers):
            raise ValueError("num_customers mismatch")
        ids = {c.node_id for c in self.customers}
        if ids != set(range(1, self.num_customers + 1)):
            raise ValueError("Customer IDs must be contiguous from 1..n")
        if set(self.truck_ids()) != set(range(1, self.num_trucks + 1)):
            raise ValueError("Truck IDs must be contiguous from 1..|T|")
        if set(self.drone_ids()) != set(range(1, self.num_drones + 1)):
            raise ValueError("Drone IDs must be contiguous from 1..|D|")
        for cls in PRIORITY_CLASSES:
            if cls not in self.priority_penalties:
                raise ValueError(f"Missing priority penalty for {cls}")
            if self.priority_penalties[cls] < 0:
                raise ValueError("Priority penalties must be non-negative")
        if self.constants is None:
            raise ValueError("constants must be provided")

    @property
    def num_customers(self) -> int:
        return len(self.customers)

    @property
    def num_trucks(self) -> int:
        return len(self.trucks)

    @property
    def num_drones(self) -> int:
        return len(self.drones)

    @property
    def customer_ids(self) -> Tuple[int, ...]:
        return tuple(c.node_id for c in self.customers)

    @property
    def depot(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def truck_ids(self) -> List[int]:
        return [t.truck_id for t in self.trucks]

    def drone_ids(self) -> List[int]:
        return [d.drone_id for d in self.drones]

    def all_nodes(self) -> List[int]:
        return [0] + list(self.customer_ids)

    def customer_map(self) -> Dict[int, Customer]:
        return {c.node_id: c for c in self.customers}

    def demand(self, node: int) -> float:
        if node == 0:
            return 0.0
        return self.customer_map()[node].demand_kg

    def service_time(self, node: int, mode: str = "truck") -> float:
        if node == 0:
            return 0.0
        c = self.customer_map()[node]
        return c.service_time_min

    def class_share(self) -> Tuple[float, float, float]:
        class_counts = [0, 0, 0]
        for c in self.customers:
            if c.priority == "high":
                class_counts[0] += 1
            elif c.priority == "medium":
                class_counts[1] += 1
            else:
                class_counts[2] += 1
        n = max(1, self.num_customers)
        return tuple(v / n for v in class_counts)

    def validate_service(self, assignments: Dict[tuple[int, int], int], by="truck") -> None:
        """Validate assignment dictionary keys/values."""
        expected_nodes = set(self.customer_ids)
        served_nodes = {c for (_vid, c), v in assignments.items() if round(v) >= 0.5}
        if served_nodes - expected_nodes:
            raise ValueError(f"Service assignments include invalid nodes for {by}: {served_nodes - expected_nodes}")

    def customer(self, node_id: int) -> Customer:
        if node_id == 0:
            raise ValueError("Use depot (0) is not a customer")
        return self.customer_map()[node_id]

    def coordinate_map(self) -> Dict[int, Tuple[float, float]]:
        coords = {0: (0.0, 0.0)}
        for customer in self.customers:
            coords[customer.node_id] = (customer.x, customer.y)
        return coords

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "region": self.region,
            "constants": to_serializable_dict(self.constants),
            "priority_penalties": dict(self.priority_penalties),
            "customers": [to_serializable_dict(c) for c in self.customers],
            "trucks": [to_serializable_dict(t) for t in self.trucks],
            "drones": [to_serializable_dict(d) for d in self.drones],
            "metadata": self.metadata,
        }


@dataclass
class DecisionArc:
    """Binary arc decision for a vehicle between two nodes."""

    i: int
    j: int
    vehicle: int


@dataclass
class SyncEvent:
    """Truck-drone synchronization event at node i."""

    node: int
    drone_id: int
    truck_id: int
    event_type: str


@dataclass
class SolutionData:
    """MILP/heuristic solution representation used by validators and reporters."""

    instance_name: str
    status: str
    objective: float
    components: Dict[str, float]
    run_time_seconds: float = 0.0

    x_truck: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    x_drone: Dict[Tuple[int, int, int], int] = field(default_factory=dict)

    u_truck: Dict[Tuple[int, int], int] = field(default_factory=dict)
    u_drone: Dict[Tuple[int, int], int] = field(default_factory=dict)

    z1: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    z2: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    y_loaded: Dict[Tuple[int, int], int] = field(default_factory=dict)

    a_truck: Dict[Tuple[int, int], float] = field(default_factory=dict)
    l_truck: Dict[Tuple[int, int], float] = field(default_factory=dict)
    a_drone: Dict[Tuple[int, int], float] = field(default_factory=dict)
    l_drone: Dict[Tuple[int, int], float] = field(default_factory=dict)
    waiting_truck: Dict[Tuple[int, int], float] = field(default_factory=dict)
    waiting_drone: Dict[Tuple[int, int], float] = field(default_factory=dict)

    load_truck: Dict[Tuple[int, int], float] = field(default_factory=dict)
    battery_drone: Dict[Tuple[int, int], float] = field(default_factory=dict)
    tardiness: Dict[int, float] = field(default_factory=dict)

    sync_events: List[SyncEvent] = field(default_factory=list)
    truck_routes: Dict[int, List[int]] = field(default_factory=dict)
    drone_routes: Dict[int, List[int]] = field(default_factory=dict)

    seed: Optional[int] = None
    tags: Tuple[str, ...] = ()

    def served_by_truck(self, tol: float = 0.5) -> set[int]:
        return {j for (_, j), v in self.u_truck.items() if round(float(v)) == 1 and v >= tol}

    def served_by_drone(self, tol: float = 0.5) -> set[int]:
        return {j for (_, j), v in self.u_drone.items() if round(float(v)) == 1 and v >= tol}

    def served_all_customers(self, customers: Sequence[int], tol: float = 0.5) -> bool:
        customer_set = set(customers)
        served_union = self.served_by_truck(tol) | self.served_by_drone(tol)
        return customer_set.issubset(served_union)

    def set_objective_components(self, components: Dict[str, float]) -> None:
        self.components = dict(components)
        self.objective = sum(components.values())


@dataclass
class ExperimentSettings:
    """Runtime and output settings for experiment scripts."""

    seed: int = 2026
    output_dir: str = "outputs"
    save_json: bool = True
    save_csv: bool = True
    run_small_sizes: Tuple[int, ...] = (20, 50, 100)
    run_large_sizes: Tuple[int, ...] = (150, 200, 250)


@dataclass
class ExperimentResult:
    """Summary record persisted per experiment instance-scenario."""

    instance_name: str
    method: str
    status: str
    runtime_seconds: float
    objective: float
    gap_to_baseline: Optional[float] = None
    notes: str = ""
    stats: Dict[str, float] = field(default_factory=dict)


def to_serializable_dict(obj: object) -> object:
    """Convert dataclass-like objects to plain dictionaries for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_serializable_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    raise TypeError(f"Unsupported object type {type(obj)}")


def dump_instance_json(instance: InstanceData, path: str | Path) -> None:
    """Write an InstanceData object into JSON."""
    path = Path(path)
    payload = {
        "name": instance.name,
        "seed": instance.seed,
        "region": instance.region,
        "constants": to_serializable_dict(instance.constants),
        "priority_penalties": to_serializable_dict(instance.priority_penalties),
        "customers": [to_serializable_dict(c) for c in instance.customers],
        "trucks": [to_serializable_dict(t) for t in instance.trucks],
        "drones": [to_serializable_dict(d) for d in instance.drones],
        "metadata": instance.metadata,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

