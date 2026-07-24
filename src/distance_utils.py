"""Distance utilities for truck (Manhattan) and drone (Euclidean) travel times."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def manhattan_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compute Manhattan distance in Euclidean coordinate units."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compute Euclidean distance in Euclidean coordinate units."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(np.hypot(dx, dy))


def compute_matrix(coords: Iterable[Tuple[float, float]], metric: str = "manhattan") -> np.ndarray:
    """Return pairwise distance matrix for nodes in `coords`."""
    coords = list(coords)
    n = len(coords)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    mat = np.zeros((n, n), dtype=float)
    metric_fn = manhattan_distance if metric == "manhattan" else euclidean_distance

    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0.0
            else:
                mat[i, j] = metric_fn(coords[i], coords[j])

    return mat


def travel_time_from_distance(distance_km: float, speed_kmph: float) -> float:
    """Convert distance in km and speed in km/h into minutes."""
    if speed_kmph <= 0:
        raise ValueError("speed_kmph must be positive")
    return float(distance_km / speed_kmph * 60.0)


def seconds_to_minutes(seconds: float) -> float:
    return float(seconds) / 60.0


def minutes_to_seconds(minutes: float) -> float:
    return float(minutes) * 60.0


def as_coordinate_list(depot: Tuple[float, float], customers: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [depot, *list(customers)]


def infer_coords_from_instance(instance):
    """Build a node-indexed coordinate dictionary from an InstanceData-like object."""
    coords = {0: (0.0, 0.0)}
    for customer in instance.customers:
        coords[int(customer.node_id)] = (float(customer.x), float(customer.y))
    return coords
