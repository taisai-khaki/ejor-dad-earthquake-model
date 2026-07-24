"""Acceptance rules for NILS iterations."""
from __future__ import annotations

import math
import random


def improved(current_obj: float, candidate_obj: float) -> bool:
    return candidate_obj < current_obj - 1e-9


def accept_with_simulated_annealing(current_obj: float, candidate_obj: float, temperature: float, rng: random.Random) -> bool:
    if improved(current_obj, candidate_obj):
        return True
    if temperature <= 1e-9:
        return False
    delta = candidate_obj - current_obj
    prob = math.exp(-delta / max(1e-9, temperature))
    return rng.random() < prob


def metropolis(current_obj: float, candidate_obj: float, temperature: float, rng: random.Random) -> bool:
    return accept_with_simulated_annealing(current_obj, candidate_obj, temperature, rng)
