from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ejor_dad.fixed_y import FixedPlanResult, FixedYResult, evaluate_fixed_plan, evaluate_fixed_y
from ejor_dad.model import DADInstance
from ejor_dad.states import nominal_probabilities


@dataclass(frozen=True)
class RoadRetrofitChannelResult:
    """Road-retrofit decomposition at fixed exposure and capacity decisions."""

    no_retrofit: FixedPlanResult
    conditional_consequence: FixedPlanResult
    decision_dependent_probability: FixedPlanResult
    conditional_consequence_improvement: float
    probability_channel_improvement: float
    total_road_improvement: float


@dataclass(frozen=True)
class OptimizedRoadRetrofitChannelResult:
    """Road-channel decomposition with exposure and capacity reoptimized."""

    no_retrofit: FixedYResult
    conditional_consequence: FixedYResult
    decision_dependent_probability: FixedYResult
    conditional_consequence_improvement: float
    probability_channel_improvement: float
    total_road_improvement: float


def decompose_road_retrofit_channels(instance: DADInstance, z: Sequence[float], w: Sequence[float], y: Sequence[float]) -> RoadRetrofitChannelResult:
    """Separate road channels while holding exposure and capacity fixed."""
    y_vec = np.asarray(y, dtype=float)
    if y_vec.shape != (len(instance.links),):
        raise ValueError("y must have one value for every link.")
    zero_y = np.zeros(len(instance.links), dtype=float)
    nominal_without_retrofit = nominal_probabilities(instance.links, instance.states, zero_y, instance.hazard_regimes)
    no_retrofit = evaluate_fixed_plan(instance, z, w, zero_y)
    conditional_consequence = evaluate_fixed_plan(instance, z, w, y_vec, nominal_distribution_override=nominal_without_retrofit)
    decision_dependent_probability = evaluate_fixed_plan(instance, z, w, y_vec)
    conditional_improvement = float(no_retrofit.objective - conditional_consequence.objective)
    probability_improvement = float(conditional_consequence.objective - decision_dependent_probability.objective)
    return RoadRetrofitChannelResult(
        no_retrofit=no_retrofit,
        conditional_consequence=conditional_consequence,
        decision_dependent_probability=decision_dependent_probability,
        conditional_consequence_improvement=conditional_improvement,
        probability_channel_improvement=probability_improvement,
        total_road_improvement=float(no_retrofit.objective - decision_dependent_probability.objective),
    )


def decompose_optimized_road_channels(instance: DADInstance, y: Sequence[float], epsilon: float = 1e-6, max_iterations: int = 200) -> OptimizedRoadRetrofitChannelResult:
    """Separate road channels while reoptimizing exposure and capacity."""
    y_vec = np.asarray(y, dtype=float)
    if y_vec.shape != (len(instance.links),):
        raise ValueError("y must have one value for every link.")
    zero_y = np.zeros(len(instance.links), dtype=float)
    nominal_without_retrofit = nominal_probabilities(instance.links, instance.states, zero_y, instance.hazard_regimes)
    no_retrofit = evaluate_fixed_y(instance, zero_y, epsilon=epsilon, max_iterations=max_iterations)
    conditional_consequence = evaluate_fixed_y(instance, y_vec, epsilon=epsilon, max_iterations=max_iterations, nominal_distribution_override=nominal_without_retrofit)
    decision_dependent_probability = evaluate_fixed_y(instance, y_vec, epsilon=epsilon, max_iterations=max_iterations)
    conditional_improvement = float(no_retrofit.objective - conditional_consequence.objective)
    probability_improvement = float(conditional_consequence.objective - decision_dependent_probability.objective)
    return OptimizedRoadRetrofitChannelResult(
        no_retrofit=no_retrofit,
        conditional_consequence=conditional_consequence,
        decision_dependent_probability=decision_dependent_probability,
        conditional_consequence_improvement=conditional_improvement,
        probability_channel_improvement=probability_improvement,
        total_road_improvement=float(no_retrofit.objective - decision_dependent_probability.objective),
    )

