from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from ejor_dad.model import HazardRegime, Link, State


def generate_failure_states(links: Sequence[Link], max_failures: int | None = 2, include_tail: bool = True) -> list[State]:
    """Create the paper's reduced state support plus an optional tail state."""
    link_ids = [link.id for link in links]
    if max_failures is None:
        max_failures = len(link_ids); include_tail = False
    states=[]
    for cardinality in range(max_failures+1):
        for failed in combinations(link_ids,cardinality):
            state_id='intact' if not failed else 'fail__'+'__'.join(failed)
            states.append(State(id=state_id,failed_links=failed))
    if include_tail and max_failures<len(link_ids):states.append(State(id=f'tail_ge_{max_failures+1}_failures',is_tail=True))
    return states


def generate_regime_failure_states(links: Sequence[Link], regimes: Sequence[HazardRegime]) -> list[State]:
    """Create complete road states within each spatial facility-outage regime."""
    link_ids=[link.id for link in links]; states=[]
    for regime in regimes:
        for cardinality in range(len(link_ids)+1):
            for failed in combinations(link_ids,cardinality):
                road_id='intact' if not failed else 'fail__'+'__'.join(failed)
                states.append(State(id=f'{regime.id}__{road_id}',failed_links=failed,failed_centers=regime.failed_centers,hazard_regime_id=regime.id))
    return states


def nominal_probabilities(links: Sequence[Link], states: Sequence[State], y: Sequence[float], hazard_regimes: Sequence[HazardRegime] | None = None) -> np.ndarray:
    """Compute decision-dependent probabilities, optionally under a regime mixture."""
    y_vec=np.asarray(y,dtype=float)
    if len(y_vec)!=len(links):raise ValueError(f'Expected {len(links)} retrofit decisions, received {len(y_vec)}.')
    base_phi=np.array([link.failure_probability(float(level)) for link,level in zip(links,y_vec)],dtype=float)
    floors=np.array([link.residual_failure_probability for link in links],dtype=float)
    link_index={link.id:i for i,link in enumerate(links)}
    regime_lookup={regime.id:regime for regime in hazard_regimes or []}
    probabilities=np.zeros(len(states));tail_index=None;non_tail_total=0.0
    for state_index,state in enumerate(states):
        if state.is_tail:tail_index=state_index;continue
        if hazard_regimes is None:
            phi=base_phi;regime_weight=1.0
        else:
            if state.hazard_regime_id not in regime_lookup:raise ValueError(f'State {state.id} has no valid hazard regime.')
            regime=regime_lookup[state.hazard_regime_id];multipliers=np.array([regime.link_failure_multipliers.get(link.id,1.0) for link in links])
            phi=np.clip(floors+multipliers*(base_phi-floors),floors,1.0);regime_weight=regime.probability
        failed=set(state.failed_links);probability=regime_weight
        for link in links:
            link_phi=phi[link_index[link.id]];probability*=link_phi if link.id in failed else 1.0-link_phi
        probabilities[state_index]=probability;non_tail_total+=probability
    if tail_index is not None:probabilities[tail_index]=max(0.0,1.0-non_tail_total)
    total=probabilities.sum()
    if total<=0:raise ValueError('Nominal probabilities sum to zero.')
    return probabilities/total


def state_failure_lookup(states: Sequence[State]) -> Mapping[str,set[str]]:
    return {state.id:set(state.failed_links) for state in states}
