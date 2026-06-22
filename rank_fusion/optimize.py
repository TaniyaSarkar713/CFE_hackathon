"""MRR-driven optimisation of the weighted-RRF fusion (plan §5).

MRR is non-differentiable (it depends on discrete ranks), so we treat it as a
**selection criterion**, not a loss: search over the fusion hyper-parameters
(per-source weights w_m and smoothing k) and keep the setting with the best
**dev** MRR. Two complementary searches are provided:

  - random search over the (w, k) space, then
  - coordinate ascent that refines each weight / k locally.

Everything is selected on `dev` only; report once on `test` (no leakage).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .fusion import make_rrf_rank_fn
from .metrics import corpus_mrr
from .schema import Query


@dataclass
class RRFConfig:
    weights: Dict[str, float]
    k: float = 60.0
    dev_mrr: float = 0.0
    history: List[float] = field(default_factory=list)

    def rank_fn(self):
        return make_rrf_rank_fn(self.weights, self.k)


def _score(queries: Sequence[Query], weights: Dict[str, float], k: float,
           per_direction: bool,
           costs: Optional[Dict[str, float]] = None,
           lambda_cost: float = 0.0) -> float:
    """Objective J = MRR_dev - lambda * (cost of the active sources) (Level 5).

    A source only incurs cost when its weight is non-zero, so the optimiser can
    *drop* an expensive ranker if it does not earn its keep (the plan's
    compute-risk governor, §9.5). With lambda_cost=0 this is plain dev MRR.
    """
    mrr = corpus_mrr(queries, make_rrf_rank_fn(weights, k), per_direction)
    if costs and lambda_cost:
        active_cost = sum(costs.get(s, 0.0) for s, w in weights.items() if w > 0)
        return mrr - lambda_cost * active_cost
    return mrr


def tune_rrf(
    dev_queries: Sequence[Query],
    sources: Sequence[str],
    n_random: int = 400,
    k_choices: Sequence[float] = (1, 5, 10, 20, 40, 60, 100),
    weight_high: float = 2.0,
    coord_rounds: int = 3,
    per_direction: bool = True,
    costs: Optional[Dict[str, float]] = None,
    lambda_cost: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
) -> RRFConfig:
    """Search (weights, k) to maximise dev MRR (optionally cost-penalised).

    Returns the best `RRFConfig`. Weights are searched in [0, weight_high];
    k from `k_choices` plus continuous refinement during coordinate ascent.
    Pass `costs` + `lambda_cost` for the cost-aware objective (Level 5).
    """
    rng = random.Random(seed)
    sources = list(sources)

    def score(w, k):
        return _score(dev_queries, w, k, per_direction, costs, lambda_cost)

    # --- baseline: uniform weights, default k ----------------------------- #
    best = RRFConfig(weights={s: 1.0 for s in sources}, k=60.0)
    best.dev_mrr = score(best.weights, best.k)
    best.history.append(best.dev_mrr)

    # --- random search ---------------------------------------------------- #
    for _ in range(n_random):
        w = {s: round(rng.uniform(0.0, weight_high), 3) for s in sources}
        if sum(w.values()) == 0:  # avoid the degenerate all-zero weighting
            w[rng.choice(sources)] = 1.0
        k = rng.choice(list(k_choices))
        m = score(w, k)
        if m > best.dev_mrr:
            best = RRFConfig(weights=w, k=float(k), dev_mrr=m)
        best.history.append(best.dev_mrr)

    # --- coordinate ascent refinement ------------------------------------ #
    weights = dict(best.weights)
    k = best.k
    grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for _round in range(coord_rounds):
        improved = False
        for s in sources:
            base = weights[s]
            for cand in grid:
                weights[s] = cand
                m = score(weights, k)
                if m > best.dev_mrr:
                    best = RRFConfig(weights=dict(weights), k=k, dev_mrr=m)
                    base = cand
                    improved = True
            weights[s] = base
        # refine k around the current best
        for cand_k in (max(1.0, k * 0.5), k, k * 1.5, k * 2.0):
            m = score(weights, cand_k)
            if m > best.dev_mrr:
                best = RRFConfig(weights=dict(weights), k=cand_k, dev_mrr=m)
                k = cand_k
                improved = True
        best.history.append(best.dev_mrr)
        if verbose:
            print(f"  [coord round {_round}] dev MRR={best.dev_mrr:.4f} "
                  f"k={best.k} weights={best.weights}")
        if not improved:
            break

    return best
