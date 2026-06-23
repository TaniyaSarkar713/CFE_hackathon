"""Tune RRF weights + k to maximise dev MRR.

Strategy: random search (400 samples) then coordinate ascent.
Everything is selected on dev only — report once on test.

Usage
-----
from ranking import tune_rrf, corpus_mrr

cfg = tune_rrf(dataset.dev, sources=["agent_a", "agent_b"],
               beta_signals=["graph_relevance", "graph_novelty",
                             "graph_recency", "citation_velocity"])

print(f"Best dev MRR: {cfg.dev_mrr:.4f}")
print(f"Weights: {cfg.weights}")
print(f"Beta: {cfg.beta}")

# evaluate on test
from ranking import evaluate
results = evaluate(dataset.test, cfg.rank_fn())
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .fusion import make_rrf_rank_fn, feature_bias_term, weighted_rrf
from .metrics import corpus_mrr
from .schema import Query


@dataclass
class RRFConfig:
    """Best found (weights, k, beta) configuration."""
    weights:  Dict[str, float]
    k:        float = 60.0
    beta:     Dict[str, float] = field(default_factory=dict)   # S2 signal weights
    dev_mrr:  float = 0.0
    history:  List[float] = field(default_factory=list)

    def rank_fn(self):
        """Build a callable rank_fn(query) -> List[str] from this config."""
        return make_rrf_rank_fn(self.weights, self.k,
                                beta=self.beta if self.beta else None)


def _score(queries, weights, k, beta, per_direction=True):
    def rf(q):
        bias = {d: sum(beta.get(s, 0.0) * v
                       for s, v in q.features.get(d, {}).items())
                for d in q.candidate_ids} if beta else None
        ordered, _ = weighted_rrf(q.rankings, weights, k,
                                   q.candidate_ids, bias)
        return ordered
    return corpus_mrr(queries, rf, per_direction)


def tune_rrf(
    dev_queries:   Sequence[Query],
    sources:       Sequence[str],
    beta_signals:  Sequence[str]   = (),   # S2 feature names to tune beta for
    n_random:      int              = 400,
    k_choices:     Sequence[float]  = (1, 5, 10, 20, 40, 60, 100),
    weight_high:   float            = 2.0,
    coord_rounds:  int              = 3,
    per_direction: bool             = True,
    seed:          int              = 0,
    verbose:       bool             = False,
) -> RRFConfig:
    """Search (weights, k, beta) to maximise dev MRR.

    sources      : list of ranking source names (e.g. ["agent_a", "agent_b"])
    beta_signals : S2 graph signal names to include as additive feature terms
                   (e.g. ["graph_relevance", "graph_novelty", "citation_velocity"])
    """
    rng     = random.Random(seed)
    sources = list(sources)
    bsigs   = list(beta_signals)

    def score(w, k, b):
        return _score(dev_queries, w, k, b, per_direction)

    # baseline: uniform weights, k=60, beta=0
    best_w = {s: 1.0 for s in sources}
    best_b = {s: 0.0 for s in bsigs}
    best   = RRFConfig(weights=best_w, k=60.0, beta=best_b)
    best.dev_mrr = score(best.weights, best.k, best.beta)
    best.history.append(best.dev_mrr)

    # random search
    for _ in range(n_random):
        w = {s: round(rng.uniform(0.0, weight_high), 3) for s in sources}
        if sum(w.values()) == 0:
            w[rng.choice(sources)] = 1.0
        b = {s: round(rng.uniform(0.0, 0.1), 4) for s in bsigs}
        k = rng.choice(list(k_choices))
        m = score(w, k, b)
        if m > best.dev_mrr:
            best = RRFConfig(weights=w, k=float(k), beta=b, dev_mrr=m)
        best.history.append(best.dev_mrr)

    # coordinate ascent refinement
    weights = dict(best.weights)
    beta    = dict(best.beta)
    k       = best.k
    w_grid  = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    b_grid  = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15]

    for rnd in range(coord_rounds):
        improved = False
        for s in sources:
            base = weights[s]
            for cand in w_grid:
                weights[s] = cand
                m = score(weights, k, beta)
                if m > best.dev_mrr:
                    best = RRFConfig(dict(weights), k, dict(beta), m)
                    base = cand; improved = True
            weights[s] = base
        for s in bsigs:
            base = beta[s]
            for cand in b_grid:
                beta[s] = cand
                m = score(weights, k, beta)
                if m > best.dev_mrr:
                    best = RRFConfig(dict(weights), k, dict(beta), m)
                    base = cand; improved = True
            beta[s] = base
        for ck in (max(1.0, k * 0.5), k, k * 1.5, k * 2.0):
            m = score(weights, ck, beta)
            if m > best.dev_mrr:
                best = RRFConfig(dict(weights), ck, dict(beta), m)
                k = ck; improved = True
        best.history.append(best.dev_mrr)
        if verbose:
            print(f"  [round {rnd}] dev MRR={best.dev_mrr:.4f} "
                  f"k={best.k} w={best.weights} β={best.beta}")
        if not improved: break

    return best
