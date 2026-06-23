"""Rank fusion algorithms.

Primary: weighted RRF
    score(c) = Σ_source  w_source / (k + rank_source(c))
             + Σ_signal  beta_signal * feature_signal(c)   ← S2 graph signals

The beta term lets S2 graph signals (relevance, novelty, recency, velocity)
add directly to the RRF score without needing a separate ranking list.

Also: Borda count and CombSUM as ablation baselines.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def _sort(scores: Dict[str, float],
          candidate_ids: Optional[Sequence[str]] = None) -> List[str]:
    if candidate_ids is not None:
        for d in candidate_ids: scores.setdefault(d, 0.0)
    return sorted(scores, key=lambda d: (-scores[d], str(d)))


def weighted_rrf(
    rankings:     Dict[str, Sequence[str]],
    weights:      Optional[Dict[str, float]]  = None,
    k:            float                        = 60.0,
    candidate_ids: Optional[Sequence[str]]    = None,
    feature_bias: Optional[Dict[str, float]]  = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Weighted RRF with optional additive feature term (S2 graph signals).

    Parameters
    ----------
    k : float or dict {source -> k}  (per-source smoothing, advanced)
    feature_bias : {candidate_id -> additive score} built by feature_bias_term()
    """
    def k_of(src):
        return float(k.get(src, 60.0)) if isinstance(k, dict) else float(k)

    scores: Dict[str, float] = defaultdict(float)
    for src, ids in rankings.items():
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        if w == 0.0: continue
        for rank, d in enumerate(ids, 1):
            scores[d] += w / (k_of(src) + rank)
    if feature_bias:
        for d, b in feature_bias.items():
            scores[d] += b
    ordered = _sort(dict(scores), candidate_ids)
    return ordered, dict(scores)


def feature_bias_term(query, beta: Dict[str, float]) -> Dict[str, float]:
    """Additive term: beta . x(c) using a Query's per-candidate feature vectors.

    Use this to incorporate S2 graph signals (graph_relevance, graph_novelty,
    graph_recency, citation_velocity) directly into the RRF score.

    Example
    -------
    beta = {"graph_relevance": 0.05, "graph_novelty": 0.03,
            "citation_velocity": 0.04}
    bias = feature_bias_term(query, beta)
    ordered, scores = weighted_rrf(query.rankings, weights, k,
                                   query.candidate_ids, feature_bias=bias)
    """
    out: Dict[str, float] = {}
    for d in query.candidate_ids:
        feats = query.features.get(d, {})
        out[d] = sum(beta.get(name, 0.0) * val for name, val in feats.items())
    return out


def borda_count(
    rankings:     Dict[str, Sequence[str]],
    weights:      Optional[Dict[str, float]] = None,
    candidate_ids: Optional[Sequence[str]]  = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Borda: candidate at rank r in list of length n scores w*(n-r)."""
    scores: Dict[str, float] = defaultdict(float)
    for src, ids in rankings.items():
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        n = len(ids)
        for rank, d in enumerate(ids, 1):
            scores[d] += w * (n - rank)
    return _sort(dict(scores), candidate_ids), dict(scores)


def combsum(
    rankings:      Dict[str, Sequence[str]],
    source_scores: Optional[Dict[str, Dict[str, float]]] = None,
    weights:       Optional[Dict[str, float]] = None,
    candidate_ids: Optional[Sequence[str]]   = None,
) -> Tuple[List[str], Dict[str, float]]:
    """CombSUM over min-max normalised per-source scores (fallback: 1/rank)."""
    scores: Dict[str, float] = defaultdict(float)
    for src, ids in rankings.items():
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        if source_scores and src in source_scores:
            raw = {d: source_scores[src].get(d, 0.0) for d in ids}
            lo, hi = min(raw.values(), default=0.0), max(raw.values(), default=1.0)
            span = (hi - lo) or 1.0
            for d, s in raw.items(): scores[d] += w * (s - lo) / span
        else:
            for rank, d in enumerate(ids, 1): scores[d] += w / rank
    return _sort(dict(scores), candidate_ids), dict(scores)


def make_rrf_rank_fn(weights=None, k=60.0, beta=None):
    """Build a rank_fn(query) -> List[str] for the metrics module."""
    def rank_fn(query):
        bias = feature_bias_term(query, beta) if beta else None
        ordered, _ = weighted_rrf(query.rankings, weights=weights, k=k,
                                   candidate_ids=query.candidate_ids,
                                   feature_bias=bias)
        return ordered
    return rank_fn


def make_single_source_rank_fn(source: str):
    """Returns one source's ranking — use for per-agent MRR baseline."""
    def rank_fn(query):
        ranked = list(query.rankings.get(source, []))
        seen   = set(ranked)
        ranked += [d for d in query.candidate_ids if d not in seen]
        return ranked
    return rank_fn
