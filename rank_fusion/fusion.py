"""Rank-fusion algorithms.

The finalizer is **weighted Reciprocal Rank Fusion** (plan §5, stage 1):

    RRF(d) = sum_{m in sources}  w_m / (k + r_m(d))

where r_m(d) is d's 1-indexed rank in source m (the term is dropped when d is
unranked by m), w_m is the source weight, and k is the smoothing constant
(k=60 is the original RRF value from Cormack et al. 2009).

Also included: Borda count and CombSUM, useful as fusion baselines in the
ablation table.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def _ordered_by_score(
    scores: Dict[str, float], candidate_ids: Optional[Sequence[str]] = None
) -> List[str]:
    """Sort candidates by score desc; deterministic id tie-break.

    If `candidate_ids` is given, every candidate appears in the output even if
    no source ranked it (score 0) — so the fused list spans the whole pool.
    """
    if candidate_ids is not None:
        for d in candidate_ids:
            scores.setdefault(d, 0.0)
    return sorted(scores.keys(), key=lambda d: (-scores[d], str(d)))


def weighted_rrf(
    rankings: Dict[str, Sequence[str]],
    weights: Optional[Dict[str, float]] = None,
    k=60.0,
    candidate_ids: Optional[Sequence[str]] = None,
    feature_bias: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Weighted RRF with optional per-source k and an additive feature term.

        score(c) = Sum_a  w_a / (k_a + r_a(c))   +   beta . x(c)

    Parameters
    ----------
    k:
        Either a single smoothing constant (float) or a ``source -> k_a`` dict
        for per-source smoothing (plan §5 / Level 2). Missing sources fall back
        to 60.0.
    feature_bias:
        Optional ``candidate id -> additive score`` (the ``beta . x(c)`` term);
        build it with :func:`feature_bias_term`.
    """
    def k_of(src: str) -> float:
        if isinstance(k, dict):
            return float(k.get(src, 60.0))
        return float(k)

    scores: Dict[str, float] = defaultdict(float)
    for src, ids in rankings.items():
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        if w == 0.0:
            continue
        ks = k_of(src)
        for rank, d in enumerate(ids, start=1):
            scores[d] += w / (ks + rank)
    if feature_bias:
        for d, b in feature_bias.items():
            scores[d] += b
    ordered = _ordered_by_score(dict(scores), candidate_ids)
    return ordered, dict(scores)


def feature_bias_term(
    query, beta: Dict[str, float]
) -> Dict[str, float]:
    """Compute the additive ``beta . x(c)`` term from a Query's per-candidate features."""
    out: Dict[str, float] = {}
    for d in query.candidate_ids:
        feats = query.features.get(d, {})
        out[d] = sum(beta.get(name, 0.0) * val for name, val in feats.items())
    return out


def borda_count(
    rankings: Dict[str, Sequence[str]],
    weights: Optional[Dict[str, float]] = None,
    candidate_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Borda: a candidate at rank r in a list of length n scores (n - r)."""
    scores: Dict[str, float] = defaultdict(float)
    for src, ids in rankings.items():
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        n = len(ids)
        for rank, d in enumerate(ids, start=1):
            scores[d] += w * (n - rank)
    ordered = _ordered_by_score(dict(scores), candidate_ids)
    return ordered, dict(scores)


def combsum(
    rankings: Dict[str, Sequence[str]],
    source_scores: Optional[Dict[str, Dict[str, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
    candidate_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """CombSUM over per-source scores (min-max normalised within each source).

    If `source_scores` is None, falls back to a rank-derived score 1/rank.
    """
    scores: Dict[str, float] = defaultdict(float)
    for src, ids in rankings.items():
        w = 1.0 if weights is None else float(weights.get(src, 0.0))
        if source_scores and src in source_scores:
            raw = {d: source_scores[src].get(d, 0.0) for d in ids}
            lo, hi = min(raw.values(), default=0.0), max(raw.values(), default=0.0)
            span = (hi - lo) or 1.0
            for d, s in raw.items():
                scores[d] += w * (s - lo) / span
        else:
            for rank, d in enumerate(ids, start=1):
                scores[d] += w * (1.0 / rank)
    ordered = _ordered_by_score(dict(scores), candidate_ids)
    return ordered, dict(scores)


def make_rrf_rank_fn(weights: Optional[Dict[str, float]] = None, k=60.0,
                     beta: Optional[Dict[str, float]] = None):
    """Convenience: build a `rank_fn(query) -> List[str]` for the metrics module.

    `beta` (optional) applies the additive feature term using each Query's
    `.features`.
    """

    def rank_fn(query):
        bias = feature_bias_term(query, beta) if beta else None
        ordered, _ = weighted_rrf(
            query.rankings, weights=weights, k=k,
            candidate_ids=query.candidate_ids, feature_bias=bias,
        )
        return ordered

    return rank_fn


def make_single_source_rank_fn(source: str):
    """A `rank_fn` that just returns one source's ranking (for per-ranker MRR)."""

    def rank_fn(query):
        ranked = list(query.rankings.get(source, []))
        seen = set(ranked)
        ranked += [d for d in query.candidate_ids if d not in seen]
        return ranked

    return rank_fn
