"""Diversity-aware re-ranking via Maximal Marginal Relevance (plan §9.3).

The fused ranking optimises relevance, but if several near-duplicate directions
all score high they crowd the top-K and bury a *distinct* true direction. Since
Paper MRR averages 1/rank over the true directions, covering distinct gaps near
the top directly helps MRR when a paper has several future-work directions.

MMR (Carbonell & Goldstein 1998) greedily trades relevance against redundancy:

    next = argmax_{d in remaining}  λ·rel(d) − (1−λ)·max_{s in selected} sim(d, s)

λ=1 reproduces the relevance-only order; lower λ diversifies. `sim` is a
candidate–candidate similarity (cosine over embeddings / TF-IDF, or any callable).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def cosine_sim_callable(ids: Sequence[str], vectors) -> Callable[[str, str], float]:
    """Build a sim(a, b) callable from per-candidate vectors (cosine)."""
    V = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    U = V / norm
    S = U @ U.T
    idx = {d: i for i, d in enumerate(ids)}

    def sim(a: str, b: str) -> float:
        if a in idx and b in idx:
            return float(S[idx[a], idx[b]])
        return 0.0

    return sim


def feature_embedding_sim(query, prefix: str = "emb_") -> Optional[Callable]:
    """Cosine sim callable from embedding dims stored on `Query.features`."""
    dims = sorted({k for f in query.features.values() for k in f
                   if k.startswith(prefix)})
    if not dims:
        return None
    ids = list(query.candidate_ids)
    vecs = [[query.features.get(d, {}).get(dim, 0.0) for dim in dims] for d in ids]
    return cosine_sim_callable(ids, vecs)


def mmr_rerank(
    ranked_ids: Sequence[str],
    scores: Dict[str, float],
    sim: Callable[[str, str], float],
    lambda_: float = 0.7,
    k: Optional[int] = None,
) -> List[str]:
    """Greedy MMR re-ranking. Returns the full re-ordered list (diverse first)."""
    ids = list(ranked_ids)
    if not ids:
        return ids
    vals = np.array([scores.get(i, 0.0) for i in ids], dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    rel = {i: (scores.get(i, 0.0) - lo) / span for i in ids}

    k = k or len(ids)
    selected: List[str] = []
    remaining = list(ids)
    while remaining and len(selected) < k:
        best, best_val = None, -np.inf
        for d in remaining:
            div = max((sim(d, s) for s in selected), default=0.0)
            val = lambda_ * rel[d] - (1.0 - lambda_) * div
            if val > best_val:
                best_val, best = val, d
        selected.append(best)
        remaining.remove(best)
    selected.extend([d for d in ids if d not in selected])  # leftover, original order
    return selected


def make_mmr_rank_fn(base_rank_fn, sim_provider: Callable, lambda_: float = 0.7,
                     score_provider: Optional[Callable] = None):
    """Wrap a fusion `rank_fn` with an MMR pass for the metrics harness.

    `sim_provider(query) -> sim callable` (e.g. `feature_embedding_sim`);
    `score_provider(query) -> {id: score}` defaults to rank-derived scores.
    """
    def rank_fn(query):
        ranked = base_rank_fn(query)
        sim = sim_provider(query)
        if sim is None:
            return ranked
        if score_provider is not None:
            scores = score_provider(query)
        else:  # higher position -> higher relevance proxy
            n = len(ranked)
            scores = {d: float(n - i) for i, d in enumerate(ranked)}
        return mmr_rerank(ranked, scores, sim, lambda_=lambda_)

    return rank_fn
