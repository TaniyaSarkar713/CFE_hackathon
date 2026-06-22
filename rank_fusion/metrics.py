"""Ranking metrics for the free-text future-direction task.

Primary metric is **MRR**, computed per the plan (§1):

    For each true direction g_j:  rank_j = min{ i : candidate at position i
                                                    matches g_j }      (inf if none)
                                  RR_j   = 1 / rank_j                  (0 if none)
    Paper MRR = mean_j RR_j           (averaged over the true directions)
    Corpus MRR = mean over papers of Paper MRR

We also expose the classic *first-relevant* MRR (treat the whole relevant set
as one target), plus Recall@K and binary nDCG@K for the diagnostics in §8.

All functions take a `ranked_ids` list (best first) and a `Query`'s relevance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set

from .schema import Query


# --------------------------------------------------------------------------- #
# Per-query metrics
# --------------------------------------------------------------------------- #
def _positions(ranked_ids: Sequence[str]) -> Dict[str, int]:
    """Map candidate id -> 1-indexed position (first occurrence wins)."""
    pos: Dict[str, int] = {}
    for i, d in enumerate(ranked_ids, start=1):
        if d not in pos:
            pos[d] = i
    return pos


def reciprocal_rank_first(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """Classic RR: 1 / rank of the first relevant candidate (0 if none)."""
    for i, d in enumerate(ranked_ids, start=1):
        if d in relevant_ids:
            return 1.0 / i
    return 0.0


def paper_mrr(
    ranked_ids: Sequence[str],
    relevance: Dict[str, Set[str]],
    per_direction: bool = True,
) -> float:
    """MRR for a single paper.

    per_direction=True  -> mean over true directions of (1 / best matching rank)
    per_direction=False -> first-relevant RR over the union of relevant ids
    """
    if not per_direction:
        union: Set[str] = set()
        for c in relevance.values():
            union |= c
        return reciprocal_rank_first(ranked_ids, union)

    pos = _positions(ranked_ids)
    rrs: List[float] = []
    for _g, cands in relevance.items():
        ranks = [pos[c] for c in cands if c in pos]
        rrs.append(1.0 / min(ranks) if ranks else 0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevance: Dict[str, Set[str]], k: int) -> float:
    """Fraction of true directions with at least one matching candidate in top-K."""
    if not relevance:
        return 0.0
    topk = set(ranked_ids[:k])
    hit = sum(1 for cands in relevance.values() if cands & topk)
    return hit / len(relevance)


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Binary nDCG@K over the union relevant set."""
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# Corpus-level evaluation over a fused-ranking function
# --------------------------------------------------------------------------- #
def corpus_mrr(
    queries: Sequence[Query],
    rank_fn,
    per_direction: bool = True,
) -> float:
    """Mean Paper MRR over `queries`.

    `rank_fn(query) -> List[str]` returns the fused/ordered candidate ids.
    Queries with no ground truth are skipped.
    """
    vals: List[float] = []
    for q in queries:
        if not q.has_ground_truth():
            continue
        ranked = rank_fn(q)
        vals.append(paper_mrr(ranked, q.relevance, per_direction=per_direction))
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(
    queries: Sequence[Query],
    rank_fn,
    ks: Sequence[int] = (1, 5, 10),
    per_direction: bool = True,
) -> Dict[str, float]:
    """Full metric bundle: MRR, MRR@K, Recall@K, nDCG@K (corpus-averaged)."""
    out: Dict[str, List[float]] = {"MRR": []}
    for k in ks:
        out[f"MRR@{k}"] = []
        out[f"Recall@{k}"] = []
        out[f"nDCG@{k}"] = []

    for q in queries:
        if not q.has_ground_truth():
            continue
        ranked = rank_fn(q)
        out["MRR"].append(paper_mrr(ranked, q.relevance, per_direction))
        rel_union = q.relevant_ids
        for k in ks:
            out[f"MRR@{k}"].append(
                paper_mrr(ranked[:k], q.relevance, per_direction)
            )
            out[f"Recall@{k}"].append(recall_at_k(ranked, q.relevance, k))
            out[f"nDCG@{k}"].append(ndcg_at_k(ranked, rel_union, k))

    return {m: (sum(v) / len(v) if v else 0.0) for m, v in out.items()}
