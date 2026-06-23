"""MRR evaluation harness.

Primary metric: Mean Reciprocal Rank (MRR)
  For each true direction g_j:
      rank_j = position of best matching candidate in the ranked list
      RR_j   = 1 / rank_j  (0 if no match found)
  Paper MRR  = mean over g_j of RR_j
  Corpus MRR = mean over papers of Paper MRR

Also provides Recall@K and nDCG@K for the ablation table.
"""
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Set
from .schema import Query


def _positions(ranked_ids: Sequence[str]) -> Dict[str, int]:
    pos: Dict[str, int] = {}
    for i, d in enumerate(ranked_ids, 1):
        if d not in pos: pos[d] = i
    return pos


def reciprocal_rank_first(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for i, d in enumerate(ranked_ids, 1):
        if d in relevant_ids: return 1.0 / i
    return 0.0


def paper_mrr(ranked_ids: Sequence[str], relevance: Dict[str, Set[str]],
              per_direction: bool = True) -> float:
    """MRR for one paper.
    per_direction=True  → mean over each true direction's best-rank RR
    per_direction=False → classic first-relevant RR
    """
    if not per_direction:
        union: Set[str] = set()
        for c in relevance.values(): union |= c
        return reciprocal_rank_first(ranked_ids, union)
    pos = _positions(ranked_ids)
    rrs: List[float] = []
    for cands in relevance.values():
        ranks = [pos[c] for c in cands if c in pos]
        rrs.append(1.0 / min(ranks) if ranks else 0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevance: Dict[str, Set[str]], k: int) -> float:
    if not relevance: return 0.0
    topk = set(ranked_ids[:k])
    hit  = sum(1 for cands in relevance.values() if cands & topk)
    return hit / len(relevance)


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids: return 0.0
    dcg = sum(1.0 / math.log2(i + 1)
              for i, d in enumerate(ranked_ids[:k], 1) if d in relevant_ids)
    ideal = min(len(relevant_ids), k)
    idcg  = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def corpus_mrr(queries: Sequence[Query], rank_fn, per_direction: bool = True) -> float:
    """Mean Paper MRR across all papers that have ground truth."""
    vals = [paper_mrr(rank_fn(q), q.relevance, per_direction)
            for q in queries if q.has_ground_truth()]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(queries: Sequence[Query], rank_fn,
             ks: Sequence[int] = (1, 5, 10),
             per_direction: bool = True) -> Dict[str, float]:
    """Full metric bundle: MRR, MRR@K, Recall@K, nDCG@K."""
    out: Dict[str, List[float]] = {"MRR": []}
    for k in ks:
        out[f"MRR@{k}"] = []; out[f"Recall@{k}"] = []; out[f"nDCG@{k}"] = []

    for q in queries:
        if not q.has_ground_truth(): continue
        ranked = rank_fn(q)
        out["MRR"].append(paper_mrr(ranked, q.relevance, per_direction))
        rel = q.relevant_ids
        for k in ks:
            out[f"MRR@{k}"].append(paper_mrr(ranked[:k], q.relevance, per_direction))
            out[f"Recall@{k}"].append(recall_at_k(ranked, q.relevance, k))
            out[f"nDCG@{k}"].append(ndcg_at_k(ranked, rel, k))

    return {m: (sum(v) / len(v) if v else 0.0) for m, v in out.items()}
