"""Level-1 diagnostics: candidate-pool recall is the ceiling (plan §8.2).

No ranker can place a true direction it was never given. So *before* tuning
ordering, measure how often the candidate pool even contains a match:

  - Oracle MRR  — the MRR upper bound if every present match sat at rank 1
                  (i.e. fraction of true directions with >=1 matching candidate).
  - Candidate Recall@K — per candidate source and for the union pool.

Also provides bootstrap confidence intervals for any corpus-MRR number, to
guard against over-reading a tiny dev set (the "main warning").
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .metrics import corpus_mrr
from .schema import Query


# --------------------------------------------------------------------------- #
# Oracle MRR — the ranking-independent ceiling
# --------------------------------------------------------------------------- #
def _pool(query: Query, source: Optional[str]) -> Set[str]:
    """The candidate id set available from `source` (None = the full pool)."""
    if source is None:
        return set(query.candidate_ids)
    if source in query.candidate_sources:
        return set(query.candidate_sources[source])
    # fall back to a ranker's ranked list as its candidate set
    return set(query.rankings.get(source, []))


def oracle_mrr(queries: Sequence[Query], source: Optional[str] = None) -> float:
    """Mean over papers of (fraction of true directions matchable in the pool).

    This is the best MRR any ranker over `source`'s pool could achieve: put each
    matched direction at rank 1 (RR=1), unmatched directions contribute 0.
    """
    vals: List[float] = []
    for q in queries:
        if not q.has_ground_truth():
            continue
        pool = _pool(q, source)
        hits = sum(1 for cands in q.relevance.values() if cands & pool)
        vals.append(hits / len(q.relevance))
    return sum(vals) / len(vals) if vals else 0.0


def candidate_recall(
    queries: Sequence[Query], source: Optional[str] = None, k: Optional[int] = None
) -> float:
    """Recall over true directions for a candidate `source`'s pool.

    If `k` is given and the source has an order (a ranker / ordered list), only
    its top-K candidates count; otherwise the whole (unordered) pool counts.
    """
    vals: List[float] = []
    for q in queries:
        if not q.has_ground_truth():
            continue
        if k is not None and source is not None:
            ordered = q.candidate_sources.get(source) or q.rankings.get(source, [])
            pool = set(ordered[:k])
        else:
            pool = _pool(q, source)
        hits = sum(1 for cands in q.relevance.values() if cands & pool)
        vals.append(hits / len(q.relevance))
    return sum(vals) / len(vals) if vals else 0.0


def candidate_recall_table(
    queries: Sequence[Query], sources: Sequence[str], k: Optional[int] = None
) -> Dict[str, float]:
    """Recall per candidate source plus the union pool (the achievable ceiling)."""
    out: Dict[str, float] = {s: candidate_recall(queries, s, k) for s in sources}
    out["union"] = candidate_recall(queries, None, k)
    return out


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval for a corpus-MRR number
# --------------------------------------------------------------------------- #
def bootstrap_mrr_ci(
    queries: Sequence[Query],
    rank_fn: Callable,
    n_boot: int = 1000,
    alpha: float = 0.05,
    per_direction: bool = True,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (point estimate, lower, upper) for corpus MRR via paper resampling."""
    labelled = [q for q in queries if q.has_ground_truth()]
    point = corpus_mrr(labelled, rank_fn, per_direction)
    if not labelled:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(labelled)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [labelled[rng.randrange(n)] for _ in range(n)]
        boots.append(corpus_mrr(sample, rank_fn, per_direction))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


# --------------------------------------------------------------------------- #
# Ranker validation: does the ranker separate good candidates from bad?
# --------------------------------------------------------------------------- #
def topk_bottomk_separation(
    queries: Sequence[Query], rank_fn: Callable, k: int = 10
) -> Dict[str, float]:
    """Top-K vs bottom-K relevance separation (Si et al. 2024, Table 1).

    A sanity check that a ranker actually orders candidates: the fraction of
    *relevant* candidates among its top-K should clearly exceed the fraction
    among its bottom-K. Returns mean top-K relevance, mean bottom-K relevance,
    and the gap (averaged over papers). A positive, large gap = the ranker works.
    """
    tops: List[float] = []
    bottoms: List[float] = []
    for q in queries:
        if not q.has_ground_truth():
            continue
        ranked = rank_fn(q)
        rel = q.relevant_ids
        if len(ranked) < 2 * k:                       # need disjoint top/bottom
            kk = max(1, len(ranked) // 2)
        else:
            kk = k
        top = ranked[:kk]
        bottom = ranked[-kk:]
        tops.append(sum(1 for d in top if d in rel) / kk)
        bottoms.append(sum(1 for d in bottom if d in rel) / kk)
    mt = sum(tops) / len(tops) if tops else 0.0
    mb = sum(bottoms) / len(bottoms) if bottoms else 0.0
    return {"top_k_relevance": mt, "bottom_k_relevance": mb, "gap": mt - mb, "k": k}
