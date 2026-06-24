"""MLRC-Bench-style benchmarking: improvement over a baseline ranker.

MLRC-Bench scores a method by its **absolute improvement over a provided
baseline** (and relative improvement toward an upper bound). We adopt the same
discipline for the future-direction task:

  - Baseline = a **random ordering** of the candidate pool (the trivial floor any
    ranker must beat), averaged over shuffles.
  - Upper bound = the **oracle / candidate-recall ceiling** (every matchable true
    direction placed at rank 1) — the best MRR any ranker over this pool could get.
  - For each method: MRR, absolute improvement (method − baseline), relative
    improvement over baseline (%), and the fraction of the achievable gap captured
    (method − baseline) / (ceiling − baseline).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Sequence

from .diagnostics import oracle_mrr
from .metrics import paper_mrr


def _mrr(queries, rank_fn, mrr_k, per_direction):
    vals = []
    for q in queries:
        if not q.has_ground_truth():
            continue
        ranked = rank_fn(q)
        if mrr_k:
            ranked = ranked[:mrr_k]
        vals.append(paper_mrr(ranked, q.relevance, per_direction))
    return sum(vals) / len(vals) if vals else 0.0


def random_baseline_mrr(queries: Sequence, mrr_k: Optional[int] = None,
                        per_direction: bool = True, n_shuffles: int = 25,
                        seed: int = 0) -> float:
    """Expected MRR of a uniformly random ordering of each paper's candidates."""
    rng = random.Random(seed)
    vals = []
    for q in queries:
        if not q.has_ground_truth():
            continue
        cands = list(q.candidate_ids)
        rr = []
        for _ in range(n_shuffles):
            perm = cands[:]
            rng.shuffle(perm)
            ranked = perm[:mrr_k] if mrr_k else perm
            rr.append(paper_mrr(ranked, q.relevance, per_direction))
        vals.append(sum(rr) / len(rr))
    return sum(vals) / len(vals) if vals else 0.0


def mlrc_benchmark(
    queries: Sequence,
    methods: Dict[str, Callable],
    mrr_k: Optional[int] = None,
    per_direction: bool = True,
    n_shuffles: int = 25,
    seed: int = 0,
) -> Dict:
    """MLRC-style report: each method's MRR and improvement over the random
    baseline, plus the fraction of the oracle-achievable gap it captures.

    `methods` maps name -> rank_fn(query) -> ranked ids.
    Returns {baseline, oracle, rows:[{method, mrr, abs, rel, captured}]}.
    """
    base = random_baseline_mrr(queries, mrr_k, per_direction, n_shuffles, seed)
    ceil = oracle_mrr(queries)                       # candidate-recall ceiling
    span = (ceil - base) or 1.0
    rows = [{"method": "random baseline", "mrr": base,
             "abs": 0.0, "rel": 0.0, "captured": 0.0}]
    for name, fn in methods.items():
        m = _mrr(queries, fn, mrr_k, per_direction)
        absimp = m - base
        rows.append({"method": name, "mrr": m, "abs": absimp,
                     "rel": (absimp / base if base else 0.0),
                     "captured": absimp / span})
    return {"baseline": base, "oracle": ceil, "rows": rows}


def format_benchmark(report: Dict) -> str:
    """Render the MLRC benchmark report as a text table."""
    out = [f"   {'method':<24}{'MRR':>8}{'Δ base':>9}{'rel%':>8}{'%ceil':>8}"]
    out.append("   " + "-" * 57)
    for r in report["rows"]:
        out.append(f"   {r['method']:<24}{r['mrr']:>8.4f}{r['abs']:>+9.4f}"
                   f"{100*r['rel']:>7.0f}%{100*r['captured']:>7.0f}%")
    out.append("   " + "-" * 57)
    out.append(f"   oracle / recall ceiling: {report['oracle']:.4f}")
    return "\n".join(out)
