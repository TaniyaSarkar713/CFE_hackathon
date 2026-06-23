"""Best-of selection: never do worse than the best single source.

Fusion is *meant* to beat every individual ranker, but it is not guaranteed —
a weak/noisy source (e.g. S2 paper titles that don't match the ground truth) can
dilute a strong one under uniform RRF. So instead of trusting fusion blindly,
we evaluate **each source alone (LLM1, LLM2, S2-KG) and the fused method**, and
emit the ranking with the best MRR.

Done properly this is *model selection on dev MRR* (pick the winning strategy on
a dev split, apply it to test) — so it's leakage-free and deployable, and the
output is provably ≥ the best single source on dev. `compare_strategies` gives
the per-strategy MRR table for reporting; `select_best_strategy` returns the
winning rank_fn.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from .fusion import make_single_source_rank_fn, make_rrf_rank_fn
from .metrics import corpus_mrr
from .consensus import corpus_mrr_at_k


def _score(queries, rank_fn, mrr_k, per_direction):
    if mrr_k is None:
        return corpus_mrr(queries, rank_fn, per_direction)
    return corpus_mrr_at_k(queries, rank_fn, mrr_k, per_direction)


def strategy_rank_fns(
    sources: Sequence[str],
    fused_rank_fns: Optional[Dict[str, Callable]] = None,
    k: float = 60.0,
) -> Dict[str, Callable]:
    """The set of strategies to compare: each source alone + the fused method(s).

    `fused_rank_fns` lets you plug in a tuned-RRF or consensus rank_fn; if omitted,
    a uniform RRF over `sources` is used as the fused baseline.
    """
    fns: Dict[str, Callable] = {f"single:{s}": make_single_source_rank_fn(s)
                                for s in sources}
    if fused_rank_fns:
        fns.update(fused_rank_fns)
    else:
        fns["fused:RRF"] = make_rrf_rank_fn({s: 1.0 for s in sources}, k)
    return fns


def compare_strategies(
    queries: Sequence,
    sources: Sequence[str],
    fused_rank_fns: Optional[Dict[str, Callable]] = None,
    mrr_k: Optional[int] = None,
    per_direction: bool = True,
    k: float = 60.0,
) -> List[Dict]:
    """MRR of each source alone and the fused method on `queries`, sorted desc.

    `mrr_k=None` -> unbounded MRR (the honest primary); pass an int for MRR@K.
    """
    fns = strategy_rank_fns(sources, fused_rank_fns, k)
    rows = [{"name": n, "mrr": _score(queries, f, mrr_k, per_direction)}
            for n, f in fns.items()]
    rows.sort(key=lambda r: -r["mrr"])
    return rows


def select_best_strategy(
    dev_queries: Sequence,
    sources: Sequence[str],
    fused_rank_fns: Optional[Dict[str, Callable]] = None,
    mrr_k: Optional[int] = None,
    per_direction: bool = True,
    k: float = 60.0,
) -> Dict:
    """Pick the strategy (source or fusion) with the best **dev** MRR.

    Returns {best_name, best_rank_fn, best_dev_mrr, table}. Apply
    `best_rank_fn` to new papers — the output is never worse than the best single
    source on dev (the selector falls back to it if fusion underperforms).
    """
    fns = strategy_rank_fns(sources, fused_rank_fns, k)
    scored = [(n, f, _score(dev_queries, f, mrr_k, per_direction))
              for n, f in fns.items()]
    scored.sort(key=lambda x: -x[2])
    best_name, best_fn, best_mrr = scored[0]
    return {
        "best_name": best_name,
        "best_rank_fn": best_fn,
        "best_dev_mrr": best_mrr,
        "table": [{"name": n, "dev_mrr": m} for n, _f, m in scored],
    }
