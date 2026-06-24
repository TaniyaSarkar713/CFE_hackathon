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
from .metrics import corpus_mrr, paper_mrr
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


def select_top_sources(
    source_to_queries: Dict[str, Sequence],
    k: int = 2,
    mrr_k: Optional[int] = None,
    per_direction: bool = False,
    score_all: bool = True,
) -> Dict:
    """Rank *independent* sources (each with its OWN candidate pool) by MRR and
    return the top-k.

    Use this when several rankers produce DIFFERENT candidate lists — e.g. N LLM
    agents each emitting their own ranked `future_directions`. Each source maps
    to its own list of `Query` whose `candidate_ids` are already in that source's
    ranked order; we score each by mean Paper MRR against `relevance`.

    `score_all=True` (default, **fair**) scores every supplied paper — a paper
    where this source matched no ground-truth direction counts as MRR 0, so all
    sources share the same denominator. `score_all=False` scores only matched
    papers (ranking quality given a hit), which favours selective sources.

    Returns {ranked: [{source, mrr, n_papers}], top_k: [source names]}.
    """
    rows: List[Dict] = []
    for name, queries in source_to_queries.items():
        if score_all:
            vals = []
            for q in queries:
                ranked = list(q.candidate_ids)
                if mrr_k:
                    ranked = ranked[:mrr_k]
                vals.append(paper_mrr(ranked, q.relevance, per_direction))
            mrr = sum(vals) / len(vals) if vals else 0.0
            n = len(queries)
        else:
            labelled = [q for q in queries if q.has_ground_truth()]
            mrr = _score(labelled, lambda q: list(q.candidate_ids), mrr_k, per_direction)
            n = len(labelled)
        rows.append({"source": name, "mrr": mrr, "n_papers": n})
    rows.sort(key=lambda r: -r["mrr"])
    return {"ranked": rows, "top_k": [r["source"] for r in rows[:k]]}


def _complexity(name: str) -> int:
    """Strategy simplicity rank: single source < uniform fusion < tuned fusion."""
    if name.startswith("single:"):
        return 0
    if "uniform" in name.lower():
        return 1
    return 2


def select_best_strategy_cv(
    dev_queries: Sequence,
    sources: Sequence[str],
    fused_builders: Optional[Dict[str, Callable]] = None,
    mrr_k: Optional[int] = None,
    per_direction: bool = True,
    n_folds: int = 5,
    margin: float = 0.01,
    seed: int = 0,
) -> Dict:
    """Pick the strategy with the best **cross-validated** dev MRR, preferring the
    simplest strategy within `margin` of the best.

    Unlike `select_best_strategy` (single dev pass), this re-fits each fused
    strategy on the fold's TRAIN split and scores it on the held-out fold — so a
    fusion that only wins by **overfitting** the full dev set gets a lower CV
    score and is not chosen. The `margin` + simplicity preference then breaks
    near-ties toward the single best source (which generalises better than a
    fusion that barely edges it out).

    `fused_builders` maps a strategy name to `builder(train_queries) -> rank_fn`
    (e.g. a consensus/RRF tuner). Single sources need no builder. The winner is
    finally re-fit on the FULL dev set for application to test.

    Returns {best_name, best_rank_fn, best_cv_mrr, table}.
    """
    labelled = [q for q in dev_queries if q.has_ground_truth()]
    builders = dict(fused_builders or {})
    names = [f"single:{s}" for s in sources] + list(builders)

    folds = [labelled[i::n_folds] for i in range(n_folds)]
    cv: Dict[str, List[float]] = {n: [] for n in names}
    for i in range(n_folds):
        val = folds[i]
        train = [q for j, f in enumerate(folds) if j != i for q in f]
        if not val or not train:
            continue
        for s in sources:
            cv[f"single:{s}"].append(
                _score(val, make_single_source_rank_fn(s), mrr_k, per_direction))
        for name, builder in builders.items():
            cv[name].append(_score(val, builder(train), mrr_k, per_direction))

    cv_mean = {n: (sum(v) / len(v) if v else 0.0) for n, v in cv.items()}
    best_cv = max(cv_mean.values()) if cv_mean else 0.0
    contenders = [n for n in names if cv_mean[n] >= best_cv - margin]
    best_name = min(contenders, key=lambda n: (_complexity(n), -cv_mean[n]))

    if best_name.startswith("single:"):
        best_fn = make_single_source_rank_fn(best_name.split(":", 1)[1])
    else:
        best_fn = builders[best_name](labelled)        # re-fit on full dev
    table = sorted(({"name": n, "cv_mrr": cv_mean[n]} for n in names),
                   key=lambda r: -r["cv_mrr"])
    return {"best_name": best_name, "best_rank_fn": best_fn,
            "best_cv_mrr": cv_mean[best_name], "table": table}


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
