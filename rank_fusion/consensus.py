"""Consensus-aware rank fusion (GPT + Claude + S2/KG).

Encodes the intuition: when both LLMs independently rank the same future
direction near the top, that's strong evidence; the Semantic Scholar knowledge
graph is a *supporting* signal — strongest when it backs an LLM-consensus
direction — and the data (dev MRR) decides how much each matters.

Two stages:

  Stage 1 — interpretable consensus-aware RRF:
      S(c) = Σ_m w_m/(k+r_m(c))                     # per-source reciprocal rank
             + η1·A12_top1(c) + η3·A12_top3(c)      # LLM-LLM agreement boosts
             + ρ·A12_top3(c)·R_s2(c)                # S2 supports the consensus
    Tuned over (w1,w2,w3,k,η1,η3,ρ,t) on dev MRR@K. The ρ term is a genuine
    interaction that additive RRF cannot represent.

  Stage 2 — learned meta-ranker over the same quantities as features
    (`CONSENSUS_KEYS`), so the model can weight agreement × S2-support nonlinearly.

Candidates from the three sources are first *clustered* semantically
(`cluster_sources` / `build_consensus_query`) so differently-phrased versions of
the same direction share one cluster id and accumulate per-source ranks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .matching import match_relevance, similarity_matrix
from .metrics import paper_mrr
from .schema import Query

_BIG = 10_000.0
CONSENSUS_KEYS = ["a12_top1", "a12_top3", "a12_top5", "a12_top3_x_s2",
                  "min_llm_rank", "mean_llm_rank", "in_s2"]


# --------------------------------------------------------------------------- #
# Stage 1: consensus-aware RRF score
# --------------------------------------------------------------------------- #
@dataclass
class ConsensusConfig:
    weights: Dict[str, float] = field(
        default_factory=lambda: {"llm1": 1.0, "llm2": 1.0, "s2": 0.5})
    k: float = 60.0
    eta1: float = 0.0           # boost when both LLMs rank c at #1
    eta3: float = 0.0           # boost when both LLMs rank c in top-t
    rho: float = 0.0            # boost when S2 also supports the LLM consensus
    t: int = 3                  # the agreement window
    llm_sources: Tuple[str, str] = ("llm1", "llm2")
    s2_source: str = "s2"
    dev_mrr: float = 0.0

    def rank_fn(self):
        return make_consensus_rank_fn(self)


def _positions(query: Query) -> Dict[str, Dict[str, int]]:
    return {s: {c: i + 1 for i, c in enumerate(ids)}
            for s, ids in query.rankings.items()}


def consensus_score(query: Query, cfg: ConsensusConfig) -> Dict[str, float]:
    pos = _positions(query)
    l1, l2 = cfg.llm_sources
    s2 = cfg.s2_source
    scores: Dict[str, float] = {}
    for c in query.candidate_ids:
        S = 0.0
        for s, w in cfg.weights.items():
            r = pos.get(s, {}).get(c)
            if r:
                S += w / (cfg.k + r)
        r1, r2 = pos.get(l1, {}).get(c), pos.get(l2, {}).get(c)
        a_top1 = 1.0 if (r1 == 1 and r2 == 1) else 0.0
        a_topt = 1.0 if (r1 and r2 and r1 <= cfg.t and r2 <= cfg.t) else 0.0
        S += cfg.eta1 * a_top1 + cfg.eta3 * a_topt
        r3 = pos.get(s2, {}).get(c)
        R3 = 1.0 / (cfg.k + r3) if r3 else 0.0
        S += cfg.rho * a_topt * R3
        scores[c] = S
    return scores


def make_consensus_rank_fn(cfg: ConsensusConfig) -> Callable[[Query], List[str]]:
    def fn(query: Query) -> List[str]:
        sc = consensus_score(query, cfg)
        for c in query.candidate_ids:
            sc.setdefault(c, 0.0)
        return sorted(query.candidate_ids, key=lambda c: (-sc[c], str(c)))
    return fn


def corpus_mrr_at_k(queries: Sequence[Query], rank_fn: Callable, k: int = 3,
                    per_direction: bool = True) -> float:
    vals = [paper_mrr(rank_fn(q)[:k], q.relevance, per_direction)
            for q in queries if q.has_ground_truth()]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Stage 1 tuner: search (w, k, eta1, eta3, rho, t) on dev MRR@K
# --------------------------------------------------------------------------- #
def tune_consensus(
    dev_queries: Sequence[Query],
    llm_sources: Tuple[str, str] = ("llm1", "llm2"),
    s2_source: str = "s2",
    mrr_k: int = 3,
    n_random: int = 600,
    coord_rounds: int = 3,
    use_consensus_terms: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> ConsensusConfig:
    """Maximise dev MRR@`mrr_k` over the consensus-aware RRF parameters.

    `use_consensus_terms=False` pins eta1=eta3=rho=0 -> a plain weighted-RRF
    tuner under the *same* objective, for a fair consensus-vs-RRF baseline.
    """
    rng = random.Random(seed)
    srcs = [llm_sources[0], llm_sources[1], s2_source]

    def score(cfg):
        return corpus_mrr_at_k(dev_queries, cfg.rank_fn(), mrr_k)

    best = ConsensusConfig(weights={s: 1.0 for s in srcs}, k=60.0,
                           llm_sources=llm_sources, s2_source=s2_source)
    best.dev_mrr = score(best)

    k_choices = (1, 2, 5, 10, 20, 60)
    eta_choices = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0) if use_consensus_terms else (0.0,)
    t_choices = (1, 2, 3, 5)
    for _ in range(n_random):
        cfg = ConsensusConfig(
            weights={s: round(rng.uniform(0, 2), 3) for s in srcs},
            k=float(rng.choice(k_choices)),
            eta1=rng.choice(eta_choices), eta3=rng.choice(eta_choices),
            rho=rng.choice(eta_choices), t=rng.choice(t_choices),
            llm_sources=llm_sources, s2_source=s2_source)
        if sum(cfg.weights.values()) == 0:
            cfg.weights[srcs[0]] = 1.0
        m = score(cfg)
        if m > best.dev_mrr:
            cfg.dev_mrr = m
            best = cfg

    # coordinate ascent on each continuous knob
    w_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    for _round in range(coord_rounds):
        improved = False

        def try_set(mutate):
            nonlocal best, improved
            trial = _clone(best)
            mutate(trial)
            m = score(trial)
            if m > best.dev_mrr:
                trial.dev_mrr = m
                best = trial
                improved = True

        for s in srcs:
            for cand in w_grid:
                try_set(lambda t, s=s, c=cand: t.weights.__setitem__(s, c))
        for attr in ("eta1", "eta3", "rho"):
            for cand in eta_choices:
                try_set(lambda t, a=attr, c=cand: setattr(t, a, c))
        for cand in k_choices:
            try_set(lambda t, c=cand: setattr(t, "k", float(c)))
        for cand in t_choices:
            try_set(lambda t, c=cand: setattr(t, "t", c))

        if verbose:
            print(f"  [round {_round}] dev MRR@{mrr_k}={best.dev_mrr:.4f} "
                  f"eta1={best.eta1} eta3={best.eta3} rho={best.rho} "
                  f"t={best.t} w={ {s: round(best.weights[s], 2) for s in srcs} }")
        if not improved:
            break
    return best


def _clone(cfg: ConsensusConfig) -> ConsensusConfig:
    return ConsensusConfig(weights=dict(cfg.weights), k=cfg.k, eta1=cfg.eta1,
                           eta3=cfg.eta3, rho=cfg.rho, t=cfg.t,
                           llm_sources=cfg.llm_sources, s2_source=cfg.s2_source,
                           dev_mrr=cfg.dev_mrr)


# --------------------------------------------------------------------------- #
# Stage 2 features: consensus signals onto Query.features
# --------------------------------------------------------------------------- #
def add_consensus_features(query: Query, llm_sources: Tuple[str, str] = ("llm1", "llm2"),
                           s2_source: str = "s2", k: float = 60.0) -> Query:
    """Write the consensus features (`CONSENSUS_KEYS`) onto Query.features (in place).

    Feed them to the meta-ranker via `MetaRanker(extra_feature_keys=CONSENSUS_KEYS)`.
    """
    pos = _positions(query)
    l1, l2 = llm_sources
    for c in query.candidate_ids:
        r1, r2 = pos.get(l1, {}).get(c), pos.get(l2, {}).get(c)
        r3 = pos.get(s2_source, {}).get(c)
        both = r1 and r2
        a1 = 1.0 if (r1 == 1 and r2 == 1) else 0.0
        a3 = 1.0 if (both and r1 <= 3 and r2 <= 3) else 0.0
        a5 = 1.0 if (both and r1 <= 5 and r2 <= 5) else 0.0
        R3 = 1.0 / (k + r3) if r3 else 0.0
        query.features.setdefault(c, {}).update({
            "a12_top1": a1, "a12_top3": a3, "a12_top5": a5,
            "a12_top3_x_s2": a3 * R3,
            "min_llm_rank": float(min(r1 or _BIG, r2 or _BIG)),
            "mean_llm_rank": float(np.mean([r1 or _BIG, r2 or _BIG])),
            "in_s2": 1.0 if r3 else 0.0,
        })
    return query


# --------------------------------------------------------------------------- #
# Cross-source semantic clustering (align differently-phrased directions)
# --------------------------------------------------------------------------- #
def cluster_sources(
    source_to_texts: Dict[str, Sequence[str]],
    embed_fn: Optional[Callable] = None,
    threshold: float = 0.6,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """Cluster ranked text lists from each source into shared cluster ids.

    Returns (cluster_ids, rankings {source -> ordered cluster ids by that
    source's best rank}, cluster_text {cluster id -> representative text}).
    A cluster's rank within a source = the best (smallest) rank of any of its
    members in that source's list.
    """
    items = [(s, rank, txt) for s, texts in source_to_texts.items()
             for rank, txt in enumerate(texts, start=1)]
    if not items:
        return [], {s: [] for s in source_to_texts}, {}
    texts = [t for _, _, t in items]
    sims = similarity_matrix(texts, texts, embed_fn)      # [n, n] cosine

    cluster_of: List[int] = [-1] * len(items)
    reps: List[int] = []                                   # representative item index per cluster
    for i in range(len(items)):
        assigned = -1
        for cidx, ridx in enumerate(reps):
            if sims[i, ridx] >= threshold:
                assigned = cidx
                break
        if assigned == -1:
            assigned = len(reps)
            reps.append(i)
        cluster_of[i] = assigned

    n_clusters = len(reps)
    cluster_ids = [f"k{c}" for c in range(n_clusters)]
    cluster_text = {f"k{c}": texts[reps[c]] for c in range(n_clusters)}
    # best rank per (source, cluster)
    best_rank: Dict[str, Dict[int, int]] = {s: {} for s in source_to_texts}
    for (s, rank, _txt), cl in zip(items, cluster_of):
        cur = best_rank[s].get(cl)
        if cur is None or rank < cur:
            best_rank[s][cl] = rank
    rankings = {s: [f"k{cl}" for cl in sorted(br, key=lambda cl: br[cl])]
                for s, br in best_rank.items()}
    return cluster_ids, rankings, cluster_text


def build_consensus_query(
    query_id: str,
    source_to_texts: Dict[str, Sequence[str]],
    gt_texts: Optional[Sequence[str]] = None,
    embed_fn: Optional[Callable] = None,
    threshold: float = 0.6,
    tau: float = 0.12,
    llm_sources: Tuple[str, str] = ("llm1", "llm2"),
    s2_source: str = "s2",
    k: float = 60.0,
) -> Query:
    """Full real-pipeline entry: three sources' ranked texts -> a fusion Query
    (clustered candidates, per-source rankings, eval relevance, consensus features).
    """
    cluster_ids, rankings, cluster_text = cluster_sources(
        source_to_texts, embed_fn, threshold)
    relevance: Dict[str, set] = {}
    if gt_texts and cluster_ids:
        relevance = match_relevance(cluster_ids, [cluster_text[c] for c in cluster_ids],
                                    list(gt_texts), tau=tau, embed_fn=embed_fn)
    q = Query(query_id, cluster_ids, rankings, relevance,
              features={c: {} for c in cluster_ids})
    add_consensus_features(q, llm_sources, s2_source, k)
    return q
