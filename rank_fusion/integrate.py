"""Adapters: feed the candidate-harvesting agent's output and the Semantic
Scholar graph signals into the rank-fusion engine.

Two integration points, both via the `Query` contract:

1. AGENT OUTPUT -> Query.
   `pool_to_query` turns the agent's harvested candidate pool into a `Query`:
   candidate ids, a "retrieval" ranking (candidates ordered by similarity to the
   paper — the agent IS a fusion source), provenance in `candidate_sources`, the
   eval `relevance` (vs masked future-work), and optional per-candidate features.

2. S2 GRAPH SIGNALS -> features -> fusion.
   Naz's `GraphSignalExtractor` gives {graph_relevance, graph_novelty,
   graph_recency, citation_velocity} per direction. `graph_signal_fn` wraps it
   into a `text -> {recency, popularity, novelty, relevance}` callable;
   `attach_features` writes those onto `Query.features`. From there the signals
   enter fusion three ways (pick any / all):
     a) as an extra RRF source     — `add_horizon_source` ranks by horizon_prior;
     b) as the RRF feature-bias term — `make_rrf_rank_fn(beta=...)` (in fusion.py);
     c) as learned meta-ranker features — `MetaRanker(extra_feature_keys=HORIZON_KEYS)`;
   and they render as Direction-Card horizon badges automatically (explain.py).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .matching import match_relevance, similarity_matrix
from .schema import Query

# Naz's graph signal names -> the plan's horizon feature names (§7)
GRAPH_SIGNAL_MAP = {
    "graph_recency": "recency",
    "citation_velocity": "popularity",
    "graph_novelty": "novelty",
    "graph_relevance": "relevance",
}
HORIZON_KEYS = ["recency", "popularity", "novelty", "relevance"]
DEFAULT_HORIZON_WEIGHTS = {"recency": 1.0, "popularity": 0.5,
                           "novelty": 1.0, "relevance": 1.0}


# --------------------------------------------------------------------------- #
# 1. Agent output -> Query
# --------------------------------------------------------------------------- #
def pool_to_query(
    query_id: str,
    candidates: Sequence,                 # list of agent.Candidate (.id/.text/.provenance)
    paper_text: str = "",
    gt_texts: Optional[Sequence[str]] = None,
    rankings: Optional[Dict[str, List[str]]] = None,
    signal_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    tau: float = 0.10,
    embed_fn: Optional[Callable] = None,
    retrieval_source: str = "retrieval",
) -> Query:
    """Build a fusion `Query` from the agent's harvested candidate pool.

    `rankings` may carry the GPT/Claude lists once they exist; the agent always
    contributes a `retrieval_source` ranking (similarity to the paper).
    """
    ids = [c.id for c in candidates]
    texts = [c.text for c in candidates]
    text_of = dict(zip(ids, texts))

    candidate_sources: Dict[str, List[str]] = {}
    for c in candidates:
        kind = (getattr(c, "provenance", "") or "agent").split(":")[0]
        candidate_sources.setdefault(f"agent_{kind}", []).append(c.id)

    q_rankings = dict(rankings or {})
    if retrieval_source not in q_rankings:
        if paper_text and ids:
            sims = similarity_matrix(texts, [paper_text], embed_fn)[0]  # [n_cand]
            order = [ids[i] for i in np.argsort(-sims)]
        else:
            order = list(ids)                 # fall back to discovery order
        q_rankings[retrieval_source] = order

    features: Dict[str, Dict[str, float]] = {}
    if signal_fn:
        for cid in ids:
            features[cid] = signal_fn(text_of[cid])

    relevance: Dict[str, set] = {}
    if gt_texts:
        relevance = match_relevance(ids, texts, list(gt_texts), tau=tau, embed_fn=embed_fn)

    return Query(query_id, ids, q_rankings, relevance,
                 candidate_sources=candidate_sources, features=features)


# --------------------------------------------------------------------------- #
# 2. S2 graph signals -> features -> fusion
# --------------------------------------------------------------------------- #
def graph_signal_fn(extractor, ctx, top_k: int = 5) -> Callable[[str], Dict[str, float]]:
    """Wrap Naz's `GraphSignalExtractor` into a `text -> {recency,popularity,...}`.

        from semantic_scholar_ranking.s2_client import GraphSignalExtractor
        ext = GraphSignalExtractor(client); ctx = ext.build_context(corpus_id)
        sig = graph_signal_fn(ext, ctx)
    """
    def fn(text: str) -> Dict[str, float]:
        raw = extractor.signals_for_text(text, ctx, top_k=top_k)
        return {GRAPH_SIGNAL_MAP.get(k, k): float(v) for k, v in raw.items()}
    return fn


def attach_features(query: Query, texts_by_id: Dict[str, str],
                    signal_fn: Callable[[str], Dict[str, float]]) -> Query:
    """Write per-candidate signals onto `Query.features` (in place)."""
    for cid in query.candidate_ids:
        sig = signal_fn(texts_by_id.get(cid, ""))
        query.features.setdefault(cid, {}).update(sig)
    return query


def horizon_prior(feat: Dict[str, float],
                  weights: Optional[Dict[str, float]] = None) -> float:
    w = weights or DEFAULT_HORIZON_WEIGHTS
    return float(sum(w.get(k, 0.0) * feat.get(k, 0.0) for k in w))


def add_horizon_source(query: Query, weights: Optional[Dict[str, float]] = None,
                       name: str = "horizon") -> Query:
    """Add a fusion source that ranks candidates by their horizon_prior (in place)."""
    scored = [(cid, horizon_prior(query.features.get(cid, {}), weights))
              for cid in query.candidate_ids]
    query.rankings[name] = [cid for cid, _ in sorted(scored, key=lambda x: -x[1])]
    return query


def beta_from_horizon_weights(weights: Optional[Dict[str, float]] = None
                              ) -> Dict[str, float]:
    """Build the `beta` dict for `make_rrf_rank_fn(beta=...)` from horizon weights."""
    return dict(weights or DEFAULT_HORIZON_WEIGHTS)


# --------------------------------------------------------------------------- #
# Wire Naz's Agent C (S2 knowledge-graph agent) in as the `s2` fusion source
# --------------------------------------------------------------------------- #
def s2_agent_source(agent_output) -> List[str]:
    """Convert `S2DirectionAgent.run()` output into the `s2` ranked text list.

    `agent_output` is the tuple `(cand_ids, cand_texts, ranked_ids, features)`
    returned by Naz's Agent C. Returns the candidate **texts** ordered best-first
    (by the agent's horizon_prior) — ready to drop into `build_consensus_query`'s
    `source_to_texts["s2"]` so the S2 directions are clustered and fused with the
    two LLM sources.
    """
    cand_ids, cand_texts, ranked_ids, _features = agent_output
    id2text = dict(zip(cand_ids, cand_texts))
    return [id2text[c] for c in ranked_ids if c in id2text]


def s2_agent_features(agent_output, id_map: Optional[Dict[str, str]] = None
                      ) -> Dict[str, Dict[str, float]]:
    """Agent C's per-candidate graph signals, optionally re-keyed to cluster ids.

    Use to also feed the S2 signals (recency/novelty/relevance) into the horizon
    source or the meta-ranker via `attach_features`.
    """
    _ids, _texts, _ranked, features = agent_output
    if id_map is None:
        return dict(features)
    return {id_map[k]: v for k, v in features.items() if k in id_map}
