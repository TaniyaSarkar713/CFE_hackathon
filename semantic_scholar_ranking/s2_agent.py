"""Agent C — Semantic Scholar autonomous future-direction generator.

Searches the S2 citation graph iteratively to produce a ranked list of
candidate future directions grounded in what the field actually pursued
(forward-citing paper titles as realized next-step signals).

The agent self-decides whether to expand the search based on the TF-IDF
diversity of the current top-K candidates (higher diversity = better coverage
of distinct research directions).  It stops when:
  - diversity >= diversity_threshold, OR
  - max_hops reached, OR
  - no new papers found in frontier expansion

Ranking: candidates ordered by horizon_prior = weighted sum of
  graph_recency  (0.40) — how recent the citing papers are
  graph_novelty  (0.35) — how underexplored the direction is
  graph_relevance(0.25) — how close to the seed paper's neighbourhood

The resulting "agent_c" ranking is a third source for the RRF fusion
alongside "agent_a" (LLM future_directions) and "agent_b" (key_gaps +
main_themes).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .s2_client import S2Client, GraphSignalExtractor, GraphContext

logger = logging.getLogger(__name__)

# Horizon-prior weights (tunable via optimize.py beta search)
HORIZON_WEIGHTS: Dict[str, float] = {
    "graph_recency":    0.40,
    "graph_novelty":    0.35,
    "graph_relevance":  0.25,
}


class S2DirectionAgent:
    """Autonomous S2 agent that generates and ranks future-direction candidates.

    Parameters
    ----------
    extractor          : GraphSignalExtractor bound to an S2Client
    max_hops           : maximum graph expansion depth (1 = direct citations only)
    min_candidates     : minimum pool size before diversity check triggers
    diversity_threshold: stop when top-K TF-IDF diversity reaches this (0-1)
    max_per_hop        : max citations to fetch per seed per hop
    """

    def __init__(
        self,
        extractor:           GraphSignalExtractor,
        max_hops:            int   = 2,
        min_candidates:      int   = 8,
        diversity_threshold: float = 0.55,
        max_per_hop:         int   = 40,
    ):
        self.extractor           = extractor
        self.client: S2Client    = extractor.client
        self.max_hops            = max_hops
        self.min_candidates      = min_candidates
        self.diversity_threshold = diversity_threshold
        self.max_per_hop         = max_per_hop

    # ------------------------------------------------------------------
    def run(
        self,
        paper_id: str,
        ctx:      Optional[GraphContext],
        title:    Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, float]]]:
        """Iteratively search the citation graph for future-direction candidates.

        Parameters
        ----------
        paper_id : local paper ID (used as namespace prefix for candidate IDs)
        ctx      : pre-computed GraphContext from GraphSignalExtractor.build_context()
                   (may be None if S2 resolution failed)
        title    : paper title used to resolve the real S2 CorpusId

        Returns
        -------
        cand_ids   : list of candidate IDs  (e.g. "{paper_id}_C1", "_C2", ...)
        cand_texts : matching text for each candidate (paper title from S2)
        ranked_ids : cand_ids ordered best-first by horizon_prior score
        features   : {cand_id -> {signal: value}} (same format as agent_a/b features)
        """
        # Resolve real S2 CorpusId so the citation fetch hits the right paper
        real_id = paper_id
        if title:
            resolved = self.client.resolve_corpus_id(title)
            if resolved:
                logger.info("  [S2Agent] resolved '%s' -> CorpusID:%s", paper_id, resolved)
                real_id = resolved
            else:
                logger.warning("  [S2Agent] title search failed for '%s' — using raw id", paper_id)

        pool: Dict[str, Dict] = {}      # s2_paperId -> {text, signals, score}
        frontier  = [real_id]
        seen_ids  = {real_id}

        for hop in range(self.max_hops):
            # Collect forward-citing papers from all seeds in frontier
            raw_papers: List[dict] = []
            for seed in frontier:
                raw_papers.extend(self.client.get_citations(seed, limit=self.max_per_hop))

            # Filter: new papers with a title
            new_ids = [
                p["paperId"] for p in raw_papers
                if p.get("paperId") and p["paperId"] not in seen_ids
            ]
            new_ids = list(dict.fromkeys(new_ids))  # deduplicate, preserve order
            if not new_ids:
                logger.info("  [S2Agent] hop=%d — no new papers found, stopping", hop + 1)
                break

            # Batch-fetch with embeddings for scoring
            nb_full = self.client.get_papers_batch(new_ids)
            nb_map  = {
                p.get("paperId"): p
                for p in nb_full
                if p and p.get("paperId") and p.get("title")
            }

            added = 0
            for s2_id in new_ids:
                paper = nb_map.get(s2_id)
                if not paper:
                    continue
                seen_ids.add(s2_id)
                text = paper["title"]

                # Score via existing graph signals (proxy-embeds the title)
                if ctx is not None:
                    signals = self.extractor.signals_for_text(text, ctx)
                else:
                    signals = {k: 0.5 for k in HORIZON_WEIGHTS}

                score = sum(HORIZON_WEIGHTS.get(sig, 0.0) * val
                            for sig, val in signals.items())
                pool[s2_id] = {"text": text, "signals": signals, "score": score}
                added += 1

            logger.info(
                "  [S2Agent] hop=%d  fetched=%d  added=%d  pool=%d",
                hop + 1, len(new_ids), added, len(pool),
            )

            # Self-assess: check diversity of current top candidates
            if len(pool) >= self.min_candidates:
                top_texts = [
                    v["text"]
                    for v in sorted(pool.values(), key=lambda x: -x["score"])[:20]
                ]
                div = _tfidf_diversity(top_texts)
                logger.info(
                    "  [S2Agent] diversity=%.3f (threshold=%.2f)",
                    div, self.diversity_threshold,
                )
                if div >= self.diversity_threshold:
                    logger.info("  [S2Agent] diversity threshold met — stopping search")
                    break

            # Expand frontier to the top-scoring papers from this hop
            frontier = [
                s2_id for s2_id in
                sorted(pool, key=lambda k: -pool[k]["score"])[:5]
            ]

        if not pool:
            logger.warning("  [S2Agent] no candidates found for %s", paper_id)
            return [], [], [], {}

        # Build output ordered by horizon_prior
        ranked_s2 = sorted(pool, key=lambda k: -pool[k]["score"])

        cand_ids:   List[str]                    = []
        cand_texts: List[str]                    = []
        features:   Dict[str, Dict[str, float]]  = {}

        for i, s2_id in enumerate(ranked_s2, 1):
            cid = f"{paper_id}_C{i}"
            cand_ids.append(cid)
            cand_texts.append(pool[s2_id]["text"])
            features[cid] = pool[s2_id]["signals"]

        logger.info(
            "  [S2Agent] produced %d candidates (diversity-stopped after %d hops)",
            len(cand_ids), self.max_hops,
        )
        return cand_ids, cand_texts, list(cand_ids), features


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _tfidf_diversity(texts: List[str]) -> float:
    """1 - mean pairwise TF-IDF cosine similarity across texts (0=identical, 1=distinct)."""
    if len(texts) < 2:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec  = TfidfVectorizer(stop_words="english", max_features=500)
        mat  = vec.fit_transform(texts)
        sims = cosine_similarity(mat)
        n    = len(texts)
        avg_sim = (float(sims.sum()) - n) / max(n * (n - 1), 1)
        return float(1.0 - avg_sim)
    except Exception:
        return 0.5
