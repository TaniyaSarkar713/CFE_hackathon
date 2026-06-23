"""
rank_fusion.py
==============
Reciprocal Rank Fusion (RRF) module for future research direction recommendation.

Architecture context
--------------------
  ┌─────────────────────────┐
  │  Summarisation LLM       │  → SummaryContext (seed paper + entities + themes)
  └────────────┬────────────┘
               │ seed_paper_id
               ▼
  ┌─────────────────────────┐
  │  SemanticScholarClient   │  → GraphContext (SPECTER2 embeddings, neighbours)
  └────────────┬────────────┘
               │
  ┌────────────┴────────────┐
  │  Agent A   │   Agent B   │  → DirectionList (ranked candidate future directions)
  └────────────┬────────────┘
               │
  ┌────────────▼────────────┐
  │   RankFusionModule       │
  │                          │
  │  signals fused via RRF:  │
  │   1. agent_a_rank        │
  │   2. agent_b_rank        │
  │   3. graph_relevance     │  cosine sim to neighbourhood centroid
  │   4. graph_novelty       │  distance from densest cluster (novel = sparse)
  │   5. graph_recency       │  mean year of closest neighbours
  └────────────┬────────────┘
               │
  ┌────────────▼────────────┐
  │  List[RankedDirection]   │  final fused ranking + per-signal breakdown
  └─────────────────────────┘

RRF formula:  score(d) = Σ_signal  w_signal / (k + rank_signal(d))
Default k=60, all weights=1.0 (tunable toward MRR later).

Usage
-----
  python rank_fusion.py                        # live S2 API calls
  python rank_fusion.py --no-graph             # agent-only RRF (offline demo)
  python rank_fusion.py --api-key <YOUR_KEY>   # authenticated (higher rate limit)
"""

from __future__ import annotations

import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ===========================================================================
# 1.  DATA CONTRACTS
#     These mirror the output schemas of the summarisation LLM and both agents.
#     Replace the simulate_* functions below with real agent output parsers.
# ===========================================================================

@dataclass
class SummaryContext:
    """
    Output of the summarisation LLM agent.

    Fields produced by the LLM; seed_paper_id is the S2 / ArXiv identifier
    the user passes in alongside the paper PDF/text.
    """
    seed_paper_id: str        # e.g. ArXiv:1706.03762 or S2 hash
    seed_title: str
    seed_abstract: str
    key_entities: List[str]   # fine-grained concepts, e.g. ["multi-head attention", "ViT"]
    key_themes: List[str]     # coarser topics, e.g. ["representation learning"]
    limitations: List[str]    # explicit limitations flagged by the LLM
    open_questions: List[str] # open questions extracted from conclusion / discussion


@dataclass
class CandidateDirection:
    """
    A single future-research direction proposed by Agent A or B.

    The `embedding` field is populated by GraphFeatureExtractor at fusion time;
    the agents do not need to provide it.
    """
    id: str                        # unique per session, e.g. "A-0", "B-3"
    title: str
    description: str
    keywords: List[str] = field(default_factory=list)
    # -- populated internally, not by agents --
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class DirectionList:
    """Ordered list of candidate directions from a single agent (best-first)."""
    agent_id: str                       # "A" or "B"
    directions: List[CandidateDirection]


# ===========================================================================
# 2.  SEMANTIC SCHOLAR CLIENT
#     Thin wrapper around the S2 Graph API with caching and rate-limit backoff.
#     Base URL: https://api.semanticscholar.org/graph/v1
# ===========================================================================

S2_GRAPH_BASE   = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_URL   = f"{S2_GRAPH_BASE}/paper/search"
PAPER_FIELDS    = "title,abstract,year,citationCount,embedding"


class SemanticScholarClient:
    """
    Minimal Semantic Scholar Graph API client.

    Authentication
    --------------
    Pass api_key for the authenticated tier (1 RPS, stable).
    Without a key you share the public 1000 RPS pool — fine for demos, but
    fragile under heavy load.  Free API keys at: https://www.semanticscholar.org/product/api

    Caching
    -------
    All responses are cached in-memory for the lifetime of the object, so
    repeated calls for the same paper (common during direction embedding) are free.
    """

    def __init__(self, api_key: Optional[str] = None, requests_per_second: float = 0.9):
        self._headers      = {"x-api-key": api_key} if api_key else {}
        self._min_interval = 1.0 / requests_per_second
        self._last_call    = 0.0
        self._cache: Dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Low-level GET with retry / backoff
    # -----------------------------------------------------------------------

    def _get(self, url: str, params: dict) -> dict:
        cache_key = url + repr(sorted(params.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Token-bucket rate limiting
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, headers=self._headers, timeout=20)
                self._last_call = time.time()

                if resp.status_code == 200:
                    data = resp.json()
                    self._cache[cache_key] = data
                    return data
                elif resp.status_code == 429:
                    sleep_for = 2 ** attempt
                    logger.warning("S2 rate limited — retrying in %ds", sleep_for)
                    time.sleep(sleep_for)
                elif resp.status_code == 404:
                    logger.warning("S2 404 for %s", url)
                    return {}
                else:
                    logger.error("S2 HTTP %d for %s", resp.status_code, url)
                    return {}
            except requests.RequestException as exc:
                logger.error("Request error: %s", exc)
                time.sleep(2 ** attempt)

        return {}

    # -----------------------------------------------------------------------
    # Public API helpers
    # -----------------------------------------------------------------------

    def get_paper(self, paper_id: str) -> dict:
        """Fetch metadata + SPECTER2 embedding for a single paper."""
        url = f"{S2_GRAPH_BASE}/paper/{paper_id}"
        return self._get(url, {"fields": PAPER_FIELDS})

    def get_citations(self, paper_id: str, limit: int = 50) -> List[dict]:
        """
        Papers that *cite* the given paper (forward neighbours).
        Each element has a 'citingPaper' sub-dict with PAPER_FIELDS.
        """
        url = f"{S2_GRAPH_BASE}/paper/{paper_id}/citations"
        data = self._get(url, {"fields": PAPER_FIELDS, "limit": limit})
        return [item.get("citingPaper", {}) for item in data.get("data", [])]

    def get_references(self, paper_id: str, limit: int = 50) -> List[dict]:
        """
        Papers that the given paper *cites* (backward neighbours / references).
        Each element has a 'citedPaper' sub-dict with PAPER_FIELDS.
        """
        url = f"{S2_GRAPH_BASE}/paper/{paper_id}/references"
        data = self._get(url, {"fields": PAPER_FIELDS, "limit": limit})
        return [item.get("citedPaper", {}) for item in data.get("data", [])]

    def search_paper(self, query: str, limit: int = 3) -> List[dict]:
        """
        Keyword-based paper search.
        Used to proxy-embed candidate directions: find the closest existing paper
        to a direction's text and use its SPECTER2 vector as the direction's embedding.
        """
        data = self._get(S2_SEARCH_URL, {"query": query, "fields": PAPER_FIELDS, "limit": limit})
        return data.get("data", [])


# ===========================================================================
# 3.  GRAPH FEATURE EXTRACTOR
#     Builds the seed-paper's neighbourhood from S2 and computes three
#     graph-derived scores per candidate direction.
# ===========================================================================

@dataclass
class GraphContext:
    """The citation neighbourhood of the seed paper, ready for scoring."""
    seed_embedding: np.ndarray            # SPECTER2 vector of the seed paper, shape (768,)
    neighbor_embeddings: List[np.ndarray] # SPECTER2 vectors of citations + references
    neighbor_years: List[int]             # publication years of the same neighbours
    neighbor_citation_counts: List[int]   # citation counts  of the same neighbours
    neighborhood_centroid: np.ndarray     # mean of neighbor_embeddings


class GraphFeatureExtractor:
    """
    Pulls the seed paper's neighbourhood from Semantic Scholar and computes
    three graph-derived ranking signals per candidate direction.

    Signal definitions
    ------------------
    graph_relevance  ∈ [0, 1]
        Cosine similarity between the direction's proxy embedding and the
        neighbourhood centroid.  High = the direction is closely related to
        the existing cluster of work around the seed.

    graph_novelty    ∈ [0, 1]
        1 − (mean cosine similarity to the top-k closest neighbours).
        High = the direction sits in a sparse / underexplored region of the
        literature graph (novel and risky).

    graph_recency    ∈ [0, 1]
        Mean normalised publication year of the top-k closest neighbours.
        High = the area is active and recently published.
    """

    REFERENCE_YEAR = 2000   # lower bound for year normalisation
    CURRENT_YEAR   = 2024   # upper bound

    def __init__(self, client: SemanticScholarClient):
        self.client = client

    # -----------------------------------------------------------------------
    # Build graph neighbourhood for the seed paper
    # -----------------------------------------------------------------------

    def build_graph_context(self, seed_paper_id: str) -> Optional[GraphContext]:
        logger.info("Fetching seed paper '%s' from S2 ...", seed_paper_id)
        seed_data = self.client.get_paper(seed_paper_id)
        if not seed_data:
            logger.warning("Seed paper not found in S2. Graph signals will be skipped.")
            return None

        seed_emb = self._extract_embedding(seed_data)
        if seed_emb is None:
            logger.warning("No SPECTER2 embedding for seed paper. Graph signals skipped.")
            return None

        logger.info("Fetching forward citations ...")
        citations = self.client.get_citations(seed_paper_id, limit=50)
        logger.info("Fetching backward references ...")
        references = self.client.get_references(seed_paper_id, limit=50)

        neighbor_embs, years, counts = [], [], []
        for paper in citations + references:
            emb = self._extract_embedding(paper)
            if emb is not None:
                neighbor_embs.append(emb)
                years.append(paper.get("year") or self.CURRENT_YEAR)
                counts.append(paper.get("citationCount") or 0)

        if not neighbor_embs:
            logger.warning("No neighbour embeddings available. Graph signals skipped.")
            return None

        logger.info("Built graph context: %d neighbours with embeddings.", len(neighbor_embs))
        centroid = np.mean(neighbor_embs, axis=0)
        return GraphContext(
            seed_embedding=seed_emb,
            neighbor_embeddings=neighbor_embs,
            neighbor_years=years,
            neighbor_citation_counts=counts,
            neighborhood_centroid=centroid,
        )

    # -----------------------------------------------------------------------
    # Embed a candidate direction via S2 proxy search
    # -----------------------------------------------------------------------

    def embed_direction(self, direction: CandidateDirection) -> Optional[np.ndarray]:
        """
        We don't run a local SPECTER2 model.  Instead we search S2 for the
        most relevant existing paper matching the direction's text, and use
        *that* paper's SPECTER2 embedding as a proxy.  This is fast, free,
        and surprisingly accurate for well-phrased directions.
        """
        query = direction.title + " " + " ".join(direction.keywords[:4])
        results = self.client.search_paper(query, limit=3)
        for paper in results:
            emb = self._extract_embedding(paper)
            if emb is not None:
                logger.info(
                    "  Embedded '%s' via proxy paper '%s'",
                    direction.title[:40], paper.get("title", "?")[:40]
                )
                return emb
        logger.warning("  No proxy embedding found for '%s'.", direction.title[:40])
        return None

    # -----------------------------------------------------------------------
    # Compute per-direction graph signals
    # -----------------------------------------------------------------------

    def compute_direction_signals(
        self,
        direction: CandidateDirection,
        ctx: GraphContext,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Returns a dict of three graph-derived scores, all in [0, 1].
        Falls back to 0.5 (neutral) if the direction has no embedding.
        """
        emb = direction.embedding
        if emb is None:
            return {"graph_relevance": 0.5, "graph_novelty": 0.5, "graph_recency": 0.5}

        # -- graph_relevance --------------------------------------------------
        raw_relevance   = self._cosine(emb, ctx.neighborhood_centroid)
        graph_relevance = self._norm_cosine(raw_relevance)

        # -- graph_novelty  ---------------------------------------------------
        # Similarity to each neighbour; take mean of top-k (densest region)
        sims = [self._cosine(emb, n) for n in ctx.neighbor_embeddings]
        top_k_sims = sorted(sims, reverse=True)[:top_k]
        avg_density = self._norm_cosine(float(np.mean(top_k_sims)))
        graph_novelty = 1.0 - avg_density   # inverse of density

        # -- graph_recency  ---------------------------------------------------
        top_k_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        top_years = [ctx.neighbor_years[i] for i in top_k_idx]
        normed = [
            (y - self.REFERENCE_YEAR) / max(self.CURRENT_YEAR - self.REFERENCE_YEAR, 1)
            for y in top_years
        ]
        graph_recency = float(np.clip(np.mean(normed), 0.0, 1.0))

        return {
            "graph_relevance": graph_relevance,
            "graph_novelty":   graph_novelty,
            "graph_recency":   graph_recency,
        }

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_embedding(paper: dict) -> Optional[np.ndarray]:
        """Parse the embedding.vector list from an S2 paper dict."""
        emb_field = paper.get("embedding") or {}
        vector = emb_field.get("vector") if isinstance(emb_field, dict) else None
        if vector:
            return np.array(vector, dtype=np.float32)
        return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _norm_cosine(cos: float) -> float:
        """Shift cosine similarity from [-1, 1] to [0, 1]."""
        return (cos + 1.0) / 2.0


# ===========================================================================
# 4.  RANK FUSION MODULE
#     Merges + deduplicates both agent lists, scores each candidate on all
#     five signals, converts scores to ranks, and applies weighted RRF.
# ===========================================================================

@dataclass
class RankedDirection:
    """One entry in the final fused ranking."""
    direction:    CandidateDirection
    rrf_score:    float
    signal_scores: Dict[str, float]   # raw score per signal ∈ [0, 1]
    signal_ranks:  Dict[str, int]     # rank per signal (1 = best)
    final_rank:   int = 0


class RankFusionModule:
    """
    Reciprocal Rank Fusion over five heterogeneous signals.

    RRF formula
    -----------
        score(d) = Σ_signal  weight_signal / (k + rank_signal(d))

    k = 60   (standard RRF constant; dampens the influence of top ranks)

    Default weights are equal.  After you have MRR ground truth, expose
    these weights to grid-search / Bayesian optimisation (see notes in README).

    Parameters
    ----------
    client   : SemanticScholarClient  (shared, re-use for caching)
    weights  : dict mapping signal name → float  (default all 1.0)
    k        : int  RRF smoothing constant (default 60)
    use_graph: bool  Set False to run agent-only fusion without S2 calls
    """

    SIGNALS = ["agent_a", "agent_b", "graph_relevance", "graph_novelty", "graph_recency"]

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "agent_a":         1.0,
        "agent_b":         1.0,
        "graph_relevance": 1.0,
        "graph_novelty":   1.0,
        "graph_recency":   0.5,
    }

    def __init__(
        self,
        client:    Optional[SemanticScholarClient] = None,
        weights:   Optional[Dict[str, float]]      = None,
        k:         int  = 60,
        use_graph: bool = True,
    ):
        self.client    = client or SemanticScholarClient()
        self.extractor = GraphFeatureExtractor(self.client)
        self.weights   = weights or dict(self.DEFAULT_WEIGHTS)
        self.k         = k
        self.use_graph = use_graph

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def fuse(
        self,
        summary_ctx: SummaryContext,
        list_a:      DirectionList,
        list_b:      DirectionList,
    ) -> List[RankedDirection]:
        """
        Produce a merged, deduplicated, RRF-ranked list of candidate directions.

        Steps
        -----
        1. Merge and deduplicate both agent lists (by direction id).
        2. Build the citation graph neighbourhood for the seed paper.
        3. Proxy-embed each candidate direction via S2 search.
        4. Compute per-direction scores for all five signals.
        5. Convert scores → ranks.
        6. Apply weighted RRF.
        7. Sort by RRF score and return with final ranks populated.
        """

        # Step 1: merge
        pool = self._merge_and_deduplicate(list_a, list_b)
        logger.info("Direction pool after deduplication: %d directions", len(pool))

        # Step 2-3: graph context + direction embeddings
        graph_ctx = None
        if self.use_graph:
            graph_ctx = self.extractor.build_graph_context(summary_ctx.seed_paper_id)
            if graph_ctx is not None:
                logger.info("Proxy-embedding %d directions via S2 search ...", len(pool))
                for d in pool:
                    d.embedding = self.extractor.embed_direction(d)

        # Step 4: all signal scores
        signal_scores = self._compute_all_signal_scores(pool, list_a, list_b, graph_ctx)

        # Step 5: scores → ranks (rank 1 = highest score)
        signal_ranks = {
            sig: self._rank_from_scores(signal_scores[sig])
            for sig in self.SIGNALS
        }

        # Step 6: weighted RRF
        n = len(pool)
        results: List[RankedDirection] = []
        for d in pool:
            rrf = sum(
                self.weights.get(sig, 1.0) / (self.k + signal_ranks[sig].get(d.id, n + 1))
                for sig in self.SIGNALS
            )
            results.append(
                RankedDirection(
                    direction     = d,
                    rrf_score     = rrf,
                    signal_scores = {sig: signal_scores[sig].get(d.id, 0.5) for sig in self.SIGNALS},
                    signal_ranks  = {sig: signal_ranks[sig].get(d.id, n + 1) for sig in self.SIGNALS},
                )
            )

        # Step 7: sort and label
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        for i, r in enumerate(results):
            r.final_rank = i + 1

        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _merge_and_deduplicate(
        self,
        list_a: DirectionList,
        list_b: DirectionList,
    ) -> List[CandidateDirection]:
        """
        Union of both lists, first-seen wins on duplicate ids.
        If both agents produce the same direction id (e.g. "A-2" appears in B too),
        the Agent A version is kept and its relative position in B is still used
        for the agent_b ranking signal.
        """
        seen: Dict[str, CandidateDirection] = {}
        for d in list_a.directions + list_b.directions:
            if d.id not in seen:
                seen[d.id] = d
        return list(seen.values())

    def _compute_all_signal_scores(
        self,
        pool:      List[CandidateDirection],
        list_a:    DirectionList,
        list_b:    DirectionList,
        graph_ctx: Optional[GraphContext],
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns {signal_name: {direction_id: float_score ∈ [0, 1]}}.

        Agent signals use reciprocal position as the raw score so that
        rank-1 → 1.0, rank-2 → 0.5, rank-3 → 0.33, … (avoids arbitrary scaling).
        Graph signals come directly from GraphFeatureExtractor ∈ [0, 1].
        Directions absent from an agent list receive score 0 (treated as rank n+1 in RRF).
        """
        def agent_scores(dl: DirectionList) -> Dict[str, float]:
            return {d.id: 1.0 / (i + 1) for i, d in enumerate(dl.directions)}

        scores: Dict[str, Dict[str, float]] = {
            "agent_a":         agent_scores(list_a),
            "agent_b":         agent_scores(list_b),
            "graph_relevance": {},
            "graph_novelty":   {},
            "graph_recency":   {},
        }

        for d in pool:
            if graph_ctx is not None:
                graph_sigs = self.extractor.compute_direction_signals(d, graph_ctx)
            else:
                # Neutral fallback — graph signals contribute equally, breaking no ties
                graph_sigs = {"graph_relevance": 0.5, "graph_novelty": 0.5, "graph_recency": 0.5}

            scores["graph_relevance"][d.id] = graph_sigs["graph_relevance"]
            scores["graph_novelty"][d.id]   = graph_sigs["graph_novelty"]
            scores["graph_recency"][d.id]   = graph_sigs["graph_recency"]

        return scores

    @staticmethod
    def _rank_from_scores(scores: Dict[str, float]) -> Dict[str, int]:
        """Convert {id: score} → {id: rank} with rank 1 = highest score."""
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return {did: i + 1 for i, did in enumerate(sorted_ids)}


# ===========================================================================
# 5.  SIMULATED DATA
#     Replace these with your real agent / LLM output parsers.
#     "Attention Is All You Need" is used as the seed because it has a rich
#     S2 graph, making it a good integration test.
# ===========================================================================

def simulate_summary_context() -> SummaryContext:
    """Mimics output from the summarisation LLM agent."""
    return SummaryContext(
        # "Attention Is All You Need" — Semantic Scholar paper ID
        seed_paper_id  = "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        seed_title     = "Attention Is All You Need",
        seed_abstract  = (
            "We propose the Transformer, a model architecture eschewing recurrence "
            "entirely in favour of an attention mechanism to draw global dependencies "
            "between input and output."
        ),
        key_entities   = [
            "self-attention", "multi-head attention", "positional encoding",
            "encoder-decoder", "sequence transduction",
        ],
        key_themes     = ["neural machine translation", "language modelling", "sequence-to-sequence"],
        limitations    = [
            "quadratic complexity in sequence length",
            "fixed absolute positional encoding",
            "no explicit recurrence — long-range memory via attention only",
        ],
        open_questions = [
            "Can attention replace convolution in vision tasks?",
            "How to scale efficient attention to very long sequences (100k+ tokens)?",
            "Are learned relative position representations better than sinusoidal?",
        ],
    )


def simulate_agent_a_output() -> DirectionList:
    """Mimics Agent A's ranked list of future directions (best-first)."""
    return DirectionList(
        agent_id="A",
        directions=[
            CandidateDirection(
                id="A-0",
                title="Linear Attention Mechanisms for Long Sequences",
                description=(
                    "Replace softmax attention with kernel approximations (e.g. Random Features) "
                    "to achieve O(n) time and memory complexity, enabling Transformers to process "
                    "100k+ token contexts without truncation."
                ),
                keywords=["linear attention", "kernel methods", "long context", "efficiency"],
            ),
            CandidateDirection(
                id="A-1",
                title="Vision Transformers with Hierarchical Window Attention",
                description=(
                    "Apply self-attention over local image-patch windows using a shifted-window "
                    "strategy, capturing both local texture and global context without O(n²) cost."
                ),
                keywords=["vision transformer", "image recognition", "window attention", "ViT"],
            ),
            CandidateDirection(
                id="A-2",
                title="Relative Positional Encodings for Better Length Generalisation",
                description=(
                    "Replace absolute sinusoidal encodings with relative position biases learned "
                    "during training, yielding stronger generalisation to sequence lengths unseen "
                    "at training time."
                ),
                keywords=["positional encoding", "relative positions", "length generalisation"],
            ),
            CandidateDirection(
                id="A-3",
                title="Cross-lingual Transfer via Massively Multilingual Pre-training",
                description=(
                    "Pre-train a single Transformer on 100+ languages jointly so that shared "
                    "cross-lingual representations emerge, enabling zero-shot transfer to low-resource "
                    "languages without any fine-tuning data."
                ),
                keywords=["multilingual NLP", "cross-lingual transfer", "mBERT", "XLM"],
            ),
            CandidateDirection(
                id="A-4",
                title="Sparse Mixture-of-Experts Routing in Transformer Layers",
                description=(
                    "Route each token to a learned sparse subset of expert FFN layers, dramatically "
                    "increasing effective model capacity without proportional increases in compute "
                    "or memory per forward pass."
                ),
                keywords=["mixture of experts", "sparse gating", "scalability", "Switch Transformer"],
            ),
        ],
    )


def simulate_agent_b_output() -> DirectionList:
    """
    Mimics Agent B's ranked list.
    Intentionally contains one duplicate (id='A-2') to test deduplication,
    plus directions that are novel relative to Agent A's list.
    """
    return DirectionList(
        agent_id="B",
        directions=[
            CandidateDirection(
                id="B-0",
                title="Memory-Efficient Attention via Chunked Recomputation",
                description=(
                    "Compute attention in fixed-size sequence chunks and recompute activations "
                    "during the backward pass (gradient checkpointing), reducing peak GPU memory "
                    "from O(n²) to O(n) and enabling very long training sequences."
                ),
                keywords=["memory efficiency", "gradient checkpointing", "flash attention", "IO-aware"],
            ),
            CandidateDirection(
                id="B-1",
                title="Protein Structure Prediction with Pairwise Attention",
                description=(
                    "Model inter-residue contact maps using attention over amino-acid pairs, "
                    "capturing long-range dependencies in protein sequences that CNNs miss, "
                    "leading to near-experimental accuracy in structure prediction."
                ),
                keywords=["protein folding", "bioinformatics", "contact prediction", "AlphaFold"],
            ),
            # -- Intentional duplicate of A-2 to exercise deduplication logic --
            CandidateDirection(
                id="A-2",
                title="Relative Positional Encodings for Better Length Generalisation",
                description="Replace absolute sinusoidal encodings with relative position biases.",
                keywords=["positional encoding", "relative positions"],
            ),
            CandidateDirection(
                id="B-3",
                title="Decision Transformers for Offline Reinforcement Learning",
                description=(
                    "Frame offline RL as return-conditioned sequence modelling: predict the next "
                    "action given a history of (state, action, return-to-go) triples using a causal "
                    "Transformer, bypassing explicit value-function estimation."
                ),
                keywords=["reinforcement learning", "decision transformer", "offline RL", "sequence modelling"],
            ),
            CandidateDirection(
                id="B-4",
                title="Retrieval-Augmented Generation with Dense Passage Retrieval",
                description=(
                    "Augment a Transformer generator with a dense bi-encoder retriever that fetches "
                    "relevant passages from a large corpus at inference time, combining parametric "
                    "and non-parametric memory for knowledge-intensive NLP tasks."
                ),
                keywords=["retrieval augmented generation", "open-domain QA", "dense retrieval", "RAG"],
            ),
        ],
    )


# ===========================================================================
# 6.  DISPLAY UTILITIES
# ===========================================================================

def print_results(results: List[RankedDirection]) -> None:
    """Formatted console output of the fused ranking."""
    w = 72
    sigs = ["agent_a", "agent_b", "graph_relevance", "graph_novelty", "graph_recency"]
    short = {"agent_a": "AgA", "agent_b": "AgB",
             "graph_relevance": "Rel", "graph_novelty": "Nov", "graph_recency": "Rec"}

    print("\n" + "=" * w)
    print("FINAL FUSED RANKING  (RRF + Graph Signals)")
    print("=" * w)
    header = f"{'#':<4} {'ID':<6} {'RRF':>7}  " + "  ".join(f"{short[s]:>4}" for s in sigs)
    print(header + "  TITLE")
    print("-" * w)
    for r in results:
        ss = r.signal_scores
        row = (
            f"{r.final_rank:<4} {r.direction.id:<6} {r.rrf_score:>7.4f}  "
            + "  ".join(f"{ss.get(s, 0):>4.2f}" for s in sigs)
            + f"  {r.direction.title[:40]}"
        )
        print(row)
    print("=" * w)

    print("\nPer-signal rank breakdown  (1 = best in that signal):")
    print(f"{'ID':<6}", end="")
    for s in sigs:
        print(f"  {short[s]:>5}", end="")
    print()
    for r in results:
        print(f"{r.direction.id:<6}", end="")
        for s in sigs:
            print(f"  {r.signal_ranks.get(s, '—'):>5}", end="")
        print()

    print("\nTop-3 directions (detail):")
    print("-" * w)
    for r in results[:3]:
        print(f"[#{r.final_rank}] {r.direction.title}")
        print(f"     {r.direction.description[:120]}...")
        print(f"     keywords: {', '.join(r.direction.keywords)}")
        print()


# ===========================================================================
# 7.  CLI ENTRY POINT
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RRF rank fusion demo for future-direction recommendation."
    )
    p.add_argument(
        "--api-key", default=None,
        help="Semantic Scholar API key (optional but recommended). "
             "Get one free at semanticscholar.org/product/api"
    )
    p.add_argument(
        "--no-graph", action="store_true",
        help="Disable graph signals — run agent-only RRF fusion without S2 API calls."
    )
    p.add_argument(
        "--k", type=int, default=60,
        help="RRF smoothing constant (default 60)."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- load simulated data (replace with real agent / LLM output) ---
    summary_ctx = simulate_summary_context()
    list_a      = simulate_agent_a_output()
    list_b      = simulate_agent_b_output()

    print(f"\nSeed paper    : {summary_ctx.seed_title}")
    print(f"Key entities  : {', '.join(summary_ctx.key_entities[:4])}")
    print(f"Limitations   : {', '.join(summary_ctx.limitations[:2])}")
    print(f"Directions    : {len(list_a.directions)} from Agent A, "
          f"{len(list_b.directions)} from Agent B")
    print(f"Graph signals : {'ENABLED (live S2 API)' if not args.no_graph else 'DISABLED'}")

    # --- run fusion ---
    client = SemanticScholarClient(api_key=args.api_key)
    module = RankFusionModule(client=client, k=args.k, use_graph=not args.no_graph)

    results = module.fuse(summary_ctx, list_a, list_b)
    print_results(results)