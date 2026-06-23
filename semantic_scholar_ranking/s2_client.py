"""Semantic Scholar API client + graph feature extractor.

Fetches the citation neighbourhood for a seed paper and computes four
graph-derived signals per candidate direction:

  graph_relevance   — cosine sim to neighbourhood centroid (related to existing work)
  graph_novelty     — 1 - density in nearest-k cluster (underexplored area)
  graph_recency     — mean normalised year of nearest-k neighbours (active area)
  citation_velocity — mean cit/year of nearest-k neighbours (trending area)

Usage
-----
client    = S2Client(api_key="YOUR_KEY")
extractor = GraphSignalExtractor(client)
ctx       = extractor.build_context("11812686")   # CorpusID
signals   = extractor.signals_for_direction("linear attention mechanisms", ctx)
# {"graph_relevance": 0.72, "graph_novelty": 0.41, ...}
"""
from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

S2_BASE   = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH = f"{S2_BASE}/paper/search"
FIELDS    = "title,abstract,year,citationCount,embedding"


# ===========================================================================
# S2 API Client
# ===========================================================================

class S2Client:
    """Semantic Scholar Graph API wrapper with caching and backoff.

    api_key : free at semanticscholar.org/product/api — gives 1 RPS dedicated
    rps     : requests per second (0.5 is safe for unauthenticated)
    """

    def __init__(self, api_key: Optional[str] = None, rps: float = 0.5):
        self._headers      = {"x-api-key": api_key} if api_key else {}
        self._min_interval = 1.0 / rps
        self._last_call    = 0.0
        self._cache: Dict[str, Any] = {}

    def _get(self, url: str, params: dict) -> dict:
        key = url + repr(sorted(params.items()))
        if key in self._cache:
            return self._cache[key]
        gap = self._min_interval - (time.time() - self._last_call)
        if gap > 0: time.sleep(gap)
        for attempt in range(5):
            try:
                r = requests.get(url, params=params,
                                 headers=self._headers, timeout=20)
                self._last_call = time.time()
                if r.status_code == 200:
                    d = r.json(); self._cache[key] = d; return d
                elif r.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("S2 rate-limited, retrying in %ds", wait)
                    time.sleep(wait)
                elif r.status_code in (400, 404):
                    logger.warning("S2 %d: %s", r.status_code, url)
                    return {}
                else:
                    logger.error("S2 HTTP %d: %s", r.status_code, url)
                    return {}
            except requests.RequestException as e:
                logger.error("Request error: %s", e)
                time.sleep(2 ** attempt)
        return {}

    def get_paper(self, corpus_id: str) -> dict:
        return self._get(f"{S2_BASE}/paper/CorpusID:{corpus_id}",
                         {"fields": FIELDS})

    def get_citations(self, corpus_id: str, limit: int = 50) -> List[dict]:
        data = self._get(f"{S2_BASE}/paper/CorpusID:{corpus_id}/citations",
                         {"fields": FIELDS, "limit": limit})
        if not data: return []
        return [i.get("citingPaper", {}) for i in (data.get("data") or [])]

    def get_references(self, corpus_id: str, limit: int = 50) -> List[dict]:
        data = self._get(f"{S2_BASE}/paper/CorpusID:{corpus_id}/references",
                         {"fields": FIELDS, "limit": limit})
        if not data: return []
        return [i.get("citedPaper", {}) for i in (data.get("data") or [])]

    def search(self, query: str, limit: int = 3) -> List[dict]:
        data = self._get(S2_SEARCH, {"query": query, "fields": FIELDS,
                                     "limit": limit})
        if not data: return []
        return data.get("data") or []


# ===========================================================================
# Graph Signal Extractor
# ===========================================================================

class GraphContext:
    """Pre-computed neighbourhood for one seed paper."""
    __slots__ = ("seed_emb", "neighbor_embs", "years", "counts", "centroid")

    def __init__(self, seed_emb, neighbor_embs, years, counts):
        self.seed_emb      = seed_emb
        self.neighbor_embs = neighbor_embs
        self.years         = years
        self.counts        = counts
        self.centroid      = np.mean(neighbor_embs, axis=0) if neighbor_embs else seed_emb


class GraphSignalExtractor:
    REF_YEAR     = 2000
    CURRENT_YEAR = 2025

    def __init__(self, client: S2Client):
        self.client = client

    # -----------------------------------------------------------------------
    def build_context(self, corpus_id: str) -> Optional[GraphContext]:
        """Fetch seed paper's neighbourhood from S2. Returns None on failure."""
        logger.info("  [S2] fetching CorpusID:%s", corpus_id)
        seed = self.client.get_paper(corpus_id)
        if not seed:
            logger.warning("  [S2] not found — graph signals unavailable")
            return None
        seed_emb = self._emb(seed)
        if seed_emb is None:
            logger.warning("  [S2] no SPECTER2 embedding — graph signals unavailable")
            return None

        logger.info("  [S2] fetching citations + references ...")
        neighbours = (self.client.get_citations(corpus_id, 50) +
                      self.client.get_references(corpus_id, 50))
        embs, years, counts = [], [], []
        for n in neighbours:
            e = self._emb(n)
            if e is not None:
                embs.append(e)
                years.append(n.get("year") or self.CURRENT_YEAR)
                counts.append(n.get("citationCount") or 0)
        if not embs:
            logger.warning("  [S2] no neighbour embeddings — graph signals unavailable")
            return None
        logger.info("  [S2] %d neighbours", len(embs))
        return GraphContext(seed_emb, embs, years, counts)

    # -----------------------------------------------------------------------
    def signals_for_direction(
        self, direction_text: str, ctx: GraphContext, top_k: int = 5
    ) -> Dict[str, float]:
        """Compute all four graph signals for a direction string.

        Proxy-embeds the direction by searching S2 for the closest existing paper.
        Falls back to 0.5 (neutral) if no embedding found.
        """
        emb = self._proxy_embed(direction_text)
        if emb is None or ctx is None:
            return {"graph_relevance": 0.5, "graph_novelty": 0.5,
                    "graph_recency": 0.5, "citation_velocity": 0.5}
        return self._compute(emb, ctx, top_k)

    def signals_for_text(
        self, text: str, ctx: GraphContext, top_k: int = 5
    ) -> Dict[str, float]:
        """Same as signals_for_direction — explicit alias for clarity."""
        return self.signals_for_direction(text, ctx, top_k)

    # -----------------------------------------------------------------------
    def _proxy_embed(self, text: str) -> Optional[np.ndarray]:
        """Search S2 for the closest paper to `text`, return its SPECTER2 vector."""
        for paper in self.client.search(text[:200], limit=3):
            e = self._emb(paper)
            if e is not None:
                return e
        return None

    def _compute(self, emb: np.ndarray, ctx: GraphContext,
                 top_k: int) -> Dict[str, float]:
        sims = [self._cos(emb, n) for n in ctx.neighbor_embs]

        # graph_relevance
        relevance = self._norm(self._cos(emb, ctx.centroid))

        # graph_novelty
        top_sims = sorted(sims, reverse=True)[:top_k]
        novelty  = 1.0 - self._norm(float(np.mean(top_sims)))

        # graph_recency + citation_velocity
        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        span = max(self.CURRENT_YEAR - self.REF_YEAR, 1)
        top_yrs = [ctx.years[i] for i in idx]
        recency = float(np.clip(
            np.mean([(y - self.REF_YEAR) / span for y in top_yrs]), 0, 1))
        velocities = [ctx.counts[i] / max(self.CURRENT_YEAR - ctx.years[i], 1)
                      for i in idx]
        citation_velocity = min(float(np.mean(velocities)) / 100.0, 1.0)

        return {"graph_relevance":   relevance,
                "graph_novelty":     novelty,
                "graph_recency":     recency,
                "citation_velocity": citation_velocity}

    @staticmethod
    def _emb(paper: dict) -> Optional[np.ndarray]:
        ef = paper.get("embedding") or {}
        v  = ef.get("vector") if isinstance(ef, dict) else None
        return np.array(v, dtype=np.float32) if v else None

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

    @staticmethod
    def _norm(c: float) -> float:
        return (c + 1.0) / 2.0
