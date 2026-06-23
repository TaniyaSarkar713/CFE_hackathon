"""
semantic_scholar_module.py
==========================


Signals fused via RRF
---------------------
  agent_a           — Agent A ordering (future_directions list)
  agent_b           — Agent B ordering (key_gaps + main_themes)
  graph_relevance   — cosine sim to S2 neighbourhood centroid
  graph_novelty     — 1 - density in nearest-k cluster (unexplored = novel)
  graph_recency     — mean normalised year of nearest-k neighbours
  citation_velocity — mean citation-velocity of nearest-k neighbours (trending)

Usage
-----
  python semantic_scholar_module.py
  python semantic_scholar_module.py --paper-id 11812686
  python semantic_scholar_module.py --no-graph
  python semantic_scholar_module.py --output results.json
  python semantic_scholar_module.py --api-key s2k-VqZPfeS1kZk3Mqd5ac7i1PKN0rv15d8wrKm7LNUR

Install
-------
  pip install numpy requests
"""

from __future__ import annotations

import os
import json
import time
import logging
import argparse
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import requests
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths relative to this file  (Data/ is one level up)
# ---------------------------------------------------------------------------
_HERE            = os.path.dirname(os.path.abspath(__file__))
_ROOT            = os.path.dirname(_HERE)
DEFAULT_SUMMARIES_DIR = os.path.join(_ROOT, "Data", "Paper_summaries")
DEFAULT_PAPERS_DIR    = os.path.join(_ROOT, "Data", "papers")


# ===========================================================================
# DATA CONTRACTS
# ===========================================================================

@dataclass
class SummaryContext:
    seed_paper_id:   str
    topic_and_scope: str
    main_themes:     List[str]
    main_findings:   List[str]
    key_gaps:        List[str]


@dataclass
class CandidateDirection:
    id:        str
    title:     str
    rationale: str
    keywords:  List[str]            = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class DirectionList:
    agent_id:   str
    directions: List[CandidateDirection]


@dataclass
class RankedDirection:
    direction:     CandidateDirection
    rrf_score:     float
    signal_scores: Dict[str, float]
    signal_ranks:  Dict[str, int]
    source_agent:  str
    final_rank:    int = 0


# ===========================================================================
# DATA LOADER
# ===========================================================================

class DataLoader:
    """
    Agent A  <-  future_directions  (explicit LLM proposals, id order = rank)
    Agent B  <-  key_gaps + main_themes  (gap-closure perspective)
    """

    def __init__(self, summaries_dir: str = DEFAULT_SUMMARIES_DIR):
        self.summaries_dir = summaries_dir

    def list_paper_ids(self) -> List[str]:
        pattern = os.path.join(self.summaries_dir, "*_future_directions.json")
        return sorted(
            os.path.basename(f).replace("_future_directions.json", "")
            for f in glob.glob(pattern)
        )

    def load(self, paper_id: str):
        """Returns (SummaryContext, DirectionList_A, DirectionList_B)."""
        path = os.path.join(self.summaries_dir, f"{paper_id}_future_directions.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._build_context(data), self._build_agent_a(data), self._build_agent_b(data)

    def _build_context(self, data: dict) -> SummaryContext:
        return SummaryContext(
            seed_paper_id   = str(data["file"]),
            topic_and_scope = data.get("topic_and_scope", ""),
            main_themes     = data.get("main_themes", []),
            main_findings   = data.get("main_findings", []),
            key_gaps        = [g["description"] for g in data.get("key_gaps", [])],
        )

    def _build_agent_a(self, data: dict) -> DirectionList:
        dirs = []
        for item in data.get("future_directions", []):
            dirs.append(CandidateDirection(
                id        = f"A-{item['id']}",
                title     = item["title"],
                rationale = item.get("rationale", ""),
                keywords  = self._keywords(item.get("rationale", "")),
            ))
        return DirectionList(agent_id="A", directions=dirs)

    def _build_agent_b(self, data: dict) -> DirectionList:
        dirs = []
        for i, gap in enumerate(data.get("key_gaps", []), 1):
            desc = gap["description"]
            dirs.append(CandidateDirection(
                id        = f"B-gap-{i}",
                title     = self._gap_title(desc),
                rationale = f"Addresses an explicit gap: {desc}",
                keywords  = self._keywords(desc),
            ))
        for i, theme in enumerate(data.get("main_themes", []), 1):
            dirs.append(CandidateDirection(
                id        = f"B-theme-{i}",
                title     = f"Investigate and advance: {theme}",
                rationale = (
                    f"Central theme: '{theme}'. "
                    "Future work should systematically deepen understanding here."
                ),
                keywords  = self._keywords(theme),
            ))
        return DirectionList(agent_id="B", directions=dirs)

    @staticmethod
    def _gap_title(gap: str) -> str:
        first = gap.split(";")[0].strip()
        return f"Close the gap: {first[:97]}..." if len(first) > 100 else f"Close the gap: {first}"

    @staticmethod
    def _keywords(text: str, n: int = 5) -> List[str]:
        STOP = {
            "about","across","after","although","among","around","because",
            "before","between","during","findings","further","however",
            "including","limited","literature","number","obtain","often",
            "other","outcomes","paper","research","results","should",
            "studies","study","there","these","their","though","through",
            "toward","under","which","while","within","without",
        }
        words = [w.lower().strip(".,;:()'\"") for w in text.split() if len(w) > 5]
        seen, unique = set(), []
        for w in words:
            if w not in seen and w not in STOP:
                seen.add(w); unique.append(w)
        return unique[:n]


# ===========================================================================
# SEMANTIC SCHOLAR CLIENT
# ===========================================================================

S2_BASE   = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH = f"{S2_BASE}/paper/search"
# citationVelocity not a direct field — we compute it from citationCount + year
FIELDS    = "title,abstract,year,citationCount,embedding"


class SemanticScholarClient:
    """S2 Graph API wrapper with caching and exponential-backoff on 429s."""

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
        if gap > 0:
            time.sleep(gap)
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, headers=self._headers, timeout=20)
                self._last_call = time.time()
                if r.status_code == 200:
                    d = r.json(); self._cache[key] = d; return d
                elif r.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("S2 rate-limited, retrying in %ds", wait)
                    time.sleep(wait)
                elif r.status_code == 404:
                    logger.warning("S2 404 — paper not found: %s", url)
                    return {}
                elif r.status_code == 400:
                    logger.warning("S2 400 — bad request (paper may have no citations/refs): %s", url)
                    return {}
                else:
                    logger.error("S2 HTTP %d: %s", r.status_code, url); return {}
            except requests.RequestException as e:
                logger.error("Request error: %s", e); time.sleep(2 ** attempt)
        return {}

    def get_paper(self, corpus_id: str) -> dict:
        return self._get(f"{S2_BASE}/paper/CorpusID:{corpus_id}", {"fields": FIELDS})

    def get_citations(self, corpus_id: str, limit: int = 50) -> List[dict]:
        data = self._get(
            f"{S2_BASE}/paper/CorpusID:{corpus_id}/citations",
            {"fields": FIELDS, "limit": limit}
        )
        if not data:
            return []
        return [item.get("citingPaper", {}) for item in (data.get("data") or [])]

    def get_references(self, corpus_id: str, limit: int = 50) -> List[dict]:
        data = self._get(
            f"{S2_BASE}/paper/CorpusID:{corpus_id}/references",
            {"fields": FIELDS, "limit": limit}
        )
        if not data:
            return []
        return [item.get("citedPaper", {}) for item in (data.get("data") or [])]

    def search_paper(self, query: str, limit: int = 3) -> List[dict]:
        data = self._get(S2_SEARCH, {"query": query, "fields": FIELDS, "limit": limit})
        if not data:
            return []
        return data.get("data") or []


# ===========================================================================
# GRAPH FEATURE EXTRACTOR
# ===========================================================================

@dataclass
class GraphContext:
    seed_embedding:          np.ndarray
    neighbor_embeddings:     List[np.ndarray]
    neighbor_years:          List[int]
    neighbor_citation_counts: List[int]       # needed for citation_velocity
    neighborhood_centroid:   np.ndarray


class GraphFeatureExtractor:
    REF_YEAR     = 2000
    CURRENT_YEAR = 2025

    def __init__(self, client: SemanticScholarClient):
        self.client = client

    def build_graph_context(self, corpus_id: str) -> Optional[GraphContext]:
        logger.info("  [S2] fetching seed CorpusID:%s", corpus_id)
        seed = self.client.get_paper(corpus_id)
        if not seed:
            logger.warning("  [S2] seed not found — graph signals skipped")
            return None
        seed_emb = self._emb(seed)
        if seed_emb is None:
            logger.warning("  [S2] no SPECTER2 embedding — graph signals skipped")
            return None

        logger.info("  [S2] fetching citations + references ...")
        citations  = self.client.get_citations(corpus_id, 50)
        references = self.client.get_references(corpus_id, 50)
        neighbours = citations + references
        if not neighbours:
            logger.warning("  [S2] no citations or references returned — graph signals skipped")
            return None

        embs, years, counts = [], [], []
        for n in neighbours:
            e = self._emb(n)
            if e is not None:
                embs.append(e)
                years.append(n.get("year") or self.CURRENT_YEAR)
                counts.append(n.get("citationCount") or 0)

        if not embs:
            logger.warning("  [S2] no neighbour embeddings — graph signals skipped")
            return None

        logger.info("  [S2] %d neighbours with embeddings", len(embs))
        return GraphContext(
            seed_embedding           = seed_emb,
            neighbor_embeddings      = embs,
            neighbor_years           = years,
            neighbor_citation_counts = counts,
            neighborhood_centroid    = np.mean(embs, axis=0),
        )

    def embed_direction(self, d: CandidateDirection) -> Optional[np.ndarray]:
        """Proxy-embed: find closest S2 paper, use its SPECTER2 vector."""
        query = d.title + " " + " ".join(d.keywords[:4])
        for r in self.client.search_paper(query, 3):
            e = self._emb(r)
            if e is not None:
                return e
        return None

    def compute_signals(
        self, d: CandidateDirection, ctx: GraphContext, top_k: int = 5
    ) -> Dict[str, float]:
        """
        Returns four graph signals, all ∈ [0, 1]:

        graph_relevance     — how close the direction is to the seed's neighbourhood
        graph_novelty       — how underexplored the direction's region is
        graph_recency       — how recently active that region is
        citation_velocity   — how fast nearby papers are accumulating citations
                              (citations / age in years) — high = trending area
        """
        if d.embedding is None:
            return {
                "graph_relevance":   0.5,
                "graph_novelty":     0.5,
                "graph_recency":     0.5,
                "citation_velocity": 0.5,
            }

        emb  = d.embedding
        sims = [self._cos(emb, n) for n in ctx.neighbor_embeddings]

        # -- graph_relevance --------------------------------------------------
        relevance = self._norm(self._cos(emb, ctx.neighborhood_centroid))

        # -- graph_novelty  ---------------------------------------------------
        top_sims  = sorted(sims, reverse=True)[:top_k]
        novelty   = 1.0 - self._norm(float(np.mean(top_sims)))

        # -- graph_recency  ---------------------------------------------------
        idx      = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        top_yrs  = [ctx.neighbor_years[i] for i in idx]
        span     = max(self.CURRENT_YEAR - self.REF_YEAR, 1)
        recency  = float(np.clip(
            np.mean([(y - self.REF_YEAR) / span for y in top_yrs]), 0, 1
        ))

        # -- citation_velocity -----------------------------------------------
        # citations per year of age for the top-k nearest neighbours
        # captures whether the direction's area is actively accumulating citations
        velocities = []
        for i in idx:
            count = ctx.neighbor_citation_counts[i]
            age   = max(self.CURRENT_YEAR - ctx.neighbor_years[i], 1)
            velocities.append(count / age)
        raw_vel = float(np.mean(velocities))
        # normalise: cap at 100 cit/yr (high-impact threshold), scale to [0,1]
        citation_velocity = min(raw_vel / 100.0, 1.0)

        return {
            "graph_relevance":   relevance,
            "graph_novelty":     novelty,
            "graph_recency":     recency,
            "citation_velocity": citation_velocity,
        }

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


# ===========================================================================
# RANK FUSION MODULE
# ===========================================================================

class RankFusionModule:
    """
    Reciprocal Rank Fusion:  score(d) = Σ_signal  weight / (k + rank_signal(d))

    6 signals total:
      agent_a, agent_b              — the two agent rankings
      graph_relevance, graph_novelty,
      graph_recency, citation_velocity  — S2 graph signals
    """

    SIGNALS = [
        "agent_a", "agent_b",
        "graph_relevance", "graph_novelty", "graph_recency", "citation_velocity",
    ]

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "agent_a":           1.0,
        "agent_b":           1.0,
        "graph_relevance":   1.0,
        "graph_novelty":     1.0,
        "graph_recency":     0.5,
        "citation_velocity": 1.0,   # new — rewards trending areas
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

    def fuse(
        self,
        ctx:    SummaryContext,
        list_a: DirectionList,
        list_b: DirectionList,
    ) -> List[RankedDirection]:

        pool = self._merge(list_a, list_b)
        logger.info("  Pool: %d unique directions (A=%d, B=%d)",
                    len(pool), len(list_a.directions), len(list_b.directions))

        graph_ctx = None
        if self.use_graph:
            graph_ctx = self.extractor.build_graph_context(ctx.seed_paper_id)
            if graph_ctx:
                logger.info("  Proxy-embedding %d directions via S2 search ...", len(pool))
                for d in pool:
                    d.embedding = self.extractor.embed_direction(d)

        sig_scores = self._score_all(pool, list_a, list_b, graph_ctx)
        sig_ranks  = {s: self._to_ranks(sig_scores[s]) for s in self.SIGNALS}

        a_ids = {d.id for d in list_a.directions}
        b_ids = {d.id for d in list_b.directions}
        n     = len(pool)

        results = []
        for d in pool:
            rrf = sum(
                self.weights.get(s, 1.0) / (self.k + sig_ranks[s].get(d.id, n + 1))
                for s in self.SIGNALS
            )
            src = ("A+B" if d.id in a_ids and d.id in b_ids
                   else "A" if d.id in a_ids else "B")
            results.append(RankedDirection(
                direction     = d,
                rrf_score     = rrf,
                signal_scores = {s: sig_scores[s].get(d.id, 0.5) for s in self.SIGNALS},
                signal_ranks  = {s: sig_ranks[s].get(d.id, n + 1)  for s in self.SIGNALS},
                source_agent  = src,
            ))

        results.sort(key=lambda x: x.rrf_score, reverse=True)
        for i, r in enumerate(results):
            r.final_rank = i + 1
        return results

    def _merge(self, a, b):
        seen, pool = set(), []
        for d in a.directions + b.directions:
            if d.id not in seen:
                seen.add(d.id); pool.append(d)
        return pool

    def _score_all(self, pool, a, b, graph_ctx):
        def ags(dl):
            return {d.id: 1.0 / (i + 1) for i, d in enumerate(dl.directions)}

        neutral = {"graph_relevance": 0.5, "graph_novelty": 0.5,
                   "graph_recency": 0.5,   "citation_velocity": 0.5}

        scores = {
            "agent_a": ags(a), "agent_b": ags(b),
            "graph_relevance": {}, "graph_novelty": {},
            "graph_recency": {},   "citation_velocity": {},
        }
        for d in pool:
            sigs = (self.extractor.compute_signals(d, graph_ctx)
                    if graph_ctx else neutral)
            for sig, val in sigs.items():
                scores[sig][d.id] = val
        return scores

    @staticmethod
    def _to_ranks(scores):
        order = sorted(scores, key=lambda x: scores[x], reverse=True)
        return {did: i + 1 for i, did in enumerate(order)}


# ===========================================================================
# OUTPUT
# ===========================================================================

def print_paper_results(paper_id, ctx, results):
    sigs  = ["agent_a", "agent_b", "graph_relevance",
             "graph_novelty", "graph_recency", "citation_velocity"]
    short = {"agent_a": "AgA", "agent_b": "AgB", "graph_relevance": "Rel",
             "graph_novelty": "Nov", "graph_recency": "Rec", "citation_velocity": "Vel"}
    w = 86
    print(f"\n{'='*w}")
    print(f"Paper : {paper_id}")
    print(f"Scope : {ctx.topic_and_scope[:115]}...")
    print(f"{'='*w}")
    hdr = f"{'#':<4} {'Src':<5} {'RRF':>7}  " + "  ".join(f"{short[s]:>4}" for s in sigs)
    print(hdr + "  Title")
    print(f"{'-'*w}")
    for r in results:
        ss  = r.signal_scores
        row = (f"{r.final_rank:<4} {r.source_agent:<5} {r.rrf_score:>7.4f}  "
               + "  ".join(f"{ss.get(s,0):>4.2f}" for s in sigs)
               + f"  {r.direction.title[:38]}")
        print(row)
    print(f"{'='*w}")
    print("Top 3:")
    for r in results[:3]:
        print(f"  [{r.final_rank}][{r.source_agent}] {r.direction.title}")
        print(f"       {r.direction.rationale[:110]}...")


def results_to_dict(paper_id, ctx, results):
    return {
        "paper_id":          paper_id,
        "topic_and_scope":   ctx.topic_and_scope,
        "ranked_directions": [
            {
                "rank":          r.final_rank,
                "id":            r.direction.id,
                "source_agent":  r.source_agent,
                "title":         r.direction.title,
                "rationale":     r.direction.rationale,
                "rrf_score":     round(r.rrf_score, 6),
                "signal_scores": {k: round(v, 4) for k, v in r.signal_scores.items()},
                "signal_ranks":  r.signal_ranks,
            }
            for r in results
        ]
    }


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Rank Fusion for CFE Hackathon future directions"
    )
    p.add_argument("--summaries-dir", default=DEFAULT_SUMMARIES_DIR,
                   help=f"Path to Paper_summaries folder (default: {DEFAULT_SUMMARIES_DIR})")
    p.add_argument("--paper-id",  default=None,
                   help="Run on a single paper ID only. Omit to run all.")
    p.add_argument("--no-graph",  action="store_true",
                   help="Skip S2 graph signals (offline mode)")
    p.add_argument("--api-key",   default=None,
                   help="S2 API key — get one free at semanticscholar.org/product/api")
    p.add_argument("--k",         type=int, default=60,
                   help="RRF smoothing constant (default 60)")
    p.add_argument("--output",    default=None,
                   help="Save results to this JSON file")
    args = p.parse_args()

    loader = DataLoader(args.summaries_dir)
    client = SemanticScholarClient(api_key=args.api_key)
    module = RankFusionModule(client=client, k=args.k, use_graph=not args.no_graph)

    paper_ids = [args.paper_id] if args.paper_id else loader.list_paper_ids()

    print(f"\nRank Fusion — CFE Hackathon")
    print(f"Summaries : {args.summaries_dir}")
    print(f"Papers    : {len(paper_ids)}")
    print(f"Signals   : {', '.join(RankFusionModule.SIGNALS)}")
    print(f"Graph     : {'OFF (--no-graph)' if args.no_graph else 'ON (live S2 API)'}")

    all_results = []
    for pid in paper_ids:
        logger.info("Processing %s ...", pid)
        try:
            ctx, list_a, list_b = loader.load(pid)
        except FileNotFoundError:
            logger.error("No JSON for paper %s — skipping", pid)
            continue
        results = module.fuse(ctx, list_a, list_b)
        print_paper_results(pid, ctx, results)
        all_results.append(results_to_dict(pid, ctx, results))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved -> {args.output}")

    print(f"\nDone. {len(all_results)} papers processed.")
    return all_results


if __name__ == "__main__":
    main()