"""Semantic Scholar clients for the candidate-harvesting agent.

Two implementations of the client contract the agent expects
(`get_paper / get_citations / get_references / search`):

  - `EnrichedS2Client` — subclasses Naz's `S2Client` to additionally request
    `tldr`, `fieldsOfStudy` (= keywords) and ids (`paperId`, `externalIds`) so
    the agent can read neighbours' keywords and follow the graph. Naz's client
    is untouched. Use this in production (hits the live S2 API).

  - `LocalCorpusS2Client` — an OFFLINE adapter over `Data/Paper_summaries/`.
    "Neighbours" are the most TF-IDF-similar *other* local papers; each neighbour
    carries its real `future_directions`, so the agent harvests genuine
    next-steps with no network. Lets us run + test the whole loop end-to-end.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

# --- production client (extends Naz's S2Client with richer fields) ---------- #
try:
    from semantic_scholar_ranking.s2_client import S2Client, S2_BASE, S2_SEARCH

    _RICH = "title,abstract,year,citationCount,embedding,paperId,externalIds,fieldsOfStudy,tldr"

    class EnrichedS2Client(S2Client):
        """Naz's client, but asking S2 for keywords (fieldsOfStudy), tldr, and ids."""

        def get_paper(self, corpus_id: str) -> dict:
            return self._get(f"{S2_BASE}/paper/CorpusID:{corpus_id}", {"fields": _RICH})

        def get_citations(self, corpus_id: str, limit: int = 50) -> List[dict]:
            d = self._get(f"{S2_BASE}/paper/CorpusID:{corpus_id}/citations",
                          {"fields": _RICH, "limit": limit})
            return [i.get("citingPaper", {}) for i in (d.get("data") or [])] if d else []

        def get_references(self, corpus_id: str, limit: int = 50) -> List[dict]:
            d = self._get(f"{S2_BASE}/paper/CorpusID:{corpus_id}/references",
                          {"fields": _RICH, "limit": limit})
            return [i.get("citedPaper", {}) for i in (d.get("data") or [])] if d else []

        def search(self, query: str, limit: int = 5) -> List[dict]:
            d = self._get(S2_SEARCH, {"query": query, "fields": _RICH, "limit": limit})
            return d.get("data") or [] if d else []
except Exception:  # semantic_scholar_ranking not importable -> offline only
    EnrichedS2Client = None  # type: ignore


# --- offline adapter over the local Data/ corpus ---------------------------- #
class LocalCorpusS2Client:
    """Offline stand-in: serves the local PubMed corpus as an S2-like graph."""

    def __init__(self, data_dir: str = "Data", exclude: Optional[str] = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.papers: Dict[str, dict] = {}
        for path in sorted(glob.glob(os.path.join(
                data_dir, "Paper_summaries", "*_future_directions.json"))):
            pid = os.path.basename(path).replace("_future_directions.json", "")
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            themes = s.get("main_themes", [])
            self.papers[pid] = {
                "paperId": pid,
                "title": s.get("topic_and_scope", "")[:200],
                "abstract": " ".join(themes + s.get("main_findings", [])),
                "themes": themes,
                "fieldsOfStudy": themes[:6],
                "future_directions": s.get("future_directions", []),
                "year": 2024, "citationCount": 0,
                "_simtext": s.get("topic_and_scope", "") + " " + " ".join(themes),
            }
        self.exclude = exclude
        self.ids = list(self.papers.keys())
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._mat = self._vec.fit_transform([self.papers[i]["_simtext"] for i in self.ids])
        self._pos = {i: k for k, i in enumerate(self.ids)}

    # -- contract --------------------------------------------------------- #
    def get_paper(self, corpus_id: str) -> dict:
        return self.papers.get(corpus_id, {})

    def _nearest(self, text: str, limit: int, drop: set) -> List[dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        q = self._vec.transform([text])
        sims = cosine_similarity(q, self._mat)[0]
        order = sims.argsort()[::-1]
        out = []
        for k in order:
            pid = self.ids[k]
            if pid in drop or (self.exclude and pid == self.exclude):
                continue
            out.append(self.papers[pid])
            if len(out) >= limit:
                break
        return out

    def get_citations(self, corpus_id: str, limit: int = 50) -> List[dict]:
        p = self.papers.get(corpus_id)
        if not p:
            return []
        return self._nearest(p["_simtext"], limit, drop={corpus_id})

    def get_references(self, corpus_id: str, limit: int = 50) -> List[dict]:
        return self.get_citations(corpus_id, limit)

    def search(self, query: str, limit: int = 5) -> List[dict]:
        return self._nearest(query, limit, drop=set())
