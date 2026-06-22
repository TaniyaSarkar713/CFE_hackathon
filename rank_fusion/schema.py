"""Core data structures for the rank-fusion / MRR harness.

A `Query` is one paper (its future-work section masked). It carries:
  - a candidate pool (the directions the rankers may reorder),
  - one ranked list per *source* (gpt, claude, retrieval, citation, ...),
  - the ground-truth relevance: which candidates match each true future
    direction g_j (from the masked future-work section).

Everything downstream (MRR, RRF, optimisation, meta-ranker) consumes this
single structure, so synthetic data and real `Data/` data plug in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Query:
    """One paper / ranking instance.

    Attributes
    ----------
    query_id:
        Stable id for the paper (e.g. the PubMed id).
    candidate_ids:
        Full candidate pool C the rankers see (deduplicated direction ids).
    rankings:
        ``source name -> ordered list of candidate ids`` (best first). A source
        need not rank every candidate; missing candidates are treated as
        unranked by that source.
    relevance:
        ``true-direction id g_j -> set of candidate ids that match it``. This is
        the masked ground truth, mapped onto the candidate pool by the match
        function (exact id for synthetic, embedding/TF-IDF cosine for real text).
    candidate_sources:
        Optional provenance ``generator name -> candidate ids it contributed``
        (generative / retrieval / citation / limitation-template). Used by the
        candidate-recall diagnostic (Level 1): the rankers can only reorder what
        the generators surfaced, so pool recall is the ceiling. Defaults to empty.
    features:
        Optional per-candidate numeric feature vectors ``candidate id -> {name:
        value}`` (e.g. sim_to_limitations, novelty, recency). Consumed by the
        feature-augmented RRF bias term and the learned rankers.
    """

    query_id: str
    candidate_ids: List[str]
    rankings: Dict[str, List[str]]
    relevance: Dict[str, Set[str]] = field(default_factory=dict)
    candidate_sources: Dict[str, List[str]] = field(default_factory=dict)
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def sources(self) -> List[str]:
        return list(self.rankings.keys())

    @property
    def relevant_ids(self) -> Set[str]:
        """Union of all candidate ids that match some true direction."""
        out: Set[str] = set()
        for cands in self.relevance.values():
            out |= cands
        return out

    def has_ground_truth(self) -> bool:
        return any(len(c) > 0 for c in self.relevance.values())


@dataclass
class Dataset:
    """A collection of queries plus a train/dev/test split by query id."""

    queries: List[Query]
    train_ids: List[str] = field(default_factory=list)
    dev_ids: List[str] = field(default_factory=list)
    test_ids: List[str] = field(default_factory=list)

    def _subset(self, ids: List[str]) -> List[Query]:
        idset = set(ids)
        return [q for q in self.queries if q.query_id in idset]

    @property
    def train(self) -> List[Query]:
        return self._subset(self.train_ids) if self.train_ids else self.queries

    @property
    def dev(self) -> List[Query]:
        return self._subset(self.dev_ids) if self.dev_ids else self.queries

    @property
    def test(self) -> List[Query]:
        return self._subset(self.test_ids) if self.test_ids else self.queries

    def all_sources(self) -> List[str]:
        seen: List[str] = []
        for q in self.queries:
            for s in q.sources:
                if s not in seen:
                    seen.append(s)
        return seen
