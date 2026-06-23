"""Core data structures shared by every module in this folder.

A Query is one paper. It holds:
  - candidate_ids   : the full pool of direction ids (Agent A + Agent B merged)
  - rankings        : {"agent_a": [...], "agent_b": [...]} ordered best-first
  - relevance       : ground truth map  {gt_id -> set of matching candidate ids}
  - features        : per-candidate S2 graph signals  {cand_id -> {signal: value}}
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Query:
    query_id:          str
    candidate_ids:     List[str]
    rankings:          Dict[str, List[str]]
    relevance:         Dict[str, Set[str]]  = field(default_factory=dict)
    candidate_sources: Dict[str, List[str]] = field(default_factory=dict)
    features:          Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def sources(self) -> List[str]:
        return list(self.rankings.keys())

    @property
    def relevant_ids(self) -> Set[str]:
        out: Set[str] = set()
        for cands in self.relevance.values():
            out |= cands
        return out

    def has_ground_truth(self) -> bool:
        return any(len(c) > 0 for c in self.relevance.values())


@dataclass
class Dataset:
    queries:   List[Query]
    train_ids: List[str] = field(default_factory=list)
    dev_ids:   List[str] = field(default_factory=list)
    test_ids:  List[str] = field(default_factory=list)

    def _subset(self, ids):
        s = set(ids)
        return [q for q in self.queries if q.query_id in s]

    @property
    def train(self): return self._subset(self.train_ids) if self.train_ids else self.queries
    @property
    def dev(self):   return self._subset(self.dev_ids)   if self.dev_ids   else self.queries
    @property
    def test(self):  return self._subset(self.test_ids)  if self.test_ids  else self.queries

    def all_sources(self) -> List[str]:
        seen: List[str] = []
        for q in self.queries:
            for s in q.sources:
                if s not in seen: seen.append(s)
        return seen
