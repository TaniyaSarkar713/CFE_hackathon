"""White-space gap-bridging generator (plan §9.7 + §7 adjacency-gated novelty).

Turns the citation neighbourhood into *structured, entity-grounded* candidate
future directions — not paper titles. The idea: an entity that is **adjacent to
the seed** (it co-occurs with the seed's concepts in neighbour papers) but **not
yet part of the seed itself** marks an unexplored edge — a direction the field is
moving toward that this paper hasn't taken.

For each candidate bridge entity e (e ∉ seed concepts):
    adjacency(e) = # (neighbour, seed-entity) co-occurrences for e   (relevance gate)
    support(e)   = # neighbours mentioning e                          (popularity)
    novelty(e)   = 1 / (1 + support(e))                               (sparse = white space)
    score(e)     = adjacency(e) · novelty(e) · (1 + gap_affinity(e))

adjacency **gates** novelty (a rare off-topic phrase is not "white space" — it
must connect to the seed). `gap_affinity` optionally boosts bridges that sit near
the paper's open `key_gaps`. Entities never co-occurring with the seed are
dropped. The result is a ranked list of bridge directions usable as a candidate
source for fusion (`whitespace_source`) and entity-grounded for matching/cards.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set

from .entities import keyphrases


@dataclass
class WhitespaceCandidate:
    entity: str                 # the bridge entity (the unexplored concept)
    direction: str              # templated, entity-grounded direction text
    score: float
    adjacency: float            # strength of connection to the seed
    support: int                # how many neighbours mention it
    connects: List[str] = field(default_factory=list)   # seed entities it bridges to


def whitespace_candidates(
    seed_entities: Optional[Set[str]],
    neighbor_docs: Sequence[str],
    seed_text: str = "",
    gap_entities: Optional[Set[str]] = None,
    entity_fn: Callable[[str], Set[str]] = keyphrases,
    min_support: int = 2,
    max_df_ratio: float = 0.4,
    top_k: int = 20,
    template: str = "Investigate {entity} in relation to {connect}",
) -> List[WhitespaceCandidate]:
    """Generate ranked white-space (unexplored-edge) future directions.

    `seed_entities` are the paper's *main concepts* (themes / key_terms) — the
    exclusion set. `gap_entities` (the paper's open `key_gaps`) are NOT excluded;
    they only boost bridges that sit near the open gaps. Pass `seed_text` instead
    of `seed_entities` to extract them with `entity_fn`.
    """
    seed_ents = set(seed_entities) if seed_entities is not None else entity_fn(seed_text)
    gap_ents = gap_entities or set()
    doc_ents = [entity_fn(d) for d in neighbor_docs]

    df: Counter = Counter(e for ents in doc_ents for e in ents)
    adjacency: Counter = Counter()                       # e -> co-occurrences with seed
    connect_strength = defaultdict(Counter)              # e -> {seed entity -> count}
    gap_affinity: Counter = Counter()                    # e -> co-occurrences with gaps

    for ents in doc_ents:
        seed_in = ents & seed_ents
        gap_in = ents & gap_ents
        if not seed_in and not gap_in:
            continue
        for e in ents - seed_ents:                       # bridge candidates only
            for s in seed_in:
                adjacency[e] += 1
                connect_strength[e][s] += 1
            gap_affinity[e] += len(gap_in - {e})

    n_docs = max(len(doc_ents), 1)
    max_df = max(min_support, int(max_df_ratio * n_docs))
    cands: List[WhitespaceCandidate] = []
    for e, adj in adjacency.items():
        support = df[e]
        # must be adjacent, recurring, but NOT ubiquitous (generic terms appear
        # everywhere -> not "white space")
        if adj < 1 or support < min_support or support > max_df:
            continue
        relevance = adj / support                        # fraction of e's docs tied to the seed
        novelty = 1.0 / (1.0 + support)                  # sparse = white space
        gap_boost = 1.0 + (0.5 if gap_ents and gap_affinity[e] > 0 else 0.0)
        score = relevance * novelty * gap_boost
        connects = [s for s, _ in connect_strength[e].most_common(3)]
        anchor = connects[0] if connects else next(iter(seed_ents), "the paper")
        cands.append(WhitespaceCandidate(
            entity=e, direction=template.format(entity=e, connect=anchor),
            score=float(score), adjacency=float(adj), support=support,
            connects=connects))

    cands.sort(key=lambda c: (-c.score, c.entity))
    return cands[:top_k]


def whitespace_source(candidates: Sequence[WhitespaceCandidate]) -> List[str]:
    """Ranked direction texts (best-first) — a candidate source for fusion."""
    return [c.direction for c in candidates]
