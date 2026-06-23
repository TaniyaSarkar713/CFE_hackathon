"""Entity-grounded re-ranking of the S2 knowledge-graph candidates.

Agent C returns S2 neighbours ranked by *graph topology* (recency / novelty /
relevance from SPECTER2). That orders by "what's trending near the paper", not by
"what future work the paper actually leaves open". Entities fix this: rank each
S2 candidate by how well its entities overlap the seed's **open-gap entities**
(from `key_gaps`, available at inference) while down-weighting entities the paper
has **already covered** — i.e. surface the candidates that point at the paper's
*white space*.

The entity extractor is pluggable (`entity_fn`):
  - default `keyphrases` — lightweight, dependency-free (lowercased 1-2-grams
    minus stopwords); good enough to demonstrate the lift;
  - pass a real biomedical NER (the model behind `future_sec_NER_ground_truth`)
    or KeyBERT for the strongest results, or precomputed entity sets per candidate.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

_STOP = set("""
a an the of to in for on and or with without within into onto from by at as is are was were
be been being this that these those it its their our we they he she his her them you your i
study studies research future work directions direction need needs needed should could may might
can will would also further more using used use based approach method methods results data paper
among across between through over under about via per such other others including include included
how why what when where which who whose toward towards given however therefore thus hence overall
focus focuses focused focusing emphasis emphasise consideration considerations working work worked
particular particularly specific specifically various several different key important role context
analysis analyses analyse analyze understanding understand assessment evaluate evaluation evaluating
high low new novel current existing potential significant relevant related associated within
""".split())

_TOKEN = re.compile(r"[a-z][a-z0-9\-]{2,}")


def keyphrases(text: str, max_n: int = 2) -> Set[str]:
    """Lightweight default entity/keyphrase extractor: 1-2-grams minus stopwords."""
    toks = [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]
    out: Set[str] = set(toks)
    for n in range(2, max_n + 1):
        for i in range(len(toks) - n + 1):
            out.add(" ".join(toks[i:i + n]))
    return out


def _overlap(cand: Set[str], target: Set[str]) -> Tuple[int, Set[str]]:
    hits = cand & target
    return len(hits), hits


# --------------------------------------------------------------------------- #
# Core re-ranker (operates on entity sets)
# --------------------------------------------------------------------------- #
def entity_rerank(
    candidate_ids: Sequence[str],
    candidate_entities: Dict[str, Set[str]],
    gap_entities: Set[str],
    covered_entities: Optional[Set[str]] = None,
    novelty_weight: float = 0.5,
) -> Tuple[List[str], Dict[str, Dict]]:
    """Re-rank candidates by open-gap entity alignment (white-space gated).

    score(c) = |E(c) ∩ gap_entities|  +  novelty_weight · 1{E(c) introduces an
                                          entity the paper has not already covered}

    Primary signal is overlap with the seed's open-gap entities; `covered_entities`
    only feeds the (secondary) novelty bonus, so a genuinely gap-aligned candidate
    is never zeroed just because the paper *mentions* that gap.

    Returns (ranked ids best-first, per-candidate info with matched entities).
    """
    covered = covered_entities or set()

    info: Dict[str, Dict] = {}
    for cid in candidate_ids:
        ce = candidate_entities.get(cid, set())
        gap_hit, matched = _overlap(ce, gap_entities)
        novel = len(ce - covered)
        score = gap_hit + novelty_weight * (1.0 if novel > 0 else 0.0)
        info[cid] = {"score": float(score), "gap_hits": gap_hit,
                     "matched_entities": sorted(matched), "n_entities": len(ce)}
    ranked = sorted(candidate_ids,
                    key=lambda c: (-info[c]["score"], str(c)))
    return ranked, info


# --------------------------------------------------------------------------- #
# Text convenience (extracts entities first)
# --------------------------------------------------------------------------- #
def entity_rerank_texts(
    candidate_ids: Sequence[str],
    candidate_texts: Sequence[str],
    gap_text: str,
    covered_text: str = "",
    entity_fn: Callable[[str], Set[str]] = keyphrases,
    novelty_weight: float = 0.5,
) -> Tuple[List[str], Dict[str, Dict]]:
    """Re-rank S2 candidate *texts* by overlap with the seed's open-gap entities.

    `gap_text`     — concatenated `key_gaps` (the open future-work space).
    `covered_text` — what the paper already did (findings/methods), to gate novelty.
    """
    cand_ents = {cid: entity_fn(t) for cid, t in zip(candidate_ids, candidate_texts)}
    gap_ents = entity_fn(gap_text)
    covered = entity_fn(covered_text) if covered_text else set()
    return entity_rerank(candidate_ids, cand_ents, gap_ents, covered, novelty_weight)


def rerank_s2_source(
    s2_texts: Sequence[str],
    gap_text: str,
    covered_text: str = "",
    entity_fn: Callable[[str], Set[str]] = keyphrases,
    novelty_weight: float = 0.5,
) -> List[str]:
    """Re-rank a plain `s2` ranked text list (from `s2_agent_source`) by entity
    alignment to the seed's gaps; returns the re-ordered texts for fusion."""
    ids = [f"c{i}" for i in range(len(s2_texts))]
    text_of = dict(zip(ids, s2_texts))
    ranked, _info = entity_rerank_texts(ids, s2_texts, gap_text, covered_text,
                                        entity_fn, novelty_weight)
    return [text_of[i] for i in ranked]
