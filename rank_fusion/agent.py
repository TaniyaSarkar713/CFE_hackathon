"""Autonomous candidate-harvesting agent (the one *agentic* component).

This is the independent AI agent of ResearchHorizon: given a paper with its
future-work masked, it runs tools in a loop — read the citation neighbourhood
via Naz's Semantic Scholar client, harvest neighbours' **keywords + realized
future directions**, add them to the recommender's candidate pool — and **stops
on its own** when newly harvested candidates stop improving coverage of the
paper's open gaps (the coverage/recall harness). It decides at runtime which
neighbour to expand and which gap to search for next, so the tool-call trace is
different for every paper (Anthropic/Willison's definition of an agent).

Everything downstream (rank → fuse → explain → MRR) stays a deterministic,
metric-calibrated workflow. The agent only *gathers*; the metric still decides
the order. The boundary is a single artifact: the candidate pool.

Client contract (duck-typed — Naz's `S2Client`, an enriched subclass, or an
offline adapter all satisfy it):
    get_citations(corpus_id, limit) -> list[paper dict]
    get_references(corpus_id, limit) -> list[paper dict]
    search(query, limit) -> list[paper dict]
A *paper dict* may carry: title, abstract, tldr ({"text":..}|str), fieldsOfStudy
(list = keywords), externalIds/paperId (for expansion), and — for the offline
adapter — future_directions (list of {title, rationale}) = realized next steps.

Leakage note: the **stopping** signal uses the paper's `key_gaps` (derived from
the paper body by Agent 1 — available at inference). The masked ground-truth
future-work is used ONLY to *evaluate* the harvested pool afterwards
(`evaluate_pool`), never to drive the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .dedup import dedup_candidates
from .matching import match_relevance


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    id: str
    text: str
    keywords: List[str]
    provenance: str            # "expand:<id>" | "search:<gap>" | "seed"
    source_id: Optional[str]   # neighbour corpus id (for productivity tracking)
    source_title: str = ""


@dataclass
class Action:
    kind: str                  # "expand" | "search"
    arg: str                   # corpus_id or query
    provenance: str
    cost: int                  # S2 calls this action spends


@dataclass
class AgentState:
    """Read-only snapshot handed to the decision policy each round."""
    seed_id: str
    gaps: List[str]
    pool_size: int
    covered: List[bool]
    uncovered_gap_indices: List[int]
    frontier: List[Dict]       # [{id, title, productive, expanded}]
    searched: set
    calls_used: int
    max_calls: int
    rounds: int


@dataclass
class AgentConfig:
    max_rounds: int = 12
    max_s2_calls: int = 30
    patience: int = 2          # stop after this many no-improvement rounds
    per_action_limit: int = 15  # citations/references/search page size
    match_tau: float = 0.12     # coverage match threshold (TF-IDF cosine)
    dedup_threshold: float = 0.8
    embed_fn: Optional[Callable] = None   # None -> TF-IDF (for dedup + coverage)


# --------------------------------------------------------------------------- #
# Helpers: extract text / keywords / id from a paper dict
# --------------------------------------------------------------------------- #
def _clean(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _tldr_text(paper: dict) -> str:
    t = paper.get("tldr")
    if isinstance(t, dict):
        return t.get("text", "") or ""
    return t or ""


def candidate_texts_from_paper(paper: dict) -> List[str]:
    """One-or-more direction texts harvested from a neighbour.

    Offline neighbours expose `future_directions` (their realized next steps) ->
    one candidate each. Real S2 neighbours have no future-work section, so we use
    the neighbour's contribution (tldr -> abstract -> title) as a proxy direction.
    """
    fds = paper.get("future_directions")
    if fds:
        out = []
        for d in fds:
            txt = _clean(f"{d.get('title','')}. {d.get('rationale','')}")
            if len(txt) > 8:
                out.append(txt)
        if out:
            return out
    for src in (_tldr_text(paper), paper.get("abstract", ""), paper.get("title", "")):
        txt = _clean(src)
        if len(txt) > 8:
            return [txt[:400]]
    return []


def keywords_of(paper: dict) -> List[str]:
    kw = paper.get("fieldsOfStudy") or paper.get("keywords") or paper.get("themes") or []
    return [str(k) for k in kw][:8] if isinstance(kw, (list, tuple)) else []


def corpus_id_of(paper: dict) -> Optional[str]:
    ext = paper.get("externalIds") or {}
    return (ext.get("CorpusId") or ext.get("CorpusID")
            or paper.get("corpusId") or paper.get("paperId"))


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class CandidateHarvestAgent:
    def __init__(self, client, config: Optional[AgentConfig] = None,
                 policy: Optional[Callable[[AgentState], Optional[Action]]] = None):
        self.client = client
        self.cfg = config or AgentConfig()
        self.policy = policy            # swap in an LLM policy here; else heuristic

    # -- tools ------------------------------------------------------------- #
    def _run_action(self, action: Action) -> List[dict]:
        lim = self.cfg.per_action_limit
        if action.kind == "expand":
            cites = self.client.get_citations(action.arg, lim) or []
            refs = self.client.get_references(action.arg, lim) or []
            return cites + refs
        if action.kind == "search":
            return self.client.search(action.arg, lim) or []
        return []

    # -- coverage harness (inference-safe: uses gaps, not the masked GT) ---- #
    def _coverage(self, pool: Dict[str, Candidate], gaps: Sequence[str]) -> Dict:
        if not gaps:
            return {"n_covered": 0, "total": 0, "covered": [], "uncovered": []}
        if not pool:
            return {"n_covered": 0, "total": len(gaps),
                    "covered": [False] * len(gaps), "uncovered": list(range(len(gaps)))}
        ids = list(pool.keys())
        texts = [pool[i].text for i in ids]
        rel = match_relevance(ids, texts, list(gaps), tau=self.cfg.match_tau,
                              embed_fn=self.cfg.embed_fn)
        covered = [(f"g{j}" in rel) for j in range(len(gaps))]
        return {"n_covered": sum(covered), "total": len(gaps), "covered": covered,
                "uncovered": [j for j, c in enumerate(covered) if not c]}

    # -- candidate harvesting + dedup-add ---------------------------------- #
    def _harvest(self, neighbours: List[dict], provenance: str,
                 counter: int) -> List[Candidate]:
        out: List[Candidate] = []
        for nb in neighbours:
            sid = corpus_id_of(nb)
            title = _clean(nb.get("title", ""))
            kws = keywords_of(nb)
            for txt in candidate_texts_from_paper(nb):
                out.append(Candidate(id=f"c{counter + len(out)}", text=txt,
                                     keywords=kws, provenance=provenance,
                                     source_id=sid, source_title=title))
        return out

    def _dedup_add(self, pool: Dict[str, Candidate],
                   new: List[Candidate]) -> List[Candidate]:
        if not new:
            return []
        existing_ids = list(pool.keys())
        new_ids = [c.id for c in new]
        new_by_id = {c.id: c for c in new}
        ids = existing_ids + new_ids
        texts = [pool[i].text for i in existing_ids] + [new_by_id[i].text for i in new_ids]
        kept, _rep, _backend = dedup_candidates(
            ids, texts, threshold=self.cfg.dedup_threshold,
            order=ids, embed_fn=self.cfg.embed_fn)
        kept_set = set(kept)
        added = [new_by_id[i] for i in new_ids if i in kept_set]
        for c in added:
            pool[c.id] = c
        return added

    # -- default autonomous policy ----------------------------------------- #
    def _decide(self, st: AgentState) -> Optional[Action]:
        # Goal-directed: first try to *directly* cover open gaps by searching for
        # their terms; only then explore the citation graph.
        # 1) the seed expansion seeds the frontier (round 1)
        for f in st.frontier:
            if f["id"] == st.seed_id and not f["expanded"]:
                return Action("expand", f["id"], f"expand:{f['id']}", cost=2)
        # 2) targeted search for each uncovered gap not yet searched
        for j in st.uncovered_gap_indices:
            q = st.gaps[j]
            if q not in st.searched:
                return Action("search", q, f"search:gap{j}", cost=1)
        # 3) expand frontier nodes, productive (gap-covering) ones first
        for want_productive in (True, False):
            for f in st.frontier:
                if not f["expanded"] and f["id"] and f["productive"] == want_productive:
                    return Action("expand", f["id"], f"expand:{f['id']}", cost=2)
        return None                       # nothing left -> stop

    def _extend_frontier(self, frontier: List[Dict], neighbours: List[dict],
                         productive_ids: set, seen: set):
        for nb in neighbours:
            cid = corpus_id_of(nb)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            frontier.append({"id": cid, "title": _clean(nb.get("title", "")),
                             "productive": cid in productive_ids, "expanded": False})

    # -- main loop --------------------------------------------------------- #
    def run(self, seed_id: str, gaps: Sequence[str]) -> Dict:
        """Harvest candidates until coverage stops improving. Returns
        {pool, trace, coverage, stop_reason}."""
        gaps = list(gaps)
        pool: Dict[str, Candidate] = {}
        counter = 0
        trace: List[Dict] = []
        frontier = [{"id": seed_id, "title": "seed", "productive": True, "expanded": False}]
        seen_ids = {seed_id}
        searched: set = set()
        calls = 0
        stale = 0
        stop_reason = "max_rounds"

        for rnd in range(1, self.cfg.max_rounds + 1):
            cov = self._coverage(pool, gaps)
            state = AgentState(seed_id, gaps, len(pool), cov["covered"],
                               cov["uncovered"], frontier, searched, calls,
                               self.cfg.max_s2_calls, rnd)
            decide = self.policy or self._decide
            action = decide(state)
            if action is None:
                stop_reason = "frontier_exhausted"
                break
            if calls + action.cost > self.cfg.max_s2_calls:
                stop_reason = "budget_exhausted"
                break

            neighbours = self._run_action(action)
            calls += action.cost
            if action.kind == "expand":
                for f in frontier:
                    if f["id"] == action.arg:
                        f["expanded"] = True
            else:
                searched.add(action.arg)

            new = self._harvest(neighbours, action.provenance, counter)
            counter += len(new)
            added = self._dedup_add(pool, new)
            productive_ids = {c.source_id for c in added if c.source_id}
            self._extend_frontier(frontier, neighbours, productive_ids, seen_ids)

            cov2 = self._coverage(pool, gaps)
            gain = cov2["n_covered"] - cov["n_covered"]
            trace.append({"round": rnd, "action": action.provenance,
                          "neighbours": len(neighbours), "added": len(added),
                          "covered": cov2["n_covered"], "total": cov2["total"],
                          "gain": gain, "calls": calls, "pool": len(pool)})

            # "don't help anymore" = coverage stopped improving (the harness),
            # regardless of whether non-covering candidates were still added.
            stale = stale + 1 if gain <= 0 else 0
            if gaps and cov2["n_covered"] >= len(gaps):
                stop_reason = "all_gaps_covered"
                break
            if stale >= self.cfg.patience:
                stop_reason = f"no_improvement_x{self.cfg.patience}"
                break
        else:
            stop_reason = "max_rounds"

        return {"pool": list(pool.values()), "trace": trace,
                "coverage": self._coverage(pool, gaps), "stop_reason": stop_reason,
                "s2_calls": calls}


# --------------------------------------------------------------------------- #
# Evaluation harness: does the harvested pool recall the MASKED ground truth?
# (eval only — never used by the stopping rule)
# --------------------------------------------------------------------------- #
def evaluate_pool(candidates: Sequence[Candidate], gt_future_work: Sequence[str],
                  tau: float = 0.12, embed_fn: Optional[Callable] = None) -> Dict:
    """Recall + oracle-MRR of the harvested pool against the masked future-work."""
    if not candidates or not gt_future_work:
        return {"recall": 0.0, "oracle_mrr": 0.0, "n_candidates": len(candidates),
                "n_gt": len(gt_future_work)}
    ids = [c.id for c in candidates]
    texts = [c.text for c in candidates]
    rel = match_relevance(ids, texts, list(gt_future_work), tau=tau, embed_fn=embed_fn)
    n_gt = len(gt_future_work)
    covered = sum(1 for j in range(n_gt) if f"g{j}" in rel)
    # oracle MRR: each covered gt direction could sit at rank 1 -> RR 1
    return {"recall": covered / n_gt, "oracle_mrr": covered / n_gt,
            "n_candidates": len(candidates), "n_gt": n_gt, "n_covered_gt": covered}
