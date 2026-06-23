"""Load the real CFE Hackathon data into Query objects for the MRR harness.

Expected folder structure (Data/ is one level above your code folder):
  CFE_hackathon/
    Data/
      papers/                      <- full paper text (for BM25, optional)
        {id}.txt
      Paper_summaries/             <- LLM-generated future directions (Agent A)
        {id}_future_directions.json

Agent A  = future_directions list from the JSON   (explicit LLM proposals)
Agent B  = key_gaps + main_themes from same JSON  (gap-closure perspective)

Ground truth (proxy)
--------------------
We don't have a separate annotated ground truth file, so we use a
self-consistency proxy: each future_direction from Agent A is treated as
a reference sentence, and we match it against the full candidate pool
(Agent A + Agent B directions) via TF-IDF cosine.  A candidate matches
a reference if cosine >= tau (default 0.15).

This measures how well the fused ranking recovers Agent A's own directions
when they're mixed into a larger pool with Agent B's directions — a valid
test of whether fusion helps surface the highest-quality proposals.

S2 graph signals
----------------
If you pass an S2Client, the loader will fetch graph signals for each paper
and attach them to query.features[candidate_id].  Pass s2_client=None to
skip (offline / fast mode).
"""
from __future__ import annotations

import glob
import json
import logging
import os
import random
from typing import Dict, List, Optional, Set, Tuple

from .schema import Dataset, Query
from .s2_client import S2Client, GraphSignalExtractor

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_DATA_DIR = os.path.join(_ROOT, "Data")


# ===========================================================================
# Direction builders (Agent A and Agent B)
# ===========================================================================

def _keywords(text: str, n: int = 6) -> str:
    STOP = {
        "about","across","after","although","among","around","because","before",
        "between","during","findings","further","however","including","limited",
        "literature","number","obtain","often","other","outcomes","paper",
        "research","results","should","studies","study","there","these","their",
        "though","through","toward","under","which","while","within","without",
    }
    words = [w.lower().strip(".,;:()'\"") for w in text.split() if len(w) > 5]
    seen, unique = set(), []
    for w in words:
        if w not in seen and w not in STOP:
            seen.add(w); unique.append(w)
    return " ".join(unique[:n])


def _build_candidates(data: dict, paper_id: str) -> Tuple[
    List[str], List[str], Dict[str, str], Dict[str, List[str]]
]:
    """Build candidate pool from both agents.

    Returns
    -------
    all_ids    : ordered candidate id list
    all_texts  : matching text for each candidate (title + rationale)
    id_to_text : {cand_id -> text}
    sources    : {"agent_a": [...ids], "agent_b": [...ids]}
    """
    a_ids, b_ids   = [], []
    id_to_text: Dict[str, str] = {}

    # Agent A — future_directions list
    for item in data.get("future_directions", []):
        cid  = f"{paper_id}_A{item['id']}"
        text = f"{item.get('title','')}. {item.get('rationale','')}"
        a_ids.append(cid)
        id_to_text[cid] = text

    # Agent B — key_gaps → directions
    for i, gap in enumerate(data.get("key_gaps", []), 1):
        desc = gap["description"]
        title = (f"Close the gap: {desc.split(';')[0].strip()[:90]}")
        cid  = f"{paper_id}_Bgap{i}"
        id_to_text[cid] = f"{title}. {desc}"
        b_ids.append(cid)

    # Agent B — main_themes → directions
    for i, theme in enumerate(data.get("main_themes", []), 1):
        cid  = f"{paper_id}_Btheme{i}"
        id_to_text[cid] = f"Investigate and advance: {theme}. {theme}"
        b_ids.append(cid)

    # deduplicate preserving order
    seen, all_ids = set(), []
    for cid in a_ids + b_ids:
        if cid not in seen:
            seen.add(cid); all_ids.append(cid)

    all_texts = [id_to_text[cid] for cid in all_ids]
    return all_ids, all_texts, id_to_text, {"agent_a": a_ids, "agent_b": b_ids}


# ===========================================================================
# Ground truth builder (TF-IDF proxy)
# ===========================================================================

def _build_relevance(
    ref_texts:  List[str],
    ref_ids:    List[str],
    cand_ids:   List[str],
    cand_texts: List[str],
    tau:        float = 0.15,
) -> Dict[str, Set[str]]:
    """Match reference directions to candidates via TF-IDF cosine.

    ref_texts  : Agent A's future_directions as reference sentences
    cand_texts : full candidate pool texts (Agent A + Agent B)
    tau        : cosine threshold for a match
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("scikit-learn not installed — ground truth unavailable. "
                       "Run: pip install scikit-learn")
        return {}

    if not ref_texts or not cand_texts:
        return {}

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(ref_texts + cand_texts)
    rmat = mat[:len(ref_texts)]
    cmat = mat[len(ref_texts):]
    sims = cosine_similarity(rmat, cmat)   # [n_ref, n_cand]

    relevance: Dict[str, Set[str]] = {}
    for ri, ref_id in enumerate(ref_ids):
        matches = {cand_ids[ci] for ci in range(len(cand_ids))
                   if sims[ri, ci] >= tau}
        if matches:
            relevance[ref_id] = matches
    return relevance


# ===========================================================================
# S2 graph signals
# ===========================================================================

def _build_s2_features(
    paper_id:  str,
    cand_ids:  List[str],
    id_to_text: Dict[str, str],
    extractor: GraphSignalExtractor,
) -> Dict[str, Dict[str, float]]:
    """Fetch S2 signals for each candidate and return features dict."""
    ctx = extractor.build_context(paper_id)
    if ctx is None:
        return {}

    features: Dict[str, Dict[str, float]] = {}
    logger.info("  [S2] computing signals for %d candidates ...", len(cand_ids))
    for cid in cand_ids:
        text = id_to_text.get(cid, "")
        features[cid] = extractor.signals_for_direction(text, ctx)
    return features


# ===========================================================================
# Main loader
# ===========================================================================

def load_dataset(
    data_dir:    str        = DEFAULT_DATA_DIR,
    tau:         float      = 0.15,
    s2_client:   Optional[S2Client] = None,
    dev_frac:    float      = 0.3,
    test_frac:   float      = 0.2,
    seed:        int        = 42,
    paper_ids:   Optional[List[str]] = None,
) -> Tuple[Dataset, Dict[str, dict]]:
    """Load all papers from Data/Paper_summaries into a Dataset.

    Parameters
    ----------
    data_dir   : path to the Data/ folder
    tau        : TF-IDF cosine threshold for ground-truth matching
    s2_client  : if provided, fetches S2 graph signals (graph_relevance etc.)
                 and adds them to query.features
    dev_frac   : fraction of papers for dev split (for weight optimisation)
    test_frac  : fraction for test (for final evaluation)
    paper_ids  : run on specific IDs only (None = all papers)

    Returns
    -------
    (Dataset, diagnostics_dict)
    """
    summ_dir = os.path.join(data_dir, "Paper_summaries")
    extractor = GraphSignalExtractor(s2_client) if s2_client else None

    # discover papers
    pattern   = os.path.join(summ_dir, "*_future_directions.json")
    all_files = sorted(glob.glob(pattern))
    if paper_ids:
        all_files = [f for f in all_files
                     if os.path.basename(f).replace("_future_directions.json","")
                     in set(paper_ids)]

    queries: List[Query] = []
    diag:    Dict[str, dict] = {}

    for fp in all_files:
        pid = os.path.basename(fp).replace("_future_directions.json", "")
        logger.info("Loading paper %s ...", pid)

        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        fds = data.get("future_directions", [])
        if not fds:
            logger.warning("  no future_directions — skipping")
            continue

        # build candidate pool
        all_ids, all_texts, id_to_text, sources = _build_candidates(data, pid)

        # ground truth: match Agent A's directions against the full pool
        ref_ids   = sources["agent_a"]
        ref_texts = [id_to_text[r] for r in ref_ids]
        relevance = _build_relevance(ref_texts, ref_ids, all_ids, all_texts, tau)

        # S2 graph signals
        features: Dict[str, Dict[str, float]] = {}
        if extractor:
            features = _build_s2_features(pid, all_ids, id_to_text, extractor)

        q = Query(
            query_id          = pid,
            candidate_ids     = all_ids,
            rankings          = {"agent_a": sources["agent_a"],
                                 "agent_b": sources["agent_b"]},
            relevance         = relevance,
            candidate_sources = sources,
            features          = features,
        )
        queries.append(q)
        diag[pid] = {
            "n_agent_a":     len(sources["agent_a"]),
            "n_agent_b":     len(sources["agent_b"]),
            "n_pool":        len(all_ids),
            "n_gt_matched":  len(relevance),
            "has_s2":        bool(features),
        }
        logger.info("  pool=%d  gt_matched=%d  s2=%s",
                    len(all_ids), len(relevance), bool(features))

    if not queries:
        return Dataset([]), diag

    # train/dev/test split
    rng = random.Random(seed)
    ids = [q.query_id for q in queries]
    rng.shuffle(ids)
    n         = len(ids)
    n_test    = max(1, int(n * test_frac))
    n_dev     = max(1, int(n * dev_frac))
    test_ids  = ids[:n_test]
    dev_ids   = ids[n_test:n_test + n_dev]
    train_ids = ids[n_test + n_dev:]

    logger.info("Dataset: %d papers  train=%d dev=%d test=%d",
                n, len(train_ids), len(dev_ids), len(test_ids))

    return Dataset(queries, train_ids, dev_ids, test_ids), diag
