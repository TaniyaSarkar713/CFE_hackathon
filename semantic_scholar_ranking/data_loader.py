"""Load the real CFE Hackathon data into Query objects for the MRR harness.

Expected folder structure (Data/ is one level above your code folder):
  CFE_hackathon/
    Data/
      Ground Truth Future Directions/  <- real GT text, one file per paper
        {id}.txt
      Paper_summaries/                 <- LLM-generated future directions
        {id}_future_directions.json

Agent A  = future_directions list from the JSON   (explicit LLM proposals)
Agent B  = key_gaps + main_themes from same JSON  (gap-closure perspective)

Ground truth
------------
We use the real GT text from `Data/Ground Truth Future Directions/{id}.txt`.
The text is split into sentences; each sentence is a true direction g_j.
A candidate matches g_j if TF-IDF cosine >= tau (default 0.15).
Papers without a GT file are skipped (no ground truth → no MRR signal).

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
import re
import random
from typing import Dict, List, Optional, Set, Tuple

from .schema import Dataset, Query
from .s2_client import S2Client, GraphSignalExtractor
from .s2_agent import S2DirectionAgent

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

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
# Ground truth builder
# ===========================================================================

def _split_gt_sentences(text: str) -> List[str]:
    """Split a GT paragraph into individual sentences (min 15 chars)."""
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 15]
    return parts or [text]


def _build_relevance(
    ref_texts:  List[str],
    ref_ids:    List[str],
    cand_ids:   List[str],
    cand_texts: List[str],
    tau:        float = 0.15,
) -> Dict[str, Set[str]]:
    """Match GT sentences to candidates via TF-IDF cosine.

    ref_texts  : ground-truth sentences from the real GT file
    ref_ids    : stable ids for each GT sentence (e.g. "{pid}_gt1")
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

def _read_paper_title(data_dir: str, paper_id: str) -> Optional[str]:
    """Read the first line of Data/papers/{id}.txt as the paper title."""
    path = os.path.join(data_dir, "papers", f"{paper_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        line = f.readline().strip()
    return line or None


def _build_s2_features(
    paper_id:   str,
    cand_ids:   List[str],
    id_to_text: Dict[str, str],
    extractor:  GraphSignalExtractor,
    title:      Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Fetch S2 signals for each candidate and return features dict."""
    ctx = extractor.build_context(paper_id, title=title)
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

    agent_c = S2DirectionAgent(extractor) if extractor else None

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

        # ground truth: use real GT file sentences
        gt_path = os.path.join(data_dir, "Ground Truth Future Directions", f"{pid}.txt")
        if not os.path.exists(gt_path):
            logger.warning("  no GT file for %s — skipping (no ground truth)", pid)
            continue
        with open(gt_path, encoding="utf-8") as gf:
            gt_sents = _split_gt_sentences(gf.read())
        if not gt_sents:
            logger.warning("  GT file for %s is empty — skipping", pid)
            continue
        ref_ids   = [f"{pid}_gt{i}" for i, _ in enumerate(gt_sents, 1)]
        ref_texts = gt_sents
        relevance = _build_relevance(ref_texts, ref_ids, all_ids, all_texts, tau)

        # S2 graph signals + Agent C candidates
        features: Dict[str, Dict[str, float]] = {}
        c_ids: List[str] = []
        ctx = None
        if extractor:
            paper_title = _read_paper_title(data_dir, pid)
            ctx      = extractor.build_context(pid, title=paper_title)
            features = _build_s2_features(pid, all_ids, id_to_text, extractor,
                                          title=paper_title)

        if agent_c and ctx is not None:
            c_ids, c_texts, c_ranked, c_feats = agent_c.run(pid, ctx, title=paper_title)
            # Merge into pool (Agent C candidates may not be in Agent A/B lists)
            for cid, text in zip(c_ids, c_texts):
                if cid not in id_to_text:
                    all_ids.append(cid)
                    all_texts.append(text)
                    id_to_text[cid] = text
            features.update(c_feats)
            sources["agent_c"] = c_ranked
            # Re-run relevance to include agent_c candidates
            relevance = _build_relevance(ref_texts, ref_ids, all_ids, all_texts, tau)

        rankings = {"agent_a": sources["agent_a"], "agent_b": sources["agent_b"]}
        if "agent_c" in sources:
            rankings["agent_c"] = sources["agent_c"]

        q = Query(
            query_id          = pid,
            candidate_ids     = all_ids,
            rankings          = rankings,
            relevance         = relevance,
            candidate_sources = sources,
            features          = features,
        )
        queries.append(q)
        diag[pid] = {
            "n_agent_a":    len(sources["agent_a"]),
            "n_agent_b":    len(sources["agent_b"]),
            "n_agent_c":    len(sources.get("agent_c", [])),
            "n_pool":       len(all_ids),
            "n_gt_sents":   len(gt_sents),
            "n_gt_matched": len(relevance),
            "has_s2":       bool(features),
        }
        logger.info("  pool=%d  A=%d B=%d C=%d  gt_sents=%d  gt_matched=%d  s2=%s",
                    len(all_ids), len(sources["agent_a"]), len(sources["agent_b"]),
                    len(sources.get("agent_c", [])),
                    len(gt_sents), len(relevance), bool(features))

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
