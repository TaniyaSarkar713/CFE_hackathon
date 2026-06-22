"""Adapter that loads the real `Data/` folder into the harness format.

Each `Paper_summaries/<id>_future_directions.json` holds one model's ordered
`future_directions` (the candidate list). The matching `Ground Truth Future
Directions/<id>.txt` is the masked future-work text. Since we have no scientific
embedding model installed, we approximate the §1 match function with a
**TF-IDF cosine** between each candidate (title + rationale) and the ground-truth
sentences; a candidate matches a true direction if cosine >= `tau`.

This proves the MRR harness runs on the real text. Fusion still needs a second
independent ranker (the pipeline's Claude ranker), which `Data/` does not yet
contain — so the full fusion demo lives in `run_synthetic.py`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from .schema import Dataset, Query

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 15]
    return parts or [text]


def load_real_dataset(
    data_dir: str,
    tau: float = 0.12,
    min_chars: int = 10,
) -> Tuple[Dataset, Dict[str, dict]]:
    """Load papers that have BOTH a summary and a ground-truth file.

    Returns (dataset, diagnostics). Each Query has a single source "summary"
    whose ranking is the directions in their given id order; `relevance` maps
    each ground-truth sentence to the candidate ids that match it at >= tau.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    summ_dir = os.path.join(data_dir, "Paper_summaries")
    gt_dir = os.path.join(data_dir, "Ground Truth Future Directions")

    queries: List[Query] = []
    diag: Dict[str, dict] = {}

    for fn in sorted(os.listdir(summ_dir)):
        if not fn.endswith("_future_directions.json"):
            continue
        pid = fn.replace("_future_directions.json", "")
        gt_path = os.path.join(gt_dir, f"{pid}.txt")
        if not os.path.exists(gt_path):
            continue

        with open(os.path.join(summ_dir, fn), encoding="utf-8") as f:
            summary = json.load(f)
        dirs = summary.get("future_directions", [])
        if not dirs:
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt_text = f.read()
        if len(gt_text) < min_chars:
            continue

        cand_texts: List[str] = []
        cand_ids: List[str] = []
        for d in dirs:
            cid = f"{pid}_d{d.get('id')}"
            cand_ids.append(cid)
            cand_texts.append(f"{d.get('title','')}. {d.get('rationale','')}")

        gt_sents = _split_sentences(gt_text)
        if not gt_sents:
            continue

        # TF-IDF over candidate texts + GT sentences in one shared vocabulary
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform(cand_texts + gt_sents)
        cmat = mat[: len(cand_texts)]
        gmat = mat[len(cand_texts):]
        sims = cosine_similarity(gmat, cmat)  # [n_gt, n_cand]

        relevance: Dict[str, Set[str]] = {}
        for gi in range(sims.shape[0]):
            matches = {cand_ids[ci] for ci in range(len(cand_ids))
                       if sims[gi, ci] >= tau}
            if matches:
                relevance[f"g{gi}"] = matches

        rankings = {"summary": list(cand_ids)}  # given id order = the model's ranking
        queries.append(Query(pid, cand_ids, rankings, relevance))
        diag[pid] = {
            "n_candidates": len(cand_ids),
            "n_gt_sentences": len(gt_sents),
            "n_matched_gt": len(relevance),
            "max_sim": float(sims.max()) if sims.size else 0.0,
        }

    # single split (these are eval-only); put everything in dev/test
    ids = [q.query_id for q in queries]
    ds = Dataset(queries, train_ids=ids, dev_ids=ids, test_ids=ids)
    return ds, diag
