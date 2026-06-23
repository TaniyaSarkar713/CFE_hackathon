"""The §1 match function: does a recommended direction cover a true one?

Recommendations and ground truth are free text, so MRR needs a match
predicate `match(d_i, g_j) ∈ {0,1}` (plan §1):

    match(d_i, g_j) = 1  if  sim(d_i, g_j) >= tau            # embedding cosine
                      or  entailment_judge(d_i, g_j) == "covers"

This module builds the `relevance` dict for a `Query` from candidate texts and
the masked ground-truth statements, with **pluggable** similarity and an
optional entailment judge:

  - `embed_fn`  : texts -> matrix of row vectors. Default = TF-IDF (always
    available). Swap in SPECTER2 / SciNCL / SciBERT / all-mpnet (the D4
    dependency) without touching the rest of the pipeline.
  - `entailment_fn(cand_text, gt_text) -> bool` : the optional LLM judge used
    ONLY for eval matching (the D5 dependency); it never ranks.

`tau` should be calibrated on a small human-labelled set (`calibrate_tau`).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def _tfidf_embed(texts: Sequence[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)


def _cosine(A, B) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(A, B)


def similarity_matrix(cand_texts: Sequence[str], gt_texts: Sequence[str],
                      embed_fn: Optional[Callable] = None) -> np.ndarray:
    """Return the [n_gt, n_cand] cosine similarity matrix."""
    if embed_fn is None:                              # shared TF-IDF vocabulary
        M = _tfidf_embed(list(cand_texts) + list(gt_texts))
        C, G = M[: len(cand_texts)], M[len(cand_texts):]
    else:
        C = np.asarray(embed_fn(list(cand_texts)), dtype=float)
        G = np.asarray(embed_fn(list(gt_texts)), dtype=float)
    return _cosine(G, C)


def match_relevance(
    candidate_ids: Sequence[str],
    candidate_texts: Sequence[str],
    gt_texts: Sequence[str],
    tau: float = 0.12,
    embed_fn: Optional[Callable] = None,
    entailment_fn: Optional[Callable[[str, str], bool]] = None,
) -> Dict[str, Set[str]]:
    """Build the `relevance` dict {g_j -> set of candidate ids that match it}.

    A candidate matches a true direction if cosine >= tau OR the entailment
    judge says it "covers" that direction.
    """
    sims = similarity_matrix(candidate_texts, gt_texts, embed_fn)
    relevance: Dict[str, Set[str]] = {}
    for gi in range(sims.shape[0]):
        matched: Set[str] = set()
        for ci, cid in enumerate(candidate_ids):
            if sims[gi, ci] >= tau:
                matched.add(cid)
            elif entailment_fn is not None and entailment_fn(
                    candidate_texts[ci], gt_texts[gi]):
                matched.add(cid)
        if matched:
            relevance[f"g{gi}"] = matched
    return relevance


def calibrate_tau(
    pairs: Sequence[Tuple[str, str]],
    labels: Sequence[int],
    embed_fn: Optional[Callable] = None,
    grid: Sequence[float] = tuple(round(0.02 * i, 2) for i in range(1, 26)),
) -> Dict:
    """Pick tau maximising F1 against a human-labelled set of (cand, gt) pairs.

    `labels[i] = 1` if the human judged pair i a true match. Returns the best
    tau plus its precision/recall/F1 — the §1 calibration step, and a guard
    against the score being an artefact of one threshold (sweep it, report it).
    """
    cand = [c for c, _ in pairs]
    gt = [g for _, g in pairs]
    # per-pair cosine (diagonal of the cand×gt similarity for aligned pairs)
    if embed_fn is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        M = vec.fit_transform(cand + gt)
        C, G = M[: len(cand)], M[len(cand):]
        sims = np.array([cosine_similarity(C[i], G[i])[0, 0] for i in range(len(cand))])
    else:
        C = np.asarray(embed_fn(cand), float)
        G = np.asarray(embed_fn(gt), float)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
        sims = np.sum(Cn * Gn, axis=1)

    y = np.asarray(labels, dtype=int)
    best = {"tau": grid[0], "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for tau in grid:
        pred = (sims >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best["f1"]:
            best = {"tau": float(tau), "f1": f1, "precision": prec, "recall": rec}
    return best
