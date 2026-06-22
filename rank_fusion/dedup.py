"""Candidate deduplication — the `dedup(C)` step in the candidate pool (plan §3).

Concrete recipe from Si et al. (2024): embed every candidate with
**all-MiniLM-L6-v2** (Sentence-Transformers) and drop any candidate whose cosine
similarity to an already-kept candidate is **>= 0.8**. This collapses the
generative + retrieval + citation pools' near-duplicate directions into one
representative, which (a) keeps the candidate pool tight and (b) prevents
near-duplicates from crowding the ranked top-K.

Embeddings are pluggable: pass `embed_fn`, or rely on the default which uses
Sentence-Transformers if installed and **falls back to TF-IDF cosine** (always
available via scikit-learn) so the step works before the embedding model lands.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def embed_minilm(texts: Sequence[str], model_name: str = "all-MiniLM-L6-v2"):
    """Embed with Sentence-Transformers all-MiniLM-L6-v2 (L2-normalised)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(list(texts), normalize_embeddings=True))


def _tfidf_cosine(texts: Sequence[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
    return cosine_similarity(X)


def _similarity_matrix(texts, embed_fn) -> Tuple[np.ndarray, str]:
    """Return (cosine matrix, backend name). MiniLM if available, else TF-IDF."""
    if embed_fn is not None:
        E = np.asarray(embed_fn(texts), dtype=float)
        norm = np.linalg.norm(E, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        U = E / norm
        return U @ U.T, "custom"
    try:
        E = embed_minilm(texts)
        return E @ E.T, "all-MiniLM-L6-v2"
    except Exception:
        return _tfidf_cosine(texts), "tfidf-fallback"


def dedup_candidates(
    ids: Sequence[str],
    texts: Sequence[str],
    threshold: float = 0.8,
    order: Optional[Sequence[str]] = None,
    embed_fn: Optional[Callable] = None,
) -> Tuple[List[str], Dict[str, str], str]:
    """Greedy dedup. Returns (kept ids, duplicate->representative map, backend).

    Candidates are processed in `order` (default: the given id order = priority,
    so the highest-priority member of a duplicate group is the representative).
    """
    ids = list(ids)
    text_of = dict(zip(ids, texts))
    seq = list(order) if order is not None else ids
    sims, backend = _similarity_matrix([text_of[i] for i in seq], embed_fn)
    pos = {d: i for i, d in enumerate(seq)}

    kept: List[str] = []
    rep: Dict[str, str] = {}
    for d in seq:
        di = pos[d]
        dup_of = None
        for kd in kept:
            if sims[di, pos[kd]] >= threshold:
                dup_of = kd
                break
        if dup_of is None:
            kept.append(d)
        else:
            rep[d] = dup_of
    return kept, rep, backend
