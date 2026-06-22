"""
Future Direction Ranker — RRF-inspired scoring
===============================================
Ranks future directions by relevance, drawing on the design patterns from
the project's meta_ranker.py (feature engineering + linear scoring) and
learned_rankers.py (pairwise comparisons with hard-negative awareness).

Architecture mirrors the uploaded codebase:
  - Feature engineering per candidate (like _query_features in meta_ranker.py)
  - Standardisation (like _FeatureSpace in learned_rankers.py)
  - RRF-style smoothed scoring (like weighted_rrf in optimize.py)
  - Weighted linear combination (like PairwiseRankNet.w @ features)

The "sources" here are the ground-truth entities and predicted entities,
analogous to how the original code fuses multiple retrieval source rankings.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Feature engineering — per future-direction feature vector
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> set[str]:
    """Simple word-level tokenization for entity matching."""
    return set(re.findall(r"[a-z][a-z0-9\-]+", text.lower()))


def _entity_present(entity: str, text_lower: str, tokens: set[str]) -> bool:
    """Check if an entity appears in text (substring or token match)."""
    entity_lower = entity.lower().strip()
    # Substring match for multi-word entities
    if entity_lower in text_lower:
        return True
    # Token overlap for single-word entities
    entity_tokens = set(entity_lower.split())
    if entity_tokens and entity_tokens.issubset(tokens):
        return True
    return False


def _build_features(
    fd: dict,
    gt_entities: list[str],
    pred_entities: list[dict],
    n_total_fds: int,
    k: float = 60.0,
) -> dict:
    """Build a feature dict for one future direction.

    Mirrors _query_features() from meta_ranker.py:
    - per-source rank/presence/reciprocal-rank → per-entity overlap/weight
    - aggregate features: rrf_score, best_rank, mean_rank, num_sources, etc.
    """
    text = f"{fd.get('title', '')} {fd.get('rationale', '')}".lower()
    tokens = _tokenize(text)

    # --- Ground-truth entity features (analogous to source rankings) ---
    gt_hits = []
    for ent in gt_entities:
        if _entity_present(ent, text, tokens):
            gt_hits.append(ent)

    gt_overlap = len(gt_hits)
    gt_coverage = gt_overlap / max(len(gt_entities), 1)

    # --- Predicted entity features (weighted, like RRF scores) ---
    pred_rrf = 0.0
    pred_hits = 0
    pred_weight_sum = 0.0
    for ent in pred_entities:
        ent_text = ent["text"] if isinstance(ent, dict) else str(ent)
        ent_weight = ent.get("weight", 1.0) if isinstance(ent, dict) else 1.0
        if _entity_present(ent_text, text, tokens):
            pred_hits += 1
            pred_weight_sum += ent_weight
            # RRF-style: weight / (k + rank), treating weight as inverse rank
            pred_rrf += ent_weight / (k + 1.0)

    pred_coverage = pred_hits / max(len(pred_entities), 1)

    # --- Text quality features ---
    rationale = fd.get("rationale", "")
    rationale_len = len(rationale)
    rationale_sentences = len(re.findall(r"[.!?]+", rationale))
    specificity = len(set(re.findall(r"[a-z]{5,}", rationale.lower())))  # unique long words

    # --- Position feature (like rank in source) ---
    fd_id = fd.get("id", n_total_fds)
    position_score = 1.0 / fd_id  # earlier = higher (positional bias)

    # --- Agreement feature (both sources agree — like disagreement in meta_ranker) ---
    agreement = 1.0 if (gt_overlap > 0 and pred_hits > 0) else 0.0

    return {
        "gt_overlap": gt_overlap,
        "gt_coverage": gt_coverage,
        "pred_hits": pred_hits,
        "pred_coverage": pred_coverage,
        "pred_rrf": pred_rrf,
        "pred_weight_sum": pred_weight_sum,
        "rationale_len": rationale_len,
        "rationale_sentences": rationale_sentences,
        "specificity": specificity,
        "position_score": position_score,
        "agreement": agreement,
        "gt_hit_entities": gt_hits,
    }


# --------------------------------------------------------------------------- #
# Scoring — weighted linear model (like PairwiseRankNet with fixed weights)
# --------------------------------------------------------------------------- #
# These weights mirror the structure of PairwiseRankNet.w from learned_rankers.py.
# In a production system with labelled data, you'd learn them via pairwise BPR
# or the SoftMRR surrogate. Here we use domain-informed fixed weights.
DEFAULT_WEIGHTS = {
    "gt_coverage":        0.30,   # how much of the paper's entities are addressed
    "pred_rrf":           0.25,   # RRF-smoothed predicted entity relevance
    "pred_coverage":      0.10,   # fraction of predicted entities present
    "agreement":          0.10,   # both GT and predicted entities present
    "specificity":        0.10,   # lexical richness of the rationale
    "rationale_sentences": 0.05,  # depth of argumentation
    "position_score":     0.05,   # mild positional prior
    "rationale_len":      0.05,   # length signal (normalised)
}


def _score_fd(features: dict, weights: dict = None) -> float:
    """Compute a scalar relevance score from features.

    Analogous to: scores = Xs @ self.w  in PairwiseRankNet.rank_fn()
    """
    w = weights or DEFAULT_WEIGHTS
    score = 0.0
    for feat_name, weight in w.items():
        val = features.get(feat_name, 0.0)
        score += weight * val
    return score


def _standardise(values: list[float]) -> list[float]:
    """Z-score standardisation, matching _FeatureSpace in learned_rankers.py."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return []
    mu = arr.mean()
    sigma = arr.std()
    if sigma < 1e-12:
        return [0.0] * len(arr)
    return ((arr - mu) / sigma).tolist()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def rank_future_directions(
    future_directions: list[dict],
    ground_truth_entities: list[str],
    predicted_entities: list[dict | str],
    weights: dict = None,
    k: float = 60.0,
) -> list[dict]:
    """Rank future directions by entity-based relevance scoring.

    Args:
        future_directions: list of {"id", "title", "rationale"} dicts
        ground_truth_entities: entities from the paper (the "labels")
        predicted_entities: entities from FDs, each {"text", "weight", "source"}
        weights: optional custom scoring weights
        k: RRF smoothing constant (default 60, matching optimize.py)

    Returns:
        Ranked list of FDs with score metadata appended.
    """
    if not future_directions:
        return []

    # Normalise predicted entities to dicts
    pred_norm = []
    for ent in (predicted_entities or []):
        if isinstance(ent, str):
            pred_norm.append({"text": ent, "weight": 1.0, "source": "unknown"})
        else:
            pred_norm.append(ent)

    n = len(future_directions)

    # --- Build features for each FD ---
    all_features = []
    for fd in future_directions:
        feats = _build_features(fd, ground_truth_entities or [], pred_norm, n, k)
        all_features.append(feats)

    # --- Standardise continuous features (like _FeatureSpace.fit/transform) ---
    continuous = ["gt_coverage", "pred_rrf", "pred_coverage", "specificity",
                  "rationale_len", "rationale_sentences"]
    for feat_name in continuous:
        raw_vals = [f[feat_name] for f in all_features]
        std_vals = _standardise(raw_vals)
        # Use sigmoid to map to [0, 1] after standardisation
        for i, sv in enumerate(std_vals):
            all_features[i][feat_name] = 1.0 / (1.0 + math.exp(-sv))

    # --- Score each FD ---
    scored = []
    for fd, feats in zip(future_directions, all_features):
        raw_score = _score_fd(feats, weights)
        scored.append({
            **fd,
            "score": round(raw_score, 4),
            "entity_overlap": feats["gt_overlap"],
            "entity_coverage_pct": round(feats["gt_coverage"] * 100, 1),
            "rrf_contribution_pct": round(feats["pred_rrf"] * 100, 1),
            "matched_entities": feats["gt_hit_entities"],
            "agreement": feats["agreement"] > 0,
        })

    # --- Sort descending by score (like np.argsort(-scores)) ---
    scored.sort(key=lambda x: -x["score"])

    # Assign ranks
    for i, fd in enumerate(scored):
        fd["rank"] = i + 1

    return scored


# --------------------------------------------------------------------------- #
# CLI test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sample_fds = [
        {"id": 1, "title": "Standardize national prevention framework",
         "rationale": "Policy heterogeneity was a core finding. A national framework would provide consistent guidance on staff roles and coordination."},
        {"id": 2, "title": "Create implementation protocol",
         "rationale": "Only a minority of policies contained explicit implementation plans. A protocol specifying governance and timelines is needed."},
    ]
    gt = ["policy heterogeneity", "implementation", "governance", "staff training"]
    pred = [
        {"text": "national framework", "weight": 1.5, "source": "future_direction"},
        {"text": "protocol", "weight": 1.2, "source": "future_direction"},
    ]

    results = rank_future_directions(sample_fds, gt, pred)
    for r in results:
        print(f"  #{r['rank']}  score={r['score']:.4f}  {r['title']}")
        print(f"         entities matched: {r['matched_entities']}")
