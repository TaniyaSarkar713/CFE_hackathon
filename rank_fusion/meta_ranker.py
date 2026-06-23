"""Learned meta-ranker (plan §5, stage 2).

RRF is hand-designed fusion. The meta-ranker instead *learns* when to trust
each source: one row per (query, candidate), pointwise label = "candidate
matches a true direction". We score every candidate and re-rank by predicted
relevance. Per the plan, the loss is only a differentiable proxy — the model /
hyper-parameters are **selected by dev MRR**, the real objective.

Default model is logistic regression (sklearn, always available, "simplest" in
the plan). If LightGBM is installed, `model="lgbm"` uses an LGBMRanker grouped
by paper instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fusion import weighted_rrf
from .metrics import corpus_mrr
from .schema import Query

_BIG_RANK = 10_000.0  # rank assigned to a candidate a source did not rank


# --------------------------------------------------------------------------- #
# Feature engineering: rank-fusion + agreement features per candidate
# --------------------------------------------------------------------------- #
def feature_names(sources: Sequence[str],
                  extra_feature_keys: Optional[Sequence[str]] = None) -> List[str]:
    names: List[str] = []
    for s in sources:
        names += [f"rank_{s}", f"present_{s}", f"rr_{s}"]
    names += [
        "rrf_score", "best_rank", "mean_rank", "rank_std",
        "num_sources", "frac_sources", "disagreement",
    ]
    if extra_feature_keys:
        names += list(extra_feature_keys)        # Query.features columns (e.g. graph signals)
    return names


def _query_features(query: Query, sources: Sequence[str], k: float = 60.0,
                    extra_feature_keys: Optional[Sequence[str]] = None
                    ) -> Tuple[List[str], np.ndarray]:
    """Return (candidate_ids, feature matrix) for one query.

    `extra_feature_keys` pulls per-candidate values from `Query.features` (e.g.
    Semantic Scholar graph signals: recency / popularity / novelty / relevance),
    appended after the rank-fusion features so the learned ranker can use them.
    """
    pos: Dict[str, Dict[str, int]] = {}
    for s in sources:
        pos[s] = {d: i for i, d in enumerate(query.rankings.get(s, []), start=1)}
    _, rrf_scores = weighted_rrf(query.rankings, k=k,
                                 candidate_ids=query.candidate_ids)

    rows: List[List[float]] = []
    cand_ids = list(query.candidate_ids)
    n_src = len(sources)
    for d in cand_ids:
        feats: List[float] = []
        ranks_present: List[float] = []
        for s in sources:
            r = pos[s].get(d)
            if r is None:
                feats += [_BIG_RANK, 0.0, 0.0]
            else:
                feats += [float(r), 1.0, 1.0 / (k + r)]
                ranks_present.append(float(r))
        num = len(ranks_present)
        best_rank = min(ranks_present) if ranks_present else _BIG_RANK
        mean_rank = float(np.mean(ranks_present)) if ranks_present else _BIG_RANK
        rank_std = float(np.std(ranks_present)) if len(ranks_present) > 1 else 0.0
        feats += [
            rrf_scores.get(d, 0.0), best_rank, mean_rank, rank_std,
            float(num), num / n_src if n_src else 0.0, rank_std,
        ]
        if extra_feature_keys:
            fd = query.features.get(d, {})
            feats += [float(fd.get(key, 0.0)) for key in extra_feature_keys]
        rows.append(feats)
    return cand_ids, np.asarray(rows, dtype=float)


def _labels(query: Query, cand_ids: Sequence[str]) -> np.ndarray:
    rel = query.relevant_ids
    return np.asarray([1.0 if d in rel else 0.0 for d in cand_ids], dtype=float)


# --------------------------------------------------------------------------- #
# The meta-ranker
# --------------------------------------------------------------------------- #
class MetaRanker:
    def __init__(self, sources: Sequence[str], model: str = "logistic",
                 k: float = 60.0, seed: int = 0,
                 extra_feature_keys: Optional[Sequence[str]] = None):
        self.sources = list(sources)
        self.model_kind = model
        self.k = k
        self.seed = seed
        self.model = None
        self.extra_feature_keys = list(extra_feature_keys) if extra_feature_keys else None
        self.feature_names = feature_names(self.sources, self.extra_feature_keys)

    # -- assemble a design matrix over many queries ------------------------ #
    def _build_xy(self, queries: Sequence[Query]):
        X_parts, y_parts, groups = [], [], []
        for q in queries:
            if not q.has_ground_truth():
                continue
            cand_ids, X = _query_features(q, self.sources, self.k, self.extra_feature_keys)
            if X.shape[0] == 0:
                continue
            X_parts.append(X)
            y_parts.append(_labels(q, cand_ids))
            groups.append(X.shape[0])
        if not X_parts:
            raise ValueError("No labelled queries to train the meta-ranker on.")
        return np.vstack(X_parts), np.concatenate(y_parts), groups

    def fit(self, train_queries: Sequence[Query], C: float = 1.0):
        X, y, groups = self._build_xy(train_queries)
        if self.model_kind == "lgbm":
            self._fit_lgbm(X, y, groups)
        else:
            self._fit_logistic(X, y, C)
        return self

    def _fit_logistic(self, X, y, C):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Drop zero-variance columns (e.g. present_* / num_sources when every
        # source ranks the full pool): they carry no signal and the all-zero
        # scaled column triggers spurious BLAS warnings on some numpy builds.
        self.keep_cols = np.std(X, axis=0) > 1e-12
        if not self.keep_cols.any():
            self.keep_cols[:] = True
        Xk = X[:, self.keep_cols]
        self.scaler = StandardScaler().fit(Xk)
        Xs = self.scaler.transform(Xk)
        self.model = LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000, random_state=self.seed
        ).fit(Xs, y)

    def _fit_lgbm(self, X, y, groups):
        import lightgbm as lgb

        self.scaler = None
        self.model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=200, learning_rate=0.05,
            num_leaves=15, random_state=self.seed, min_child_samples=5,
        )
        self.model.fit(X, y.astype(int), group=groups)

    # -- scoring / ranking ------------------------------------------------- #
    def _score_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.model_kind == "lgbm":
            return self.model.predict(X)
        Xs = self.scaler.transform(X[:, self.keep_cols])
        return self.model.predict_proba(Xs)[:, 1]

    def rank_fn(self):
        def fn(query: Query) -> List[str]:
            cand_ids, X = _query_features(query, self.sources, self.k,
                                          self.extra_feature_keys)
            if X.shape[0] == 0:
                return list(query.candidate_ids)
            scores = self._score_matrix(X)
            order = np.argsort(-scores, kind="stable")
            return [cand_ids[i] for i in order]
        return fn


def fit_select_meta_ranker(
    train_queries: Sequence[Query],
    dev_queries: Sequence[Query],
    sources: Sequence[str],
    model: str = "logistic",
    C_grid: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    k: float = 60.0,
    per_direction: bool = True,
    extra_feature_keys: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Tuple[MetaRanker, float, Dict]:
    """Train on `train`, **select hyper-parameters by dev MRR**, return best.

    `extra_feature_keys` pulls those columns from `Query.features` (e.g.
    consensus features). Returns (best_ranker, best_dev_mrr, info).
    """
    best_ranker: Optional[MetaRanker] = None
    best_mrr = -1.0
    best_info: Dict = {}

    if model == "lgbm":
        mr = MetaRanker(sources, model="lgbm", k=k, seed=seed,
                        extra_feature_keys=extra_feature_keys).fit(train_queries)
        dev = corpus_mrr(dev_queries, mr.rank_fn(), per_direction)
        return mr, dev, {"model": "lgbm"}

    for C in C_grid:
        mr = MetaRanker(sources, model="logistic", k=k, seed=seed,
                        extra_feature_keys=extra_feature_keys)
        mr.fit(train_queries, C=C)
        dev = corpus_mrr(dev_queries, mr.rank_fn(), per_direction)
        if dev > best_mrr:
            best_mrr, best_ranker, best_info = dev, mr, {"C": C}

    return best_ranker, best_mrr, best_info
