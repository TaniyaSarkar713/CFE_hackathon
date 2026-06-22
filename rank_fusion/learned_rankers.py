"""Learned rankers beyond RRF: pairwise RankNet/BPR (Level 3) and a
differentiable soft-MRR surrogate (Level 4).

Both reuse the meta-ranker's per-candidate features (rank in each source, RRF
score, agreement/disagreement, ...), standardise them on the train split, and
are **model-selected by dev MRR** — the loss is only a smooth proxy.

Pure numpy + scipy, so no LightGBM / torch needed.

  - PairwiseRankNet  minimises  -log sigma(s(c+) - s(c-))  over (positive, hard
    negative) pairs. Hard negatives are non-matching candidates that the sources
    ranked highly (high RRF score) — exactly the confusable cases, which is what
    you want when positives are sparse (BPR/RankNet).

  - SoftMRRRanker  maximises a smooth surrogate of MRR using a soft rank
        softrank(c+) = 1 + sum_{c != c+} sigma((s(c) - s(c+)) / tau)
    so the objective is differentiable and directly aligned with the metric.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .meta_ranker import _query_features, _labels
from .metrics import corpus_mrr
from .schema import Query


# --------------------------------------------------------------------------- #
# Shared feature assembly + standardisation
# --------------------------------------------------------------------------- #
class _FeatureSpace:
    """Builds standardised per-query feature matrices, sharing one scaler."""

    def __init__(self, sources: Sequence[str], k: float = 60.0):
        self.sources = list(sources)
        self.k = k
        self.mean_ = None
        self.std_ = None
        self.keep_ = None

    def fit(self, queries: Sequence[Query]):
        mats = [self._raw(q)[1] for q in queries
                if q.has_ground_truth() and self._raw(q)[1].shape[0]]
        X = np.vstack(mats)
        self.keep_ = np.std(X, axis=0) > 1e-12
        if not self.keep_.any():
            self.keep_[:] = True
        Xk = X[:, self.keep_]
        self.mean_ = Xk.mean(axis=0)
        self.std_ = Xk.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def _raw(self, query: Query) -> Tuple[List[str], np.ndarray]:
        return _query_features(query, self.sources, self.k)

    def transform(self, query: Query) -> Tuple[List[str], np.ndarray]:
        cand_ids, X = self._raw(query)
        if X.shape[0] == 0:
            return cand_ids, X
        Xs = (X[:, self.keep_] - self.mean_) / self.std_
        return cand_ids, Xs

    @property
    def dim(self) -> int:
        return int(self.keep_.sum())


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(np.clip(z, -30, 30))
    return e / (e.sum() + 1e-12)


# --------------------------------------------------------------------------- #
# Level 3 — pairwise RankNet / BPR with hard-negative mining
# --------------------------------------------------------------------------- #
class PairwiseRankNet:
    def __init__(self, sources, k=60.0, lr=0.1, epochs=300, n_hard=10,
                 l2=1e-3, seed=0):
        self.fs = _FeatureSpace(sources, k)
        self.lr, self.epochs, self.n_hard, self.l2, self.seed = lr, epochs, n_hard, l2, seed
        self.w = None

    def _pairs(self, queries):
        """Yield (x_pos - x_neg) difference vectors across all queries."""
        rng = np.random.default_rng(self.seed)
        diffs = []
        rrf_col = None
        names = None
        for q in queries:
            if not q.has_ground_truth():
                continue
            cand_ids, Xs = self.fs.transform(q)
            if Xs.shape[0] == 0:
                continue
            y = _labels(q, cand_ids)
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            # hard negatives: non-matches the sources scored highest.
            # use the raw RRF-score feature to rank negatives (pre-standardised
            # column index is found once).
            _, Xraw = self.fs._raw(q)
            from .meta_ranker import feature_names
            if names is None:
                names = feature_names(self.fs.sources)
                rrf_col = names.index("rrf_score")
            neg_sorted = neg[np.argsort(-Xraw[neg, rrf_col])]
            hard = neg_sorted[: self.n_hard]
            rand = rng.choice(neg, size=min(self.n_hard, len(neg)), replace=False)
            chosen_neg = np.unique(np.concatenate([hard, rand]))
            for p in pos:
                for n in chosen_neg:
                    diffs.append(Xs[p] - Xs[n])
        return np.asarray(diffs) if diffs else np.zeros((0, self.fs.dim))

    def fit(self, train_queries):
        self.fs.fit(train_queries)
        D = self._pairs(train_queries)  # (n_pairs, dim), each should score > 0
        self.w = np.zeros(self.fs.dim)
        if D.shape[0] == 0:
            return self
        for _ in range(self.epochs):
            margin = D @ self.w
            g = -(1.0 - _sigmoid(margin))           # dL/d(margin)
            grad = (g[:, None] * D).mean(axis=0) + self.l2 * self.w
            self.w -= self.lr * grad
        return self

    def rank_fn(self):
        def fn(query):
            cand_ids, Xs = self.fs.transform(query)
            if Xs.shape[0] == 0:
                return list(query.candidate_ids)
            scores = Xs @ self.w
            order = np.argsort(-scores, kind="stable")
            return [cand_ids[i] for i in order]
        return fn


# --------------------------------------------------------------------------- #
# Level 3 (listwise) — ListNet: top-one cross-entropy over the whole list
# --------------------------------------------------------------------------- #
class ListNetRanker:
    """ListNet (Cao et al. 2007): the listwise learning-to-rank loss.

    For each paper (a query group) the candidate scores and the binary relevance
    labels are each turned into a **top-one probability distribution** via
    softmax; the loss is the cross-entropy between them:

        P_s = softmax(X w),  P_y = softmax(y)
        L   = - sum_j P_y(j) log P_s(j)      (summed over papers)

    Unlike pointwise (each candidate alone) or pairwise (each pair), this scores
    the *whole list* at once. For a linear scorer the gradient is clean:
    dL/dw = X^T (P_s - P_y). Model-selected by dev MRR.
    """

    def __init__(self, sources, k=60.0, lr=0.05, epochs=400, l2=1e-3,
                 grad_clip=0.1, seed=0):
        self.fs = _FeatureSpace(sources, k)
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed
        self.grad_clip = grad_clip
        self.w = None

    def _groups(self, queries):
        data = []
        for q in queries:
            if not q.has_ground_truth():
                continue
            cand_ids, Xs = self.fs.transform(q)
            if Xs.shape[0] == 0:
                continue
            y = _labels(q, cand_ids)
            if y.sum() == 0:                       # need >=1 positive per group
                continue
            data.append((Xs, _softmax(y)))         # cache the target distribution
        return data

    def fit(self, train_queries):
        self.fs.fit(train_queries)
        data = self._groups(train_queries)
        self.w = np.zeros(self.fs.dim)
        if not data:
            return self
        for _ in range(self.epochs):
            grad = np.zeros(self.fs.dim)
            for Xs, Py in data:
                Ps = _softmax(Xs @ self.w)
                grad += Xs.T @ (Ps - Py)           # dL/dw for softmax cross-entropy
            grad = grad / len(data) + self.l2 * self.w
            # clip the gradient norm: full-batch GD on this listwise loss can
            # otherwise overshoot and converge to a sign-flipped solution. The
            # clip is well-scaled because features are standardised.
            gnorm = float(np.linalg.norm(grad))
            if gnorm > self.grad_clip:
                grad *= self.grad_clip / gnorm
            self.w -= self.lr * grad
        return self

    def rank_fn(self):
        def fn(query):
            cand_ids, Xs = self.fs.transform(query)
            if Xs.shape[0] == 0:
                return list(query.candidate_ids)
            order = np.argsort(-(Xs @ self.w), kind="stable")
            return [cand_ids[i] for i in order]
        return fn


# --------------------------------------------------------------------------- #
# Level 4 — differentiable soft-MRR surrogate (linear scorer)
# --------------------------------------------------------------------------- #
class SoftMRRRanker:
    def __init__(self, sources, k=60.0, tau=1.0, l2=1e-2, seed=0):
        self.fs = _FeatureSpace(sources, k)
        self.tau, self.l2, self.seed = tau, l2, seed
        self.w = None

    def _assemble(self, queries):
        data = []
        for q in queries:
            if not q.has_ground_truth():
                continue
            cand_ids, Xs = self.fs.transform(q)
            if Xs.shape[0] == 0:
                continue
            idx = {d: i for i, d in enumerate(cand_ids)}
            dirs = []
            for cands in q.relevance.values():
                mi = [idx[c] for c in cands if c in idx]
                if mi:
                    dirs.append(np.asarray(mi))
            if dirs:
                data.append((Xs, dirs))
        return data

    def _neg_soft_mrr(self, w, data):
        tau = self.tau
        total, count = 0.0, 0
        for Xs, dirs in data:
            s = Xs @ w
            for mi in dirs:
                rep = s[mi].max()                       # representative positive
                others = s.sum() - s[mi].sum()          # not used directly
                # soft rank = 1 + sum over non-positives of sigma((s_c - rep)/tau)
                contrib = _sigmoid((s - rep) / tau)
                contrib[mi] = 0.0                       # exclude the matches
                softrank = 1.0 + contrib.sum()
                total += 1.0 / softrank
                count += 1
        if count == 0:
            return 0.0
        return -(total / count) + self.l2 * float(w @ w)

    def fit(self, train_queries):
        from scipy.optimize import minimize
        self.fs.fit(train_queries)
        data = self._assemble(train_queries)
        w0 = np.zeros(self.fs.dim)
        if not data:
            self.w = w0
            return self
        res = minimize(self._neg_soft_mrr, w0, args=(data,),
                       method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-7})
        self.w = res.x
        return self

    def rank_fn(self):
        def fn(query):
            cand_ids, Xs = self.fs.transform(query)
            if Xs.shape[0] == 0:
                return list(query.candidate_ids)
            scores = Xs @ self.w
            order = np.argsort(-scores, kind="stable")
            return [cand_ids[i] for i in order]
        return fn


# --------------------------------------------------------------------------- #
# Model selection by dev MRR
# --------------------------------------------------------------------------- #
def fit_select_ranknet(train, dev, sources, k=60.0,
                       lr_grid=(0.05, 0.1, 0.3), epochs_grid=(200, 500),
                       n_hard=10, per_direction=True, seed=0):
    best, best_mrr, best_info = None, -1.0, {}
    for lr in lr_grid:
        for ep in epochs_grid:
            m = PairwiseRankNet(sources, k=k, lr=lr, epochs=ep,
                                n_hard=n_hard, seed=seed).fit(train)
            d = corpus_mrr(dev, m.rank_fn(), per_direction)
            if d > best_mrr:
                best, best_mrr, best_info = m, d, {"lr": lr, "epochs": ep}
    return best, best_mrr, best_info


def fit_select_listnet(train, dev, sources, k=60.0,
                       lr_grid=(0.02, 0.05, 0.1), epochs_grid=(300, 600),
                       per_direction=True, seed=0):
    best, best_mrr, best_info = None, -1.0, {}
    for lr in lr_grid:
        for ep in epochs_grid:
            m = ListNetRanker(sources, k=k, lr=lr, epochs=ep, seed=seed).fit(train)
            d = corpus_mrr(dev, m.rank_fn(), per_direction)
            if d > best_mrr:
                best, best_mrr, best_info = m, d, {"lr": lr, "epochs": ep}
    return best, best_mrr, best_info


def fit_select_soft_mrr(train, dev, sources, k=60.0,
                        tau_grid=(0.5, 1.0, 2.0), per_direction=True, seed=0):
    best, best_mrr, best_info = None, -1.0, {}
    for tau in tau_grid:
        m = SoftMRRRanker(sources, k=k, tau=tau, seed=seed).fit(train)
        d = corpus_mrr(dev, m.rank_fn(), per_direction)
        if d > best_mrr:
            best, best_mrr, best_info = m, d, {"tau": tau}
    return best, best_mrr, best_info
