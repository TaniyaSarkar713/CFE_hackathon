"""Statistical significance for system comparisons (eval rigor, Si et al. 2024).

Bootstrap CIs (in `diagnostics.py`) say how stable one system's MRR is. This
module says whether the *gap between two systems* is real, not noise — the
"our gains are significant (p<0.05)" claim for the ablation table.

Because every system is evaluated on the **same papers**, the right test is
*paired*: compare the per-paper MRR differences. We provide

  - `paired_bootstrap_delta` — non-parametric, robust to MRR's non-normality
    (the primary test);
  - `paired_ttest` / `welch_ttest` — parametric complements (Welch's unequal-
    variance t-test is what Si et al. use for independent samples, e.g. human
    vs. AI ideas);
  - `bonferroni` — family-wise correction when comparing many systems at once;
  - `compare_to_baseline` — one call: every system vs a baseline, corrected.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .metrics import paper_mrr
from .schema import Query


def per_paper_mrr(queries: Sequence[Query], rank_fn: Callable,
                  per_direction: bool = True) -> np.ndarray:
    """Per-paper MRR vector (labelled papers only), aligned across systems."""
    vals = [paper_mrr(rank_fn(q), q.relevance, per_direction)
            for q in queries if q.has_ground_truth()]
    return np.asarray(vals, dtype=float)


# --------------------------------------------------------------------------- #
# Paired bootstrap on per-paper MRR deltas (primary, non-parametric)
# --------------------------------------------------------------------------- #
def paired_bootstrap_delta(
    queries: Sequence[Query], rank_fn_a: Callable, rank_fn_b: Callable,
    n_boot: int = 2000, alpha: float = 0.05, per_direction: bool = True,
    seed: int = 0,
) -> Dict:
    """Bootstrap the mean per-paper MRR difference (A − B).

    Returns the point delta, a (1−alpha) CI, and a two-sided p-value (the
    fraction of resamples on the wrong side of zero, doubled). A CI that
    excludes 0 ⇔ significant at `alpha`.
    """
    a = per_paper_mrr(queries, rank_fn_a, per_direction)
    b = per_paper_mrr(queries, rank_fn_b, per_direction)
    diff = a - b
    n = len(diff)
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0, "n": 0}

    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    boots.sort()
    lo = float(boots[int((alpha / 2) * n_boot)])
    hi = float(boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
    frac_le0 = float(np.mean(boots <= 0.0))
    frac_ge0 = float(np.mean(boots >= 0.0))
    p = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    return {"delta": float(diff.mean()), "lo": lo, "hi": hi,
            "p_value": p, "n": n, "significant": (lo > 0 or hi < 0)}


# --------------------------------------------------------------------------- #
# Parametric t-tests
# --------------------------------------------------------------------------- #
def paired_ttest(queries, rank_fn_a, rank_fn_b, per_direction: bool = True) -> Dict:
    """Paired t-test on per-paper MRR (same papers → paired is more powerful)."""
    from scipy import stats
    a = per_paper_mrr(queries, rank_fn_a, per_direction)
    b = per_paper_mrr(queries, rank_fn_b, per_direction)
    t, p = stats.ttest_rel(a, b)
    return {"delta": float(a.mean() - b.mean()), "t": float(t),
            "p_value": float(p), "n": len(a)}


def welch_ttest(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict:
    """Welch's unequal-variance t-test for INDEPENDENT samples.

    Use when the two groups are different items (Si et al.: human vs. AI ideas),
    not the same papers re-ranked. For same-paper system comparison use
    `paired_ttest` / `paired_bootstrap_delta`.
    """
    from scipy import stats
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "t": float(t), "p_value": float(p)}


# --------------------------------------------------------------------------- #
# Multiple-comparison correction + one-call comparison table
# --------------------------------------------------------------------------- #
def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[Dict]:
    """Bonferroni family-wise correction over a set of p-values."""
    m = len(p_values)
    return [{"p": float(p), "p_corrected": min(1.0, float(p) * m),
             "significant": (p * m) < alpha} for p in p_values]


def compare_to_baseline(
    queries: Sequence[Query], systems: Dict[str, Callable], baseline: str,
    n_boot: int = 2000, alpha: float = 0.05, per_direction: bool = True, seed: int = 0,
) -> List[Dict]:
    """Compare every system to `baseline`; Bonferroni-correct across the family.

    `systems` maps name -> rank_fn. Returns one row per non-baseline system with
    the MRR delta, paired-bootstrap CI / p, and the corrected significance.
    """
    base_fn = systems[baseline]
    names = [n for n in systems if n != baseline]
    rows = []
    for n in names:
        res = paired_bootstrap_delta(queries, systems[n], base_fn,
                                     n_boot=n_boot, alpha=alpha,
                                     per_direction=per_direction, seed=seed)
        rows.append({"system": n, "vs": baseline, **res})
    corr = bonferroni([r["p_value"] for r in rows], alpha=alpha)
    for r, c in zip(rows, corr):
        r["p_corrected"] = c["p_corrected"]
        r["significant_corrected"] = c["significant"]
    return rows
