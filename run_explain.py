"""Demo: Direction Cards — the explainability layer (plan §6).

Two parts:
  A) SYNTHETIC — full interpretability machinery on multi-source fused rankings:
       exact attribution, causal ablation probe, defeasible logical reasoning,
       calibrated confidence/abstention, and a faithfulness check.
  B) REAL Data/ — evidence grounding: link a generated future_direction to the
       summary's key_gaps (explicit/implicit) with quoted spans.

    python run_explain.py
"""

from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

from rank_fusion import (
    make_synthetic_dataset, tune_rrf, linear_shap, fit_select_meta_ranker,
    make_rrf_rank_fn, explain_ranking, render_card_markdown, Calibrator,
    ground_evidence,
)
from rank_fusion.fusion import weighted_rrf

DATA = os.path.join(os.path.dirname(__file__), "Data")


def _score_label_pairs(queries, cfg):
    """Flatten (fused score, is_relevant) over all candidates for calibration."""
    scores, labels = [], []
    for q in queries:
        _, s = weighted_rrf(q.rankings, cfg.weights, cfg.k, q.candidate_ids)
        rel = q.relevant_ids
        for cid in q.candidate_ids:
            scores.append(s.get(cid, 0.0))
            labels.append(1 if cid in rel else 0)
    return scores, labels


def part_a_synthetic(seed=7):
    print("=" * 72)
    print("A) SYNTHETIC — Direction Cards with full interpretability")
    print("=" * 72)
    ds = make_synthetic_dataset(n_papers=300, attach_features=True, seed=seed)
    sources = ds.all_sources()

    cfg = tune_rrf(ds.dev, sources, n_random=300, coord_rounds=3, seed=seed)
    print(f"Fusion: tuned RRF (k={cfg.k:g}); attribution = exact per-source RRF "
          f"decomposition.\n")

    # calibrate confidence on dev (held-out): fused score -> P(relevant), ECE
    tr_scores, tr_labels = _score_label_pairs(ds.train, cfg)
    dev_scores, dev_labels = _score_label_pairs(ds.dev, cfg)
    cal = Calibrator().fit(tr_scores, tr_labels)
    print(f"Confidence calibration ECE (held-out dev) = "
          f"{cal.ece(dev_scores, dev_labels):.4f}\n")

    rank_fn = make_rrf_rank_fn(cfg.weights, cfg.k)
    # show cards for a few papers to surface recommend / caveat / abstain
    shown = 0
    for q in ds.test:
        ranked = rank_fn(q)
        # attribution stays consistent with the ranker: RRF cards use exact
        # per-source RRF attribution (no meta_ranker passed).
        cards = explain_ranking(q, ranked, top_k=3, weights=cfg.weights, k=cfg.k,
                                calibrator=cal)
        verdicts = {c.verdict for c in cards}
        # pick papers that show some variety in verdicts
        if shown >= 2 and len(verdicts) < 2:
            continue
        print(f"------ Paper {q.query_id} (true directions: "
              f"{sorted(q.relevant_ids)}) ------")
        for c in cards:
            print(render_card_markdown(c))
            print()
        shown += 1
        if shown >= 3:
            break

    # learned-model attribution path: exact SHAP for the logistic meta-ranker
    mr, _, _ = fit_select_meta_ranker(ds.train, ds.dev, sources, seed=seed)
    q0 = ds.test[0]
    top = mr.rank_fn()(q0)[0]
    shap = linear_shap(mr, q0, top)
    drivers = sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:5]
    print("Meta-ranker (logistic) SHAP for its #1 pick on "
          f"{q0.query_id} [{top}]:")
    print("   " + ", ".join(f"{n} {v:+.3f}" for n, v in drivers))
    print("   (exact: SHAP values sum with the intercept to the model's logit)\n")


def part_b_real():
    print("=" * 72)
    print("B) REAL Data/ — evidence grounding against key_gaps")
    print("=" * 72)
    summ_dir = os.path.join(DATA, "Paper_summaries")
    gt_dir = os.path.join(DATA, "Ground Truth Future Directions")
    pid = "11816194"
    with open(os.path.join(summ_dir, f"{pid}_future_directions.json")) as f:
        summary = json.load(f)
    gaps = summary.get("key_gaps", [])
    print(f"Paper {pid} — key_gaps:")
    for g in gaps:
        print(f"   [{g.get('status')}] {g.get('description')[:80]}")
    print()

    print("Evidence-grounded reasoning (single ranker in Data/, so the verdict "
          "keys off documented gaps; the\nfull argumentation engine adds the "
          "GPT/Claude/retrieval signals once the fusion pipeline feeds in).\n")
    for d in summary.get("future_directions", [])[:5]:
        text = f"{d['title']}. {d['rationale']}"
        addressed, spans = ground_evidence(text, gaps, tau=0.08)
        explicit = any("explicit" in a for a in addressed)
        implied = bool(addressed) and not explicit
        if explicit:
            verdict = "✅ recommend (grounded in an EXPLICIT limitation)"
        elif implied:
            verdict = "⚠️ recommend w/ caveat (addresses an INFERRED gap)"
        else:
            verdict = "⛔ abstain (no documented gap matched)"
        print(f"#{d['id']} {d['title']}")
        print(f"   addresses: {addressed[0] if addressed else '— (no gap matched)'}")
        print(f"   verdict: {verdict}")
        print()


if __name__ == "__main__":
    part_a_synthetic()
    part_b_real()
