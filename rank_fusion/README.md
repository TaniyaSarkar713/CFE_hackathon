# `rank_fusion` — MRR-calibrated rank fusion (Agent 3)

Taniya's workstream from the ResearchHorizon plan (§5 + §8): the **rank-fusion
engine, MRR harness, and MRR-driven optimisation**. Built and validated on
**synthetic rankings** so it drops straight into Agent 3 once Nourah's scaffold
delivers the real GPT/Claude ranked lists. Also runs on the real `Data/` folder.

```
Agent 2a (GPT) ranks C ─┐
Agent 2b (Claude) ranks C ─┼─►  Agent 3: weighted RRF  ──► tune (w, k) by dev MRR
retrieval / citation ────┘         └─► learned meta-ranker (stretch)
                                            └─► argmax MRR_dev  ──► ranked directions
```

## What's here

| File | Role |
|------|------|
| `schema.py` | `Query` (one paper: candidate pool, per-source rankings, free-text ground truth, optional candidate provenance + features) and `Dataset` (train/dev/test split). |
| `metrics.py` | **MRR harness** — per-direction Paper MRR (plan §1), corpus MRR, MRR@K, Recall@K, nDCG@K, and `evaluate()` for the full bundle. |
| `diagnostics.py` | **Level 1** — `oracle_mrr` (the ranking-independent ceiling), per-source `candidate_recall_table`, and `bootstrap_mrr_ci` (95% CIs). |
| `fusion.py` | **Weighted RRF** `Σ_a w_a/(k_a+r_a(c)) + β·x(c)` — per-source `k_a` and an additive feature term (Level 2), plus Borda & CombSUM baselines. |
| `optimize.py` | **MRR-driven tuning** — random search + coordinate ascent over `(w, k)` to maximise **dev** MRR, optionally **cost-aware** `MRR − λ·cost` (Level 5). |
| `meta_ranker.py` | **Learned meta-ranker** — pointwise logistic (or LightGBM `LGBMRanker`) over rank-fusion + agreement features; selected by dev MRR (Level 3). |
| `learned_rankers.py` | **Pairwise RankNet/BPR** (hard-negative mining), **listwise ListNet** (top-one cross-entropy), and the **differentiable soft-MRR surrogate** (Levels 3–4) — pure numpy/scipy. |
| `explain.py` | **Explainability layer (§6)** — Direction Cards: exact attribution, causal ablation probe, defeasible logical reasoning, calibration/abstention, faithfulness check, evidence grounding. |
| `synthetic.py` | `make_synthetic_dataset` (per-source skill + diverse noise; optional horizon features) and `make_candidate_generation_dataset` (generators with partial recall, for Level 1). |
| `realdata.py` | Adapter for `Data/` — matches generated `future_directions` to masked ground-truth text via TF-IDF cosine (stand-in for SPECTER2 until embeddings land). |

Top-level demos: **`run_synthetic.py`** (staged Level 1→5 ablation with CIs),
**`run_realdata.py`** (MRR on the real `Data/` files + tau sweep), and
**`run_explain.py`** (Direction Cards on synthetic + evidence grounding on real `Data/`).

## Explainability — Direction Cards (§6)

Every recommended direction ships as an auditable **Direction Card**, not a
black-box rank. The design borrows from interpretability practice (SHAP;
Ivanitskiy, *Mechanistic Interpretability*):

| Layer | Idea | Interp analogue |
|-------|------|-----------------|
| **Attribution** | decompose the score into per-source / per-feature contributions | input attribution / saliency (exact for RRF & linear SHAP) |
| **Ablation probe** | leave-one-signal-out re-rank: "remove citation → drops to #7" | causal intervention / activation patching |
| **Logical reasoning** | PRO/CON arguments resolved under **grounded argumentation semantics** → verdict (recommend / caveat / abstain) + trace | defeasible (non-monotonic) logic |
| **Calibration** | fused score → calibrated P(relevant), ECE reported; abstain on thin evidence | model-risk-aware (Goodhart) |
| **Faithfulness** | the shown rationale must cite the factors that *actually* drove the rank | faithfulness audit |
| **Evidence grounding** | link a direction to the `key_gaps` it addresses (explicit/implied), with quoted spans | extractive grounding |

The attribution is **faithful by construction**: for RRF the score is literally
`Σ w_a/(k_a+r_a)`, so the per-source decomposition is exact; for the logistic
meta-ranker the logit is linear in standardised features, so SHAP values are
exact (`coef_j·z_j`) and sum with the intercept back to the logit (both asserted
in tests). The natural-language rationale is generated *from* the attribution, so
it cannot drift into a plausible-but-wrong post-hoc story.

```python
from rank_fusion import explain_ranking, render_card_markdown, make_rrf_rank_fn
ranked = make_rrf_rank_fn(cfg.weights, cfg.k)(query)
cards = explain_ranking(query, ranked, top_k=3, weights=cfg.weights, k=cfg.k)
print(render_card_markdown(cards[0]))
```

## Staged optimisation strategy (the whole pipeline, not just a model)

The objective is `max_θ MRR_dev(FinalRanker_θ)` over the *whole* pipeline. MRR
can only improve if the true direction is in the candidate pool, so we optimise
in stages — measure the ceiling first, then ordering:

| Level | What it optimises | Where |
|------:|-------------------|-------|
| **1** | candidate-pool **recall** (oracle MRR ceiling, per-source recall) | `diagnostics.py` |
| **2** | weighted RRF `Σ w_a/(k_a+r_a) + β·x`, tuned on dev MRR | `fusion.py` + `optimize.py` |
| **3** | learned meta-ranker — pointwise (logistic / LightGBM), **pairwise RankNet+BPR**, **listwise ListNet** | `meta_ranker.py`, `learned_rankers.py` |
| **4** | **differentiable soft-MRR** surrogate (soft ranks) | `learned_rankers.py` |
| **5** | **cost-aware** objective `MRR − λ·cost` (compute-risk governor) | `optimize.py` |

Discipline against over-fitting a tiny dev set: temporal split, `bootstrap_mrr_ci`,
and the full ablation table — exactly the MLRC-Bench evaluation discipline.

## Quick start

```bash
pip install -r rank_fusion/requirements.txt   # numpy, scikit-learn, scipy
python run_synthetic.py        # ablation: per-source → uniform RRF → tuned RRF → meta-ranker
python run_realdata.py         # MRR harness on the real Data/ folder
python tests/test_rank_fusion.py   # 8 unit + behavioural tests
```

## Representative synthetic result (seed 7)

Level 1 — candidate recall (the ceiling): union pool **Recall 0.95** vs best
single generator 0.74; oracle MRR ceiling 0.95.

Levels 2–5 — ranking & fusion:

| System | Dev MRR | Test MRR | 95% CI (test) |
|--------|--------:|---------:|:-------------:|
| best single ranker (gpt) | 0.3998 | 0.4272 | |
| uniform RRF (k=60) | 0.3540 | 0.4161 | |
| **tuned RRF (argmax dev MRR)** | **0.4512** | **0.4475** | [0.391, 0.505] |
| cost-aware RRF (drops claude+citation) | 0.4123 | 0.4308 | |
| meta-ranker: logistic | 0.4046 | 0.4400 | |
| meta-ranker: pairwise RankNet | 0.4345 | 0.4490 | [0.392, 0.507] |
| soft-MRR surrogate | 0.3726 | 0.4157 | |

MRR-tuned fusion and the RankNet meta-ranker beat every individual ranker *and*
uniform RRF, and the gain transfers to held-out test. The CIs overlap — the
honest reading on a small dev set, which is exactly why we report them.

## Plugging in the real pipeline

Build one `Query` per paper and call the same harness — nothing else changes:

```python
from rank_fusion import Query, weighted_rrf, tune_rrf, corpus_mrr, make_rrf_rank_fn

q = Query(
    query_id="11816194",
    candidate_ids=[...],                       # deduplicated candidate pool C
    rankings={"gpt": [...], "claude": [...],   # each agent's ordered ids
              "retrieval": [...], "citation": [...]},
    relevance={"g0": {...}, "g1": {...}},      # GT direction -> matching candidate ids
)

cfg = tune_rrf(dev_queries, sources=["gpt","claude","retrieval","citation"])  # dev MRR
fused, _ = weighted_rrf(q.rankings, cfg.weights, cfg.k, q.candidate_ids)
print("test MRR:", corpus_mrr(test_queries, cfg.rank_fn()))
```

The only real-pipeline dependency is the **match function** that fills
`relevance` (which candidate matches which masked ground-truth direction).
`realdata.py` uses TF-IDF cosine today; swap in SPECTER2/SciNCL cosine + the
entailment judge (plan §1) when available — the rest of the engine is unchanged.

## Notes / next steps (plan §6)

- LightGBM path is optional (`MetaRanker(..., model="lgbm")`); falls back to
  logistic, which needs only scikit-learn.
- Once rank fusion is locked, the meta-ranker's per-candidate features
  (`rank_*`, `rrf_score`, agreement/disagreement) are the natural inputs to the
  explainability layer — SHAP over these features gives the Direction-Card
  "top_factors" directly.
