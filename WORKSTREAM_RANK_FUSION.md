# Workstream Status — Rank Fusion, MRR Optimization & Explainability (Taniya)

Status doc for the **Agent 3** workstream of ResearchHorizon: the rank-fusion
engine, the MRR-driven optimization, and the §6 explainability layer. Everything
lives in the **`rank_fusion/`** package and is validated end-to-end on synthetic
data and on the real `Data/` folder.

- **Plan reference:** `plan.pdf` (§1 MRR, §5 fusion, §6 explainability, §8 eval).
- **Package README (deeper):** [`rank_fusion/README.md`](rank_fusion/README.md)
- **Last updated:** 2026-06-22

---

## 1. What this workstream delivers

> Given a paper with its future-work section masked, two (or more) agents rank
> candidate future directions; a **rank fusion** engine fuses those lists into
> one final ranking **calibrated directly to dev MRR** (not to any single LLM's
> judgment); and every recommendation ships as an **auditable Direction Card**.

The final decision is made by the metric, and the output is explainable — which
limitation a direction addresses, which agents/sources drove its rank, and how
confident the system is.

---

## 2. Done — component inventory

All code is pure `numpy` / `scikit-learn` / `scipy` (no LightGBM/torch required;
LightGBM is an optional path). **21/21 tests pass.**

### Core engine
| File | What it does | Plan |
|------|--------------|------|
| `rank_fusion/schema.py` | `Query` (paper: candidate pool, per-source rankings, free-text ground truth, optional provenance + features) and `Dataset` (train/dev/test split). The contract every stage shares. | §1 |
| `rank_fusion/metrics.py` | **MRR harness** — per-direction Paper MRR, corpus MRR, MRR@K, Recall@K, nDCG@K, `evaluate()` bundle. | §1, §8 |
| `rank_fusion/fusion.py` | **Weighted RRF** `Σ_a w_a/(k_a+r_a(c)) + β·x(c)` (per-source `k_a`, additive feature term) + Borda / CombSUM baselines. | §5 |
| `rank_fusion/optimize.py` | **MRR-driven tuning** of `(weights, k)` via random search + coordinate ascent on **dev** MRR; optional **cost-aware** objective `MRR − λ·cost`. | §5 |
| `rank_fusion/meta_ranker.py` | **Learned meta-ranker** — pointwise logistic (or LightGBM `LGBMRanker`) over rank-fusion + agreement features; hyper-params selected by dev MRR. | §5 |
| `rank_fusion/learned_rankers.py` | **Pairwise RankNet/BPR** (hard-negative mining) and the **differentiable soft-MRR surrogate** (scipy L-BFGS). | §4, §9 |
| `rank_fusion/diagnostics.py` | **Candidate-recall ceiling** — `oracle_mrr`, per-source `candidate_recall_table`, and `bootstrap_mrr_ci` (95% CIs). | §8.2 |

### Explainability layer (§6)
| File | What it does |
|------|--------------|
| `rank_fusion/explain.py` | **Direction Cards** with: exact additive **attribution** (per-source RRF decomposition + exact linear-SHAP for the logistic meta-ranker), causal **ablation probe** (leave-one-signal-out), **defeasible logical reasoning** (PRO/CON arguments under grounded argumentation semantics → recommend / caveat / abstain + reasoning trace), **calibration + ECE + abstention**, a **faithfulness check**, and **evidence grounding** to the summary's `key_gaps`. Markdown renderer included. |

### Data generators / adapters
| File | What it does |
|------|--------------|
| `rank_fusion/synthetic.py` | `make_synthetic_dataset` (per-source *skill* + diverse noise; optional horizon features) and `make_candidate_generation_dataset` (generators with partial recall, for the Level-1 ceiling demo). |
| `rank_fusion/realdata.py` | Loads the real `Data/` folder; matches generated `future_directions` to masked ground-truth text via **TF-IDF cosine** (stand-in for SPECTER2 until embeddings land). |

### Demos & tests
| File | What it does |
|------|--------------|
| `run_synthetic.py` | Staged Level 1→5 ablation (recall ceiling → tuned RRF → meta-rankers → soft-MRR → cost-aware) with bootstrap CIs. |
| `run_realdata.py` | MRR harness on the real `Data/` files + tau-sensitivity sweep. |
| `run_explain.py` | Direction Cards on synthetic data + evidence grounding on real `Data/`. |
| `demo_researchhorizon.py` | **End-to-end on real papers**: 4 (simulated) agents → fused ranking → Direction Cards for paper 11816106. |
| `tests/test_rank_fusion.py` | 21 unit + behavioural tests (metric correctness, fusion mechanics, tuning generalises, attribution sums to score, SHAP sums to logit, logical verdicts, faithfulness, evidence grounding, end-to-end card). |

### Representative results (synthetic, seed 7)
| System | Dev MRR | Test MRR |
|--------|--------:|---------:|
| best single ranker | 0.3998 | 0.4272 |
| uniform RRF | 0.3540 | 0.4161 |
| **tuned RRF (argmax dev MRR)** | **0.4512** | **0.4475** |
| pairwise RankNet | 0.4345 | 0.4490 |

On the real task (`demo_researchhorizon.py`, 11 papers, 8-seed avg): best single
agent MRR **0.751** → tuned RRF **0.816**; the fused top-3 for paper 11816106 are
all three real future-work directions the authors pursued.

### How to run
```bash
pip install -r rank_fusion/requirements.txt   # numpy, scikit-learn, scipy
python demo_researchhorizon.py     # the headline end-to-end demo on real papers
python run_synthetic.py            # full Level 1→5 ablation
python run_realdata.py             # MRR on real Data/
python run_explain.py              # Direction Cards
python tests/test_rank_fusion.py   # 21 tests
```

---

## 3. Upstream dependencies (what's needed to go from synthetic → production)

The engine is **decoupled** behind the `Query` contract, so plugging in the real
pipeline is a data-wiring task, not a rewrite. Each dependency below is the
*only* thing standing between a synthetic demo and a real one.

| # | Need | Owner | Consumed by | Current stand-in |
|---|------|-------|-------------|------------------|
| D1 | **Agent-1 comprehension JSON** per paper (problem, limitations, `key_gaps`, key_terms) | Nourah (Agent 1) | evidence grounding, meta-ranker features | real `Data/Paper_summaries/*.json` already provide `key_gaps` + `future_directions` |
| D2 | **Real GPT & Claude ranked lists** over the candidate pool | Nourah (Agents 2a/2b) | `Query.rankings["gpt"/"claude"]` → fusion | simulated rankers in `demo_researchhorizon.py` |
| D3 | **Candidate pool C** = generative ∪ retrieval ∪ citation (deduped) | Nourah + Naz | `Query.candidate_ids`, `candidate_sources` | real `future_directions` (single source); synthetic generators for Level 1 |
| D4 | **Scientific embeddings** (SPECTER2 / SciNCL) for the match function `match(d_i, g_j)` | Naz / shared | MRR labelling, retrieval candidates | **TF-IDF cosine** in `realdata.py` |
| D5 | **Entailment judge** (LLM) + human-labeled `tau` calibration set | shared | MRR match function, `tau` selection | fixed `tau` + tau-sweep diagnostic |
| D6 | **Citation graph** (Semantic Scholar / OpenAlex): forward citations, recency/popularity/novelty | Naz (§7) | `Query.features` (horizon priors), citation candidates | synthetic horizon features via `attach_features=True` |
| D7 | **Temporal split** by publication date (train/dev/test, no leakage) | shared | tuning & reporting | index-order split in generators |

**Integration point:** build one `Query` per paper with the real `rankings`,
`candidate_ids`, `features`, and `relevance` (filled by the D4/D5 match function),
then call `tune_rrf(...)` and `explain_ranking(...)` exactly as the demos do.

---

## 4. What's left

### A. Wiring (unblocks once upstream lands)
- [ ] Replace simulated rankers with real GPT/Claude lists (**D2**) — drop into `Query.rankings`.
- [ ] Swap TF-IDF for SPECTER2/SciNCL cosine + entailment judge (**D4/D5**) in `realdata.py`'s match function.
- [ ] Feed citation-graph horizon priors (**D6**) into `Query.features`; re-run the Connected-Papers ablation (§7) to report their marginal MRR lift.
- [ ] Apply the real temporal split (**D7**); re-tune on dev, report **once** on test.

### B. Engine hardening
- [ ] Out-of-fold candidate/feature generation for the meta-ranker (plan §8.3 leakage control).
- [ ] Diversity-aware re-ranking (MMR) so the top-K covers distinct gaps (§9.3) — directly helps MRR when a paper has several true directions.
- [ ] Disagreement-routed tie-break: spend a pairwise (PRP) call only where GPT/Claude disagree most (§9.4), gated by the cost-aware governor (already in `optimize.py`).
- [ ] Cross-encoder reranker distilled from the LLM rankers (§9.2) — cheap at inference.

### C. Explainability polish
- [ ] **JSON export** of Direction Cards for the `foresight.jsx` frontend (`DirectionCard.to_dict()` exists; add a top-level writer).
- [ ] Real `tau` + entailment-judge agreement report (Cohen's κ vs humans) for the faithfulness/calibration story.
- [ ] Calibrate confidence on the full temporal-test corpus (current ECE is from small dev) and report it in the ablation table.

### D. Reporting for the submission
- [ ] Fill the §10 ablation table with **real** numbers once D1–D7 land.
- [ ] Candidate-recall diagnostic table (§8.2) on real candidate sources — show recall is the ceiling *before* showing fusion lifts ordering.
- [ ] Match-function sensitivity (sweep `tau`) on real data (`run_realdata.py` already does this for TF-IDF).

### Not in scope for this workstream
Agent-1 comprehension, Agent-2 prompt engineering, candidate generation, and the
Semantic Scholar / Connected-Papers integration (Nourah & Naz). This workstream
consumes their outputs through the `Query` contract.

---

## 5. One honest caveat baked into the design
**Verdict ≠ confidence.** The Direction Card's *verdict* is the logical decision
from the evidence (recommend / caveat / abstain); *confidence* is the calibrated
P(relevant). They are deliberately separate axes — a logically-supported lead can
still carry modest calibrated confidence. This is the model-risk-aware behaviour,
not a bug, and it's why the system can abstain ("insufficient evidence") instead
of guessing.
