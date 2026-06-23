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
| `diagnostics.py` | **Level 1 + validation** — `oracle_mrr` (ceiling), per-source `candidate_recall_table`, `bootstrap_mrr_ci` (95% CIs), and `topk_bottomk_separation` (does the ranker separate good from bad? — Si et al. Table 1). |
| `significance.py` | **System-comparison rigor** — `paired_bootstrap_delta`, `paired_ttest` / `welch_ttest`, `bonferroni`, and `compare_to_baseline` ("is the MRR gain real after correction?"). |
| `matching.py` | **The §1 match function** — `match_relevance` with a pluggable `embed_fn` (TF-IDF default; SPECTER2/SciBERT = the D4 dependency) + optional entailment judge (D5), and `calibrate_tau`. |
| `agent.py` | **Autonomous candidate-harvesting agent** — runs Naz's S2 client in a tool-loop to read neighbours' keywords + future directions, adds non-duplicate candidates, and **stops when coverage of the paper's gaps stops improving**. The one agentic component; everything downstream stays a deterministic workflow. |
| `integrate.py` | **Adapters into fusion** — `pool_to_query` (agent pool → `Query`), and `graph_signal_fn` / `attach_features` / `add_horizon_source` to feed Naz's S2 graph signals in as features, an extra RRF source, or meta-ranker features. |
| `consensus.py` | **Consensus-aware fusion** (GPT + Claude + S2/KG) — cross-source clustering, RRF + LLM-agreement (η) + agreement×S2-support (ρ) terms, an MRR@K tuner over (w, η, ρ, k, t), and consensus features for the meta-ranker. |
| `select.py` | **Best-of selection** — `compare_strategies` / `select_best_strategy`: score each source alone (LLM1, LLM2, S2-KG) *and* the fused method by dev MRR and emit the winner, so the output is never worse than the best single source (fusion can dilute under a noisy source). |
| `fusion.py` | **Weighted RRF** `Σ_a w_a/(k_a+r_a(c)) + β·x(c)` — per-source `k_a` and an additive feature term (Level 2), plus Borda & CombSUM baselines. |
| `optimize.py` | **MRR-driven tuning** — random search + coordinate ascent over `(w, k)` to maximise **dev** MRR, optionally **cost-aware** `MRR − λ·cost` (Level 5). |
| `meta_ranker.py` | **Learned meta-ranker** — pointwise logistic (or LightGBM `LGBMRanker`) over rank-fusion + agreement features; selected by dev MRR (Level 3). |
| `learned_rankers.py` | **Pairwise RankNet/BPR** (hard-negative mining), **listwise ListNet** (top-one cross-entropy), and the **differentiable soft-MRR surrogate** (Levels 3–4) — pure numpy/scipy. |
| `explain.py` | **Explainability layer (§6)** — Direction Cards: exact attribution, causal ablation probe, defeasible logical reasoning, calibration/abstention, faithfulness check, evidence grounding. |
| `diversity.py` | **MMR re-ranking** (§9.3) — trade relevance vs. redundancy so the top-K covers distinct gaps; lifts per-direction MRR when near-duplicates would crowd it out. |
| `pairwise.py` | **Swiss-tournament** aggregation of pairwise comparisons into a global ranking (~N·n/2 calls vs n²/2) — for a pairwise Agent-2, cheaper than Bradley-Terry/Elo. |
| `dedup.py` | **Candidate dedup** (Si et al.) — all-MiniLM-L6-v2 cosine ≥ 0.8 (TF-IDF fallback) collapses near-duplicate directions in the pool to one representative. |
| `synthetic.py` | `make_synthetic_dataset`, `make_candidate_generation_dataset` (Level 1), and `make_redundancy_dataset` (near-duplicate clusters, for MMR). |
| `realdata.py` | Adapter for `Data/` — matches generated `future_directions` to masked ground-truth text via TF-IDF cosine (stand-in for SPECTER2 until embeddings land). |

Top-level demos: **`run_synthetic.py`** (staged Level 1→5 ablation with CIs),
**`run_realdata.py`** (MRR on the real `Data/` files + tau sweep),
**`run_explain.py`** (Direction Cards), **`run_diversity.py`** (MMR + Swiss
tournament + dedup), **`run_significance.py`** (paired-bootstrap + Bonferroni
+ ranker separation), **`run_candidate_agent.py`** (the autonomous agent),
**`run_integration.py`** (agent output + S2 graph signals → fusion), and
**`run_consensus.py`** (consensus-aware GPT+Claude+S2 fusion).

## Consensus-aware fusion (`consensus.py`)

Encodes the intuition *"when both LLMs independently rank the same direction
near the top, that's strong evidence; the S2 knowledge graph is a supporting
signal — strongest when it backs the LLM consensus — and dev MRR decides how
much each matters."* Candidates from the three sources are first **clustered**
semantically (`cluster_sources` / `build_consensus_query`) so differently-phrased
versions of one direction share a cluster and accumulate per-source ranks. Then:

```
S(c) = Σ_m w_m/(k+r_m(c))  +  η1·A12_top1(c) + η3·A12_top3(c)  +  ρ·A12_top3(c)·R_s2(c)
```
tuned over (w, η, ρ, k, t) on dev **MRR@5** (`tune_consensus`) by **multi-restart
coordinate ascent warm-started from the tuned-RRF optimum** — so the consensus fit
can never lose to plain RRF — with a Stage-2 meta-ranker over the same quantities
as features (`add_consensus_features`, `MetaRanker(extra_feature_keys=CONSENSUS_KEYS)`).

**Honest finding** (synthetic, MRR@5): with rigorous optimisation the tuner drives
η, ρ → 0 — additive RRF already captures the LLM agreement signal, so the explicit
terms add nothing over tuned RRF (an apparent @5 gain under a weaker search turned
out to be optimiser noise). The genuine win comes from the **meta-ranker** (0.752 vs
0.745), which exploits the agreement×S2 interaction *nonlinearly* — the one thing
additive RRF can't express. So the defensible pipeline is *consensus-aware RRF →
meta-ranker → top-5*, with η/ρ ready to activate if real GPT/Claude agreement is
genuinely extra-predictive.

## Incorporating the agent output and S2 graph signals (`integrate.py`)

Both upstream outputs enter through the `Query` contract:

- **Agent pool → fusion.** `pool_to_query(query_id, agent_pool, paper_text, gt_texts)`
  builds a `Query`: candidate ids, a `retrieval` ranking (similarity to the paper —
  the agent is one fusion source), provenance in `candidate_sources`, and the eval
  `relevance`. GPT/Claude lists slot into `rankings` when they exist.
- **S2 graph signals → fusion** (Naz's `GraphSignalExtractor` → `graph_signal_fn`):
  written onto `Query.features`, they enter three ways — (a) an extra RRF source
  `add_horizon_source`, (b) the RRF feature-bias term `make_rrf_rank_fn(beta=...)`,
  (c) learned features `MetaRanker(extra_feature_keys=HORIZON_KEYS)` — and render
  as Direction-Card horizon badges automatically. In the demo an informative
  signal lifts test MRR 0.31→0.45 (RRF source) and 0.41→0.64 (meta-ranker feature).

## Candidate-harvesting agent (the one agentic component)

`agent.py` is the independent AI agent of ResearchHorizon. Given a masked paper,
it runs tools in a loop — read the citation neighbourhood via Naz's S2 client,
harvest neighbours' **keywords + realized future directions**, dedup and add them
to the candidate pool — and **decides for itself when to stop**: when newly
harvested candidates stop improving coverage of the paper's open `key_gaps` (the
coverage/recall harness). The tool-call trace differs per paper (search a gap,
expand a citation, …), which is what makes it an agent rather than a workflow.
Everything downstream (rank → fuse → explain → MRR) stays deterministic — the
agent only *gathers*; the metric still decides the order.

```python
from rank_fusion import CandidateHarvestAgent, AgentConfig, evaluate_pool
from s2_tools import EnrichedS2Client, LocalCorpusS2Client   # repo root
agent = CandidateHarvestAgent(EnrichedS2Client(api_key=...))  # or LocalCorpusS2Client (offline)
res = agent.run(seed_corpus_id, gaps=[g["description"] for g in key_gaps])
# res["pool"] -> candidates; res["trace"] -> tool-loop; res["stop_reason"]
```

The decision policy is pluggable (`policy=` argument): the default is an
autonomous heuristic controller; swap in an LLM (Claude tool-use) to make the
"decide what to do next" step LLM-driven. The **stop** signal uses `key_gaps`
(inference-safe); the masked future-work is used only by `evaluate_pool` to score
the harvested pool afterwards (no leakage).

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
