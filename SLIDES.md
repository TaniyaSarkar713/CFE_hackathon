---
marp: true
title: ResearchHorizon
paginate: true
size: 16:9
---

<style>
section { font-size: 21px; padding: 40px 50px; }
h1 { font-size: 34px; color: #1f4e79; }
h3 { font-size: 22px; }
pre, code { font-size: 15px; }
table { font-size: 18px; }
blockquote { font-size: 19px; }
li { margin: 2px 0; }
</style>

# ResearchHorizon
### Multi-agent future-research-direction recommendation, calibrated to MRR

**The problem.** >7M papers/year — researchers struggle to identify *what to study
next*. Given a paper, what are the most promising directions it leaves open?

**Our task (MLRC-Bench framing).** Mask a paper's *"Future Work"* section → recommend a
**ranked list** of candidate future directions. The held-out future-work text is the
ground truth; the whole system is optimized for **Mean Reciprocal Rank (MRR)**.

**Design principle.** Several agents independently propose ranked directions; the
**final ranking is decided by validation MRR via rank fusion — not by any single LLM's
judgment** — and every recommendation ships with an evidence-grounded rationale.

> *"Mask the future-work; have 5 LLM agents + a Semantic-Scholar knowledge graph
> independently rank candidate directions; fuse them into one list calibrated to dev
> MRR; emit the best-of {agents, KG, fusion} — never worse than the best single ranker."*

MLRC-Bench spirit: **objective metric · masked GT · real ablations · compute-aware · a baseline to beat.**

---

# Complete pipeline

```
 5 LLM AGENTS                          SEMANTIC SCHOLAR (knowledge graph)
 gpt5nano·qwen·mixtral·llama17·llama70 Agent C: resolve CorpusId -> pull CONNECTED
   │ each emits a ranked                papers (refs + citations) -> entity
   │ future_directions list             co-occurrence -> WHITE-SPACE generator:
   ▼                                     "adjacent-but-unexplored" bridge directions
 STAGE 1  rank agents by MRR           STAGE 2  S2 source (entity-grounded)   (§7, §9.7)
 -> keep TOP 2  (llama70, llama17)        │
        └────────────────┬────────────────┘
                         ▼
 STAGE 3  RANK FUSION  (consensus.py)
   build_consensus_query: cluster the 3 sources (llm1 + llm2 + s2)
   weighted RRF  +  LLM-agreement(η)  +  agreement×S2-support(ρ)
   tuned on dev MRR@3 (multi-restart coordinate ascent, warm-started from RRF)
                         │
                         ▼  CV best-of fallback  (select_best_strategy_cv)
   k-fold CV on dev -> emit the SIMPLEST strategy within margin of the best
   -> provably >= best single source; robust to the dev/test gap
                         │
                         ▼
 STAGE 4  EVALUATION   MRR@3 vs masked GT · MLRC benchmark vs random baseline
   + DIRECTION CARDS (§6): attribution · ablation · logical reasoning · faithfulness
```

---

# Stages 1–2: candidate sources

**Stage 1 — LLM agents (`run_llm_selection.py`).** 5 LLMs each emit a ranked
`future_directions` list per paper. We match each against the masked GT and score corpus
**MRR on the same 806 papers** (fair denominator: misses = 0), then keep the **top 2**.

| agent | MRR | | agent | MRR |
|---|---:|---|---|---:|
| **llama17** | **0.524** | | mixtral | 0.347 |
| **llama70** | **0.520** | | gpt5nano | 0.267 |
| qwen | 0.348 | | | |

**Stage 2 — Semantic Scholar white-space.** Resolve the seed's CorpusId, **pull
connected papers** (references + citations, with abstracts), build an **entity
co-occurrence graph**, and *generate* directions from entities **adjacent to the seed
but not in it** — the unexplored edges (§9.7), boosted near the open `key_gaps`:

`score(e) = adjacency(e) · 1/(1+support(e)) · gap_boost`  ← adjacency *gates* novelty (§7)

*Live, on the SEXIT paper: pulled 23 connected papers incl. "SRH needs of immigrant
adolescents" — directly the paper's documented gap.*

---

# Stage 3: rank fusion + safety net

**Consensus RRF (`consensus.py`)** — encodes *"when both LLMs rank the same direction
near the top it's strong evidence; the S2 graph supports it"*:

`S(c) = Σ wₘ/(k+rₘ(c)) + η₁·A12_top1 + η₃·A12_top3 + ρ·A12_top3·R_s2`

tuned on **dev MRR@3** by multi-restart coordinate ascent, **warm-started from plain RRF**
(so it can never lose to RRF on dev). Candidates from the 3 sources are **clustered**
first, so differently-phrased versions of one direction accumulate per-source ranks.

**CV best-of fallback (`select_best_strategy_cv`)** — the key robustness piece:
- k-fold CV on dev, **re-fitting each fusion per fold** → a fusion that only wins by
  *overfitting* dev gets a lower CV score;
- emit the **simplest** strategy within a margin of the best CV MRR (single < RRF < consensus).

→ **Provably ≥ the best single source on dev**, and it correctly emits the best single LLM
instead of an overfit fusion that loses on test.

---

# Explainability — Direction Cards (§6)

A ranked list isn't enough — a scientist must **trust and verify** each suggestion.
Every output direction is an auditable **Direction Card** (`explain.py`):

- **Attribution** — exact per-source RRF decomposition (sums to the score) / linear-SHAP
- **Causal ablation** — leave-one-signal-out: *"remove S2 → drops to rank 7"*
- **Logical reasoning** — defeasible argumentation → *recommend / caveat / abstain* + trace
- **Calibration & abstention** — calibrated confidence (ECE); *"insufficient evidence"* when thin
- **Faithfulness check** — the shown rationale must match the *actual* ranking drivers
- **Evidence grounding** — links the direction to the `key_gaps` it addresses (quoted spans)

> Borrowed from interpretability (SHAP; Ivanitskiy *Mechanistic Interpretability*):
> the explanation reflects the real computation, not a post-hoc story.

This is the **model-risk-aware layer** — the output is *why*, not just *what*.

---

# Why it works — results & rigor

**MLRC benchmark (75 test papers, MRR@3) — improvement over a random baseline:**

| system | MRR@3 | Δ vs baseline | % of ceiling |
|---|---:|---:|---:|
| random baseline | 0.116 | — | 0% |
| **best LLM (llama70/17)** | **0.325** | **+180%** | **24%** |
| consensus RRF | 0.325 | +180% | 24% |
| S2 white-space (alone) | 0.007 | −94% | — |
| oracle / recall ceiling | 1.000 | | 100% |

- **+180% over baseline**; candidate **recall ≈ 100%** (oracle = 1.0) → the answer is in the
  pool; the open problem is *ranking* (we capture 24% of the gap).
- **The metric decides, not the LLM** · **CV best-of** emits the best generaliser (never < best single).
- **Honest finding:** fusion ≈ the best single LLM today; the S2 white-space source is *below
  random* and needs a biomedical-NER `entity_fn` + live connected papers to earn its weight.
- **Rigor:** temporal split · bootstrap CIs · Bonferroni · k-fold CV · ablation tables · **53 passing tests**.

---

# References

- **Cormack et al. 2009** — *Reciprocal Rank Fusion* (fusion backbone)
- **Sun et al. 2023** — *RankGPT* (listwise) · **Qin et al. 2023** — *PRP* (pairwise)
- **Cao et al. 2007** — *ListNet* · **Burges** — *RankNet / LambdaRank / LambdaMART*
- **Cohan et al.** — *SPECTER2 / SciNCL* (scientific embeddings, the §1 match function)
- **Lundberg & Lee 2017** — *SHAP* · **Ivanitskiy** — *Mechanistic Interpretability* (Direction Cards)
- **Si, Yang & Hashimoto 2024** — *Can LLMs Generate Novel Research Ideas?* (overgenerate→dedup→rerank; pairwise > pointwise; **S2-API agent**)
- **Baek et al. 2024** — *ResearchAgent* (citation graph + entity-centric **knowledge graph**)
- **Wang et al. 2023** — *SciMON* (novelty optimization) · **MLRC-Bench** (baseline-relative scoring)
- **Semantic Scholar Graph API / OpenAlex** — citation neighborhood ("connected papers")

*One `rank_fusion/` package: MRR harness · RRF + consensus fusion · learned rankers
(pointwise/pairwise/listwise/soft-MRR) · entity + white-space S2 · autonomous candidate
agent · Direction-Card explainability · CV best-of selection · MLRC benchmark.*
