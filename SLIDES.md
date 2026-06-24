---
marp: true
title: ResearchHorizon
paginate: true
---

# ResearchHorizon
### Multi-agent future-research-direction recommendation, calibrated to MRR

**The problem.** >7M papers/year — researchers struggle to identify *what to study
next*. Reading a paper, what are the most promising directions it leaves open?

**Our task (MLRC-Bench framing).** Mask a paper's *"Future Work"* section. Recommend
a **ranked list** of candidate future directions. The held-out future-work text is the
ground truth; the whole system is optimized for **Mean Reciprocal Rank (MRR)**.

**Design principle.** Multiple agents independently propose ranked directions; the
**final ranking is decided by validation MRR via rank fusion — not by any single
LLM's judgment** — and every recommendation ships with an evidence-grounded rationale.

> *"We mask a paper's future-work, have multiple LLM agents + a Semantic-Scholar
> knowledge graph independently rank candidate directions, fuse them into one list
> calibrated directly to dev MRR, and emit the best-of {agents, KG, fusion} — never
> worse than the best single ranker."*

MLRC-Bench spirit kept: **objective metric · masked ground truth · real ablations ·
compute-aware · a baseline to beat.**

---

# Complete pipeline

```
 5 LLM AGENTS                 SEMANTIC SCHOLAR (KG)
 gpt5nano · qwen · mixtral    Agent C: resolve CorpusId -> pull connected
 llama17  · llama70           papers (refs+citations) -> entity co-occurrence
   │  each emits a ranked       │  -> WHITE-SPACE generator: "adjacent-but-
   │  future_directions list    │     unexplored" bridge directions (§7,§9.7)
   ▼                            ▼
 STAGE 1: rank agents by MRR    STAGE 2: S2 source (entity-grounded)
 -> keep TOP 2 (llama70,17)        │
   └───────────────┬──────────────┘
                   ▼
 STAGE 3  RANK FUSION  (consensus.py)
   build_consensus_query: cluster the 3 sources (llm1+llm2+s2)
   weighted RRF + LLM-agreement(η) + agreement×S2-support(ρ)
   tuned on dev MRR@3  (multi-restart coord ascent, warm-started from RRF)
        │
        ▼  CV best-of fallback  (select_best_strategy_cv)
   k-fold CV on dev; emit the SIMPLEST strategy within margin of the best
   -> never worse than the best single source; robust to the dev/test gap
        │
        ▼
 STAGE 4  EVALUATION
   MRR@3 vs masked GT · MLRC benchmark vs random baseline · oracle ceiling
   + DIRECTION CARDS: attribution · causal ablation · logical reasoning ·
     calibration/abstention · faithfulness · evidence grounding (§6)
```

---

# Why it works — results & rigor

**MLRC benchmark (75 test papers, MRR@3):**

| system | MRR@3 | Δ vs baseline | % of ceiling |
|---|---:|---:|---:|
| random baseline | 0.116 | — | 0% |
| **best LLM (llama70/17)** | **0.325** | **+180%** | **24%** |
| consensus RRF | 0.325 | +180% | 24% |
| S2 white-space (alone) | 0.007 | −94% | — |
| oracle / recall ceiling | 1.000 | | 100% |

**Why the design holds up:**
- **Metric, not LLM, decides** — RRF/consensus tuned on dev MRR; no single model is trusted.
- **CV best-of safety net** — emits the best of {agents, KG, fusion}; *provably ≥ best single source*, and CV makes it robust to the dev/test gap (it correctly emits the best LLM, not an overfit fusion).
- **Candidate recall is ~100%** (oracle = 1.0) — the answer is in the pool; the open problem is *ranking* (we capture 24% of the gap).
- **Explainable** — every direction is a Direction Card (attribution + causal ablation + defeasible logical reasoning + abstention), so it's auditable, not a black box.

**Honest finding:** fusion ≈ the best single LLM today; the S2 white-space source is below
random and needs a biomedical-NER `entity_fn` + live connected papers to earn its weight.
**52→53 tests passing; temporal split, bootstrap CIs, Bonferroni, k-fold CV throughout.**

---

# References

- **Cormack et al. 2009** — *Reciprocal Rank Fusion* (the fusion backbone)
- **Sun et al. 2023** — *RankGPT* (listwise LLM re-ranking) · **Qin et al. 2023** — *PRP* (pairwise)
- **Cao et al. 2007** — *ListNet* (listwise loss) · **Burges** — *RankNet/LambdaMART*
- **Cohan et al.** — *SPECTER2 / SciNCL* (scientific embeddings for the §1 match function)
- **Lundberg & Lee 2017** — *SHAP* · Ivanitskiy — *Mechanistic Interpretability* (Direction Cards)
- **Si, Yang & Hashimoto 2024** — *Can LLMs Generate Novel Research Ideas?* (overgenerate→dedup→rerank; pairwise>pointwise; **S2-API agent**)
- **Baek et al. 2024** — *ResearchAgent* (citation graph + entity-centric **knowledge graph**)
- **Wang et al. 2023** — *SciMON* (novelty optimization)
- **MLRC-Bench** — objective metric, masked test, baseline-relative scoring
- **Semantic Scholar Graph API / OpenAlex** — citation neighborhood ("connected papers")

*Code: one `rank_fusion/` package — MRR harness, RRF + consensus fusion, learned
rankers (pointwise/pairwise/listwise/soft-MRR), entity + white-space S2, autonomous
candidate agent, Direction-Card explainability, CV best-of selection, MLRC benchmark.*
