---
marp: true
title: ResearchHorizon
paginate: true
size: 16:9
html: true
---

<style>
:root {
  --navy:#1f3a5f; --blue:#2767b0; --teal:#1f9e9e; --orange:#e8743b;
  --ink:#2d3748; --muted:#5a6b7b; --bg:#ffffff; --soft:#eef4fb;
}
section {
  font-family:'Helvetica Neue',Arial,sans-serif; font-size:20px;
  color:var(--ink); background:var(--bg); padding:46px 58px;
}
h1 { font-size:36px; color:var(--navy); margin:0 0 4px 0; }
h1::after { content:""; display:block; width:74px; height:5px;
  background:var(--teal); border-radius:3px; margin-top:8px; }
h2,h3 { color:var(--blue); }
strong { color:var(--navy); }
a, .accent { color:var(--teal); }
small { color:var(--muted); }
ul { margin-top:6px; } li { margin:3px 0; }
code { background:var(--soft); color:#143; padding:1px 5px; border-radius:4px; font-size:.92em; }
section::after { color:#9bb0c4; }

/* cards + flow */
.row { display:flex; gap:18px; }
.card { background:var(--soft); border-radius:10px; padding:12px 18px;
  box-shadow:0 2px 6px rgba(31,58,95,.10); border-top:4px solid var(--blue); flex:1; }
.card b { color:var(--navy); }
.card.src { border-top-color:var(--teal); }
.card.fuse { border-top-color:var(--blue); background:#e9f1fb; }
.card.safe { border-top-color:var(--orange); background:#fdf1ea; }
.card.eval { border-top-color:var(--navy); }
.arrow { text-align:center; color:var(--teal); font-size:24px; line-height:1; margin:3px 0; font-weight:bold; }

/* callout */
.callout { background:var(--navy); color:#fff; border-radius:12px; padding:14px 22px;
  font-style:italic; box-shadow:0 3px 10px rgba(31,58,95,.25); }
.callout strong { color:#ffd9a8; }
.tag { background:var(--teal); color:#fff; border-radius:20px; padding:2px 12px; font-size:.8em; font-weight:bold; }

/* tables */
table { border-collapse:collapse; width:100%; font-size:18px; box-shadow:0 2px 6px rgba(31,58,95,.08); }
th { background:var(--navy); color:#fff; padding:8px 12px; text-align:left; }
td { padding:7px 12px; border-bottom:1px solid #e3eaf2; }
tr:nth-child(even) td { background:var(--soft); }
.big { color:var(--orange); font-weight:bold; }

/* feature grid */
.grid { display:grid; grid-template-columns:1fr 1fr; gap:10px 26px; margin-top:8px; }
.feat { border-left:4px solid var(--teal); padding:4px 12px; }
.feat b { color:var(--navy); }

.lead { font-size:23px; color:var(--navy); font-weight:bold; }
.center { text-align:center; }
</style>

<!-- _paginate: false -->

<div class="center" style="margin-top:40px;">

# ResearchHorizon

### <span class="accent">Multi-agent future-research-direction recommendation, calibrated to MRR</span>

</div>

**The problem.** >7M papers/year — researchers struggle to identify *what to study next*.
Given a paper, what are the most promising directions it leaves open?

**The task** <span class="tag">MLRC-Bench framing</span> — mask a paper's *"Future Work"* section
→ recommend a **ranked list** of candidate directions. The held-out text is the ground
truth; the system is optimized for **Mean Reciprocal Rank (MRR)**.

<div class="callout">
"Mask the future-work; have <strong>5 LLM agents + a Semantic-Scholar knowledge graph</strong>
independently rank candidate directions; fuse them into one list <strong>calibrated to dev
MRR</strong>; emit the best-of {agents, KG, fusion} — <strong>never worse than the best single ranker.</strong>"
</div>

---

# Complete pipeline

<div class="row">
  <div class="card src"><b>① 5 LLM Agents</b><br><small>gpt5nano · qwen · mixtral · llama17 · llama70 — each emits a ranked <code>future_directions</code> list → <b>rank by MRR, keep top 2</b></small></div>
  <div class="card src"><b>② Semantic Scholar KG</b><br><small>resolve CorpusId → <b>pull connected papers</b> (refs + citations) → entity co-occurrence → Agent-C / white-space S2 source</small></div>
</div>

<div class="arrow">▼ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cluster the 3 sources (llm1 + llm2 + s2) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ▼</div>

<div class="card fuse"><b>③ Rank Fusion — <code>consensus.py</code></b><br><small>weighted RRF + LLM-agreement(η) + agreement×S2-support(ρ) · <b>tuned on dev MRR@3</b> (multi-restart coordinate ascent, warm-started from plain RRF)</small></div>

<div class="arrow">▼</div>

<div class="card safe"><b>④ CV Best-of Fallback — <code>select_best_strategy_cv</code></b><br><small>k-fold CV on dev · emit the <b>simplest strategy within margin</b> of the best → <b>provably ≥ best single source</b>, robust to the dev/test gap</small></div>

<div class="arrow">▼</div>

<div class="card eval"><b>⑤ Evaluation</b><br><small>MRR@3 vs masked GT · MLRC benchmark vs random baseline · <b>Direction Cards</b> (attribution · ablation · logic · faithfulness)</small></div>

---

# Stages 1–2: candidate sources

<div class="row">
<div style="flex:1.1;">

**① LLM agents** <span class="tag">run_llm_selection.py</span>
5 LLMs each emit ranked `future_directions`; matched to masked GT, scored on the
**same 806 papers** (fair: misses = 0) → keep **top 2**.

| agent | MRR | agent | MRR |
|---|---:|---|---:|
| **llama17** | **0.524** | mixtral | 0.347 |
| **llama70** | **0.520** | gpt5nano | 0.267 |
| qwen | 0.348 | | |

</div>
<div style="flex:1; padding-left:8px;">

**② Semantic Scholar** <span class="tag">live S2 API</span>
Resolve the seed's CorpusId, **pull connected papers** (refs + citations + abstracts),
build an **entity co-occurrence graph**.

<div class="feat" style="margin-top:10px;"><b>Agent C</b> ranks neighbour papers by graph signals (recency/novelty/relevance)</div>
<div class="feat"><b>White-space</b> generates entities <i>adjacent to the seed but unexplored</i> (§7,§9.7)<br><small>score = adjacency · 1/(1+support) · gap_boost</small></div>

<small>*Live, SEXIT paper: pulled 23 connected papers incl. "SRH needs of immigrant adolescents" — the paper's documented gap.*</small>

</div>
</div>

---

# Stage 3: rank fusion + safety net

**Consensus RRF** <span class="tag">consensus.py</span> — *"when both LLMs rank a direction near the top it's strong evidence; the S2 graph supports it."*

<div class="callout" style="font-style:normal; font-size:19px;">
S(c) = Σ <b>w<sub>m</sub></b>/(k+r<sub>m</sub>(c)) &nbsp;+&nbsp; <b>η₁</b>·A12_top1 + <b>η₃</b>·A12_top3 &nbsp;+&nbsp; <b>ρ</b>·A12_top3·R<sub>s2</sub>
</div>

<br>

<div class="row">
<div class="card" style="border-top-color:var(--blue);"><b>Tuned on dev MRR@3</b><br><small>multi-restart coordinate ascent, <b>warm-started from plain RRF</b> → can never lose to RRF on dev. The 3 sources are <b>clustered</b> first, so differently-phrased versions of one direction accumulate per-source ranks.</small></div>
<div class="card safe"><b>CV best-of fallback</b> <span class="tag">the robustness piece</span><br><small>k-fold CV <b>re-fits each fusion per fold</b> → a fusion that only wins by <i>overfitting</i> dev gets a lower CV score; emit the <b>simplest</b> strategy within margin (single < RRF < consensus) → emits the best single LLM, not an overfit fusion.</small></div>
</div>

---

# Explainability — Direction Cards <span class="tag" style="font-size:.55em; vertical-align:middle;">§6</span>

A ranked list isn't enough — a scientist must **trust and verify** each suggestion.
Every output is an auditable **Direction Card** (`explain.py`):

<div class="grid">
<div class="feat"><b>🎯 Attribution</b><br><small>exact per-source RRF decomposition (sums to the score) / linear-SHAP</small></div>
<div class="feat"><b>🔬 Causal ablation</b><br><small>leave-one-signal-out: <i>"remove S2 → drops to rank 7"</i></small></div>
<div class="feat"><b>⚖️ Logical reasoning</b><br><small>defeasible argumentation → <i>recommend / caveat / abstain</i> + trace</small></div>
<div class="feat"><b>📊 Calibration & abstention</b><br><small>calibrated confidence (ECE); <i>"insufficient evidence"</i> when thin</small></div>
<div class="feat"><b>✅ Faithfulness check</b><br><small>the shown rationale must match the <i>actual</i> ranking drivers</small></div>
<div class="feat"><b>📎 Evidence grounding</b><br><small>links the direction to the <code>key_gaps</code> it addresses (quoted spans)</small></div>
</div>

<div class="callout" style="margin-top:14px; font-size:18px;">
The <strong>model-risk-aware layer</strong>: borrowed from interpretability (SHAP; Ivanitskiy,
<i>Mechanistic Interpretability</i>) — the explanation reflects the <strong>real computation</strong>, not a post-hoc story.
</div>

---

# Why it works — results & rigor

<div class="row">
<div style="flex:1.2;">

**MLRC benchmark** <small>(75 test papers, MRR@3)</small>

| system | MRR@3 | Δ baseline | %ceiling |
|---|---:|---:|---:|
| random baseline | 0.07 | — | 0% |
| **best LLM (llama70/17)** | <span class="big">0.33</span> | **+340%** | **27%** |
| consensus RRF | 0.32 | +339% | 27% |
| S2 (Agent C) alone | 0.02 | −71% | — |
| oracle / recall ceiling | 1.00 | | 100% |

</div>
<div style="flex:1; padding-left:6px;">

<div class="feat"><b>+340% over baseline</b><br><small>candidate <b>recall ≈ 100%</b> (oracle=1.0) → the answer is in the pool; the open problem is <i>ranking</i></small></div>
<div class="feat"><b>Metric decides, not the LLM</b><br><small>CV best-of emits the best generaliser — <i>never &lt; best single source</i></small></div>
<div class="feat"><b>Honest finding</b><br><small>fusion ≈ the best single LLM today; the 5 LLMs are <i>correlated</i> + S2 is near-random → needs diverse LLM families (GPT+Claude) + a stronger S2</small></div>
<div class="feat"><b>Rigor</b><br><small>temporal split · bootstrap CIs · Bonferroni · k-fold CV · <b>53 passing tests</b></small></div>

</div>
</div>

---

# References

<div class="row">
<div style="flex:1;">

- **Cormack et al. 2009** — *Reciprocal Rank Fusion*
- **Sun et al. 2023** — *RankGPT* (listwise)
- **Qin et al. 2023** — *PRP* (pairwise)
- **Cao et al. 2007** — *ListNet* · Burges — *LambdaMART*
- **Cohan et al.** — *SPECTER2 / SciNCL* (embeddings)
- **Lundberg & Lee 2017** — *SHAP* · Ivanitskiy — *Mech. Interp.*

</div>
<div style="flex:1;">

- **Si, Yang & Hashimoto 2024** — *Can LLMs Generate Novel Research Ideas?* (overgenerate→rerank; **S2-API agent**)
- **Baek et al. 2024** — *ResearchAgent* (citation graph + **knowledge graph**)
- **Wang et al. 2023** — *SciMON* (novelty)
- **MLRC-Bench** — baseline-relative scoring
- **Semantic Scholar / OpenAlex** — connected papers

</div>
</div>

<div class="callout" style="margin-top:16px; font-size:17px;">
One <code>rank_fusion/</code> package — MRR harness · RRF + consensus fusion · learned rankers
(pointwise/pairwise/listwise/soft-MRR) · entity + white-space S2 · autonomous candidate agent ·
Direction-Card explainability · CV best-of selection · MLRC benchmark.
</div>
