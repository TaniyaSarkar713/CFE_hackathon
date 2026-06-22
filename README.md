# Next Future Direction Suggestion for Research Scientists

An agentic system that generates, reasons about, and ranks future research directions for scientists — built on the **Next Product Recommendation** task from [MLRC-BENCH](https://github.com/mlrc-bench), reimagined for scientific literature discovery.

---

## Overview

Researchers often struggle to identify what to study next. This system addresses that by orchestrating multiple AI agents over PubMed papers to produce ranked, evidence-grounded future direction suggestions.

The pipeline works in three stages:

1. **Summarization Agent** — Parses PubMed papers and generates structured summaries (topic, themes, findings, and key gaps).
2. **Reasoning Agents** — Two independent agents (OpenAI and Claude) perform deep reasoning over the summaries to propose candidate future directions.
3. **Ranking Engine** — A rank fusion algorithm scores and orders the generated directions using entity-based relevance and Mean Reciprocal Rank (MRR), aligned with the MLRC-BENCH evaluation protocol.

An optional **Connected Papers Agent** enriches the ranking by mapping each paper to its citation neighborhood, surfacing directions that bridge clusters of related work.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PubMed Papers                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Agent 1: Summarizer  │
               │  (Structured JSON)    │
               └───────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
  ┌─────────────────────┐  ┌─────────────────────┐
  │  Agent 2: Reasoner  │  │  Agent 3: Reasoner  │
  │  (OpenAI)           │  │  (Claude)           │
  └──────────┬──────────┘  └──────────┬──────────┘
             │                        │
             └────────┬───────────────┘
                      ▼
         ┌────────────────────────┐
         │  Rank Fusion Engine    │
         │  (Entity Relevance +  │
         │   MRR Scoring)        │
         └────────────┬──────────┘
                      │
                      ▼  (optional)
         ┌────────────────────────┐
         │  Connected Papers      │
         │  Agent (enrichment)    │
         └────────────┬──────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Ranked Future         │
         │  Directions Output     │
         └────────────────────────┘
```

---

## Structured Summary Schema

Agent 1 produces one JSON summary per paper. This schema is the contract between the summarizer and the downstream reasoning agents.

```json
{
  "file": "<PubMed ID>",
  "token_count": 6926,
  "model": "<model used>",
  "timestamp": "2025-11-21T21:39:29.682619Z",
  "topic_and_scope": "High-level description of what the paper covers and the angle it takes.",
  "main_themes": [
    "Theme 1",
    "Theme 2",
    "Theme 3"
  ],
  "main_findings": [
    "Quantitative or qualitative finding 1.",
    "Quantitative or qualitative finding 2."
  ],
  "key_gaps": [
    {
      "description": "A specific gap identified in the research.",
      "status": "explicit | implicit"
    }
  ]
}
```

The `status` field in `key_gaps` distinguishes between gaps the authors state directly (`explicit`) and gaps inferred by the summarization agent from what the paper leaves unaddressed (`implicit`).

---

## Ranking Methodology

The system adapts the **Mean Reciprocal Rank (MRR)** metric from the MLRC-BENCH Next Product Recommendation task to evaluate and order future directions.

### Entity Extraction

Named entities are extracted from three sources per paper: the title, the abstract, and the structured summary. These entities serve as the relevance signal for ranking.

### Rank Fusion

Each reasoning agent produces its own ordered list of candidate future directions. A rank fusion algorithm combines these lists, weighted by entity overlap:

- **High relevance** — Directions whose language overlaps significantly with extracted entities are promoted to the top of the fused ranking.
- **Low relevance** — Directions with minimal entity overlap are placed lower or grouped into a separate exploratory bundle, still surfaced but clearly distinguished from the core suggestions.

### Connected Papers Integration (Optional)

An additional agent can query the Connected Papers graph for a given PubMed paper, retrieving its citation neighborhood. Directions that align with trends across connected papers receive a ranking boost, rewarding suggestions that bridge or extend existing research clusters.

---

## Project Structure

```
├── README.md
├── agents/
│   ├── summarizer/          # Agent 1 — PubMed paper summarization
│   ├── reasoner_openai/     # Agent 2 — Future direction generation (OpenAI)
│   ├── reasoner_claude/     # Agent 3 — Future direction generation (Claude)
│   └── connected_papers/    # Optional — citation graph enrichment
├── ranking/
│   ├── entity_extraction.py # Entity extraction from titles, abstracts, summaries
│   ├── rank_fusion.py       # Rank fusion with entity-based relevance scoring
│   └── mrr_evaluation.py    # MRR computation for evaluation
├── data/
│   ├── raw/                 # Raw PubMed paper files
│   ├── summaries/           # Generated structured summaries (JSON)
│   └── directions/          # Generated future direction candidates
├── configs/                 # Agent prompts, model configs, API keys
├── evaluation/              # Evaluation scripts and benchmark results
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- API keys for OpenAI and Anthropic (Claude)
- Access to PubMed data (via API or local corpus)

### Installation

```bash
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
cp configs/.env.example configs/.env
# Add your API keys to configs/.env
```

### Running the Pipeline

```bash
# Step 1 — Summarize papers
python agents/summarizer/run.py --input data/raw/ --output data/summaries/

# Step 2 — Generate future directions (both agents)
python agents/reasoner_openai/run.py --summaries data/summaries/ --output data/directions/openai/
python agents/reasoner_claude/run.py --summaries data/summaries/ --output data/directions/claude/

# Step 3 — Rank and fuse
python ranking/rank_fusion.py \
  --directions data/directions/ \
  --summaries data/summaries/ \
  --output results/ranked_directions.json

# Step 4 (optional) — Enrich with Connected Papers
python agents/connected_papers/run.py --input results/ranked_directions.json --output results/enriched_directions.json

# Evaluate
python evaluation/evaluate.py --predictions results/ranked_directions.json
```

---

## Evaluation

The system is evaluated using **Mean Reciprocal Rank (MRR)**, consistent with the MLRC-BENCH Next Product Recommendation task. MRR measures how high the first relevant future direction appears in the ranked list:

```
MRR = (1 / |Q|) × Σ (1 / rank_i)
```

where `rank_i` is the position of the first relevant direction for query `i`. Higher MRR indicates that the system consistently surfaces useful directions near the top of the list.

---

## Relationship to MLRC-BENCH

This project builds on the **Next Product Recommendation** task from MLRC-BENCH. The conceptual mapping is:

| MLRC-BENCH Concept     | This Project                              |
|------------------------|-------------------------------------------|
| Product                | Future research direction                 |
| User purchase history  | Paper's themes, findings, and key gaps    |
| Product catalog        | Space of possible research directions     |
| Recommendation         | Ranked future direction suggestion        |
| Evaluation (MRR)       | MRR over ranked direction lists           |

---

## References & Related Work

Papers and codebases in [`References/`](References/) that inform the design. The
last two columns say what we take from each and which workstream it lands in
(**Fusion/Explain** = Taniya, Agent 3 · **Candidates/Agents** = Nourah, Agents
1–2 · **Citation graph** = Naz, §7).

| Reference | What it is | Key ideas we use | Where it lands |
|-----------|------------|------------------|----------------|
| **Cao et al., 2007** — *Learning to Rank: From Pairwise to Listwise Approach* (ListNet) · `References/Listwise ranking loss.pdf` | Foundational learning-to-rank paper. Defines **permutation probability** and **top-one probability**, and the **ListNet** loss = cross-entropy between the top-one softmax distributions of predicted vs. ground-truth scores. Shows listwise > pairwise because it optimizes the whole list, not isolated pairs. | The canonical **listwise loss** for the meta-ranker, completing the pointwise → pairwise → listwise menu (we already have pointwise logistic, pairwise RankNet/BPR, and a soft-MRR surrogate). | Fusion/Explain |
| **Si, Yang & Hashimoto, 2024** — *Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers* · `References/Can LLMs Generate Novel Research Ideas?.pdf` | Stanford study: 100+ experts blind-review human vs. LLM research ideas. LLM ideas judged **more novel** but slightly less feasible. Their agent does **overgenerate (4000 seeds) → dedup → rerank**. Key findings: LLMs are **poorly calibrated for pointwise scoring but good at pairwise** (Claude-3.5 71% vs GPT-4o 61%); LLMs **cannot reliably self-evaluate** and **lack diversity** when scaled. | (1) Empirical justification for **pairwise ranking** and for the core principle *the metric decides, not a single LLM judge*; (2) **Swiss-system tournament** to aggregate pairwise comparisons; (3) **dedup recipe** (all-MiniLM-L6-v2 cosine, threshold 0.8); (4) lack-of-diversity → motivates **MMR** diversity re-ranking; (5) richer idea template (problem/method/plan/test/fallback). | Fusion/Explain + Candidates/Agents |
| **Baek et al., 2024** — *ResearchAgent: Iterative Research Idea Generation over Scientific Literature with LLMs* · `References/ResearchAgent-...pdf` | LLM ideation system: starts from a core paper, expands over the **citation graph**, augments with an **entity-centric knowledge store** (concepts mined from co-occurrence across papers), and iterates via multiple **ReviewingAgents** scored on clarity / relevance / originality / feasibility / significance. | Turnkey recipe for the **citation-graph literature survey** and **entity-based relevance**; the reviewing-agent criteria become extra **Direction-Card badges / reasoning predicates** and meta-ranker features. | Citation graph + Explain |
| **Wang, Downey, Ji & Hope, 2023** — *SciMON: Scientific Inspiration Machines Optimized for Novelty* · `References/SciMON/` (code + `scimon.pdf`) | Generates literature-grounded ideas and **optimizes for novelty** via *iterative novelty boosting* (retrieve similar prior work → prompt to increase distance from it). Three knowledge modules: KG (IE entities), CT (citation network), SN (semantic neighbors). Includes a **biomedical** pipeline. | Operationalizes the **§7 novelty signal** (novelty = adjacency-gated distance from neighbor literature); its **biomedical** modules match our PubMed corpus; SciBERT semantic eval reinforces **SPECTER2/SciBERT** for the MRR match function. | Citation graph + Candidates/Agents |
| **Ivanitskiy, 2026** — *Mechanistic Interpretability* (lecture) · `References/ivanitskiy-interp.pdf` | Survey of interpretability: input attribution (**LIME/SHAP**, with game-theoretic guarantees), **saliency** ("what drove the decision"), causal interventions (**ablation / activation patching**), and the Goodhart/proxy-objective warning. | The blueprint for the **explainability layer** (already built): exact additive attribution, causal **ablation probe**, faithfulness, and abstention as the model-risk-aware response to Goodhart. | Fusion/Explain (done) |

**Status:** Ivanitskiy is already incorporated (the `rank_fusion/explain.py` layer).
The highest-value next addition from this set is **ListNet** (Cao et al.) as a
fourth meta-ranker option. Si et al. mainly provides design justification +
concrete recipes (dedup threshold, Swiss tournament, MMR). ResearchAgent and
SciMON are primarily upstream recipes (candidate generation, citation graph,
novelty) that feed this project through the `Query` contract. See
[`WORKSTREAM_RANK_FUSION.md`](WORKSTREAM_RANK_FUSION.md) for the dependency map.

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- [MLRC-BENCH](https://github.com/mlrc-bench) for the benchmark framework and evaluation methodology
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) for the research paper corpus
- [Connected Papers](https://www.connectedpapers.com/) for the citation graph exploration concept
