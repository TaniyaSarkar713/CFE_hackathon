# Running ResearchHorizon on another machine

Everything needed to set up and run the rank-fusion pipeline on a fresh machine,
plus a ready-to-paste prompt for **Claude Code** to do it for you.

---

## TL;DR

```bash
git clone https://github.com/TaniyaSarkar713/CFE_hackathon.git
cd CFE_hackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r rank_fusion/requirements.txt          # numpy, scikit-learn, scipy
find Data -name '*.zip' -execdir unzip -o {} \;       # unzip the data (see §3)
python run_pipeline.py --papers 150                   # the complete pipeline
python tests/test_rank_fusion.py                      # 53 tests should pass
```

---

## 1. Prerequisites
- **Python 3.9+** (developed on 3.9.6; works on 3.10/3.11).
- `pip` + a virtualenv (recommended).
- `unzip` (for the data archives).
- *Optional, only for live Semantic Scholar scripts:* network access + a Semantic
  Scholar API key.
- *Optional, only to re-render the slides:* Node.js + `npx` + a Chrome/Chromium
  browser (see §7).

## 2. Code dependencies

**Required** (the whole `rank_fusion/` package runs on these — see
`rank_fusion/requirements.txt`):
```
numpy>=1.21
scikit-learn>=1.0
scipy>=1.7
```

**Optional** (features degrade gracefully without them):
| Package | Enables | Fallback if missing |
|---|---|---|
| `sentence-transformers` | all-MiniLM-L6-v2 embeddings for dedup / semantic match | TF-IDF cosine |
| `lightgbm` | `MetaRanker(model="lgbm")` LGBMRanker | logistic meta-ranker |
| `requests` | the Semantic Scholar client (`semantic_scholar_ranking/`, live S2 scripts) | offline local-corpus proxy |

```bash
pip install sentence-transformers lightgbm requests   # optional
```

> The core pipeline (`run_pipeline.py`) needs **only** numpy/scikit-learn/scipy.
> No GPU, no LLM API keys — the LLM agents' outputs are precomputed files in `Data/`.

## 3. Data setup (unzip)
A fresh clone has the data as `.zip` archives; unzip them **in place**:
```bash
find Data -name '*.zip' -execdir unzip -o {} \;
```
This produces:
- `Data/Paper_summaries/<llm>_fd/` — 5 LLM agents' ranked `future_directions`
  (`gpt5nano_fd`, `llama17_fd`, `llama70_fd`, `mixtral_fd`, `qwen_fd`)
- `Data/Ground Truth Entities/future_sec_NER_ground_truth/` — structured ground
  truth (the masked future-work, split into atomic directions) — **929 papers**
- `Data/Ground Truth Future Directions/` and `Data/papers/` — raw masked
  future-work text + paper bodies (the smaller, ~15-paper subset)

## 4. Environment (only for live Semantic Scholar)
The offline scripts need nothing. For the **live S2** scripts (`run_s2_fusion.py`,
`run_s2_whitespace.py`) set a key:
```bash
export S2_API_KEY="<your semantic scholar api key>"
```
Get a free key at semanticscholar.org/product/api. **Note:** unauthenticated /
shared keys are heavily rate-limited (HTTP 429); the scripts back off and fall
back to the offline neighbourhood, so they still produce output.

## 5. Entry-point scripts
| Command | What it does | Network? |
|---|---|---|
| `python run_pipeline.py [--papers N]` | **the complete pipeline**: LLM agents → S2 → consensus fusion + CV best-of → eval + MLRC benchmark | no |
| `python run_llm_selection.py` | rank the 5 LLM agents by MRR, keep top 2 | no |
| `python run_consensus.py` | consensus fusion (LLM1+LLM2+S2) on synthetic, tune MRR@5 | no |
| `python run_explain.py` | Direction Cards (attribution, ablation, logic, evidence) | no |
| `python run_synthetic.py` | the Level 1→5 fusion ablation with bootstrap CIs | no |
| `python run_diversity.py` | MMR + Swiss tournament + dedup | no |
| `python run_significance.py` | paired-bootstrap + Bonferroni + ranker separation | no |
| `python run_s2_ablation.py` | S2 ranking: topology vs entity-rerank vs white-space | no |
| `python run_candidate_agent.py` | the autonomous candidate-harvesting agent (offline adapter) | no |
| `python run_realdata.py` | MRR harness on the real `Data/` files + tau sweep | no |
| `python run_s2_fusion.py [PMCID]` | single paper: live Agent C → fusion → best-of | **yes** |
| `python run_s2_whitespace.py [PMCID]` | pull connected papers from S2 → white-space directions | **yes** |
| `python tests/test_rank_fusion.py` | full test suite (53 tests) | no |

## 6. Package structure (so you / Claude can navigate)
```
rank_fusion/            # the self-contained engine (Taniya's workstream)
  schema, metrics, fusion, optimize        # MRR harness + RRF + tuning
  consensus, select, learned_rankers       # consensus fusion, CV best-of, LTR losses
  meta_ranker, diversity, dedup, pairwise  # meta-ranker, MMR, dedup, Swiss tournament
  diagnostics, significance, benchmark      # recall/oracle, stats, MLRC baseline
  matching, entities, whitespace            # match fn, entity re-rank, white-space gen
  agent, integrate                          # candidate agent, S2/agent adapters
  explain                                   # Direction Cards (§6)
  README.md                                 # deep docs
semantic_scholar_ranking/  # Naz's S2 client + Agent C (live S2; parallel fusion engine)
s2_tools.py                # bridges rank_fusion <-> the S2 client (Enriched + Local)
run_*.py                   # entry-point scripts (see §5)
WORKSTREAM_RANK_FUSION.md  # status, upstream dependencies, what's left
SLIDES.md / slides.pdf     # the deck
```

## 7. Using Claude Code on the new machine
Open Claude Code in the cloned repo and paste:

> *Set up and run the ResearchHorizon rank-fusion pipeline in this repo. Steps:
> (1) create a Python 3.9+ venv and `pip install -r rank_fusion/requirements.txt`;
> (2) unzip all archives with `find Data -name '*.zip' -execdir unzip -o {} \;`;
> (3) run `python run_pipeline.py --papers 150` and show me the Stage 1–4 output +
> the MLRC benchmark; (4) run `python tests/test_rank_fusion.py` to confirm 53
> tests pass. The `rank_fusion/` package is self-contained (numpy/scikit-learn/
> scipy only) and needs no API keys — the LLM agents' outputs are precomputed in
> `Data/Paper_summaries/<llm>_fd/`. For the live Semantic Scholar scripts only, I'll
> set `S2_API_KEY`. See `SETUP.md`, `rank_fusion/README.md`, and
> `WORKSTREAM_RANK_FUSION.md` for context.*

Good follow-up asks for Claude: *"explain `run_pipeline.py` stage by stage"*,
*"wire a biomedical NER as the `entity_fn` to strengthen the S2 source"*,
*"re-run the LLM selection and benchmark on the full 929-paper corpus"*.

To re-render the slides (needs Node + Chrome):
```bash
export CHROME_PATH="/path/to/Chrome"   # macOS: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
npx --yes @marp-team/marp-cli@latest SLIDES.md -o slides.pdf --allow-local-files
```

## 8. Troubleshooting
- **`ModuleNotFoundError: rank_fusion`** → run scripts from the repo root.
- **`No module named 'requests'`** → only needed for live S2 scripts: `pip install requests`.
- **HTTP 429 from Semantic Scholar** → rate-limited; use a dedicated `S2_API_KEY`,
  lower `--limit`, or rely on the offline fallback.
- **No LLM folders found** → you didn't unzip `Data/Paper_summaries/*_fd.zip` (§3).
- **Slides won't render** → install Node + a Chrome browser, or open `SLIDES.md` with
  the VS Code "Marp for VS Code" extension; the committed `slides.pdf` already exists.
