# Project Drona
### Autonomous Agentic T20 Cricket Tactical Decision Support System

> *"Drona does not guess. It queries, verifies, then speaks."*

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey?logo=sqlite)](https://sqlite.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is Project Drona?

Drona is a **locally-running AI coaching system** that answers T20 cricket tactical questions with verifiable, database-grounded answers — not hallucinated guesses.

A captain asks: *"Who should I bowl in over 18 at Wankhede, chasing 185?"*

Most AI tools make something up. Drona runs a SQL query against 278,205 real IPL deliveries, validates the result, computes a statistical confidence interval, and responds as a gritty head coach — with the raw SQL shown so you can verify every number.

**The core insight:** separate what must be deterministic (SQL + DB) from what can be generative (coach narration). The model never invents a statistic it wasn't given.

---

## The Problem With Every Existing Solution

| Tool | Problem |
|------|---------|
| ChatGPT / Gemini | Confidently fabricates cricket statistics |
| Cricinfo / CricViz | Read-only dashboards — no conversational interface |
| Generic RAG bots | Embedding similarity has no concept of numerical meaning |
| **Project Drona** | ✅ Deterministic SQL + verified stats + coach voice |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
│         "Bowl Bumrah or Chahal in over 17?"                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              ETL ENGINE  ·  Pandas + Scikit-Learn           │
│   1,169 IPL matches · 278,205 deliveries · 18 seasons       │
│   Phase classification · K-Means archetypes · Clone table   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│           SQLite DB  ·  cricket_drona.db  ·  8 tables       │
│  matchups_1v1 · bowler_phase_stats · batter_phase_stats     │
│  venue_profiles · recent_form · player_clones               │
│  batter_vs_bowling_style · pressure_performance             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              SQLGLOT GUARD  ·  Schema Validator              │
│      Rejects hallucinated columns before DB execution        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│         LANGCHAIN ReAct AGENT  ·  6 Tactical Tools          │
│  sql_executor · clone_lookup · memory_read · memory_write   │
│  momentum_detector · team_weakness_profiler                  │
│                                                             │
│  PASS 1: Fine-tuned Llama 3.1 8B → generates SQL           │
│  PASS 2: Same model, coach persona → narrates results       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│         EPISODIC MEMORY  ·  learned_lessons.json            │
│    User corrections stored · injected into future queries   │
│            Drona learns from every session                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│      STREAMLIT UI  ·  XAI Provenance Panel                  │
│  Coach advice · Raw SQL shown · Wilson CI · Feedback form   │
└─────────────────────────────────────────────────────────────┘
```

---

## 9 Novel Features

**1. Deterministic Text-to-SQL**
The model generates a SQL query. The database returns exact numbers. Only then does the model speak. Zero hallucinated statistics.

**2. Wilson Score Confidence Interval**
Every recommendation includes a mathematically rigorous 95% CI lower bound. A matchup based on 8 balls gets a very different confidence score than one based on 800. No other public cricket tool exposes this.

**3. Self-Learning SQL Correction**
Every SQL error is journaled, analysed, and converted into a correction rule. Rules are injected into future prompts automatically. The agent makes each mistake at most once.

**4. K-Means Bowling Archetypes**
551 bowlers classified into 5 tactical roles: `death_specialist`, `powerplay_enforcer`, `strike_bowler`, `economy_merchant`, `allrounder`. The agent reasons by role, not just by name.

**5. Cosine Similarity Player Clones**
Unknown or uncapped players get a statistical twin from the database. The agent uses the clone's data transparently, with similarity score shown.

**6. Pressure Performance Table**
Separate stats for second-innings, overs 17–20, high-chase situations. Categorically different from career averages — because players genuinely perform differently under pressure.

**7. Surface Behavior Index**
Pitch condition score derived purely from first-3-over economy vs venue baseline. No Hawk-Eye, no sensors, no external data.

**8. Momentum Detector**
Scores live match state from −1.0 (collapse) to +1.0 (dominant) based on wickets, boundaries, and dots in the last 12 balls. Injected into every query as situational context.

**9. Episodic Memory (Learning Agent)**
Post-match feedback is stored and retrieved before future queries. After 10 matches, Drona knows that Eden Gardens in April assists spin after over 12 — because a coach told it that.

---

## Database at a Glance

| Table | Rows | Purpose |
|-------|------|---------|
| `matchups_1v1` | 14,723 | Head-to-head batter vs bowler |
| `bowler_phase_stats` | 964 | Economy/wickets by phase + archetype |
| `batter_phase_stats` | 865 | SR/average by powerplay/middle/death |
| `venue_profiles` | 59 | Avg scores + Surface Behavior Index |
| `batter_vs_bowling_style` | 697 | Pace vs spin vulnerability |
| `recent_form` | 1,255 | Last-10-match trend per player |
| `player_clones` | 2,961 | Statistical twins for OOD fallback |
| `pressure_performance` | 705 | Death-over chase stats specifically |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd drona
python3 -m venv venv && source venv/bin/activate

# 2. Install PyTorch (RTX 5060 / Blackwell — CUDA 13.1)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install dependencies
pip install langchain langgraph langchain-ollama sqlglot streamlit \
            rich pandas numpy scikit-learn tqdm

# 4. Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & && ollama pull llama3.1:8b

# 5. Build the database (takes ~10 seconds)
python src/etl.py --matches data/matches --db db/cricket_drona.db

# 6. Validate the DB
python src/agent.py --test

# 7. Launch the UI
streamlit run src/app.py --server.port 8501
```

---

## Fine-Tuning (Optional but Recommended)

The base `llama3.1:8b` model works. The fine-tuned `drona-v1` works significantly better on schema accuracy and archetype queries.

```bash
# Stop Ollama first (VRAM conflict)
pkill ollama && sleep 3

# Install training dependencies
pip install trl datasets accelerate
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Generate 690 training examples + run LoRA fine-tuning
python src/finetune.py

# After training, convert and register
python ~/llama.cpp/convert_hf_to_gguf.py models/drona-lora/ \
  --outfile models/drona-lora.gguf --outtype q8_0
ollama serve & && cd models && ollama create drona-v1 -f Modelfile
```

---

## 5 Demo Scenarios

Try these live in the UI:

```
1. "Who should I bowl in over 18? Target 185, need to defend 34 off 18."
   → Recommends death_specialist · shows economy + Wilson CI · warns if small sample

2. "How has V Kohli performed against SP Narine historically?"
   → Queries matchups_1v1 · 129 balls · full head-to-head breakdown

3. "Our new uncapped fast bowler is playing — any data?"
   → Triggers clone_lookup · returns 3 statistical twins with similarity scores

4. "Is Chinnaswamy a good venue for spin in the powerplay?"
   → Queries venue_profiles (SBI=0.857) + batter_vs_bowling_style

5. "Who performs best in pressure chases: MS Dhoni, AT Rayudu, RG Sharma?"
   → Queries pressure_performance · ranks by death-over strike rate
```

---

## Evaluation Results

| Metric | llama3.1:8b (Base) | drona-v1 (Fine-Tuned) |
|--------|--------------------|-----------------------|
| SQL Execution Rate | 45% | TBD post-training |
| Schema Accuracy | 35% | TBD post-training |
| Result Accuracy | 45% | TBD post-training |
| Archetype Query Accuracy | ~40% | TBD post-training |

*Fine-tuned model evaluation to be updated post-training.*

---

## Hardware

- **GPU:** NVIDIA RTX 5060 Laptop (Blackwell sm_120) · 8GB VRAM
- **OS:** WSL2 Ubuntu · Python 3.12
- **Inference:** ~15–25 seconds per full query (Pass 1 + Pass 2)
- **Training:** ~2–3 hours for 3 epochs on 690 examples

---

## Known Limitations

- IPL data only (2007–2025). No international T20 or other leagues.
- Bowling style (pace/spin) is heuristically inferred — no ground truth labels in Cricsheet.
- No live ball-by-ball data feed — match state must be entered manually.
- Inference latency (~20s) is suitable for between-overs decisions, not ball-by-ball.

---

## Team

| Role | Responsibility |
|------|---------------|
| **Data Engineer** | ETL pipeline · SQLite schema · ML clustering · DB validation |
| **AI Engineer** | LLM fine-tuning · LangChain agent · Streamlit UI · prompt engineering |

---

## Project Structure

```
drona/
├── data/matches/          # 1,169 IPL match CSVs (Cricsheet Ashwin format)
├── db/
│   ├── cricket_drona.db   # 8-table SQLite database
│   ├── learned_lessons.json
│   ├── sql_error_journal.json
│   └── sql_correction_rules.json
├── models/
│   ├── drona-lora/        # LoRA adapter weights
│   ├── drona-lora.gguf    # GGUF for Ollama
│   └── Modelfile
├── src/
│   ├── etl.py             # Data pipeline
│   ├── agent.py           # ReAct agent + all tools
│   ├── sql_memory.py      # Self-learning SQL correction
│   ├── finetune.py        # LoRA training script
│   ├── app.py             # Streamlit UI
│   └── evaluate.py        # Evaluation suite
└── fine_tune/
    ├── train.jsonl         # 552 training examples
    └── val.jsonl           # 138 validation examples
```

---
## Future Work

**Immediate improvements (next semester, same hardware)**

- *Ground-truth bowling styles:* Replace the heuristic pace/spin classifier with a scraped Cricinfo dataset of ~800 bowler styles. This would make `batter_vs_bowling_style` queries significantly more accurate — currently ~15% of bowlers are classified as `unknown`.

- *Real-time data feed:* Integrate the Cricsheet live scores API or a websocket feed so the momentum detector consumes actual ball-by-ball data during a match rather than requiring manual input.

- *Larger fine-tuning corpus:* 690 examples is sufficient for schema adherence. Expanding to 2,000+ examples with more edge cases — mid-over changes, DLS scenarios, rain interruptions — would improve tactical reasoning depth.

- *Multi-league support:* Extend the ETL to ingest BBL, CPL, and SA20 Cricsheet data. The schema is identical — it's a config change, not a rewrite.

**With better hardware (RTX 4090 / A100)**

- *Full fine-tuning instead of LoRA:* LoRA at rank=16 updates ~10M of 8B parameters. Full fine-tuning would update all 8B, producing a model that reasons about cricket tactics more deeply, not just generates schema-correct SQL. Requires ~80GB VRAM — not feasible on 8GB.

- *Larger base model (Llama 3.1 70B):* The 70B model's reasoning capability would dramatically reduce the need for fine-tuning to achieve correct SQL generation. The 8B model requires explicit schema injection and error correction loops that the 70B largely avoids out of the box. Requires ~40GB VRAM in 4-bit.

- *Embedding-based memory retrieval:* Replace keyword-matched episodic memory with a sentence-transformer embedding layer. "Pitch was damp after rain" would semantically match future queries about slow surfaces — currently only exact keyword overlap triggers retrieval.

**Research directions**

- *Formal evaluation benchmark:* Publish a CricketSQL benchmark dataset — 200 questions with verified SQL answers against the IPL schema — to enable reproducible evaluation of Text-to-SQL systems in the cricket domain.

- *Uncertainty-aware recommendations:* Extend Wilson CI to a full Bayesian posterior so the agent can say not just "low confidence" but "there is a 73% chance the true dismissal rate is above 15%" — actionable probability rather than a binary warning.

- *Opponent modelling:* Train a separate model on bowling strategy patterns to predict what the opposition captain will do next, enabling pre-emptive field setting advice rather than just reactive bowling recommendations.

---

## Evaluation Results

| Metric | llama3.1:8b (Base) | drona-v1 (Fine-Tuned) |
|--------|--------------------|-----------------------|
| SQL Execution Rate | 45% | TBD post-training |
| Schema Accuracy | 35% | TBD post-training |
| Result Accuracy | 45% | TBD post-training |
| Archetype Query Accuracy | ~40% | TBD post-training |

*Fine-tuned model evaluation to be updated post-training.*

---

<p align="center">
Built as a 4th-semester B.Tech capstone in AI & Data Science<br>
<em>Named after Dronacharya — the greatest tactical coach in the Mahabharata</em>
</p>
