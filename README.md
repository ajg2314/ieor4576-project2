# IEOR 4576 — Project 2: Multi-Agent Data Analyst

A multi-agent system that automates the first three steps of a data analysis lifecycle: **Collect → Explore → Hypothesize**. The user asks an analytics question in natural language; the system retrieves real-world data, performs exploratory data analysis, and returns a grounded hypothesis with supporting evidence.

---

## Live Demo

> Deployed on Google Cloud Run — link TBD after first deployment.

---

## Project Overview

The system is built as a pipeline of four specialized agents orchestrated by a central planner. Each agent has a distinct system prompt and responsibility:

| Agent | Role |
|---|---|
| **Orchestrator** | Receives the user's question, plans the analysis, hands off to specialist agents in sequence (or parallel) |
| **Collector** | Retrieves real-world data at runtime via external API calls and/or SQL queries against a local dataset |
| **EDA Agent** | Performs exploratory data analysis: computes statistics, filters/groups data, writes and executes Python code for deeper analysis |
| **Hypothesis Agent** | Synthesizes EDA findings into a grounded hypothesis with cited evidence, data tables, and visualizations |

---

## The Three Steps

### Step 1: Collect (`agents/collector.py`)

The Collector agent retrieves data from **two distinct external sources** at runtime:

1. **Public REST API** — The agent calls a public data API (e.g., FRED, Open-Meteo, NYC Open Data, or a sports stats API) based on the user's question. The endpoint and query parameters are constructed dynamically; no data is hard-coded.
2. **SQL via DuckDB** — For structured/tabular datasets (CSV, Parquet), the agent dynamically writes and executes SQL queries using DuckDB. The dataset is large enough that it cannot be trivially dumped into context.

The Collector agent uses **structured output** (Pydantic schema) to return a normalized `DataBundle` that downstream agents consume.

### Step 2: Explore and Analyze (`agents/eda_agent.py`)

The EDA agent performs exploratory analysis using a set of tools. It does not summarize raw data — it computes specific metrics and surfaces findings that inform the hypothesis.

EDA tools available:
- `compute_statistics` — means, medians, standard deviations, correlations, growth rates
- `group_and_filter` — segments data by category, time window, or threshold
- `execute_python_code` — the agent writes pandas/numpy/scipy/matplotlib code and runs it in a sandboxed subprocess; output and generated figures are captured
- `text_analysis` — keyword/entity frequency and basic NLP operations on text fields

The EDA agent can **fan out** to run multiple analysis tools in parallel and aggregates results before handing off to the Hypothesis agent.

### Step 3: Hypothesize (`agents/hypothesis_agent.py`)

The Hypothesis agent receives EDA findings and produces a deliverable that includes:
- A natural-language summary grounded in specific data points (numbers, percentages, time ranges)
- A structured evidence block citing which data points support the claim
- One or more data visualizations (matplotlib/plotly charts saved as PNG artifacts and served to the frontend)
- A markdown report written to disk as a persistent artifact

The hypothesis is derived from runtime data, not model weights. All claims are traceable to specific EDA outputs.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│   Orchestrator  │  Plans steps, manages state, handles iteration
│   Agent         │  OpenAI Agents SDK handoff pattern
└────────┬────────┘
         │ handoff
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌────────┐  (parallel fan-out possible)
│Collector│  │  EDA   │
│ Agent   │  │ Agent  │
└────┬────┘  └────┬───┘
     │             │
     └──────┬──────┘
            ▼
    ┌───────────────┐
    │  Hypothesis   │
    │    Agent      │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │   FastAPI     │  Streams results back to frontend
    │   Backend     │
    └───────────────┘
```

---

## Requirements Checklist

### Required

| Requirement | Implementation |
|---|---|
| **Frontend** | `frontend/index.html` — chat UI with streaming response display and artifact panel |
| **Agent Framework** | OpenAI Agents SDK (`agents/` directory) — orchestrator-handoff pattern |
| **Tool Calling** | `tools/` directory — `compute_statistics`, `execute_python_code`, `group_and_filter`, API fetch tools |
| **Non-trivial Dataset** | Runtime API + DuckDB-queried dataset (100k+ rows); described in Step 1 above |
| **Multi-agent Pattern** | Orchestrator → Collector → EDA (fan-out) → Hypothesis (handoff chain + parallel sub-tasks) |
| **Deployed** | Google Cloud Run via `cloudbuild.yaml` |
| **README** | This file |

### Grab Bag

| Concept | Points | Implementation |
|---|---|---|
| **Code Execution** | 2.5 pts | `tools/code_executor.py` — agent writes Python (pandas, matplotlib) and executes it in a sandboxed subprocess at runtime |
| **Data Visualization** | 2.5 pts | `tools/visualizer.py` + `artifacts/` — matplotlib charts are generated by the code executor, saved as PNG, and served to the frontend via a static file endpoint |
| **Structured Output** | 2.5 pts | `models/schemas.py` — Pydantic schemas for `DataBundle`, `EDAFindings`, `HypothesisReport`; used at agent handoff boundaries to ensure reliable data flow |

*(Targeting 3 grab-bag concepts for full coverage + buffer)*

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Package manager | `uv` |
| Agent framework | OpenAI Agents SDK |
| LLM | Gemini 2.5 Flash via LiteLLM → Google Vertex AI |
| Backend | FastAPI + uvicorn |
| Data querying | DuckDB (SQL over CSV/Parquet) |
| Code sandbox | Python `subprocess` with timeout |
| Visualization | matplotlib / plotly |
| Deployment | Google Cloud Run |
| Container | Docker (python:3.12-slim) |

---

## Project Structure

```
ieor4576-project2/
├── app.py                    # FastAPI entrypoint; chat, health, artifact endpoints
├── pyproject.toml            # uv-managed dependencies
├── Dockerfile
├── cloudbuild.yaml
├── .env.example
│
├── agents/
│   ├── orchestrator.py       # Orchestrator: plans, routes, iterates
│   ├── collector.py          # Collector: API calls + DuckDB SQL queries
│   ├── eda_agent.py          # EDA: statistical tools + code execution fan-out
│   └── hypothesis_agent.py   # Hypothesis: synthesizes findings → report + charts
│
├── tools/
│   ├── api_client.py         # External API integration (dynamic endpoint construction)
│   ├── sql_query.py          # DuckDB tool: write + execute SQL at runtime
│   ├── code_executor.py      # Sandboxed Python execution (pandas, matplotlib)
│   ├── statistics.py         # compute_statistics, group_and_filter tools
│   └── visualizer.py        # Chart generation helpers
│
├── models/
│   └── schemas.py            # Pydantic schemas: DataBundle, EDAFindings, HypothesisReport
│
├── frontend/
│   └── index.html            # Chat UI with artifact viewer panel
│
└── artifacts/                # Runtime-generated outputs: PNGs, CSVs, markdown reports
```

---

## Running Locally

```bash
# 1. Clone and enter repo
git clone https://github.com/ajg2314/ieor4576-project2
cd ieor4576-project2

# 2. Copy env and fill in credentials
cp .env.example .env

# 3. Install dependencies with uv
uv sync

# 4. Run the server
uv run uvicorn app:app --reload --port 8080

# 5. Open http://localhost:8080
```

### Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region (e.g. `us-central1`) |
| `GEMINI_MODEL` | Model name (e.g. `gemini-2.5-flash`) |
| `DATA_API_KEY` | API key for the external data source |

---

## Deployment

```bash
# Trigger Cloud Build (builds Docker image, pushes to GCR, deploys to Cloud Run)
gcloud builds submit --config cloudbuild.yaml
```

The `cloudbuild.yaml` mirrors project 1: build → push to GCR → deploy to Cloud Run with public access.

---

## Topic

> **TBD** — The agent system is topic-agnostic. The data source (API endpoint and/or dataset file) will be configured via environment variables and the Collector agent's prompt. Swapping topics requires changing the API target and dataset, not the agent architecture.

---

## Authors

Andy Gu — IEOR 4576, Spring 2026
