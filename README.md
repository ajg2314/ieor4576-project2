# IEOR 4576 — Project 2: Multi-Agent Sector Analyst

A multi-agent financial analysis system that performs the first three steps of a data analyst workflow — **Collect → Explore → Hypothesize** — applied to public company SEC filings and live market data. The user asks a sector or company question in natural language; the system retrieves real financial data from multiple sources, runs exploratory data analysis with code execution, and returns a grounded analyst memo with charts and evidence.

Example questions:
- *"Compare NVDA, AMD, and INTC on revenue growth and margins"*
- *"Analyze the GLP-1 obesity drug market — Novo Nordisk and Eli Lilly"*
- *"Which cloud hyperscaler — AWS, Azure, Google Cloud — is growing fastest?"*
- *"Analyze JPMorgan, Goldman Sachs, and Wells Fargo on NIM and credit quality"*

---

## Live Demo

> Deployed on Google Cloud Run — https://sectoranalystfinal-558700534812.us-central1.run.app/.

---

## Architecture

The system runs a 7-step pipeline. Steps 1–3 run before data collection; step 3 fans out five specialist agents in parallel.

```
User Question
      │
      ▼
┌──────────────┐
│   Step 1     │  Planner Agent — expands the question, identifies 10–15 tickers
│   Planner    │  Output: SectorPlan (sector, tickers, focus_metrics)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Step 2     │  Peer Discovery — validates tickers via yfinance, sorts by market cap
│ PeerDiscovery│  Filters out tiny/invalid tickers, caps universe to 12 peers
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   Step 3 — PARALLEL fan-out (asyncio.gather)                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ ┌──────────┐  │
│  │ Researcher │ │ Valuation  │ │ Sentiment  │ │  Geo   │ │  Sector  │  │
│  │  (web)     │ │  (yfinance)│ │  (web)     │ │Advisor │ │Specialist│  │
│  └────────────┘ └────────────┘ └────────────┘ └────────┘ └──────────┘  │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ all results aggregated
                               ▼
┌──────────────┐
│   Step 4     │  Collector Agent — SEC EDGAR XBRL API + MD&A text scraping
│   Collector  │  Output: DataBundle (financial records → SQLite)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Step 5     │  EDA Agent — SQL queries + Python code execution + charts
│     EDA      │  Output: EDAFindings (key_insight + artifact paths)
└──────┬───────┘      ↑ (loops back to Collector if data gaps found)
       │
       ▼
┌──────────────┐
│   Step 6–7   │  Hypothesis Agent — RAG-grounded analyst memo + structured report
│  Hypothesis  │  Saves: markdown report + charts to artifacts/
└──────────────┘
```

**Multi-agent patterns used:**
- **Orchestrator-handoff**: Orchestrator coordinates each pipeline step, passing typed schemas between agents
- **Fan-out / parallel execution**: Five specialist agents run concurrently in Step 3 via `asyncio.gather`
- **Iterative refinement**: Orchestrator loops `Collect → EDA → Collect` (up to 2× ) if EDA flags data gaps
- **Agent-as-tool-call**: Hypothesis agent can invoke a Q&A consultation tool

---

## The Three Steps

### Step 1: Collect (`pipeline/collector.py`, `tools/sec_edgar.py`, `tools/market_data.py`)

The Collector agent retrieves real financial data from **three distinct sources** at runtime. No data is hard-coded.

**Source 1 — SEC EDGAR XBRL API (primary, structured):**
- The agent resolves company tickers to SEC CIK numbers using the EDGAR company-tickers index (`https://www.sec.gov/files/company_tickers.json`)
- It calls `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` per company — each response is hundreds of KB of full financial history
- Specific financial concepts are extracted: `revenue`, `net_income`, `operating_income`, `gross_profit`, `rd_expense`, `total_assets`, `total_debt`, `cash`, `eps`
- Annual (10-K) and quarterly (10-Q) data are filtered and deduplicated before returning
- Implemented in `tools/sec_edgar.py`: `get_company_financials()`, `get_sector_financials()`

**Source 2 — SEC EDGAR filing text scraping (MD&A):**
- The agent fetches the actual 10-K / 10-Q filing document from EDGAR Archives
- The Management Discussion & Analysis (MD&A) section is extracted via regex heuristics
- Captures forward-looking statements, segment commentary, and risk factors
- Implemented in `tools/sec_edgar.py`: `get_recent_filing_text()`

**Source 3 — yfinance (fallback for non-US companies):**
- European-listed ADRs (e.g. `DNNGY`, `VWDRY`, `SMEGF`) do not file with EDGAR
- `tools/market_data.py`: `get_company_financials_yf()` fetches annual income-statement data via yfinance and returns the same flat-record schema as EDGAR so the EDA agent loads them identically into SQLite

All records are stored in a **module-level side-channel** (`_record_store` in `tools/sec_edgar.py`), so the Collector agent's text output never needs to echo large JSON arrays — the orchestrator reads the records directly after the agent finishes.

The Collector returns a typed `DataBundle` (Pydantic schema, `models/schemas.py`) that is passed to the orchestrator.

---

### Step 2: Explore and Analyze (`pipeline/eda_agent.py`, `tools/statistics.py`, `tools/code_executor.py`, `tools/visualizer.py`)

All financial records are loaded into a **per-run SQLite database** (`artifacts/eda_{run_id}.db`) before the EDA agent starts. This makes the full dataset queryable without fitting it into LLM context.

The EDA agent uses six tools:

| Tool | File | What it does |
|------|------|-------------|
| `sql_query(sql)` | `pipeline/eda_agent.py` | Run any SELECT on the `financials` table — rankings, aggregations, time-series lookups |
| `run_python(code)` | `tools/code_executor.py` | Execute pandas/numpy/scipy/matplotlib code in a sandboxed subprocess; conn and df are pre-loaded |
| `plot_metric(metric)` | `pipeline/eda_agent.py` | One-liner chart for any metric across all companies (auto-loads from SQLite) |
| `plot_margins(num, den)` | `pipeline/eda_agent.py` | Compute and chart a ratio/margin (e.g. `rd_expense/revenue`) per company per year |
| `create_chart(type, data)` | `pipeline/eda_agent.py` | Custom chart with manually specified series (bar, line, waterfall) |
| `get_analysis_guidance(sector, question)` | `tools/rag_store.py` | RAG retrieval of sector-specific EDA playbooks |

**Standard charts are auto-generated** when the database loads — Revenue Trend, Operating Margin %, and Gross Margin % — guaranteeing ≥3 visuals per report even if the agent skips custom charts.

The EDA agent's tool results are captured in `_eda_observations` (a module-level list). After the agent finishes, a **second, compact LLM call** (`_synthesise_eda_findings()` in `pipeline/orchestrator.py`) synthesises the raw observations into a typed `EDAFindings` object with a `key_insight` and `recommended_hypothesis_direction`.

---

### Step 3: Hypothesize (`pipeline/hypothesis_agent.py`)

The Hypothesis agent synthesises all prior pipeline outputs into a grounded analyst memo:

**Inputs received:**
- `EDAFindings` with specific data points and chart artifact paths
- `ResearchContext` — qualitative sector intelligence from web search
- `ValuationContext` — live P/E, EV/EBITDA, YTD return from yfinance
- `SentimentContext` — market sentiment from recent news
- `GeopoliticalAnalysis` — named policies, dates, company exposure levels
- `SectorAnalysis` — state-of-the-art technology and competitive dynamics

**RAG-augmented writing:**
- Before writing, the agent calls `retrieve_report_example()` to fetch a structurally similar exemplary analyst memo from the RAG store
- Also calls `retrieve_sector_knowledge()` for domain terminology relevant to the sector

**Mandatory report sections:**
1. Executive Summary & Hypothesis
2. Industry Overview & Revenue Ranking (anchored on most-recent year, not averages)
3. Financial Analysis (CAGRs, margins, EDA-derived data with WHY rationale)
4. Technology & Innovation
5. Geopolitical & Macro Analysis (named policies, exposure table)
6. Valuation Analysis (comp table, sector medians, relative attractiveness)
7. Investment Summary (explicit recommendation, conviction level, time horizon)

**Outputs:**
- `HypothesisReport` Pydantic schema with `hypothesis`, `evidence[]`, `narrative`, `artifact_paths[]`, `confidence`
- Markdown report saved to `artifacts/report_{uuid}.md` via `save_report()` tool
- Charts generated during EDA are embedded in the report as markdown image tags

---

## Requirements Checklist

### Required

| Requirement | Implementation |
|---|---|
| **Frontend** | `frontend/index.html` — dark-themed chat UI with real-time SSE progress bar, hypothesis panel, evidence list, artifact gallery, and follow-up Q&A. Progress bar animates with live step names and time estimates. |
| **Agent Framework** | **OpenAI Agents SDK** — `Agent`, `Runner`, `function_tool` used in all agents. Agents defined in `pipeline/planner.py`, `pipeline/collector.py`, `pipeline/eda_agent.py`, `pipeline/hypothesis_agent.py`, `pipeline/researcher.py`, `pipeline/valuation_agent.py`, `pipeline/sentiment_agent.py`, `pipeline/specialists/geopolitical_advisor.py`, `pipeline/specialists/sector_specialist.py`, `pipeline/qa_agent.py` |
| **Tool Calling** | 20+ tools across agents. Key tools: `fetch_sector_financials`, `fetch_filing_text`, `fetch_company_financials_yf`, `sql_query`, `run_python`, `plot_metric`, `plot_margins`, `create_chart`, `get_analysis_guidance`, `save_report`, `retrieve_report_example`, `get_sector_valuation`, `search_web`, `fetch_page_text` |
| **Non-trivial Dataset** | SEC EDGAR XBRL facts JSON — hundreds of KB per company, covering every financial concept since IPO. Dataset is queried per concept at runtime, never loaded into context wholesale. |
| **Multi-agent Pattern** | (1) Orchestrator-handoff across 7 pipeline steps; (2) parallel fan-out of 5 specialist agents via `asyncio.gather`; (3) iterative collect→EDA→collect refinement loop |
| **Deployed** | Google Cloud Run via `Dockerfile` + `cloudbuild.yaml`. FastAPI + uvicorn serve both the frontend and API. |
| **README** | This file |

### Grab-Bag (7 of 7 implemented)

| Concept | Implementation | File / Function |
|---|---|---|
| **Code Execution** | The EDA agent writes pandas/matplotlib Python at runtime and executes it in a sandboxed `subprocess` with a 30s timeout. Stdout + artifact paths are captured and passed to the Hypothesis agent. | `tools/code_executor.py`: `execute_python()` |
| **Data Visualization** | Revenue trend, operating margin, gross margin, R&D intensity, sector bar charts, and custom charts generated by `run_python`. Saved as UUID-named PNGs to `artifacts/`, served at `/files/`, embedded in the report. | `tools/visualizer.py`: `line_chart()`, `bar_chart()`, `waterfall_chart()` |
| **Structured Output** | Pydantic schemas enforce typed data flow at every agent boundary: `SectorPlan`, `DataBundle`, `EDAFindings`, `EDAFinding`, `HypothesisReport`, `EvidencePoint`, `ValuationContext`, `SentimentContext`, `GeopoliticalAnalysis`, `SectorAnalysis`, `QAResponse` | `models/schemas.py` |
| **Artifacts** | Charts saved as PNGs; analyst memos saved as markdown to `artifacts/`. The `save_report()` tool writes the report; artifact paths are embedded in the `HypothesisReport` schema. `GET /api/artifacts/list` returns all artifacts. | `pipeline/hypothesis_agent.py`: `save_report()` |
| **Second Data Retrieval Method** | (1) SEC EDGAR XBRL API (structured/API); (2) SEC EDGAR MD&A text scraping (web scraping); (3) DuckDuckGo web search for qualitative research (web search); (4) TF-IDF RAG over curated sector knowledge files (RAG) | `tools/sec_edgar.py`, `tools/web_search.py`, `tools/rag_store.py` |
| **Parallel Execution** | Step 3 fans out five agents simultaneously: Researcher + ValuationAgent + SentimentAgent + GeopoliticalAdvisor + SectorSpecialist. All five run concurrently via `asyncio.gather`; results are awaited and merged before Collector starts. | `pipeline/orchestrator.py`: `run_analysis_with_status()` lines ~718–800 |
| **Iterative Refinement Loop** | After the first EDA pass, the orchestrator checks `_needs_refinement(findings)` — if EDA flags "missing data" or "insufficient", it loops back to the Collector for an additional fetch, then re-runs EDA. Up to `MAX_REFINEMENT_LOOPS = 2` cycles. | `pipeline/orchestrator.py`: `_needs_refinement()`, loop in `run_analysis_with_status()` |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Package manager | `uv` |
| Agent framework | OpenAI Agents SDK (`openai-agents`) |
| LLM | Gemini 2.5 Flash via LiteLLM → Google Vertex AI |
| Backend | FastAPI + uvicorn (async) |
| Streaming | Server-Sent Events (SSE) — `StreamingResponse` |
| Data source 1 | SEC EDGAR XBRL API (public, no API key) |
| Data source 2 | SEC EDGAR filing text scraping (httpx) |
| Data source 3 | yfinance (non-US companies, live valuation data) |
| Data source 4 | DuckDuckGo web search (qualitative research, no API key) |
| EDA store | SQLite (one per-run DB in `artifacts/`) |
| RAG store | TF-IDF over curated markdown files in `data/` (pure stdlib, no external vector DB) |
| Code sandbox | Python `subprocess` with 30s timeout |
| Visualization | matplotlib (Agg backend, saved to disk) |
| Deployment | Google Cloud Run |
| Container | Docker (`python:3.12-slim`) |

---

## Project Structure

```
ieor4576-project2/
├── app.py                        # FastAPI app: /analyze/stream, /qa/stream, /files/, /api/artifacts/list
├── pyproject.toml / uv.lock      # uv-managed dependencies
├── Dockerfile                    # Container definition (python:3.12-slim + uv)
├── cloudbuild.yaml               # Google Cloud Build → Cloud Run deployment
│
├── pipeline/                     # All agents
│   ├── orchestrator.py           # Main pipeline coordinator — handoffs, parallel fan-out, refinement loop
│   ├── planner.py                # Planner Agent — sector identification, ticker selection
│   ├── peer_discovery.py         # Peer Discovery — yfinance validation, market-cap sort
│   ├── collector.py              # Collector Agent — EDGAR XBRL + MD&A scraping + yfinance fallback
│   ├── eda_agent.py              # EDA Agent — SQL + Python code execution + chart generation
│   ├── hypothesis_agent.py       # Hypothesis Agent — RAG-grounded analyst memo + save_report
│   ├── researcher.py             # Researcher Agent — DuckDuckGo web search + page fetch
│   ├── valuation_agent.py        # Valuation Agent — live P/E, EV/EBITDA, YTD via yfinance
│   ├── sentiment_agent.py        # Sentiment Agent — market sentiment from web search
│   ├── qa_agent.py               # Q&A Agent — follow-up questions on the generated report
│   └── specialists/
│       ├── geopolitical_advisor.py  # Geopolitical Advisor — trade policy, export controls, subsidy analysis
│       └── sector_specialist.py     # Sector Specialist — domain expert (tech/biomedical/energy/financials)
│
├── tools/                        # Tool implementations
│   ├── sec_edgar.py              # EDGAR XBRL API: resolve_ticker, get_company_financials, get_recent_filing_text
│   ├── market_data.py            # yfinance: get_company_financials_yf, get_sector_market_data, get_ytd_return
│   ├── web_search.py             # DuckDuckGo: search_web, fetch_page_text
│   ├── code_executor.py          # Sandboxed subprocess Python execution: execute_python
│   ├── visualizer.py             # Chart generation: line_chart, bar_chart, waterfall_chart
│   ├── rag_store.py              # TF-IDF RAG: seed_all, retrieve_report_example, retrieve_sector_knowledge, retrieve_eda_playbook
│   ├── statistics.py             # compute_statistics, group_and_filter
│   ├── api_client.py             # Generic REST helper
│   └── sql_query.py              # SQL helpers
│
├── models/
│   └── schemas.py                # Pydantic schemas: SectorPlan, DataBundle, EDAFindings, HypothesisReport, ...
│
├── data/
│   ├── report_examples/          # Exemplary analyst memos (RAG source — style reference)
│   │   ├── ideal_sector_report.md
│   │   ├── semiconductor_example.md
│   │   ├── cloud_software_example.md
│   │   └── ...
│   ├── sector_knowledge/         # Domain knowledge (RAG source — EDA guidance, sector terms)
│   │   ├── semiconductors.md
│   │   ├── biomedical_pharma.md
│   │   ├── financials_sector.md
│   │   ├── energy_materials.md
│   │   ├── geopolitics.md
│   │   ├── eda_playbooks.md
│   │   └── general_finance.md
│   └── rag_store/                # Persisted TF-IDF indexes (auto-generated on first run)
│
├── frontend/
│   └── index.html                # Single-page app: chat UI + progress bar + artifact gallery + Q&A panel
│
├── artifacts/                    # Runtime-generated outputs (git-ignored)
│   ├── report_*.md               # Analyst memos in markdown
│   ├── revenue_trend_*.png       # Revenue trend charts
│   ├── operating_margin_*.png    # Margin charts
│   ├── eda_*.db                  # Per-run SQLite databases
│   └── ...
│
└── tests/                        # pytest tests
    ├── test_orchestrator.py
    ├── test_sec_edgar.py
    ├── test_code_executor.py
    └── ...
```

---

## Running Locally

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Google Cloud project with Vertex AI API enabled
- `gcloud` CLI authenticated (`gcloud auth application-default login`)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/ajg2314/ieor4576-project2
cd ieor4576-project2

# 2. Create a .env file with your GCP credentials
cat > .env <<'EOF'
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash
EOF

# 3. Install dependencies via uv
uv sync

# 4. Start the server
uv run uvicorn app:app --reload --port 8080 --reload-exclude 'artifacts'

# 5. Open in your browser
open http://localhost:8080
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (for Vertex AI) | required |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region | `us-central1` |
| `GEMINI_MODEL` | LiteLLM model name | `gemini-2.5-flash` |

No other API keys are required:
- **SEC EDGAR** is a free public API (no key, User-Agent header required by policy)
- **DuckDuckGo** web search uses the public HTML search endpoint (no key)
- **yfinance** is a scraper-based library (no key)

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend (`frontend/index.html`) |
| `POST` | `/analyze/stream` | SSE stream — runs the full pipeline for a question |
| `POST` | `/qa/stream` | SSE stream — answers a follow-up question about the last report |
| `GET` | `/api/artifacts/list` | Lists all generated charts and reports |
| `GET` | `/files/{name}` | Serves a chart or report from `artifacts/` |
| `GET` | `/health` | Health check |

### SSE Event Format

The `/analyze/stream` endpoint yields newline-delimited SSE events:

```
data: {"type": "progress", "payload": {"step": 4, "total_steps": 7, "pct": 46, "step_name": "Fetching SEC EDGAR data", "elapsed_seconds": 42, "estimated_remaining_seconds": 190}}

data: {"type": "status", "payload": "Data collected: 1200 records across 10 companies"}

data: {"type": "result", "payload": {"title": "...", "hypothesis": "...", "evidence": [...], "narrative": "...", "artifact_paths": [...], "confidence": "high"}}
```

---

## Deployment to Google Cloud Run

The project includes a `cloudbuild.yaml` that builds a Docker image, pushes it to Google Container Registry, and deploys to Cloud Run.

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy (builds image + deploys to Cloud Run in one step)
gcloud builds submit --config cloudbuild.yaml

# The service URL will be printed at the end of the build
```

**Cloud Run configuration** (set in `cloudbuild.yaml`):
- Memory: 2Gi (matplotlib + pandas need headroom)
- CPU: 2
- Timeout: 3600s (long-running analyses can take 5–8 minutes)
- Environment variables: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GEMINI_MODEL`

**Note:** `artifacts/` is an ephemeral writable directory in the container. Charts and reports are regenerated on each analysis; they are not persisted across container restarts. For production persistence, mount a Cloud Storage FUSE bucket at `/app/artifacts`.

---

## How the Pipeline Works — End-to-End

1. **User submits a question** via the frontend chat box (e.g. "Compare NVDA, AMD, and INTC on revenue growth and margins").

2. **Planner** expands the question, identifies 10–15 tickers, and selects relevant financial metrics (e.g. `revenue`, `gross_profit`, `rd_expense`).

3. **Peer Discovery** validates each ticker via yfinance, filters out illiquid or invalid symbols, and sorts by market cap.

4. **Parallel step** (Step 3) launches five agents simultaneously:
   - **Researcher**: DuckDuckGo web searches for technology trends, analyst commentary, competitive news
   - **ValuationAgent**: fetches live P/E, EV/EBITDA, YTD return for all tickers
   - **SentimentAgent**: searches for recent earnings reactions and analyst sentiment
   - **GeopoliticalAdvisor**: synthesises trade policy, export controls, and geographic risks
   - **SectorSpecialist**: writes domain-expert context on technology and competitive dynamics

5. **Collector** fetches structured financial data from SEC EDGAR for all tickers. Non-US companies fall back to yfinance. MD&A text is scraped from the first company's most recent 10-K and 10-Q.

6. **EDA** loads all records into SQLite, auto-generates 3 standard charts, then runs additional analysis: SQL ranking queries, Python code for YoY growth and CAGRs, custom charts for sector-specific metrics.

7. **Hypothesis** receives all outputs from steps 1–6, retrieves a structurally similar exemplary report from the RAG store, and writes a 6-section analyst memo grounded in specific data points. The memo is saved to `artifacts/report_{uuid}.md`.

8. **Result** is streamed to the frontend as a `type=result` SSE event. The frontend renders the hypothesis, evidence bullets, full narrative, and all chart images.

---

## Authors

Andy Gu — IEOR 4576, Spring 2026
