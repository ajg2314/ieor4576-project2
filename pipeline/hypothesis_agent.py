"""Hypothesis Agent — synthesizes EDA findings into a grounded analyst report.

Step 4 (of 5): Hypothesize. This agent:
- Retrieves exemplary report structure and domain knowledge from the RAG store
- Receives EDAFindings and qualitative research context
- Forms a hypothesis grounded in specific data points and current events/trends
- Writes an analyst memo with mandatory sections including Technology, Geopolitics
- Saves a markdown report to disk
- Returns a structured HypothesisReport
"""

from __future__ import annotations

import asyncio
import os
import re as _re
import uuid
from pathlib import Path
from agents import Agent, Runner, ItemHelpers, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import MessageOutputItem

from tools.code_executor import execute_python
from tools.rag_store import retrieve_report_example, retrieve_sector_knowledge
from models.schemas import HypothesisReport

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

# Module-level sector state — set by build_hypothesis_agent() so consultation
# tools know which sector specialist to route to.
_current_sector: str = ""

# Side-channel: records the exact path returned by save_report() so the
# orchestrator can retrieve it without trusting the agent's JSON output.
_last_saved_report_path: str = ""


def get_last_saved_report_path() -> str:
    """Return the path recorded by the most recent save_report() tool call."""
    return _last_saved_report_path


def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


def _extract_text_from_result(result) -> str:
    """Extract the last non-empty assistant text from a RunResult."""
    raw = result.final_output
    if raw and str(raw).strip():
        return str(raw)
    for item in reversed(result.new_items):
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text.strip():
                return text
    return ""


HYPOTHESIS_PROMPT = """\
You are the Hypothesis agent in a multi-agent Sector Analyst system.

You receive:
- The user's original sector/company question
- EDAFindings from the EDA agent: computed metrics, trends, anomalies, chart paths
- Qualitative Research Context (if available): technology trends, analyst views,
  expert opinions, competitive dynamics gathered from web research

STEP 1 — RETRIEVE CONTEXT (always do this first):
Call get_report_style_reference(sector, question) to see examples of well-written reports.
Call get_sector_knowledge(sector, "geopolitical risks technology trends analysis guidance") to ground your analysis.

SPECIALIST CONTEXT is pre-loaded in your input under "GEOPOLITICAL ANALYSIS" and
"SECTOR SPECIALIST ANALYSIS" sections. Use those directly for Sections 4 and 5.
Only call consult_geopolitical_advisor() or consult_sector_specialist() for specific
follow-up questions where more depth would significantly improve a particular point.
Use these consultation tools sparingly — they are slow.

STEP 2 — WRITE THE REPORT using save_report.

STEP 3 — OUTPUT the final JSON.

━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUANTITATIVE CLAIMS:
- Every number MUST come from EDAFindings. Do not invent figures.
- Always report most recent fiscal year revenue first, then growth rate (CAGR or YoY %).
  NEVER lead with historical averages — they mislead on high-growth companies.
- When revenue or margins moved significantly (>15%): EXPLAIN WHY. Connect the data
  to real-world events: product launches, AI buildout, export controls, inventory cycles,
  competitive moves, macro shocks. "Revenue declined" is incomplete. "Revenue declined
  14% in FY2024 driven by AMD market share gains in server CPUs and customers shifting
  AI workload budgets to NVIDIA GPUs" is analysis.

QUALITATIVE CLAIMS:
- Claims from research context MUST be attributed: "according to analyst reports...",
  "industry commentary suggests...", "as noted in recent expert coverage..."
- If no research context was provided, draw on the RAG knowledge retrieved in Step 1.

ATTRIBUTION FORMAT (apply throughout the report):
- Financial data from EDA → "per SEC EDGAR 10-K FY[year]" or "per yfinance annual FY[year]"
- Research context claims → use the source type noted in that context
  (e.g. "per Reuters, Mar 2026" / "per Bloomberg analyst research" / "per EIA report, 2026")
- Geopolitical claims → cite the specific policy and date (e.g. "per BIS Oct 2024 rule" / "per IRA (Aug 2022)")
- Technology claims → "per analyst research, [year]" or "per industry reports, [year]"
- NEVER use vague "industry reports" alone — always add sector/year context

CONFIDENCE:
- If evidence is weak or limited, rate 'low' and explain.
- For evidence "source" fields: use specific sources like "SEC EDGAR 10-K FY2025",
  "yfinance annual data FY2025", "Reuters Apr 2026", "BIS rule Oct 2024".
  Never use tool names (e.g. "XBRL revenue data") as the source — name the actual filing or publication.

━━ REPORT STRUCTURE (mandatory — write ALL sections) ━━━━━━━━━━━━━━━━━━━━━━━━

Open the report with a Sector Snapshot header block: a small table of key statistics
(market size, coverage universe count, sector revenue growth, forward P/E vs. S&P,
analyst recommendation). This gives the reader an at-a-glance summary before the prose.

1. Business Description & Revenue Model
   - What the sector does and how companies make money
   - Key business model types (e.g. fabless / IDM / foundry for semiconductors)
   - Main revenue drivers and cost structure
   - DEFINE every technical term on first use with a parenthetical:
     e.g. "fabless (chip designers that outsource manufacturing to foundries)"

2. Industry Overview & Competitive Positioning
   - Market size and structure; top 5-8 companies ranked by MOST RECENT FY revenue
   - Include a revenue ranking table (not an average — most recent year only)
   - Competitive moats, barriers to entry, market share dynamics
   - VISUAL 1 (mandatory): bar chart or table showing most-recent-year revenue by company
     After the visual write one blockquote line (see VISUAL RULES below).

3. Financial Analysis
   - Revenue CAGR and YoY growth for key companies
   - Margin analysis (gross, operating, net): who has pricing power and why
   - R&D intensity comparison
   - VISUAL 2 (mandatory): line chart of revenue over time for top 3-5 companies
     After the visual write one blockquote line (see VISUAL RULES below).
   - WHY rule: when any company's revenue or margin moved >15% YoY, EXPLAIN THE CAUSE.
     Connect to real-world events: product launches, AI buildout, export controls,
     inventory cycles, competitive moves, macro shocks.
     BAD: "Revenue declined 14%." GOOD: "Revenue declined 14% in FY2024 driven by
     AMD server CPU share gains and customers shifting AI budgets to NVIDIA GPUs."

4. Technology & Innovation (Forward-Looking) — MINIMUM 2 SUBSTANTIAL PARAGRAPHS
   - What technology trends are shaping the sector RIGHT NOW?
   - What is the market betting on for the NEXT 3-5 years? Explain what scenario
     investors are pricing in — not just what happened historically.
   - Which companies are positioned to win/lose from these technology transitions?
   - Example: "The market assigns NVIDIA a 30x revenue multiple not because of trailing
     revenue, but because AI infrastructure capex is committed years in advance and CUDA's
     software moat creates switching costs that competitors cannot erode near-term."
   - NOTE: If SECTOR SPECIALIST ANALYSIS contains "[Note: real-time web search unavailable...]",
     begin this section with: "Note: real-time sector research was unavailable for this run —
     the following reflects knowledge-base analysis. Re-run for the most current developments."
     Then write at least 2 substantial paragraphs using the specialist content and your knowledge.
     Do NOT leave this section empty or write a tool-error message — knowledge-base content
     is still valuable to the reader.

5. Geopolitical & Macro Environment — MANDATORY STANDALONE SECTION
   - Name specific policies, date them, quantify their impact:
     e.g. "US BIS export controls (Oct 2024) restrict AI chips above 1,800 TFLOPS"
   - Geographic concentration risks: where is manufacturing, supply chain, or revenue
     concentrated, and what is the tail risk?
   - Industrial policy tailwinds: subsidies, tax credits, domestic production incentives
   - Which companies benefit vs. suffer from current geopolitical configuration?
   - VISUAL 3 (mandatory): Company Geopolitical Exposure table
     Columns: Company | Ticker | Exposure Level | Key Risk Mechanism
     Pull the mechanism text DIRECTLY from GEOPOLITICAL ANALYSIS company_exposures.
     Do NOT reduce to just "High exposure" — include the quantified mechanism verbatim or
     lightly paraphrased with numbers (e.g. "~17% of FY2025 revenue ($22B) from China;
     H100/B200 directly restricted under Oct 2024 BIS rules").
     After the visual write one blockquote line (see VISUAL RULES below).

   - QUANTIFIED EXPOSURE PARAGRAPH (mandatory, immediately after Visual 3):
     Write 3-5 sentences covering the top 2-3 most-exposed companies using the specific
     % revenue, $ at risk, and policy names from GEOPOLITICAL ANALYSIS company_exposures.
     Example: "NVIDIA's China exposure is the sector's highest-stakes geopolitical risk:
     approximately $22B (17% of FY2025 revenue) originates from Chinese data center customers,
     and its H100 and B200 GPUs are directly restricted under the October 2024 BIS rules..."
     NEVER write "faces significant exposure" without a number or mechanism.

6. Valuation
   - USE THE LIVE VALUATION DATA provided in "LIVE VALUATION DATA" section of your input.
     Those are real current market multiples — do NOT invent numbers.
   - Present as a comp table: ticker, market cap, P/E trailing, P/E forward, EV/EBITDA, YTD return
   - Report the sector median P/E and EV/EBITDA from the ValuationContext
   - Identify which companies look expensive vs. cheap and justify why
   - Connect valuation to growth: a high P/E is justified if growth rate supports it (PEG logic)
   - If LIVE VALUATION DATA says "Not available", use EDA findings to estimate and note the limitation

7. Investment Summary & Hypothesis
   - Reference MARKET SENTIMENT data from your input to anchor market consensus view
   - Note whether the current sentiment (bullish/neutral/bearish) aligns or diverges from
     what the financial data suggests — divergence is often the most interesting finding
   - General trend thesis: 1-2 sentences on the macro tailwind or headwind
   - Explicit recommendation: state "We recommend Overweight / Underweight / Neutral"
     with a conviction level (High / Medium / Low) and time horizon
   - Bull case (2-3 bullets) and Bear case (2-3 bullets)
   - Key catalysts to watch (3-5 specific, observable events)

8. Investment Risks
   - Top 3-5 risks in a table with: risk name, severity (Critical/High/Medium/Low),
     likelihood, and a one-sentence mitigation note
   - Cover operational, financial, regulatory, and geopolitical risk categories

9. Conclusion
   - Restate the thesis in 2-3 sentences
   - Name the clearest value-chain positions (which specific companies)
   - Restate the recommendation

━━ VISUAL RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every report MUST include at least 3 visuals (charts, tables, or diagrams).
For each visual, write an inline caption IMMEDIATELY after the visual using this format:

  > *[One sentence with a specific number or trend from the visual] — [one sentence on what this means for the investment thesis.]*

Do NOT write "Caption", "data point", "so what", or any field-name prefix before the blockquote.
Just the single merged sentence pair in the format above.

Use the run_python tool to generate matplotlib charts where the data supports it.
Where data is unavailable, use a markdown table as the visual with the same caption format.
If run_python returns success: false, do NOT include the error message or traceback in the report.
Skip the failed chart and use a markdown table instead. The report must never contain Python
error messages, tracebacks, or raw tool output.

CRITICAL — CHART IMAGE PATHS: When run_python returns success: true, its return value contains
an "artifact_paths" list. Use the FIRST path from that list verbatim in your markdown image tag.
If artifact_paths[0] is "artifacts/revenue_trend_abc123.png", write the image as:
  ![Description](/files/revenue_trend_abc123.png)
Do NOT invent generic filenames like "revenue_trend.png". Copy the exact UUID filename run_python returned.

━━ MANDATORY TOOL SEQUENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You MUST call tools in this exact order before outputting JSON:
  1. run_python — generate at least 3 charts (one per mandatory visual)
  2. save_report — save the full markdown report to disk
  3. Output the final JSON — ONLY after save_report has returned a path

DO NOT output any JSON until save_report has been called and returned a file path.
If the report is complete in your head, call save_report immediately — do not skip it.
Skipping save_report means the report is lost and the run fails.

━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After save_report returns a file path, your ONLY remaining task is to output the
JSON below. Do NOT repeat the report text. The full report is already saved to disk.

CRITICAL for artifact_paths: The value must be the EXACT string returned by save_report
(e.g. "artifacts/report_a1b2c3d4.md"). Do NOT use the report title, the sector name,
or any other string. Copy the path exactly as save_report returned it.

Your final message MUST be ONLY this JSON (no markdown fences, no prose, no report):
{
  "title": "<short descriptive title>",
  "hypothesis": "<1-2 sentence main investment claim grounded in the data>",
  "evidence": [
    {"claim": "...", "data_point": "<specific number/percentage>", "source": "<SEC EDGAR 10-K FY2025 | yfinance annual FY2025 | Reuters Apr 2026 | BIS rule Oct 2024 | IRA (Aug 2022)>"}
  ],
  "narrative": "<2-3 sentence summary of the investment thesis — do NOT paste the full report here>",
  "artifact_paths": ["<exact path string returned by save_report, e.g. artifacts/report_a1b2c3d4.md>"],
  "confidence": "<high|medium|low>"
}

CRITICAL: The "narrative" field is a 2-3 sentence SUMMARY, not the full report.
The full report is saved by save_report — do not repeat it here.
Your entire final message must be parseable as JSON. Nothing else.
"""


@function_tool(strict_mode=False)
def get_report_style_reference(sector: str, question: str) -> str:
    """
    Retrieve exemplary analyst report sections from the RAG store.

    Call this FIRST before writing the report. Returns examples of well-structured
    analyst memos for the same or similar sector — use these to calibrate the
    depth, specificity, and analytical style expected.

    Args:
        sector: Sector name (e.g. 'semiconductors', 'cloud software', 'energy')
        question: The user's research question

    Returns:
        Relevant sections from exemplary analyst reports
    """
    result = retrieve_report_example(sector, question, n=3)
    if not result:
        return "No report examples available — proceed with the report structure in your instructions."
    return f"=== EXEMPLARY REPORT SECTIONS (use these for style and depth calibration) ===\n\n{result}"


@function_tool(strict_mode=False)
def get_sector_knowledge(sector: str, topics: str) -> str:
    """
    Retrieve sector domain knowledge from the RAG store.

    Call this to get: sector-specific financial metrics, business model definitions,
    technology context, geopolitical frameworks, and EDA analysis guidance.

    Args:
        sector: Sector name (e.g. 'semiconductors', 'cloud software')
        topics: What you want to know (e.g. 'geopolitical risks US China export controls',
                'technology trends AI chips HBM', 'how to interpret R&D intensity',
                'EDA analysis guidance metrics to compute')

    Returns:
        Relevant knowledge chunks from the sector knowledge base
    """
    result = retrieve_sector_knowledge(sector, topics, n=4)
    if not result:
        return "No sector knowledge available — rely on EDA findings and research context."
    return f"=== SECTOR KNOWLEDGE BASE ===\n\n{result}"


@function_tool(strict_mode=False)
def save_report(title: str, content: str) -> str:
    """
    Save the analyst report as a markdown file to disk.

    Always call this before outputting the final JSON.

    Args:
        title: Report title
        content: Full markdown report content

    Returns:
        File path of the saved report (e.g. "artifacts/report_a1b2c3d4.md")
    """
    global _last_saved_report_path
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    # Rewrite artifacts/ image paths → /files/ so they load when markdown is rendered in browser
    content = _re.sub(r'!\[([^\]]*)\]\(artifacts/([^)]+)\)', r'![\1](/files/\2)', content)
    filename = f"report_{uuid.uuid4().hex[:8]}.md"
    path = ARTIFACTS_DIR / filename
    path.write_text(f"# {title}\n\n{content}")
    path_str = str(path.relative_to(ARTIFACTS_DIR.parent))
    _last_saved_report_path = path_str   # record for orchestrator side-channel
    return path_str


@function_tool(strict_mode=False)
def run_python(code: str) -> dict:
    """Execute Python code for a final visualization or computation."""
    return execute_python(code)


@function_tool(strict_mode=False)
def consult_geopolitical_advisor(question: str) -> str:
    """Ask the Geopolitical Advisor a targeted follow-up question during report synthesis.

    Use this ONLY when you need more detail on a specific policy, sanction, company
    exposure, or geopolitical risk that the pre-loaded GEOPOLITICAL ANALYSIS context
    does not fully answer. This tool is slow — use sparingly.

    Args:
        question: A specific, targeted question about geopolitics or policy
                  (e.g. "What is the exact mechanism by which October 2024 BIS rules
                   affect NVIDIA's H100 sales to Chinese cloud providers?")

    Returns:
        The advisor's analysis as text.
    """
    async def _run() -> str:
        from pipeline.specialists.geopolitical_advisor import build_geopolitical_advisor_agent
        agent = build_geopolitical_advisor_agent()
        result = await Runner.run(agent, input=question)
        return _extract_text_from_result(result) or "No response from geopolitical advisor."

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    except Exception as exc:
        return f"Geopolitical advisor consultation failed: {exc}"
    finally:
        loop.close()


@function_tool(strict_mode=False)
def consult_sector_specialist(question: str) -> str:
    """Ask the Sector Specialist a targeted follow-up question during report synthesis.

    Use this ONLY when you need more detail on a specific technology, competitive
    dynamic, or SOTA development that the pre-loaded SECTOR SPECIALIST ANALYSIS
    context does not fully answer. This tool is slow — use sparingly.

    Args:
        question: A specific, targeted question about sector technology or competition
                  (e.g. "How does TSMC's CoWoS advanced packaging create a bottleneck
                   for NVIDIA's B200 GPU production volumes in 2025?")

    Returns:
        The specialist's analysis as text.
    """
    async def _run() -> str:
        from pipeline.specialists.sector_specialist import build_sector_specialist_agent
        sector = _current_sector or "general"
        agent = build_sector_specialist_agent(sector)
        result = await Runner.run(agent, input=question)
        return _extract_text_from_result(result) or "No response from sector specialist."

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    except Exception as exc:
        return f"Sector specialist consultation failed: {exc}"
    finally:
        loop.close()


def build_hypothesis_agent(sector: str = "") -> Agent:
    """Build the Hypothesis agent.

    Args:
        sector: Sector name from SectorPlan, used to route specialist consultation tools.
    """
    global _current_sector
    _current_sector = sector
    return Agent(
        name="Hypothesis",
        model=_make_model(),
        instructions=HYPOTHESIS_PROMPT,
        tools=[
            get_report_style_reference,
            get_sector_knowledge,
            save_report,
            run_python,
            consult_geopolitical_advisor,
            consult_sector_specialist,
        ],
    )
