---
description: Grade the most recent sector analyst report against the quality rubric. Pass an optional report path as argument, e.g. /grade-report artifacts/report_abc123.md
allowed-tools: Read, Glob, Bash
---

Grade the sector analyst report against the quality rubric below.

## Step 1 — Find the report

If the user passed a file path as an argument ($ARGUMENTS), read that file.
Otherwise, find the most recently modified `artifacts/report_*.md` file using Glob, then read it.

## Step 2 — Score each dimension

Read the full report text, then score it against every dimension below.
For each dimension, assign a score **0–10** and write 1–2 sentences of evidence (quote the report or note what is missing).

---

## THE RUBRIC (100 points total)

### SECTION COMPLETENESS (36 pts)

| # | Section | Pts | Pass criteria |
|---|---------|-----|---------------|
| 1 | **Business Description & Revenue Model** | 4 | Explains how companies in the sector make money; defines technical terms on first use |
| 2 | **Industry Overview & Competitive Positioning** | 4 | Ranks top companies by most-recent-FY revenue (not averages); includes a revenue ranking table or bar chart |
| 3 | **Financial Analysis** | 4 | Reports revenue CAGR and YoY growth; margin analysis; R&D intensity; includes a revenue-over-time line chart; explains any >15% YoY move with a real-world cause |
| 4 | **Technology & Innovation (Forward-Looking)** | 4 | Minimum 2 substantial paragraphs on what the market is betting on for 3–5 years; names specific technologies, products, or milestones |
| 5 | **Geopolitical & Macro Environment** | 4 | Named and dated policies (not generic "trade tensions"); company-level exposure with % or $ amounts; covers at least 2 of: (a) Middle East/Hormuz/Houthis, (b) China-US export controls/tariffs, (c) Russia-Ukraine/NATO |
| 6 | **Valuation** | 4 | Comp table with real multiples (P/E, EV/EBITDA, YTD return); sector median stated; identifies cheapest and most expensive vs. peers with reasoning |
| 7 | **Investment Summary & Hypothesis** | 4 | Explicit Overweight/Underweight/Neutral recommendation with conviction level and time horizon; bull case and bear case bullet points; 3–5 specific catalysts |
| 8 | **Investment Risks** | 4 | Risk table with severity, likelihood, and mitigation; covers operational, financial, regulatory, and geopolitical categories |
| 9 | **Conclusion** | 4 | Restates thesis in 2–3 sentences; names specific companies; restates recommendation |

### DATA QUALITY (28 pts)

| Dimension | Pts | Pass criteria |
|-----------|-----|---------------|
| **Data recency** | 8 | Uses data from 2025 or 2026 Q1 where available; not anchored only to 2022–2023 figures; mentions current year developments |
| **Specificity — numbers** | 8 | Quantified claims throughout (revenue in $B, growth %, margins %, market share %); avoids vague phrases like "significant growth" without a number |
| **Specificity — geopolitics** | 6 | At least one named, dated policy per geopolitical claim (e.g. "US BIS Oct 2024 export controls restrict chips >1,800 TFLOPS"); never just "geopolitical risks exist" |
| **Source attribution** | 6 | Claims attributed to sources ("SEC EDGAR 10-K", "analyst research", "industry reports"); no fabricated-looking numbers without attribution |

### REPORT HYGIENE (20 pts)

| Dimension | Pts | Pass criteria |
|-----------|-----|---------------|
| **No garbage output** | 8 | No Python tracebacks, JSON field names (e.g. `"key_policies":`), raw error messages, or XBRL artifacts in the body text |
| **Visuals present** | 6 | At least 3 visuals (charts or markdown tables) with two-part captions (data point + so-what); chart file paths referenced if generated |
| **Sector Snapshot header** | 6 | Report opens with a Sector Snapshot block (small table of key statistics) before the prose sections begin |

### ANALYTICAL QUALITY (16 pts)

| Dimension | Pts | Pass criteria |
|-----------|-----|---------------|
| **Forward-looking framing** | 8 | Report explains what the market is *pricing in*, not just what happened historically; bull/bear case tied to observable future events |
| **Cause-and-effect reasoning** | 8 | When revenue/margins move >15% YoY, the report connects the data to a real-world event (product launch, export control, inventory cycle, competitive move); avoids "revenue declined" without a because |

---

## Step 3 — Compute total and output the scorecard

Output the results in this format:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SECTOR ANALYST REPORT GRADER
  Report: <filename>
  Sector: <sector from report>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION COMPLETENESS  xx/36
  §1 Business Description          x/4  — <evidence>
  §2 Industry Overview             x/4  — <evidence>
  §3 Financial Analysis            x/4  — <evidence>
  §4 Technology & Innovation       x/4  — <evidence>
  §5 Geopolitical & Macro          x/4  — <evidence>
  §6 Valuation                     x/4  — <evidence>
  §7 Investment Summary            x/4  — <evidence>
  §8 Investment Risks              x/4  — <evidence>
  §9 Conclusion                    x/4  — <evidence>

DATA QUALITY          xx/28
  Data recency                     x/8  — <evidence>
  Specificity — numbers            x/8  — <evidence>
  Specificity — geopolitics        x/6  — <evidence>
  Source attribution               x/6  — <evidence>

REPORT HYGIENE        xx/20
  No garbage output                x/8  — <evidence>
  Visuals present                  x/6  — <evidence>
  Sector Snapshot header           x/6  — <evidence>

ANALYTICAL QUALITY    xx/16
  Forward-looking framing          x/8  — <evidence>
  Cause-and-effect reasoning       x/8  — <evidence>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOTAL SCORE:  xx/100   GRADE: <A/B/C/D/F>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRADE SCALE:  90–100 A | 80–89 B | 70–79 C | 60–69 D | <60 F

TOP 3 STRENGTHS:
  1. <specific strength with quote>
  2. <specific strength>
  3. <specific strength>

TOP 3 WEAKNESSES (most impactful improvements):
  1. <specific gap with section reference>
  2. <specific gap>
  3. <specific gap>
```

Be strict and evidence-based. Quote the report to justify scores above 7/10.
Deduct full points for any garbage output, missing sections, or vague geopolitical language.
