---
sector: general
type: eda_playbook
---

# EDA Playbook: Analysis Patterns for Sector Analysts

## Core Principle: Ask WHY, Not Just WHAT

When the data shows a significant rise or fall, the analyst's job is to explain WHY — connecting the number to known events, market dynamics, product cycles, or external shocks. Never simply report that revenue grew or declined; always ask: what caused this?

**Template for explaining revenue movements:**
- Sudden 30%+ growth → likely product launch, new market, acquisition, external demand shock (AI buildout, post-COVID restocking, commodity price surge)
- 20%+ decline → likely demand collapse, market share loss, competitive disruption, export control, inventory correction after over-ordering
- Flat/slow growth → mature market, pricing pressure, market saturation, defensive incumbent
- Accelerating growth trend → compounding market share or expanding TAM; most valuable signal

## Analysis Patterns by Question Type

### "Who is leading the sector?"
DO:
- Rank by most recent fiscal year revenue (not historical average)
- Show 3-year CAGR alongside current scale — high-CAGR smaller companies often more important than low-CAGR leaders
- Compute market share trajectory: is the leader gaining or losing share?
- Separate absolute revenue from growth trajectory: the answer may be "Intel is larger today, NVIDIA is more important"

DON'T:
- Rank by average historical revenue — leads to systematically misleading conclusions about high-growth companies
- Stop at revenue: gross margin and operating margin reveal profitability quality

SQL pattern:
```sql
-- Most recent year ranking
SELECT ticker, fiscal_year, value_billions
FROM financials
WHERE metric='revenue' AND form='10-K'
AND fiscal_year = (SELECT MAX(fiscal_year) FROM financials WHERE form='10-K')
ORDER BY value_billions DESC
```

Python pattern (CAGR):
```python
rev = df[(df.metric=='revenue') & (df.form=='10-K')].pivot_table(
    index='fiscal_year', columns='ticker', values='value_billions').sort_index()
cagr = ((rev.iloc[-1] / rev.iloc[0]) ** (1 / (len(rev)-1)) - 1) * 100
print("3-year revenue CAGR:")
print(cagr.sort_values(ascending=False).round(1))
```

### "Is the sector growing?"
DO:
- Compute sector-aggregate revenue YoY growth
- Identify which companies are driving growth (is it broad-based or concentrated?)
- Decompose: organic growth vs. acquisition-driven
- Compare sector growth to GDP growth (tech: should outgrow GDP; utilities: GDP-correlated)
- Check for inflection points: does growth accelerate at a specific year? Connect to known event.

Python pattern (YoY growth per company):
```python
rev = df[(df.metric=='revenue') & (df.form=='10-K')].pivot_table(
    index='fiscal_year', columns='ticker', values='value_billions').sort_index()
yoy = rev.pct_change() * 100
print("YoY Revenue Growth % by company:")
print(yoy.round(1).to_string())
```

### "What are the margin trends?"
DO:
- Compute gross margin, operating margin, and net margin as three separate series
- Gross margin compression = pricing pressure or cost inflation
- Operating margin expansion despite flat gross margin = operating leverage
- Net margin diverging from operating margin = interest expense or tax changes
- Compare margin levels across companies — identify structural outliers (NVIDIA gross margin 74% vs. Intel 41% signals very different business quality)

### "Is R&D investment paying off?"
DO:
- Compute R&D intensity (R&D / Revenue %) trend
- Compare against revenue growth — if R&D% rising but revenue declining, the investment isn't translating
- Compare against gross margin — rising R&D with rising gross margin = product differentiation working
- Note: some companies capitalize R&D (appears as capex), others expense it entirely (appears in R&D)

### "What is the valuation story?"
(When price/market cap data is not available from SEC, note this limitation but provide framework)
- Revenue growth + gross margin = proxy for business quality
- High growth + high gross margin = justifies premium valuation (NVIDIA, ASML)
- Low growth + low gross margin = likely value trap (legacy semiconductors)
- The market is not valuing trailing revenue — it is valuing the expected future revenue trajectory

## Interpreting Specific Patterns

### Revenue spike followed by decline
**Pattern**: Revenue grows 50%+ for 1-2 years, then falls 20-30%.
**Cause**: Almost always inventory/demand cycle. Customers over-ordered during shortage, then had excess inventory. Semiconductor industry has a well-documented ~3-year cycle.
**Example**: Micron (DRAM/NAND), Analog Devices, Texas Instruments — all experienced post-COVID inventory correction in 2022-2023.
**What to say**: "Revenue decline in [year] reflects inventory normalization following the COVID-era demand pull-forward, not structural demand loss."

### Gradual share loss over 5+ years
**Pattern**: Company revenue grows slowly while peers grow fast; market share declining.
**Cause**: Competitive disruption, product generation fall-behind, or market mix shift away from the company's core segment.
**Example**: Intel CPU market share loss to AMD in server (2020-2024); traditional auto OEM EV share loss to Tesla/BYD.
**What to say**: "Intel's server revenue declined from X% of data center market in 2019 to Y% in 2024, as AMD EPYC gained acceptance based on superior performance-per-dollar metrics. This is structural, not cyclical."

### Sudden profitability collapse
**Pattern**: Revenue flat or growing, but operating/net income collapses.
**Cause**: Input cost surge (memory prices, substrate costs), one-time impairments (goodwill writedown, restructuring), or R&D step-function increase for product development.
**What to say**: identify the specific line item driving the collapse, not just report the result.

### Revenue acceleration connecting to external events
**NVIDIA FY2024**: Revenue tripled. Why? H100 GPU launch into hyperscaler AI training market. ChatGPT released November 2022, triggering $100B+ in AI infrastructure commitments from hyperscalers in 2023.
**Cloud companies 2020**: Revenue accelerated. Why? COVID drove enterprise cloud adoption 3-5 years ahead of schedule; office closures forced remote work tool adoption.
**Energy companies 2022**: Revenue surged. Why? Russia-Ukraine war disrupted European gas supply; LNG prices spiked 500%; oil above $120/barrel.

## Chart Selection Guide

| Analysis goal | Best chart type |
|---|---|
| Revenue trend per company over time | Line chart (one line per company) |
| Current year ranking comparison | Bar chart (one bar per company) |
| Margin trend over time | Line chart with % format |
| YoY growth rates | Bar chart or line chart with % format |
| Revenue composition (segments) | Waterfall or stacked bar |
| Correlation between two metrics | Scatter (use run_python + matplotlib) |
| CAGR comparison across companies | Bar chart of CAGR values |

## What NOT to Compute

- **Simple average revenue across all history**: Systematically wrong for high-growth companies; always misleads.
- **YoY growth for a company that changed its fiscal year**: Compare apples to apples.
- **Margins when gross_profit data is missing**: Note the limitation, don't fabricate.
- **Predictions/forecasts**: Report what the data shows; the Hypothesis agent forms the forward-looking view.
