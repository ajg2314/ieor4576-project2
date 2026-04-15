---
sector: general
type: financial_concepts
---

# Core Financial Concepts for Sector Analysis

## Revenue Analysis

**CAGR (Compound Annual Growth Rate)**: ((End Value / Start Value)^(1/years)) - 1
- Above 20%: high-growth; market is betting on continued expansion
- 10-20%: solid growth; typically commands premium valuation
- 0-10%: mature; valued on earnings yield
- Negative: structural decline or cyclical trough

**YoY growth**: (Current year - Prior year) / Prior year. Always compute this for every company and every metric. The single most important number in trend analysis.

**Revenue quality indicators**:
- Recurring vs. one-time revenue (SaaS ARR vs. hardware one-time)
- Geographic diversification (single-country dependence = concentration risk)
- Customer concentration (>10% customer = material risk, must be disclosed in 10-K)
- Organic vs. acquisition-driven (M&A inflates revenue; strip it out for organic view)

## Margin Analysis

**Gross Margin = (Revenue - COGS) / Revenue**
- Software / IP: 70-85% (minimal incremental cost)
- Fabless semiconductors: 50-70%
- Foundry / manufacturing: 40-55%
- Commodity / materials: 20-40%
- Retail: 25-45%

Gross margin compression signals: pricing pressure from competition, input cost inflation, product mix shift toward lower-margin segments.
Gross margin expansion signals: pricing power, mix shift to higher-value products, operational efficiency.

**Operating Margin = Operating Income / Revenue**
Operating leverage: when revenue grows faster than operating expenses, operating margin expands. This is the primary driver of earnings growth for scaling companies.

**Net Margin = Net Income / Revenue**
Net margin below operating margin due to: interest expense (debt burden), taxes. Net margin above operating margin: tax benefits, investment gains (unusual; flag it).

**EBITDA Margin**: Adds back depreciation and amortization — useful for capital-intensive businesses where depreciation is large relative to earnings. Required for fair comparison between asset-heavy (foundries, energy) and asset-light (software) companies.

## Growth vs. Profitability

**Rule of 40** (SaaS / high-growth): Revenue growth % + FCF margin % >= 40 is healthy. Below 30 requires explanation. Useful for comparing companies with different growth/profit tradeoffs.

**Operating leverage**: For every 1% of revenue growth, what % does operating income grow? If operating income grows 2x revenue growth rate, the business has positive operating leverage — a very valuable property. Compute it: operating_income_growth / revenue_growth.

## Interpreting Inflection Points

When revenue or margins show a **sudden change** (>15% in one year):
1. Is it visible in all companies or just one? (Sector-wide = macro/external; company-specific = competitive or execution)
2. What happened that year in the real world? Connect the data to news: product launches, competitor moves, macro events, regulatory changes.
3. Is the inflection sustained (structural) or reversed (cyclical)?
4. Does the change in revenue correspond to a change in margins? (If revenue drops but margins hold = pricing power; if margins collapse along with revenue = commodity shock)

## Cash Flow Analysis

**Free Cash Flow = Operating Cash Flow - Capex**
- For capital-intensive businesses (foundries, energy, telecom): FCF can be negative during investment cycles even with positive net income. Evaluate over a full cycle.
- For software/fabless: FCF should closely track net income. Large divergence = quality concern.

**Capex intensity = Capex / Revenue**
- TSMC: 30-35% (justified for leading-edge fab)
- Intel: 25-30% (debated: is it generating returns?)
- NVIDIA: <3% (fabless; capex is mainly offices and equipment)
- Software: 1-5%

High capex is not inherently bad — it signals growth investment. The question is return on invested capital (ROIC). If capex is rising but revenue is flat, flag it.

## Debt and Balance Sheet

**Net Debt = Total Debt - Cash**: Negative net debt = net cash position (common in asset-light tech). Positive net debt = leverage. Leverage is fine if cash generation is strong and rising.

**Interest coverage ratio = EBIT / Interest Expense**: Above 5x is healthy. Below 2x is distress signal.

**Debt maturity profile**: When does debt come due? If major maturities in next 2 years and FCF is insufficient, refinancing risk is material. This matters in rising rate environments.

## SEC Filing Specifics

**10-K**: Annual report. Most reliable data. Audited. Use for YoY comparisons, margin trends, capex, segment data.

**10-Q**: Quarterly report. Unaudited but timely. Use for recent trend detection, quarterly seasonality.

**MD&A (Management Discussion & Analysis)**: Item 7 of the 10-K. Contains management's explanation of why revenue/margins moved. Critical for connecting data to cause. Always read this before concluding on revenue movements.

**Risk Factors (Item 1A)**: Forward-looking risks that management is required to disclose. If a geopolitical risk (export controls, China exposure) is in Risk Factors, management has acknowledged it as material.

**Segment reporting**: Companies often report separate revenue/profit for segments (Data Center, Client/PC, Networking). Always disaggregate — total company revenue masks the most important signal (which segment is driving growth or drag).
