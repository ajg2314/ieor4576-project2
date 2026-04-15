---
sector: biomedical_pharma
type: knowledge
---

# Biomedical & Pharmaceutical Sector Knowledge Base

## Business Models

### Drug Developers (Biopharma / Big Pharma)
Companies like Pfizer, Eli Lilly, AbbVie, and Regeneron discover and commercialize drugs. Revenue comes from product sales (proprietary drugs) and royalties. Gross margins are typically 70–85% once a drug is approved because manufacturing costs are low relative to price — the real cost is the R&D and clinical development to get to approval. Companies protect revenues through patents (typically 20 years from filing, giving 10–12 years of effective market exclusivity after approval timelines).

### Biotech (Early-Stage Drug Discovery)
Pure-play biotechs (Moderna, BioNTech, Vertex, CRISPR Therapeutics) focus on a smaller number of novel therapeutic approaches. They often carry no or minimal revenue in early stages and are valued on probability-weighted NPV (net present value) of their drug pipeline. A single Phase 3 success or failure can move the stock ±40%.

### Contract Research Organizations (CROs)
IQVIA, Labcorp, Medpace run clinical trials on behalf of pharma/biotech clients. Revenue is recurring and predictable; margins are 15–25% EBITDA. CRO demand tracks overall industry R&D spending.

### Medical Devices
Medtronic, Boston Scientific, Intuitive Surgical make devices (pacemakers, surgical robots, stents). These require FDA 510(k) clearance (equivalence to existing device) or PMA (premarket approval for novel devices). Revenue is recurring via capital sales (equipment) plus high-margin disposables/consumables.

---

## FDA Drug Approval Process

Understanding the approval pipeline is critical for valuing pharma/biotech companies. Each stage represents a probability gate.

### Stage 1: Preclinical Research
- Laboratory and animal testing to establish safety and mechanism of action
- No human subjects; 3-6 years typical duration
- Leads to Investigational New Drug (IND) application to FDA

### Stage 2: Phase I Clinical Trials
- 20–80 healthy volunteers or patients
- Primary goal: safety, dosing, pharmacokinetics (how the body processes the drug)
- Success rate: ~63% advance to Phase II
- Duration: 1–2 years

### Stage 3: Phase II Clinical Trials
- 100–300 patients with the target disease
- Primary goal: efficacy signals and side effect profile
- Success rate: ~31% advance to Phase III (from Phase I start)
- Duration: 2–3 years

### Stage 4: Phase III Clinical Trials
- 300–3,000+ patients; randomized controlled trials (RCTs)
- Primary goal: prove efficacy vs. placebo/standard of care; confirm safety
- Success rate: ~58% of Phase III trials result in NDA/BLA filing
- Duration: 2–4 years
- Most expensive stage: large Phase III trials cost $200M–$2B

### Stage 5: FDA Review — NDA/BLA
- New Drug Application (NDA) for small molecule drugs
- Biologics License Application (BLA) for biologics (antibodies, gene therapies)
- PDUFA date (Prescription Drug User Fee Act): FDA commits to review within 10–12 months
- Standard review: 12 months. Priority review (breakthrough, orphan drug): 6 months
- Complete Response Letter (CRL) = rejection; requires additional data or label changes

### Overall Success Rates
- From Phase I to approval: ~7.9% for all drug types
- Oncology: ~5.1% (highest failure rate — most complex diseases)
- Hematology: ~26% (highest success rate)
- Rare diseases / Orphan drugs: ~25% (smaller, better-characterized populations)

### Key Time-to-Market Statistics
- Average total development time (IND to approval): 10–15 years
- Average cost: $1–3B fully loaded (including failures)
- Peak sales typically reached 5–7 years post-launch before generic/biosimilar entry

---

## Patent Cliff Mechanics

A patent cliff occurs when key drugs lose patent protection, allowing generic competitors to enter and rapidly erode branded drug prices (typically 80–90% price erosion within 2 years of generic entry).

**Major upcoming patent cliffs (illustrative)**:
- AbbVie's Humira (adalimumab, $20B+ annual revenue) — US biosimilar entry began January 2023
- Pfizer's Eliquis (apixaban, $12B+) — patent expiry ~2026–2028
- Merck's Keytruda (pembrolizumab, $25B+) — biosimilar entry risk ~2028–2030
- Bristol-Myers' Revlimid — already facing generics since 2022

**How to analyze patent cliff risk**:
1. Identify % of total revenue from near-patent-expiry drugs
2. Model revenue at risk (typically 70–85% erosion in year 1–2 of generic entry)
3. Evaluate pipeline to replace lost revenue ("pipeline coverage ratio")
4. Check for biosimilar complexity — biologics are harder to replicate than small molecules

---

## Pipeline Valuation (NPV Method)

Analysts value pharma/biotech pipelines using risk-adjusted NPV (rNPV):

```
rNPV = Σ [Phase probability × Peak Sales × Margin × Duration / (1+discount rate)^year]
```

Key assumptions:
- Peak sales: based on addressable patient population × price × market share
- Phase probability: Phase I = 63%, Phase II = 31%, Phase III = 58%, NDA = 85%
- WACC (discount rate): typically 10–15% for biotech (higher for earlier stage)
- Royalty rate: 5–15% of net sales for licensed assets

**Signs of undervaluation**: rNPV of pipeline > current market cap minus net cash

---

## Key Technology Areas (SOTA)

### GLP-1 Receptor Agonists (Obesity / Diabetes)
The most significant pharmaceutical opportunity in a generation. Drugs like semaglutide (Ozempic/Wegovy — Novo Nordisk) and tirzepatide (Mounjaro/Zepbound — Eli Lilly) have demonstrated 15–22% body weight reduction in clinical trials, far superior to prior obesity drugs.

Market sizing: 650M+ obese adults globally; only ~5% are treated pharmacologically. Analysts forecast the GLP-1 market reaching $130–150B by 2030. Eli Lilly's market cap surpassed $700B by 2024 primarily on GLP-1 expectations.

Key risks: manufacturing capacity constraints (peptide synthesis), insurance coverage limitations, competition from oral formulations (orforglipron), long-term cardiovascular outcomes data.

### CRISPR Gene Editing
Vertex/CRISPR Therapeutics received FDA approval (December 2023) for Casgevy — the first CRISPR-based therapy, for sickle cell disease and beta-thalassemia. One-time cure; price: $2.2M per patient.

CRISPR editing works by using guide RNA to direct Cas9 protein to cut specific DNA sequences. Limitations: off-target edits (risk of unintended mutations), delivery mechanisms, cost of manufacturing.

Next wave: in vivo CRISPR (editing genes inside the body, not in extracted cells). Companies: Intellia Therapeutics, Prime Medicine, Beam Therapeutics.

### CAR-T Cell Therapy
Chimeric Antigen Receptor T-cell therapy engineers a patient's own immune cells to attack cancer. FDA-approved products: Kymriah (Novartis), Yescarta (Kite/Gilead), Carvykti (J&J/Legend Biotech).

Market position: highly effective for certain blood cancers (B-cell malignancies); largely ineffective for solid tumors (major R&D challenge). Manufacturing is complex and expensive (~$400–500K per patient). Next generation: allogeneic (off-the-shelf) CAR-T reduces cost and manufacturing time.

### Antibody-Drug Conjugates (ADCs)
ADCs attach a cytotoxic (cancer-killing) payload to an antibody that targets cancer cells — a "guided missile" approach. Key products: Enhertu (AstraZeneca/Daiichi Sankyo), Kadcyla (Roche), Trodelvy (Gilead).

ADCs represent one of the most active M&A areas: Pfizer acquired Seagen for $43B (2023), AstraZeneca signed a $6.9B ADC partnership with Daiichi Sankyo.

### AI in Drug Discovery
Companies including Recursion Pharmaceuticals, Schrödinger, BioNTech, and Insilico Medicine use machine learning to:
- Predict protein folding (Google DeepMind's AlphaFold2 — public release 2022)
- Design novel molecules with target properties
- Identify clinical trial patient populations
- Predict drug-drug interactions

Early-stage but increasingly important: Pfizer, Merck, AstraZeneca have major internal AI programs.

---

## Geopolitical Factors

### Drug Pricing — US Policy
The Inflation Reduction Act (IRA, 2022) allows Medicare to negotiate drug prices for the first time:
- 10 drugs subject to negotiation starting 2026 (announced August 2023)
- Negotiated prices expected 40–79% below list prices for high-cost drugs
- Risk to companies with high Medicare exposure (especially for older drugs near patent expiry)
- Estimated market impact: $100B+ in foregone pharma revenue over 10 years

### China Market Access
US-China tensions affect pharma differently than tech:
- China is a significant growth market (1.4B population, rising middle class)
- Foreign drug approval in China requires separate NMPA (National Medical Products Administration) review — adds 1–3 years vs. FDA approval
- Biosimilar competition from Chinese manufacturers is growing
- Companies with significant China revenue: AstraZeneca (~25%), Novo Nordisk (~5%)

### EU Pharmaceutical Legislation
EU Pharmaceutical Strategy (2023) proposed reducing data exclusivity from 8 to 5 years to speed generic entry. This is a risk to European pharma revenue if enacted. Ongoing as of 2024.

---

## Key Metrics for EDA and Report Analysis

| Metric | Formula | What it Means |
|--------|---------|--------------|
| R&D as % of Revenue | R&D spend / Revenue | Pipeline investment intensity; biotech: 30-60%, big pharma: 15-25% |
| Revenue per approved drug | Revenue / # approved products | Blockbuster concentration risk |
| Pipeline coverage ratio | rNPV of pipeline / Revenue at patent cliff risk | Whether pipeline can replace expiring revenue |
| Gross margin | (Revenue - COGS) / Revenue | Drug pricing power; 70-85% for pharma |
| Days Sales Outstanding (DSO) | AR / (Revenue/365) | Payer mix and collection quality |
| NDA/BLA approval rate | Approvals / Submissions (trailing 5yr) | FDA track record |
| PDUFA date risk | Binary date when FDA decision expected | Near-term binary event catalyst |

### How to Interpret Revenue Movements
- **Sudden revenue spike**: New drug launch, acquisition, or pandemic-related demand (e.g., Pfizer FY2021–2022 COVID vaccine)
- **Revenue cliff**: Patent expiry + generic entry (e.g., AbbVie post-Humira biosimilar)
- **Accelerating growth in one therapeutic area**: Market penetration of a new class (e.g., Eli Lilly GLP-1 ramp)
- **Flat revenue + rising R&D**: Pipeline investment phase; look for PDUFA dates coming up
- **Falling revenue + falling R&D**: Pipeline thinning; potential M&A target

### EDA Playbook for Biomedical Sector
1. Start with most recent FY revenue ranking (not historical average)
2. Compute R&D as % of revenue — flag companies below 10% (pipeline at risk) or above 40% (early-stage)
3. Identify patent cliff exposure: what % of revenue comes from drugs expiring in <5 years?
4. Map key PDUFA dates (upcoming FDA decisions) as catalysts
5. Compare gross margins — widening margins = pricing power or mix shift to higher-margin drugs
6. For any company with >30% revenue growth: identify which drug drove it (is it sustainable?)
7. For any company with >15% revenue decline: identify patent expiry, generic entry, or pipeline failure
