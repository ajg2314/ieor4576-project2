---
sector: semiconductors
type: domain_knowledge
---

# Semiconductor Sector: Domain Knowledge for Analysts

## Business Models

**IDM (Integrated Device Manufacturer)**: Designs AND manufactures its own chips. Intel, Samsung, and (partially) Texas Instruments. Higher capex burden; vertical integration provides control of process technology. Increasingly uncompetitive as process R&D costs exceed $15B/node below 7nm.

**Fabless**: Designs chips, outsources manufacturing entirely to foundries. NVIDIA, AMD, Qualcomm, Broadcom, ARM. Asset-light model; gross margins 55-75%. Dependent on TSMC for advanced nodes — a concentration risk.

**Foundry (Pure-play)**: Manufactures chips designed by others. TSMC (52% global foundry market share), Samsung Foundry, GlobalFoundries, SMIC. Revenue driven by volume and ASP (average selling price); margins 50-60% for TSMC, lower for others.

**OSAT (Outsourced Semiconductor Assembly and Test)**: Packages and tests chips after manufacturing. ASE Group, Amkor. Lower margin (20-35%) but critical bottleneck for advanced packaging (CoWoS, SoIC, HBM integration).

**Equipment makers**: Build machines that fabs use. ASML (EUV monopoly), Applied Materials, Lam Research, KLA. High-margin, long-cycle businesses; revenues lag capex decisions by 6-18 months.

**EDA/IP**: Design automation software and reusable chip IP. Synopsys, Cadence, ARM. Asset-light, recurring revenue. ARM licenses its CPU architecture — every mobile chip and increasingly data center uses ARM cores.

## Key Metrics to Compute

- **Gross margin**: NVIDIA ~74%, TSMC ~54%, Intel ~41%, AMD ~50%. Higher margin = better pricing power / differentiation.
- **R&D intensity (R&D/Revenue %)**: Intel ~28%, AMD ~22%, NVIDIA ~12%, TSMC ~8%. High R&D% with falling revenue = burning cash on catch-up. High R&D% with rising revenue = investing ahead of demand.
- **Capex/Revenue ratio**: TSMC ~30-35% (justified by leading-edge fab investment), Intel ~25-30% (questioned given returns), fabless <5%.
- **Revenue CAGR by segment**: Always separate data center from PC/consumer. Data center has grown 60%+ annually; PC is flat to declining.
- **Book-to-bill ratio**: Orders/shipments. Above 1.0 = demand exceeding supply; below 1.0 = inventory correction. Leading indicator by 2-3 quarters.
- **Days Sales Outstanding (DSO)** and **inventory turns**: Rising inventory = demand slowdown coming.

## Technology Trends (What Drives Revenue)

**AI accelerators**: NVIDIA H100/H200/B200 GPUs are the primary revenue driver for the AI infrastructure buildout. Training a frontier LLM requires 10,000-100,000 GPUs. Inference (running the model) requires more GPUs than training at scale. This creates durable, multi-year demand.

**HBM (High Bandwidth Memory)**: Stacked DRAM used alongside AI GPUs and CPUs to handle the data bandwidth requirements of large model inference. SK Hynix, Samsung, Micron are the only manufacturers. HBM supply is the bottleneck for AI chip production — NVIDIA cannot ship GPUs without it.

**EUV lithography**: ASML's extreme ultraviolet machines are required for chips below 7nm. Each machine costs $180-350M. ASML has a global monopoly and years-long waiting lists. Access to EUV = access to leading-edge manufacturing. China is completely excluded from EUV.

**Chiplets and advanced packaging**: Rather than shrinking transistors (hitting physics limits), designers assemble multiple smaller dies on a substrate (CoWoS, EMIB, SoIC). Enables AMD's EPYC/Ryzen approach, Intel Meteor Lake, NVIDIA GB200 NVL72 rack-scale design. Advanced packaging is now a competitive bottleneck.

**3nm and 2nm nodes**: TSMC N3 (3nm equivalent) volume in 2022; N2 (2nm) production 2025. Each node transition takes 2+ years and $20B+ in R&D. Only TSMC and Samsung can compete at this level. Intel's 18A process aims to be competitive with TSMC N2 but has not been validated at volume.

**Custom silicon (ASIC)**: Hyperscalers building their own chips (Google TPU, AWS Trainium, Meta MTIA, Microsoft Maia). Optimized for specific AI workloads; more efficient than general-purpose GPUs for inference. Threat to NVIDIA's total addressable market, but CPU-equivalent software ecosystem gap limits displacement to ~15% of inference by 2027.

## Geopolitical Context

**US-China export controls (most important)**: The U.S. Entity List and Export Administration Regulations (EAR) restrict which chips can be sold to China. Key milestones:
- 2019: Huawei placed on Entity List; TSMC, ASML cut off
- Oct 2022: H100/A100 AI chips restricted (>600 TFLOPS threshold)
- Oct 2023: H800/A800 loopholes closed; geographic expansion to 40+ countries
- Oct 2024: Comprehensive AI chip export controls; new licensing requirements for advanced logic chips; chip design software restrictions
Impact: NVIDIA China revenue fell from 22% to ~13% of total. AMD similar impact. Qualcomm less affected (mobile chips exempt). Intel data center chips affected.

**CHIPS and Science Act ($52.3B, 2022)**: U.S. government subsidies to rebuild domestic semiconductor manufacturing. Direct grants and loans to: Intel ($8.5B grant + $11B loan), TSMC Arizona ($6.6B), Samsung Texas ($6.4B), Micron ($6.1B). Requires "guardrails" — recipients cannot expand China capacity for 10 years. This is permanently reshaping fab geography.

**Taiwan concentration risk**: TSMC's Hsinchu and Tainan fabs produce ~90% of sub-5nm chips. Taiwan's cross-strait tensions create a tail risk that most supply chain risk models under-price. TSMC Arizona Phase 1 (N4P) began production in 2024; N2 planned for 2028. Geographic diversification is a decade-long project.

**Netherlands EUV export ban**: ASML cannot ship EUV machines to China (ban since 2019, formalized in Dutch export license policy 2023). In 2023, the Netherlands restricted DUV (older-generation) immersion lithography exports to China as well. This sets back Chinese domestic semiconductor development by an estimated 7-10 years.

**Japan semiconductor equipment restrictions (July 2023)**: Japan restricted 23 categories of semiconductor manufacturing equipment exports. Japan-based companies (Tokyo Electron, Shin-Etsu Chemical, JSR) are critical to the global supply chain. This closes a loophole that ASML's restrictions left open.

**South Korea position**: Samsung and SK Hynix are Korean companies with significant China manufacturing operations ($40B+ in existing China fabs). US pressure to align with export controls creates tension with Korean industrial policy. SK Hynix received waivers to continue HBM production in China fabs.

**India semiconductor ambitions**: Tata-PSMC 28nm fab, Micron OSAT in Gujarat. Production 2025-2026. Mature nodes only; not competitive with TSMC's advanced nodes. But represents long-term geographic diversification of assembly and older-generation manufacturing.

## EDA Guidance: What to Analyze for Semiconductors

When analyzing semiconductor companies:
1. NEVER use average revenue — always use most recent fiscal year revenue + 3-year and 5-year CAGR
2. Segment revenue: separate data center / AI from PC / mobile / automotive / industrial
3. Compute gross margin trend — expanding margin = pricing power; compressing = competitive pressure
4. Compute R&D intensity (R&D/Revenue) — but interpret carefully: high R&D% on declining revenue is distress, not investment
5. If capex data available: compute capex/revenue ratio and note if it's rising (growth mode) or falling (harvest mode)
6. Check for revenue concentration: does one customer or segment represent >50% of revenue? That's a risk, not just a feature
7. Look for inflection points in the data (sudden acceleration or deceleration) and explain WHY — connect to known events (COVID demand surge, inventory correction, AI buildout, export control impact)

## Interpreting Revenue Movements

**Sudden acceleration**: Look for — product launch, new market entry, AI demand surge, acquisition. For NVIDIA: FY2024 data center revenue tripled due to H100 launch into AI training market.

**Sudden deceleration or decline**: Look for — inventory correction (customers overstocked during shortage, then stopped ordering), demand shift (PC decline due to smartphone substitution), market share loss, export control impact, end of a product cycle.

**Intel-specific**: FY2023-24 decline due to combination of: (1) PC market collapse post-COVID; (2) AMD server market share gains with EPYC; (3) process technology execution failures (10nm, 7nm delays); (4) data center customers moving to NVIDIA for AI workloads.

**NVIDIA-specific**: FY2024 tripling of revenue is directly attributable to hyperscaler AI training buildout, not cyclical demand. This is a structural demand shift, not a one-time event. The correct question is: what happens when training demand matures and inference demand dominates?
