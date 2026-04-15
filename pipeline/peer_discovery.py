"""Peer Discovery — validates and enriches tickers from the SectorPlan.

Pure Python function (no LLM). Called from the orchestrator after the Planner step.

For each ticker the Planner selected:
  - Calls yfinance (in a thread pool) to fetch market cap and sector metadata
  - Marks the ticker invalid if market cap is below MIN_MARKET_CAP_B or data fetch fails
  - Sorts valid peers by market cap descending
  - Falls back to the original Planner ticker list if all fetches fail (non-fatal)

Design:
  - asyncio.to_thread wraps all yfinance calls because yfinance is synchronous/blocking
    and would stall the async event loop if called directly.
  - asyncio.gather runs all ticker fetches concurrently so 15 tickers resolve in ~2s
    instead of ~30s.
"""

from __future__ import annotations

import asyncio
import logging

from models.schemas import SectorPlan, PeerList, PeerInfo
from tools.market_data import get_stock_info

logger = logging.getLogger(__name__)

MIN_MARKET_CAP_B = 1.0  # Filter: exclude companies below $1B market cap
MAX_PEERS = 12           # Cap to avoid Gemini rate limits during SEC collection


async def _fetch_one(ticker: str) -> PeerInfo:
    """Fetch yfinance info for one ticker in a thread pool (blocking → async)."""
    try:
        data = await asyncio.to_thread(get_stock_info, ticker)
        market_cap_b = data.get("market_cap_b")
        valid = market_cap_b is not None and market_cap_b >= MIN_MARKET_CAP_B
        return PeerInfo(
            ticker=ticker,
            company_name=data.get("company_name") or ticker,
            market_cap_b=market_cap_b,
            sector=data.get("sector") or "",
            industry=data.get("industry") or "",
            valid=valid,
        )
    except Exception as exc:
        logger.warning("Peer discovery failed for %s: %s", ticker, exc)
        return PeerInfo(ticker=ticker, valid=False)


async def discover_peers(plan: SectorPlan) -> PeerList:
    """Validate and enrich tickers from a SectorPlan using live market data.

    Args:
        plan: SectorPlan output from the Planner agent.

    Returns:
        PeerList with validated PeerInfo objects and tickers sorted by market cap.
        If all fetches fail, returns a PeerList using the original plan tickers as stubs
        so the pipeline can continue.
    """
    if not plan.tickers:
        return PeerList(
            sector=plan.sector,
            peers=[],
            tickers=[],
            selection_rationale="No tickers provided by Planner.",
        )

    logger.info("Peer discovery: validating %d tickers for %s", len(plan.tickers), plan.sector)

    peers: list[PeerInfo] = list(
        await asyncio.gather(*[_fetch_one(t) for t in plan.tickers])
    )

    valid_peers = sorted(
        [p for p in peers if p.valid],
        key=lambda p: p.market_cap_b or 0.0,
        reverse=True,
    )
    invalid_peers = [p for p in peers if not p.valid]

    if invalid_peers:
        logger.info(
            "Peer discovery: filtered %d invalid tickers: %s",
            len(invalid_peers),
            [p.ticker for p in invalid_peers],
        )

    # Fallback: if ALL tickers failed, use original list as stub PeerInfos
    if not valid_peers:
        logger.warning(
            "Peer discovery: all %d tickers failed validation — falling back to original list",
            len(plan.tickers),
        )
        valid_peers = [PeerInfo(ticker=t, valid=True) for t in plan.tickers]
        rationale = (
            f"Market data unavailable — using all {len(plan.tickers)} tickers "
            f"from Planner without validation."
        )
    else:
        top = valid_peers[0]
        rationale = (
            f"{len(valid_peers)} of {len(plan.tickers)} tickers passed validation "
            f"(market cap ≥ ${MIN_MARKET_CAP_B}B). "
            f"Top peer: {top.ticker} "
            + (f"(${top.market_cap_b:.1f}B)" if top.market_cap_b else "")
        )

    # Cap to MAX_PEERS (already sorted by market cap descending — keep the largest)
    if len(valid_peers) > MAX_PEERS:
        logger.info("Peer discovery: capping from %d to %d peers", len(valid_peers), MAX_PEERS)
        valid_peers = valid_peers[:MAX_PEERS]
        rationale += f" Capped to top {MAX_PEERS} by market cap."

    tickers = [p.ticker for p in valid_peers]

    logger.info("Peer discovery complete: %d valid tickers — %s", len(tickers), tickers[:6])

    return PeerList(
        sector=plan.sector,
        peers=peers,  # include invalid peers for auditability
        tickers=tickers,
        selection_rationale=rationale,
    )
