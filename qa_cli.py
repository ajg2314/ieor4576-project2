#!/usr/bin/env python3
"""Interactive Q&A CLI for the Sector Analyst system.

Usage:
    python qa_cli.py [--report artifacts/report_xxx.md]

Commands:
    summarize         → 3-page executive summary of the report
    ask: <question>   → answer a specific question using the full report context
    help              → show available commands
    quit / exit / q   → exit the loop
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import agents
agents.set_tracing_disabled(True)

os.environ.setdefault("VERTEXAI_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
os.environ.setdefault("VERTEXAI_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))

from agents import Runner, ItemHelpers
from agents.items import MessageOutputItem

from pipeline.qa_agent import build_qa_agent
from pipeline.orchestrator import get_last_run_context
from models.schemas import QAResponse


def _find_latest_report() -> Path | None:
    artifacts = Path("artifacts")
    if not artifacts.exists():
        return None
    reports = sorted(
        artifacts.glob("report_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def _extract_text(result) -> str:
    raw = result.final_output
    if raw and str(raw).strip():
        return str(raw)
    for item in reversed(result.new_items):
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text.strip():
                return text
    return ""


async def _ask(qa_agent, question: str) -> str:
    result = await Runner.run(qa_agent, input=question)
    raw = _extract_text(result)

    # Parse JSON response and return the answer field
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        response = QAResponse.model_validate_json(cleaned)
        return response.answer
    except Exception:
        # Fall back to raw text if JSON parsing fails
        return raw or "(No response generated.)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sector Analyst Q&A CLI")
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="Path to the analyst report markdown file (default: most recent in artifacts/)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve report path
    report_path: Path | None = None
    if args.report:
        report_path = Path(args.report)
    else:
        # Try orchestrator context first (works when run in same process), then disk
        ctx = get_last_run_context()
        ctx_path = ctx.get("report_path")
        if ctx_path:
            report_path = Path(ctx_path)
        if not report_path or not report_path.exists():
            report_path = _find_latest_report()

    report_text = ""
    if report_path and report_path.exists():
        report_text = report_path.read_text()
    else:
        print("[WARNING] No report found. Q&A will rely on specialist analyses only.", file=sys.stderr)

    ctx = get_last_run_context()
    sector = ctx.get("sector", "")
    tickers = ctx.get("tickers", [])
    geo_analysis = ctx.get("geo_analysis")
    sector_analysis = ctx.get("sector_analysis")

    qa_agent = build_qa_agent(
        full_report=report_text,
        sector=sector,
        tickers=tickers,
        geo_analysis=geo_analysis,
        sector_analysis=sector_analysis,
    )

    print("\n=== Sector Analyst Q&A ===")
    print(f"Report : {report_path or 'none'}")
    print(f"Sector : {sector or 'unknown'}")
    if tickers:
        print(f"Tickers: {', '.join(tickers[:10])}" + (" ..." if len(tickers) > 10 else ""))
    print("\nCommands: summarize | ask: <question> | help | quit\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        if line in ("quit", "exit", "q"):
            print("Bye.")
            break

        if line == "help":
            print(
                "\n  summarize           → 3-page executive summary\n"
                "  ask: <question>     → answer using full report context\n"
                "  quit / exit / q     → exit\n"
            )
            continue

        # Normalize command
        if line == "summarize":
            question = "summarize"
        elif line.startswith("ask:"):
            question = line[4:].strip()
            if not question:
                print("  Usage: ask: <your question>")
                continue
        else:
            # Treat bare text as a question
            question = line

        print("  [thinking...]\n")
        try:
            answer = asyncio.run(_ask(qa_agent, question))
            print(answer)
            print()
        except Exception as exc:
            print(f"  [ERROR] {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
