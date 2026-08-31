"""Module 8 — verification harness. CLI that runs the full fetch layer
(Modules 1-7) over a ticker list and writes one human-checkable markdown
report: every fetched value, source, timestamp, which modules failed, the
MISSING list, financials basis, annual-report truncation status, and
token count, per ticker.

This is the Phase 1 hard stop. Nothing in Phase 2 (the paid LLM stages)
is trustworthy until a human has checked these numbers against the real
source — Screener, NSE, the actual annual report PDF — for every ticker
in the report this produces.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from stockbot.brief import to_markdown
from stockbot.config import PROJECT_ROOT, setup_logging
from stockbot.data_readiness import assemble_brief_for_analysis
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.models import AmbiguousMatch, TickerInfo

DEFAULT_TICKERS = ["RELIANCE", "TCS", "JYOTHYLAB"]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "manual"


class ResolutionFailed(Exception):
    pass


def _resolve_or_raise(query: str, table) -> TickerInfo:
    result = resolve_ticker(query, table)
    if result is None:
        raise ResolutionFailed(f"{query!r} did not resolve — check spelling or symbol")
    if isinstance(result, AmbiguousMatch):
        candidates = ", ".join(f"{c.symbol} ({c.company_name})" for c in result.candidates)
        raise ResolutionFailed(f"{query!r} is ambiguous: {candidates} — pass the exact NSE symbol")
    return result


def _report_for_ticker(query: str, table) -> str:
    lines = [f"## {query}", ""]

    try:
        ticker = _resolve_or_raise(query, table)
    except ResolutionFailed as exc:
        lines.append(f"**RESOLUTION FAILED**: {exc}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Resolved: **{ticker.symbol}** — {ticker.company_name} ({ticker.exchange})")
    lines.append("")

    t0 = time.monotonic()
    try:
        _brief, readiness = assemble_brief_for_analysis(ticker)
    except Exception as exc:  # noqa: BLE001 - this harness's job is to isolate one bad ticker
        lines.append(f"**FETCH FAILED (fatal — likely price/technicals)**: {exc}")
        lines.append("")
        return "\n".join(lines)
    elapsed = time.monotonic() - t0

    basis = _brief.financials.basis if _brief.financials else "MISSING"
    lines.extend(
        [
            f"- Fetched in {elapsed:.1f}s",
            f"- **Ready for LLM:** {'yes' if readiness.ready_for_llm else 'no'}",
            f"- Confidence ceiling: {_brief.confidence_ceiling}/10",
            f"- Token count: {_brief.token_count}",
            f"- Financials basis: {basis}",
            f"- Annual report truncated: {_brief.annual_report.truncated}",
            f"- Annual report sections found: {list(_brief.annual_report.sections.keys())}",
            "",
            readiness.markdown_summary(),
            "",
        ]
    )

    if _brief.missing:
        lines.append("### MISSING / degraded modules")
        lines.extend(f"- {entry}" for entry in _brief.missing)
        lines.append("")

    lines.append("### Full brief")
    lines.append("")
    lines.append(to_markdown(_brief))
    lines.append("")
    return "\n".join(lines)


def run_verification(tickers: list[str]) -> Path:
    setup_logging()
    table = load_symbol_table()

    sections = []
    for query in tickers:
        print(f"Fetching {query}...", file=sys.stderr)
        sections.append(_report_for_ticker(query, table))

    report = "\n".join(
        [
            "# Fetch Layer Verification Report",
            f"*Generated {datetime.now(UTC).isoformat()}*",
            "",
            *sections,
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"verify_{timestamp}.md"
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fetch layer over a ticker list and write a human-checkable report."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Company names or NSE symbols to verify (default: RELIANCE TCS JYOTHYLAB)",
    )
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else DEFAULT_TICKERS
    output_path = run_verification(tickers)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
