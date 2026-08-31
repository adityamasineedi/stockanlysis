"""CLI: free data preflight before paid /analyze."""

from __future__ import annotations

import argparse
import sys

from stockbot.config import setup_logging
from stockbot.data_readiness import assemble_brief_for_analysis
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.models import AmbiguousMatch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify free data sources and fallbacks before spending LLM tokens."
    )
    parser.add_argument("symbol", help="NSE symbol or company name")
    args = parser.parse_args()
    setup_logging()

    table = load_symbol_table()
    resolved = resolve_ticker(args.symbol, table)
    if resolved is None:
        print(f"Not found: {args.symbol!r}", file=sys.stderr)
        raise SystemExit(1)
    if isinstance(resolved, AmbiguousMatch):
        print(f"Ambiguous: {args.symbol!r}", file=sys.stderr)
        raise SystemExit(1)

    _brief, report = assemble_brief_for_analysis(resolved)
    print(report.markdown_summary())
    if not report.ready_for_llm:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
