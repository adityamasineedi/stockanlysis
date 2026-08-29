"""Query prescan outcomes — local JSONL or pull from Railway volume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stockbot.config import setup_logging
from stockbot.portfolio_screener.outcome_log import (
    OUTCOMES_PATH,
    format_prescan_table,
    load_prescan_outcomes,
    pull_prescan_from_railway,
    query_prescan_outcomes,
    summarize_outcomes,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Query prescan outcome log (quality/quant filters). "
            "Use --pull-railway to sync from the Railway volume first."
        ),
    )
    p.add_argument(
        "--path",
        type=Path,
        default=OUTCOMES_PATH,
        help=f"JSONL path (default: {OUTCOMES_PATH})",
    )
    p.add_argument(
        "--pull-railway",
        action="store_true",
        help="Fetch prescan_outcomes.jsonl from Railway via `railway ssh`",
    )
    p.add_argument(
        "--railway-service",
        default="stockanlysis",
        help="Railway service name for --pull-railway (default: stockanlysis)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write pulled JSONL here (default: --path)",
    )
    p.add_argument("--min-quality", type=float, default=None, help="Minimum Q score")
    p.add_argument("--min-growth", type=float, default=None, help="Minimum G score")
    p.add_argument("--min-strength", type=float, default=None, help="Minimum S score")
    p.add_argument("--min-quant", type=float, default=None, help="Minimum quant_score")
    p.add_argument(
        "--band",
        action="append",
        default=None,
        help="Candidate band filter (repeatable): STRONG_CANDIDATE, CANDIDATE, WATCHLIST",
    )
    p.add_argument(
        "--analyze-ready",
        action="store_true",
        help=(
            "Only AUTO_DEEP_ANALYSIS / SECTOR_SPECIFIC_REVIEW with cash PASS|WATCH|NOT_APPLICABLE "
            "and no HARD_EXCLUDE"
        ),
    )
    p.add_argument("--json", action="store_true", help="Print matching rows as JSON array")
    p.add_argument("--summary", action="store_true", help="Print reject_class counts only")
    return p


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    setup_logging()
    args = build_parser().parse_args(argv)

    target = args.out or args.path
    if args.pull_railway:
        try:
            pull_prescan_from_railway(service=args.railway_service, dest=target)
        except (RuntimeError, OSError) as exc:
            print(f"Railway pull failed: {exc}", file=sys.stderr)
            return 1
        print(f"Pulled prescan log → {target}")

    if not target.exists():
        print(
            f"No prescan log at {target}. Run prescans on the bot or use --pull-railway.",
            file=sys.stderr,
        )
        return 1

    if args.summary:
        print(json.dumps(summarize_outcomes(target), indent=2))
        return 0

    rows = load_prescan_outcomes(target)
    bands = set(args.band) if args.band else None
    matched = query_prescan_outcomes(
        rows,
        min_quality=args.min_quality,
        min_growth=args.min_growth,
        min_strength=args.min_strength,
        min_quant=args.min_quant,
        bands=bands,
        analyze_ready_only=bool(args.analyze_ready),
    )

    missing_q = sum(1 for r in rows if r.get("quality_score") is None)
    if missing_q and (args.min_quality is not None or args.min_growth is not None):
        print(
            f"Note: {missing_q}/{len(rows)} rows lack Q/G/S (deploy updated bot + re-prescan). "
            "Using quant_score / verdict filters only for those rows.",
            file=sys.stderr,
        )

    if args.json:
        print(json.dumps(matched, indent=2, ensure_ascii=False))
    else:
        print(f"Source: {target} · {len(rows)} tickers · {len(matched)} matched\n")
        print(format_prescan_table(matched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
