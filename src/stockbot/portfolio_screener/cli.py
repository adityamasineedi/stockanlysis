"""CLI for portfolio pre-screener / single-ticker eligibility.

Examples:
  # One stock — is it worth expensive deep analysis?
  uv run stockbot-prescreen BEL
  uv run stockbot-prescreen --ticker BEL

  # Full watchlist screen (batch)
  uv run stockbot-prescreen --universe
  uv run stockbot-prescreen --universe --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stockbot.config import setup_logging
from stockbot.portfolio_screener.eligibility import check_deep_analysis_eligibility
from stockbot.portfolio_screener.pipeline import run_prescreen_then_analyze
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pre-screen: check if a ticker suits deep analysis, "
            "or (--universe) reduce a watchlist to 10–18 candidates"
        ),
    )
    p.add_argument(
        "ticker",
        nargs="?",
        default=None,
        help="Single NSE symbol/name to eligibility-check (default mode)",
    )
    p.add_argument(
        "--ticker",
        dest="ticker_flag",
        default=None,
        help="Same as positional ticker",
    )
    p.add_argument(
        "--universe",
        action="store_true",
        help="Batch-screen the full watchlist (not a single ticker)",
    )
    p.add_argument(
        "--watchlist",
        type=Path,
        default=None,
        help="Watchlist path for --universe (default: data/portfolio/watchlist.txt)",
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="With --universe: explicit symbol list (overrides watchlist)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Quant only — skip cheap AI eligibility/ranking",
    )
    p.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip AI layer (quant-only)",
    )
    p.add_argument(
        "--run-deep",
        action="store_true",
        help="With --universe: handoff survivors to run_full_analysis",
    )
    p.add_argument(
        "--max-deep",
        type=int,
        default=None,
        help="Cap deep analyses after --universe --run-deep",
    )
    p.add_argument(
        "--ai-provider",
        choices=["auto", "openai", "deepseek", "anthropic"],
        default="auto",
        help="Cheap ranker/eligibility provider (auto: openai > deepseek > haiku)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write JSON result to this path",
    )
    return p


def _run_single(ticker: str, args: argparse.Namespace) -> int:
    config = ScreenerRunConfig(
        dry_run=bool(args.dry_run),
        skip_ai=bool(args.skip_ai or args.dry_run),
        ai_provider=args.ai_provider,
    )
    result = check_deep_analysis_eligibility(ticker, config=config)
    print(result.telegram_html().replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    print()
    print(json.dumps(result.to_dict(), indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if result.verdict in ("NOT_FOUND", "AMBIGUOUS"):
        return 1
    if not result.suitable_for_deep_analysis:
        return 2
    return 0


def _run_universe(args: argparse.Namespace) -> int:
    config = ScreenerRunConfig(
        dry_run=bool(args.dry_run),
        skip_ai=bool(args.skip_ai or args.dry_run),
        run_deep_analysis=bool(args.run_deep) and not bool(args.dry_run),
        max_deep_analyses=args.max_deep,
        ai_provider=args.ai_provider,
    )
    result = run_prescreen_then_analyze(
        symbols=args.symbols,
        watchlist_path=args.watchlist,
        config=config,
    )
    print(result.human_table)
    print()
    print(
        f"status={result.status} universe={result.universe_size} "
        f"hard_excluded={result.hard_excluded} final_candidates={result.final_candidates} "
        f"ai_model={result.ai_model}"
    )
    print(
        f"costs: AI_calls={result.costs.ai_calls} "
        f"est_inr={result.costs.estimated_cost_inr} "
        f"est_deep_savings_inr={result.costs.estimated_deep_analysis_cost_saved_inr}"
    )
    print("candidates:", ", ".join(result.deep_analysis_tickers) or "(none)")
    payload = result.to_dict()
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps({
            "status": result.status,
            "final_candidates": result.final_candidates,
            "tickers": result.deep_analysis_tickers,
            "costs": payload["costs"],
            "ai_model": result.ai_model,
        }))
    if result.status == "INSUFFICIENT_HIGH_QUALITY_CANDIDATES":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    ticker = args.ticker_flag or args.ticker

    if args.universe:
        return _run_universe(args)
    if not ticker:
        print(
            "Usage:\n"
            "  stockbot-prescreen BEL              # single-ticker eligibility\n"
            "  stockbot-prescreen --universe       # full watchlist screen\n",
            file=sys.stderr,
        )
        return 1
    return _run_single(ticker, args)


if __name__ == "__main__":
    sys.exit(main())
