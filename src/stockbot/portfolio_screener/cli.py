"""CLI for the portfolio pre-screener.

Examples:
  uv run stockbot-prescreen --dry-run
  uv run stockbot-prescreen --watchlist data/portfolio/watchlist.txt --skip-ai
  uv run stockbot-prescreen --run-deep --max-deep 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stockbot.config import setup_logging
from stockbot.portfolio_screener.pipeline import run_prescreen_then_analyze
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Portfolio pre-screener — reduce 40+ stocks to 10–18 deep-analysis candidates",
    )
    p.add_argument(
        "--watchlist",
        type=Path,
        default=None,
        help="Path to watchlist.txt or .json (default: data/portfolio/watchlist.txt)",
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Explicit symbol list (overrides watchlist)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Quant + deterministic ranking only; no AI ranker; no deep analysis",
    )
    p.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip AI ranking layer (use quant-only blend)",
    )
    p.add_argument(
        "--run-deep",
        action="store_true",
        help="Handoff selected candidates to run_full_analysis",
    )
    p.add_argument(
        "--max-deep",
        type=int,
        default=None,
        help="Cap number of deep analyses after screening",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write machine-readable ScreeningResult JSON to this path",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    config = ScreenerRunConfig(
        dry_run=bool(args.dry_run),
        skip_ai=bool(args.skip_ai or args.dry_run),
        run_deep_analysis=bool(args.run_deep) and not bool(args.dry_run),
        max_deep_analyses=args.max_deep,
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
        f"hard_excluded={result.hard_excluded} data_insufficient={result.data_insufficient} "
        f"sent_to_ai={result.sent_to_ai} final_candidates={result.final_candidates}"
    )
    print(
        f"costs: AI_calls={result.costs.ai_calls} "
        f"tokens_in={result.costs.input_tokens} tokens_out={result.costs.output_tokens} "
        f"est_inr={result.costs.estimated_cost_inr} "
        f"est_deep_savings_inr={result.costs.estimated_deep_analysis_cost_saved_inr}"
    )
    print("candidates:", ", ".join(result.deep_analysis_tickers) or "(none)")

    payload = result.to_dict()
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    else:
        # Always emit a compact JSON summary on stdout for piping
        print(json.dumps({
            "status": result.status,
            "final_candidates": result.final_candidates,
            "tickers": result.deep_analysis_tickers,
            "costs": payload["costs"],
        }))

    if result.status == "INSUFFICIENT_HIGH_QUALITY_CANDIDATES":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
