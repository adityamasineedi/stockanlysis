"""Dry-run tool: assembles the real Stage 1 and Stage 2 payloads and
reports their true input token counts via Anthropic's count_tokens
endpoint — which is free — without ever sending a billed messages.create()
or messages.parse() call. This would have shown the real Stage 2 input
size before a single rupee was spent chasing the max_tokens=8000 problem.

Usage: uv run python -m stockbot.dry_run TICKER
"""

from __future__ import annotations

import sys

from anthropic import Anthropic

from stockbot.brief import assemble_brief
from stockbot.config import settings
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.llm.extract import MODEL as STAGE1_MODEL
from stockbot.llm.extract import SYSTEM_PROMPT as STAGE1_SYSTEM_PROMPT
from stockbot.llm.extract import ExtractionResult
from stockbot.llm.extract import build_user_message as build_stage1_message
from stockbot.llm.verdict import MODEL as STAGE2_MODEL
from stockbot.llm.verdict import build_user_message as build_stage2_message
from stockbot.llm.verdict import load_master_prompt
from stockbot.models import AmbiguousMatch


def _count_tokens(client: Anthropic, model: str, system: str, user_message: str) -> int:
    result = client.messages.count_tokens(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return result.input_tokens


def dry_run(ticker_query: str) -> None:
    client = Anthropic(api_key=settings.anthropic_api_key)

    table = load_symbol_table()
    resolved = resolve_ticker(ticker_query, table)
    if resolved is None:
        print(f"Could not resolve {ticker_query!r} — check spelling or symbol")
        return
    if isinstance(resolved, AmbiguousMatch):
        print(f"Ambiguous: {[c.symbol for c in resolved.candidates]}")
        return

    print(f"Resolved: {resolved.symbol} ({resolved.company_name})")
    print("Assembling brief (free — NSE/Screener/Google News fetches, no LLM calls)...")
    brief = assemble_brief(resolved)
    print(f"Brief's own token estimate (chars/4 heuristic): {brief.token_count}")
    print(f"Confidence ceiling: {brief.confidence_ceiling}/10, missing: {brief.missing}")

    stage1_message = build_stage1_message(brief)
    stage1_tokens = _count_tokens(client, STAGE1_MODEL, STAGE1_SYSTEM_PROMPT, stage1_message)
    print(f"\nStage 1 ({STAGE1_MODEL}) real input tokens: {stage1_tokens}")

    master_prompt = load_master_prompt()
    # Placeholder ExtractionResult — this dry run only measures the Stage 2
    # INPUT size (what the real max_tokens=8000 bug should have been caught
    # by), not the eventual output length, which count_tokens can't predict.
    stage2_message = build_stage2_message(brief, ExtractionResult())
    stage2_tokens = _count_tokens(client, STAGE2_MODEL, master_prompt, stage2_message)
    print(
        f"Stage 2 ({STAGE2_MODEL}) real input tokens "
        f"(with a placeholder empty Stage 1 extraction): {stage2_tokens}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python -m stockbot.dry_run TICKER")
        sys.exit(1)
    dry_run(sys.argv[1])


if __name__ == "__main__":
    main()
