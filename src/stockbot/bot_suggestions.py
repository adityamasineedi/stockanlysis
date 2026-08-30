"""Telegram symbol suggestions — inline @-mention results and pick buttons."""

from __future__ import annotations

import logging
from typing import Literal

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
)
from telegram.helpers import escape_markdown

from stockbot.fetch.tickers import load_symbol_table
from stockbot.models import TickerInfo

logger = logging.getLogger(__name__)

PickAction = Literal["prescan", "analyze"]
CALLBACK_PREFIX = "pick"
MAX_CALLBACK_BYTES = 64
MAX_BUTTONS = 8


def load_default_pick_symbols(limit: int = 10) -> list[str]:
    """Watchlist head + recent high-score prescans for quick-pick rows."""
    symbols: list[str] = []
    seen: set[str] = set()

    def _add(sym: str) -> None:
        key = sym.strip().upper()
        if key and key not in seen:
            seen.add(key)
            symbols.append(key)

    try:
        from stockbot.portfolio_screener.data_loader import load_watchlist

        for sym in load_watchlist():
            _add(sym)
            if len(symbols) >= limit:
                return symbols[:limit]
    except FileNotFoundError:
        pass

    try:
        from stockbot.portfolio_screener.outcome_log import load_prescan_outcomes

        rows = load_prescan_outcomes()
        rows.sort(
            key=lambda row: (
                -(float(row["quant_score"]) if isinstance(row.get("quant_score"), (int, float)) else -1),
                str(row.get("ticker") or ""),
            ),
        )
        for row in rows:
            _add(str(row.get("ticker") or ""))
            if len(symbols) >= limit:
                break
    except OSError:
        logger.debug("default pick symbols: prescan log unavailable", exc_info=True)

    return symbols[:limit]


def tickers_for_symbols(symbols: list[str]) -> list[TickerInfo]:
    if not symbols:
        return []
    table = load_symbol_table()
    out: list[TickerInfo] = []
    for sym in symbols:
        hit = table[table["symbol"].str.upper() == sym.upper()]
        if hit.empty:
            continue
        row = hit.iloc[0]
        out.append(
            TickerInfo(
                symbol=row["symbol"],
                exchange=row["exchange"],
                company_name=row["company_name"],
                isin=row["isin"] if isinstance(row.get("isin"), str) and row["isin"] else None,
            )
        )
    return out


def default_pick_tickers(limit: int = MAX_BUTTONS) -> list[TickerInfo]:
    return tickers_for_symbols(load_default_pick_symbols(limit=limit))


def build_symbol_pick_keyboard(
    tickers: list[TickerInfo],
    *,
    action: PickAction,
) -> InlineKeyboardMarkup | None:
    if not tickers:
        return None
    rows: list[list[InlineKeyboardButton]] = []
    for ticker in tickers[:MAX_BUTTONS]:
        label = ticker.symbol
        if ticker.company_name:
            short = ticker.company_name
            if len(short) > 28:
                short = f"{short[:25]}…"
            label = f"{ticker.symbol} · {short}"
        callback = f"{CALLBACK_PREFIX}:{action}:{ticker.symbol}"
        if len(callback.encode("utf-8")) > MAX_CALLBACK_BYTES:
            callback = f"{CALLBACK_PREFIX}:{action}:{ticker.symbol[:12]}"
        rows.append([InlineKeyboardButton(label, callback_data=callback)])
    return InlineKeyboardMarkup(rows)


def parse_pick_callback(data: str | None) -> tuple[PickAction, str] | None:
    if not data:
        return None
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        return None
    action = parts[1]
    if action not in ("prescan", "analyze"):
        return None
    symbol = parts[2].strip().upper()
    if not symbol:
        return None
    return action, symbol


def build_inline_query_results(
    tickers: list[TickerInfo],
    *,
    action: PickAction = "prescan",
) -> list[InlineQueryResultArticle]:
    results: list[InlineQueryResultArticle] = []
    for ticker in tickers[:12]:
        cmd = f"/{action} {ticker.symbol}"
        desc = ticker.company_name or ticker.symbol
        results.append(
            InlineQueryResultArticle(
                id=f"{action}:{ticker.symbol}",
                title=ticker.symbol,
                description=desc[:256],
                input_message_content=InputTextMessageContent(message_text=cmd),
            )
        )
    return results


def suggestion_hint_markdown(bot_username: str | None) -> str:
    if bot_username:
        handle = escape_markdown(bot_username, version=2)
        return (
            f"Tip: type @{handle} + a few letters in any chat for name suggestions."
        )
    return "Tip: enable inline mode in BotFather (/setinline) then type @YourBot + letters."
