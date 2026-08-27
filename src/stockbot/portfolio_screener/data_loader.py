"""Universe / watchlist loading and batch metric assembly.

Reuses stockbot.fetch.* — does not duplicate Screener/yfinance parsers.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from stockbot.config import DATA_DIR
from stockbot.fetch.fundamentals import FundamentalsSchemaError, fetch_fundamentals
from stockbot.fetch.prices import StaleDataError, fetch_price_data
from stockbot.fetch.shareholding import fetch_shareholding
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.models import (
    AmbiguousMatch,
    Financials,
    PriceData,
    Shareholding,
    TickerInfo,
)
from stockbot.portfolio_screener.metrics import extract_metrics, fetch_market_metadata
from stockbot.portfolio_screener.models import StockMetrics

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = DATA_DIR / "portfolio" / "watchlist.txt"


@dataclass(frozen=True)
class LoadedUniverse:
    tickers: list[TickerInfo]
    unresolved: list[str]
    ambiguous: dict[str, list[str]]
    loaded_at: datetime
    source_path: Path | None


def load_watchlist(path: Path | None = None) -> list[str]:
    """Load symbols from a text or JSON watchlist.

    Text format: one symbol per line, # comments allowed.
    JSON format: {"symbols": ["TCS", ...]} or a bare list.
    """
    watchlist_path = path or DEFAULT_WATCHLIST_PATH
    if not watchlist_path.exists():
        raise FileNotFoundError(
            f"Watchlist not found at {watchlist_path}. "
            "Create data/portfolio/watchlist.txt with one NSE symbol per line."
        )
    text = watchlist_path.read_text(encoding="utf-8").strip()
    if watchlist_path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip().upper() for x in data if str(x).strip()]
        if isinstance(data, dict) and "symbols" in data:
            return [str(x).strip().upper() for x in data["symbols"] if str(x).strip()]
        raise ValueError("JSON watchlist must be a list or {\"symbols\": [...]}")

    symbols: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow "TCS, INFY" on one line
        for part in line.replace(";", ",").split(","):
            sym = part.strip().upper()
            if sym:
                symbols.append(sym)
    # preserve order, drop dupes
    seen: set[str] = set()
    ordered: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def resolve_universe(symbols: list[str]) -> LoadedUniverse:
    table = load_symbol_table()
    tickers: list[TickerInfo] = []
    unresolved: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    seen_resolved: set[str] = set()

    for raw in symbols:
        resolved = resolve_ticker(raw, table)
        if resolved is None:
            unresolved.append(raw)
            logger.warning("watchlist symbol unresolved: %s", raw)
            continue
        if isinstance(resolved, AmbiguousMatch):
            ambiguous[raw] = [c.symbol for c in resolved.candidates]
            logger.warning("watchlist symbol ambiguous: %s → %s", raw, ambiguous[raw])
            continue
        if resolved.symbol in seen_resolved:
            continue
        seen_resolved.add(resolved.symbol)
        tickers.append(resolved)

    return LoadedUniverse(
        tickers=tickers,
        unresolved=unresolved,
        ambiguous=ambiguous,
        loaded_at=datetime.now(UTC),
        source_path=None,
    )


@dataclass
class FetchBundle:
    ticker: TickerInfo
    financials: Financials | None
    price: PriceData | None
    shareholding: Shareholding | None
    market_meta: dict[str, float | str | None]
    errors: list[str]


def _fetch_one(ticker: TickerInfo) -> FetchBundle:
    errors: list[str] = []
    financials = None
    price = None
    shareholding = None
    try:
        financials = fetch_fundamentals(ticker.symbol)
    except (FundamentalsSchemaError, Exception) as exc:  # noqa: BLE001
        errors.append(f"fundamentals: {exc}")
        logger.warning("fundamentals failed for %s: %s", ticker.symbol, exc)

    try:
        price = fetch_price_data(ticker.symbol)
    except (StaleDataError, Exception) as exc:  # noqa: BLE001
        errors.append(f"price: {exc}")
        logger.warning("price failed for %s: %s", ticker.symbol, exc)

    try:
        shareholding = fetch_shareholding(ticker.symbol)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"shareholding: {exc}")
        logger.warning("shareholding failed for %s: %s", ticker.symbol, exc)

    market_meta = fetch_market_metadata(ticker.symbol)
    return FetchBundle(
        ticker=ticker,
        financials=financials,
        price=price,
        shareholding=shareholding,
        market_meta=market_meta,
        errors=errors,
    )


def fetch_universe_metrics(
    tickers: list[TickerInfo],
    *,
    max_workers: int = 4,
) -> list[StockMetrics]:
    """Fetch cheap data (no annual report, no Stage 1/2) for the universe."""
    bundles: list[FetchBundle] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            bundles.append(fut.result())

    # Stable order matching input
    by_symbol = {b.ticker.symbol: b for b in bundles}
    metrics_list: list[StockMetrics] = []
    for t in tickers:
        b = by_symbol[t.symbol]
        m = extract_metrics(
            t,
            financials=b.financials,
            price=b.price,
            shareholding=b.shareholding,
            market_meta=b.market_meta,
        )
        m.raw_notes.extend(b.errors)
        metrics_list.append(m)
    return metrics_list
