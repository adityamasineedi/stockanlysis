"""Unified product universe — watchlist ∪ SIP portfolio symbols.

Daily tips and the 3y portfolio build share one funnel. Watchlist remains the
editable text list; SIP config remains the DCA allocation surface. This module
merges both so commands never treat them as disconnected universes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from stockbot.config import WATCHLIST_PATH
from stockbot.portfolio_screener.data_loader import load_watchlist
from stockbot.portfolio_sip_schema import load_portfolio_sip_config

logger = logging.getLogger(__name__)

Source = Literal["watchlist", "sip"]


@dataclass(frozen=True)
class UniverseSymbol:
    symbol: str
    sources: frozenset[Source]
    sip_bucket_id: str | None = None
    sip_bucket_label: str | None = None


@dataclass(frozen=True)
class ProductUniverse:
    symbols: tuple[UniverseSymbol, ...]
    watchlist_path: Path
    sip_path: Path | None

    @property
    def tickers(self) -> list[str]:
        return [row.symbol for row in self.symbols]

    @property
    def watchlist_only(self) -> list[str]:
        return [r.symbol for r in self.symbols if r.sources == frozenset({"watchlist"})]

    @property
    def sip_only(self) -> list[str]:
        return [r.symbol for r in self.symbols if r.sources == frozenset({"sip"})]

    @property
    def shared(self) -> list[str]:
        both = frozenset({"watchlist", "sip"})
        return [r.symbol for r in self.symbols if r.sources == both]


def _sip_enabled_symbols(
    path: Path | None = None,
) -> tuple[dict[str, tuple[str, str]], Path | None]:
    """symbol -> (bucket_id, bucket_label) for enabled SIP names."""
    try:
        config = load_portfolio_sip_config(path)
    except Exception as exc:  # noqa: BLE001 — universe must stay usable without SIP file
        logger.warning("product_universe: SIP config unavailable (%s)", exc)
        return {}, None

    mapping: dict[str, tuple[str, str]] = {}
    for bucket in config.portfolios:
        if not bucket.enabled:
            continue
        for sym in bucket.symbols:
            if not sym.enabled:
                continue
            key = str(sym.symbol).strip().upper()
            if not key or key in mapping:
                continue
            mapping[key] = (bucket.id, bucket.label)
    resolved = path
    if resolved is None:
        try:
            from stockbot.config import resolve_sip_portfolios_path

            resolved = resolve_sip_portfolios_path()
        except Exception:  # noqa: BLE001
            resolved = None
    return mapping, resolved


def load_product_universe(
    *,
    watchlist_path: Path | None = None,
    sip_path: Path | None = None,
) -> ProductUniverse:
    """Watchlist ∪ enabled SIP symbols, order: watchlist first, then SIP-only."""
    wl_path = watchlist_path or WATCHLIST_PATH
    watchlist = [s.strip().upper() for s in load_watchlist(wl_path) if s.strip()]
    sip_map, resolved_sip = _sip_enabled_symbols(sip_path)

    rows: list[UniverseSymbol] = []
    seen: set[str] = set()
    for symbol in watchlist:
        if symbol in seen:
            continue
        seen.add(symbol)
        if symbol in sip_map:
            bucket_id, bucket_label = sip_map[symbol]
            rows.append(
                UniverseSymbol(
                    symbol=symbol,
                    sources=frozenset({"watchlist", "sip"}),
                    sip_bucket_id=bucket_id,
                    sip_bucket_label=bucket_label,
                )
            )
        else:
            rows.append(UniverseSymbol(symbol=symbol, sources=frozenset({"watchlist"})))

    for symbol, (bucket_id, bucket_label) in sorted(sip_map.items()):
        if symbol in seen:
            continue
        seen.add(symbol)
        rows.append(
            UniverseSymbol(
                symbol=symbol,
                sources=frozenset({"sip"}),
                sip_bucket_id=bucket_id,
                sip_bucket_label=bucket_label,
            )
        )

    return ProductUniverse(
        symbols=tuple(rows),
        watchlist_path=wl_path,
        sip_path=resolved_sip,
    )


def format_universe_summary(universe: ProductUniverse | None = None) -> str:
    """Short HTML blurb for workflow / progress headers."""
    uni = universe or load_product_universe()
    shared = sum(1 for r in uni.symbols if len(r.sources) > 1)
    sip_n = sum(1 for r in uni.symbols if "sip" in r.sources)
    wl_n = sum(1 for r in uni.symbols if "watchlist" in r.sources)
    return (
        f"Universe: <b>{len(uni.symbols)}</b> names "
        f"(watchlist {wl_n} · SIP {sip_n} · in both {shared})"
    )
