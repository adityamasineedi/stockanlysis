"""Offline sector labels for concentration checks.

Sectors come from ``data/portfolio/sector_map.json`` (curated for the product
universe). Unknown tickers map to ``Unclassified`` — they still count toward
capital but are shown separately rather than inventing a sector.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from stockbot.config import PORTFOLIO_DIR

logger = logging.getLogger(__name__)

DEFAULT_SECTOR_MAP_PATH = PORTFOLIO_DIR / "sector_map.json"
UNCLASSIFIED = "Unclassified"

_cache: dict[str, str] | None = None
_cache_path: Path | None = None


def load_sector_map(path: Path | None = None) -> dict[str, str]:
    """Return symbol → sector. Empty dict if the map file is missing."""
    global _cache, _cache_path
    target = path or DEFAULT_SECTOR_MAP_PATH
    if _cache is not None and _cache_path == target:
        return _cache
    if not target.exists():
        logger.warning("sector_map missing at %s — all holdings Unclassified", target)
        _cache = {}
        _cache_path = target
        return _cache
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"sector_map must be a JSON object: {target}")
    mapping = {
        str(k).strip().upper(): str(v).strip() or UNCLASSIFIED
        for k, v in raw.items()
        if str(k).strip()
    }
    _cache = mapping
    _cache_path = target
    return mapping


def clear_sector_map_cache() -> None:
    global _cache, _cache_path
    _cache = None
    _cache_path = None


def sector_for_symbol(symbol: str, *, path: Path | None = None) -> str:
    key = str(symbol or "").strip().upper()
    if not key:
        return UNCLASSIFIED
    return load_sector_map(path).get(key, UNCLASSIFIED)
