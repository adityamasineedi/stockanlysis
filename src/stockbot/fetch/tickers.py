"""Module 1 — ticker resolution.

Source: NSE's EQUITY_L.csv, discovered from the page the user supplied
(https://www.nseindia.com/static/market-data/securities-available-for-trading
links directly to it) and confirmed reachable with a plain GET, no cookie
priming needed:
    https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv

BSE is NOT wired in for a full symbol master. The two BSE sources supplied were checked and neither
is usable as a full symbol master:
  - downloads1/List_of_companies.csv is BSE's GSM (Graded Surveillance
    Measure) watchlist — ~850 flagged securities, not the ~5,000+ company
    BSE universe. Using it would silently fail to resolve almost every
    real BSE-only company.
  - corporates/List_Scrips is a client-rendered SPA with no static data or
    discoverable download link in its HTML/JS. A one-off guess at a public
    API endpoint (api.bseindia.com/BseIndiaAPI/api/ListofScrips) failed.
For BSE-only symbols, resolve_ticker falls back to yfinance (.BO) when the
query looks like a ticker symbol and Yahoo has a live price. Shareholding
then uses Screener (FII/DII) without NSE pledge/XBRL. Annual reports remain
NSE-only until a BSE filing source is verified.
"""

from __future__ import annotations

import re
import time

import httpx
import pandas as pd
from rapidfuzz import fuzz, process

from stockbot import config
from stockbot.models import AmbiguousMatch, TickerInfo

NSE_EQUITY_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
CACHE_PATH = config.SYMBOLS_DIR / "EQUITY_L.csv"
CACHE_MAX_AGE_DAYS = 30

USER_AGENT = config.HTTP_USER_AGENT

FUZZY_SCORE_CUTOFF = 60.0
DOMINANT_MARGIN = 15.0
MAX_AMBIGUOUS_CANDIDATES = 5

# Found live: a short fragment like "rel" scores only ~27 against "reliance
# industries" under token_set_ratio (it's a token-overlap scorer, and "rel"
# shares no whole token with either word), so it never even reaches the
# 60-cutoff and Reliance never appears as a candidate at all. partial_ratio
# fixes that specific case (scores it 100) but is far too permissive as a
# general-purpose scorer — it matches "bel" against "bellacasa", "orientbell"
# etc. at 100 too, since it just finds the best-aligned substring window
# anywhere in the name. The fix that doesn't trade one failure mode for a
# worse one: boost only a query that is a prefix of a company's FIRST word
# specifically (not a substring anywhere), and only for short fragments
# where this is the plausible intent.
PREFIX_BOOST_MIN_LEN = 2
PREFIX_BOOST_MAX_LEN = 8

# Well-known companies where the colloquial name is genuinely ambiguous by
# string similarity alone — e.g. "Reliance" is a substring of 7 distinct
# real NSE companies (Industries, Power, Infrastructure, Communications,
# Home Finance, Industrial Infrastructure, Chemotex) and no rapidfuzz
# scorer ranks Reliance Industries uniquely above the others; some rank it
# below Reliance Power. Picking "the flagship" is brand-recognition
# knowledge the NSE data doesn't contain, so it's encoded here explicitly
# rather than guessed at via a fuzzy-score threshold. Extend by hand as
# real cases are found — do not try to generalize this into a scoring rule.
COMMON_NAME_ALIASES: dict[str, str] = {
    "reliance": "RELIANCE",
}

_SUFFIX_RE = re.compile(r"\b(limited|ltd\.?)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_SYMBOL_LIKE_RE = re.compile(r"^[A-Z0-9&-]{2,20}$")


def _resolve_bse_only_via_yfinance(query: str) -> TickerInfo | None:
    """Accept BSE-only symbols missing from the NSE EQUITY_L master."""
    sym = query.strip().upper()
    if not _SYMBOL_LIKE_RE.fullmatch(sym):
        return None
    try:
        import yfinance as yf

        info = yf.Ticker(f"{sym}.BO").info or {}
    except Exception:
        return None
    price = info.get("regularMarketPrice") or info.get("currentPrice")
    if not price:
        return None
    name = info.get("longName") or info.get("shortName") or sym
    return TickerInfo(
        symbol=sym,
        exchange="BSE",
        company_name=str(name),
        isin=None,
    )


def normalize_company_name(name: str) -> str:
    name = _SUFFIX_RE.sub("", name)
    name = _PUNCT_RE.sub("", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def _ensure_symbol_csv_cached(force_refresh: bool = False) -> None:
    if not force_refresh and CACHE_PATH.exists():
        age_days = (time.time() - CACHE_PATH.stat().st_mtime) / 86400
        if age_days < CACHE_MAX_AGE_DAYS:
            return

    config.SYMBOLS_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(NSE_EQUITY_CSV_URL)
    response.raise_for_status()
    if not response.text.strip():
        raise RuntimeError("NSE EQUITY_L.csv download returned an empty body")

    CACHE_PATH.write_bytes(response.content)


def load_symbol_table(force_refresh: bool = False) -> pd.DataFrame:
    _ensure_symbol_csv_cached(force_refresh=force_refresh)

    df = pd.read_csv(CACHE_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "SYMBOL": "symbol",
            "NAME OF COMPANY": "company_name",
            "ISIN NUMBER": "isin",
        }
    )
    df["symbol"] = df["symbol"].str.strip()
    df["company_name"] = df["company_name"].str.strip()
    df["isin"] = df["isin"].str.strip()
    df["exchange"] = "NSE"
    df["normalized_name"] = df["company_name"].map(normalize_company_name)

    return df[["symbol", "exchange", "company_name", "isin", "normalized_name"]]


def _row_to_ticker_info(row: pd.Series) -> TickerInfo:
    isin = row.get("isin")
    return TickerInfo(
        symbol=row["symbol"],
        exchange=row["exchange"],
        company_name=row["company_name"],
        isin=isin if isinstance(isin, str) and isin else None,
    )


def _dedup_cross_exchange(
    candidates: list[tuple[pd.Series, float]],
) -> list[tuple[pd.Series, float]]:
    """Collapse the same company listed on multiple exchanges down to its
    NSE row, matching by ISIN (or normalized name if ISIN is missing).
    Today the symbol table is NSE-only so this never fires, but the
    dual-listing rule is part of resolve_ticker's contract and this keeps
    the behaviour correct the moment BSE rows are added, with no change
    to the calling logic below."""
    best: dict[str, tuple[pd.Series, float]] = {}
    order: list[str] = []
    for row, score in candidates:
        isin = row.get("isin")
        key = isin if isinstance(isin, str) and isin else row["normalized_name"]
        if key not in best:
            best[key] = (row, score)
            order.append(key)
            continue
        existing_row, existing_score = best[key]
        prefer_new = row["exchange"] == "NSE" and existing_row["exchange"] != "NSE"
        kept_row = row if prefer_new else existing_row
        best[key] = (kept_row, max(score, existing_score))
    return [best[key] for key in order]


def _finalize(
    candidates: list[tuple[pd.Series, float]],
) -> TickerInfo | AmbiguousMatch | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return _row_to_ticker_info(candidates[0][0])

    scores = [score for _, score in candidates]
    best_score = max(scores)
    top_scorers = [c for c in candidates if c[1] == best_score]
    distinct_scores = sorted(set(scores), reverse=True)
    second_best = distinct_scores[1] if len(distinct_scores) > 1 else -1.0
    dominant = len(top_scorers) == 1 and (best_score - second_best) >= DOMINANT_MARGIN

    if dominant:
        return _row_to_ticker_info(top_scorers[0][0])

    ranked = sorted(candidates, key=lambda c: c[1], reverse=True)[:MAX_AMBIGUOUS_CANDIDATES]
    return AmbiguousMatch(
        candidates=[_row_to_ticker_info(row) for row, _ in ranked],
        scores=[score for _, score in ranked],
    )


def _first_word_prefix_matches(query: str, table: pd.DataFrame) -> list[tuple[pd.Series, float]]:
    q = query.lower()
    if not (PREFIX_BOOST_MIN_LEN <= len(q) <= PREFIX_BOOST_MAX_LEN):
        return []
    matches = []
    for _, row in table.iterrows():
        normalized = row["normalized_name"]
        first_word = normalized.split(" ", 1)[0] if normalized else ""
        if first_word and first_word != q and first_word.startswith(q):
            # Proportional to how much of the real word the fragment covers
            # — "tat" covering 3/4 of "tata" scores higher than "r" covering
            # 1/8 of "reliance" would. Capped below FUZZY dominance range so
            # a clearly-longer, better fuzzy match can still win outright.
            score = min(60.0 + 30.0 * (len(q) / len(first_word)), 95.0)
            matches.append((row, score))
    return matches


def resolve_ticker(
    user_input: str, table: pd.DataFrame | None = None
) -> TickerInfo | AmbiguousMatch | None:
    if table is None:
        table = load_symbol_table()

    query = user_input.strip()
    if not query:
        return None

    exact_symbol = table[table["symbol"].str.lower() == query.lower()]
    if not exact_symbol.empty:
        return _row_to_ticker_info(exact_symbol.iloc[0])

    bse_only = _resolve_bse_only_via_yfinance(query)
    if bse_only is not None:
        return bse_only

    alias_symbol = COMMON_NAME_ALIASES.get(query.lower())
    if alias_symbol:
        aliased = table[table["symbol"] == alias_symbol]
        if not aliased.empty:
            return _row_to_ticker_info(aliased.iloc[0])

    normalized_query = normalize_company_name(query)
    if not normalized_query:
        return None

    exact_name = table[table["normalized_name"] == normalized_query]
    if not exact_name.empty:
        candidates = _dedup_cross_exchange(
            [(row, 100.0) for _, row in exact_name.iterrows()]
        )
        return _finalize(candidates)

    choices = table["normalized_name"].tolist()
    matches = process.extract(
        normalized_query,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_SCORE_CUTOFF,
        limit=10,
    )
    fuzzy_candidates = [(table.iloc[idx], score) for _, score, idx in matches]
    prefix_candidates = _first_word_prefix_matches(query, table)

    # Merge, keeping the higher score per symbol when both methods hit it.
    merged: dict[str, tuple[pd.Series, float]] = {}
    for row, score in [*fuzzy_candidates, *prefix_candidates]:
        symbol = row["symbol"]
        if symbol not in merged or score > merged[symbol][1]:
            merged[symbol] = (row, score)

    if not merged:
        bse_only = _resolve_bse_only_via_yfinance(query)
        if bse_only is not None:
            return bse_only
        return None

    candidates = _dedup_cross_exchange(list(merged.values()))
    return _finalize(candidates)
