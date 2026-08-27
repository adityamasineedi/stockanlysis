"""Module 4 — shareholding and pledge.

NSE via a confirmed-working endpoint (cookie priming from the
shareholding-pattern page itself, not the homepage — the homepage 403s):
    https://www.nseindia.com/api/corporate-share-holdings-master
      ?index=equities&symbol=<SYMBOL>
Gives promoter % and public % by quarter (most recent record first), plus
a link to the full XBRL filing per period — but no FII/DII breakdown and
no pledge percentage in the JSON itself.

BSE has no discoverable, verifiable endpoint — its shareholding pages are
the same client-rendered Angular SPA that blocked Module 1's BSE source;
one further guess (shpPledge) redirected but still served the empty app
shell rather than data. Not wired in — same documented gap as Module 1.

Screener (already fetched and cached by Module 3, reused here rather than
re-fetched — see fetch_screener_page) supplies FII/DII/public % by
quarter, filling in what NSE's JSON doesn't carry.

Promoter pledge: NOT available from either source above. The underlying
SEBI XBRL filings linked from NSE's master JSON do carry pledge-disclosure
tags (WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged, etc.), but
parsing XBRL was deliberately deferred — every company checked while
investigating this module (RELIANCE, JPASSOCIAT's latest filing) had the
flag false, so there was no live positive example to confirm the numeric
percentage tag name against, and XBRL parsing is real added scope. So
pledge_pct_of_promoter_holding is always None here — the correct, honest
value per this project's contract (None means "unconfirmed", never
"zero"), not a placeholder to silently fill in later.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from stockbot.config import HTTP_USER_AGENT
from stockbot.fetch.fundamentals import (
    FundamentalsSchemaError,
    detect_basis,
    fetch_screener_page,
    parse_screener_table,
)
from stockbot.models import Shareholding

USER_AGENT = HTTP_USER_AGENT
NSE_PRIMING_URL = "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
NSE_MASTER_URL = "https://www.nseindia.com/api/corporate-share-holdings-master"


class ShareholdingFetchError(Exception):
    pass


@retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
def _fetch_nse_master(symbol: str) -> list[dict]:
    with httpx.Client(
        timeout=20.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        client.get(NSE_PRIMING_URL)  # sets the cookies the API call below requires
        response = client.get(
            NSE_MASTER_URL,
            params={"index": "equities", "symbol": symbol},
            headers={"Accept": "application/json", "Referer": NSE_PRIMING_URL},
        )
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, list) or not data:
        raise ShareholdingFetchError(f"NSE returned no shareholding records for {symbol!r}")
    return data


def fetch_nse_shareholding(symbol: str) -> Shareholding | None:
    try:
        records = _fetch_nse_master(symbol)
    except (httpx.HTTPError, ShareholdingFetchError):
        return None

    latest = records[0]
    raw_promoter = latest.get("pr_and_prgrp")
    promoter_pct = float(raw_promoter) if raw_promoter not in (None, "") else None

    return Shareholding(
        promoter_pct=promoter_pct,
        pledge_pct_of_promoter_holding=None,  # see module docstring — not in this endpoint
        fii_pct=None,
        dii_pct=None,
        quarter=latest.get("date"),  # e.g. "30-JUN-2026", the filing's as-of date
        source="NSE",
        fetched_at=datetime.now(UTC),
    )


def _table_value(table: pd.DataFrame, label: str, column: str) -> float | None:
    if label not in table.index:
        return None
    value = table.loc[label, column]
    return float(value) if pd.notna(value) else None


def fetch_screener_shareholding(symbol: str) -> Shareholding | None:
    for basis in ("consolidated", "standalone"):
        try:
            html = fetch_screener_page(symbol, basis)
        except (FundamentalsSchemaError, httpx.HTTPError):
            continue

        soup = BeautifulSoup(html, "lxml")
        if detect_basis(soup) is None:
            continue

        try:
            table = parse_screener_table(soup, "shareholding")
        except FundamentalsSchemaError:
            continue
        if table.empty:
            continue

        latest_quarter = table.columns[-1]

        return Shareholding(
            promoter_pct=_table_value(table, "Promoters", latest_quarter),
            pledge_pct_of_promoter_holding=None,  # Screener's free tier has no pledge data (confirmed empirically)
            fii_pct=_table_value(table, "FIIs", latest_quarter),
            dii_pct=_table_value(table, "DIIs", latest_quarter),
            quarter=str(latest_quarter),
            source="Screener",
            fetched_at=datetime.now(UTC),
        )

    return None


def fetch_shareholding(symbol: str) -> Shareholding:
    """Fallback chain, one source per record — not a merge. NSE first
    (promoter % only; fii_pct/dii_pct are honestly None, since NSE's
    endpoint doesn't carry them — not a gap, just what that source has).
    Screener only if NSE fails entirely, giving promoter/FII/DII/public %
    together from one consistent source. BSE is not wired in (see module
    docstring)."""
    nse_result = fetch_nse_shareholding(symbol)
    if nse_result is not None:
        return nse_result

    screener_result = fetch_screener_shareholding(symbol)
    if screener_result is not None:
        return screener_result

    raise ShareholdingFetchError(
        f"Could not fetch shareholding data for {symbol!r} from NSE or Screener"
    )
