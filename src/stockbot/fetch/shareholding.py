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
quarter. When NSE succeeds, promoter/pledge come from NSE and FII/DII are
merged from Screener on the same record.

Promoter pledge: the NSE master JSON links to SEBI XBRL shareholding
filings. When ``WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged``
is explicitly false, pledge is set to ``0.0`` (confirmed zero). When the
flag is true but a reliable % of promoter holding cannot be parsed from
XBRL yet, pledge stays ``None`` (unconfirmed, not zero).
"""

from __future__ import annotations

import logging
import re
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


logger = logging.getLogger(__name__)

_PLEDGE_FLAG_TAG = "WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged"


def _xbrl_tag_value(xml: str, tag: str) -> str | None:
    match = re.search(rf":{re.escape(tag)}[^>]*>([^<]+)</", xml, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def parse_pledge_pct_from_nse_xbrl(xml: str) -> float | None:
    """Return 0.0 when XBRL confirms no pledge; None when unknown."""
    raw_flag = _xbrl_tag_value(xml, _PLEDGE_FLAG_TAG)
    if raw_flag is None:
        return None
    if raw_flag.lower() == "false":
        return 0.0
    if raw_flag.lower() != "true":
        return None
    # Pledge exists — numeric % of promoter holding not reliably parsed yet.
    return None


def _fetch_nse_xbrl(client: httpx.Client, url: str) -> str | None:
    try:
        response = client.get(url, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("NSE XBRL fetch failed for %s: %s", url, exc)
        return None
    return response.text


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
        with httpx.Client(
            timeout=20.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        ) as client:
            client.get(NSE_PRIMING_URL)
            response = client.get(
                NSE_MASTER_URL,
                params={"index": "equities", "symbol": symbol},
                headers={"Accept": "application/json", "Referer": NSE_PRIMING_URL},
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                raise ShareholdingFetchError(
                    f"NSE returned no shareholding records for {symbol!r}"
                )
            latest = data[0]
            raw_promoter = latest.get("pr_and_prgrp")
            promoter_pct = (
                float(raw_promoter) if raw_promoter not in (None, "") else None
            )
            pledge_pct: float | None = None
            xbrl_url = latest.get("xbrl")
            if isinstance(xbrl_url, str) and xbrl_url.strip():
                xbrl_text = _fetch_nse_xbrl(client, xbrl_url.strip())
                if xbrl_text:
                    pledge_pct = parse_pledge_pct_from_nse_xbrl(xbrl_text)
    except (httpx.HTTPError, ShareholdingFetchError):
        return None

    return Shareholding(
        promoter_pct=promoter_pct,
        pledge_pct_of_promoter_holding=pledge_pct,
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
    """NSE first for promoter % and pledge; merge Screener FII/DII when available."""
    nse_result = fetch_nse_shareholding(symbol)
    screener_result = fetch_screener_shareholding(symbol)

    if nse_result is not None:
        if screener_result is not None:
            return Shareholding(
                promoter_pct=nse_result.promoter_pct,
                pledge_pct_of_promoter_holding=nse_result.pledge_pct_of_promoter_holding,
                fii_pct=screener_result.fii_pct,
                dii_pct=screener_result.dii_pct,
                quarter=nse_result.quarter or screener_result.quarter,
                source="NSE",
                fetched_at=nse_result.fetched_at,
            )
        return nse_result

    if screener_result is not None:
        return screener_result

    raise ShareholdingFetchError(
        f"Could not fetch shareholding data for {symbol!r} from NSE or Screener"
    )
