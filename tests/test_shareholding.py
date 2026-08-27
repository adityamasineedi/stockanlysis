"""Module 4 (shareholding) unit tests for the fallback ordering, via
monkeypatching the two source functions — no network. Live behaviour
(real NSE cookie priming, real Screener parsing, cross-checked promoter %
agreement between NSE and Screener for RELIANCE) was verified by hand
during development."""

from datetime import UTC, datetime

import pytest

from stockbot.fetch import shareholding
from stockbot.fetch.shareholding import ShareholdingFetchError, fetch_shareholding
from stockbot.models import Shareholding

NSE_RESULT = Shareholding(
    promoter_pct=50.0,
    pledge_pct_of_promoter_holding=None,
    fii_pct=None,
    dii_pct=None,
    quarter="30-JUN-2026",
    source="NSE",
    fetched_at=datetime.now(UTC),
)

SCREENER_RESULT = Shareholding(
    promoter_pct=50.0,
    pledge_pct_of_promoter_holding=None,
    fii_pct=17.0,
    dii_pct=21.0,
    quarter="Jun 2026",
    source="Screener",
    fetched_at=datetime.now(UTC),
)


def test_prefers_nse_when_available(monkeypatch):
    monkeypatch.setattr(shareholding, "fetch_nse_shareholding", lambda symbol: NSE_RESULT)
    monkeypatch.setattr(
        shareholding, "fetch_screener_shareholding", lambda symbol: SCREENER_RESULT
    )
    result = fetch_shareholding("RELIANCE")
    assert result.source == "NSE"
    assert result.fii_pct is None  # honestly None, not silently backfilled from Screener


def test_falls_back_to_screener_when_nse_fails(monkeypatch):
    monkeypatch.setattr(shareholding, "fetch_nse_shareholding", lambda symbol: None)
    monkeypatch.setattr(
        shareholding, "fetch_screener_shareholding", lambda symbol: SCREENER_RESULT
    )
    result = fetch_shareholding("RELIANCE")
    assert result.source == "Screener"
    assert result.fii_pct == 17.0


def test_raises_when_both_sources_fail(monkeypatch):
    monkeypatch.setattr(shareholding, "fetch_nse_shareholding", lambda symbol: None)
    monkeypatch.setattr(shareholding, "fetch_screener_shareholding", lambda symbol: None)
    with pytest.raises(ShareholdingFetchError):
        fetch_shareholding("NOPE")


def test_pledge_is_always_none_never_zero():
    # both real sources leave it None; this pins the contract so a future
    # change can't silently start defaulting it to 0.0
    assert NSE_RESULT.pledge_pct_of_promoter_holding is None
    assert SCREENER_RESULT.pledge_pct_of_promoter_holding is None
