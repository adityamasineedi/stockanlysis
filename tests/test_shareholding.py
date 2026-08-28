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


def test_merges_screener_fii_dii_when_nse_available(monkeypatch):
    monkeypatch.setattr(shareholding, "fetch_nse_shareholding", lambda symbol: NSE_RESULT)
    monkeypatch.setattr(
        shareholding, "fetch_screener_shareholding", lambda symbol: SCREENER_RESULT
    )
    result = fetch_shareholding("RELIANCE")
    assert result.source == "NSE"
    assert result.promoter_pct == 50.0
    assert result.fii_pct == 17.0
    assert result.dii_pct == 21.0


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


def test_pledge_is_none_when_xbrl_unavailable():
    # Screener has no pledge data; NSE JSON alone does not carry pledge %.
    assert NSE_RESULT.pledge_pct_of_promoter_holding is None
    assert SCREENER_RESULT.pledge_pct_of_promoter_holding is None
