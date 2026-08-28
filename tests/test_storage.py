"""Module 11 (storage) unit tests against a temp SQLite file, with
fetch_price_data monkeypatched — no network, no real DB touched."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from stockbot import storage as storage_module
from stockbot.models import PriceData
from stockbot.storage import build_staleness_banner, get_cached, save_analysis

VERDICT_JSON = {
    "verdict": "WATCH",
    "current_price_abs": 400.0,
    "price_date": "2026-08-19",
    "confidence": 6,
}


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "DB_PATH", tmp_path / "test_analyses.sqlite3")


def _price(value: float, price_date: date | None = None) -> PriceData:
    df = pd.DataFrame({"Close": [value]})
    return PriceData(
        value,
        price_date or date(2026, 8, 25),
        df,
        df,
        500.0,
        300.0,
        "yfinance",
        datetime.now(UTC),
    )


def _save(ticker="TEST", verdict_json=None, created_offset_days=0, missing=None):
    row_id = save_analysis(
        ticker=ticker,
        verdict_json=verdict_json or VERDICT_JSON,
        report_md="# report",
        brief_text="brief",
        stage1_tokens=100,
        stage2_tokens=200,
        cost_inr=39.0,
        validation_passed=True,
        missing=missing,
    )
    if created_offset_days:
        backdated = (datetime.now(UTC) - timedelta(days=created_offset_days)).isoformat()
        with storage_module._connect() as conn:
            conn.execute("UPDATE analyses SET created_at = ? WHERE id = ?", (backdated, row_id))
    return row_id


def test_get_cached_returns_none_when_nothing_stored():
    assert get_cached("NOPE") is None


def test_get_cached_returns_recent_analysis_when_price_stable(monkeypatch):
    _save()
    monkeypatch.setattr(storage_module, "fetch_price_data", lambda ticker: _price(405.0))  # +1.25%

    result = get_cached("TEST")
    assert result is not None
    assert result.analysis.verdict_json["current_price_abs"] == 400.0  # DB row unchanged
    assert result.current_price_abs == 405.0
    assert result.price_date == date(2026, 8, 25)


def test_get_cached_preserves_missing_list_across_the_cache(monkeypatch):
    _save(missing=["MISSING: shareholding — could not fetch"])
    monkeypatch.setattr(storage_module, "fetch_price_data", lambda ticker: _price(400.0))

    result = get_cached("TEST")
    assert result is not None
    assert result.analysis.missing == ["MISSING: shareholding — could not fetch"]


def test_get_cached_refuses_when_older_than_max_age(monkeypatch):
    _save(created_offset_days=10)
    monkeypatch.setattr(storage_module, "fetch_price_data", lambda ticker: _price(400.0))

    assert get_cached("TEST", max_age_days=7) is None


def test_get_cached_refuses_when_price_moved_beyond_threshold(monkeypatch):
    _save()
    monkeypatch.setattr(storage_module, "fetch_price_data", lambda ticker: _price(340.0))  # -15%

    assert get_cached("TEST") is None


def test_get_cached_accepts_price_move_within_threshold(monkeypatch):
    _save()
    monkeypatch.setattr(storage_module, "fetch_price_data", lambda ticker: _price(430.0))  # +7.5%

    assert get_cached("TEST") is not None


def test_get_cached_refuses_when_price_fetch_fails(monkeypatch):
    _save()

    def _raise(ticker):
        raise ValueError("network error")

    monkeypatch.setattr(storage_module, "fetch_price_data", _raise)
    assert get_cached("TEST") is None


def test_get_cached_is_scoped_to_ticker(monkeypatch):
    _save(ticker="AAA")
    monkeypatch.setattr(storage_module, "fetch_price_data", lambda ticker: _price(400.0))

    assert get_cached("BBB") is None
    assert get_cached("AAA") is not None


def test_get_cached_refuses_when_original_price_missing_and_live_fetch_fails(monkeypatch):
    _save(verdict_json={"verdict": "WATCH", "price_date": "2026-08-19", "confidence": 6})

    def _raise(ticker):
        raise ValueError("network error")

    monkeypatch.setattr(storage_module, "fetch_price_data", _raise)
    assert get_cached("TEST") is None


def test_build_staleness_banner_shows_analysis_and_live_price():
    from stockbot.models import Analysis, ValidationResult

    analysis = Analysis(
        ticker="TEST",
        run_date=date(2026, 8, 19),
        verdict_json={
            **VERDICT_JSON,
            "analysis_price_abs": 400.0,
            "analysis_price_date": "2026-08-19",
            "current_price_abs": 340.0,
            "price_date": "2026-08-25",
        },
        report_md="# r",
        costs=39.0,
        validation=ValidationResult(True, []),
        missing=[],
    )
    banner = build_staleness_banner(analysis, current_price_abs=340.0)
    assert "400.00" in banner
    assert "340.00" in banner
    assert "-15.0%" in banner
    assert "fresh after new results" in banner
