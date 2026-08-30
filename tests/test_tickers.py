"""Module 1 unit tests. Uses a small fixture table built from real NSE rows
so no network call is needed — resolve_ticker(query, table) accepts an
explicit table for exactly this reason."""

import pandas as pd
import pytest

from stockbot.fetch.tickers import normalize_company_name, resolve_ticker
from stockbot.models import AmbiguousMatch, TickerInfo

FIXTURE_ROWS = [
    ("TCS", "Tata Consultancy Services Limited", "INE467B01029"),
    ("RELIANCE", "Reliance Industries Limited", "INE002A01018"),
    ("RPOWER", "Reliance Power Limited", "INE614G01033"),
    ("RCOM", "Reliance Communications Limited", "INE330H01018"),
    ("RINFRA", "Reliance Infrastructure Limited", "INE036A01016"),
    ("INFY", "Infosys Limited", "INE009A01021"),
    ("HCL-INSYS", "HCL Infosystems Limited", "INE236A01020"),
    ("HDFCAMC", "HDFC Asset Management Company Limited", "INE127D01025"),
    ("HDFCBANK", "HDFC Bank Limited", "INE040A01034"),
    ("HDFCLIFE", "HDFC Life Insurance Company Limited", "INE795G01014"),
]


@pytest.fixture
def table() -> pd.DataFrame:
    rows = [
        {
            "symbol": symbol,
            "exchange": "NSE",
            "company_name": name,
            "isin": isin,
            "normalized_name": normalize_company_name(name),
        }
        for symbol, name, isin in FIXTURE_ROWS
    ]
    return pd.DataFrame(rows)


def test_exact_symbol_match(table):
    result = resolve_ticker("TCS", table)
    assert isinstance(result, TickerInfo)
    assert result.symbol == "TCS"


def test_bse_only_symbol_via_yfinance(monkeypatch, table):
    monkeypatch.setattr(
        "stockbot.fetch.tickers._resolve_bse_only_via_yfinance",
        lambda query: TickerInfo(
            symbol=query.upper(),
            exchange="BSE",
            company_name="BSE Only Co",
            isin=None,
        ),
    )
    result = resolve_ticker("BSEONLY", table)
    assert isinstance(result, TickerInfo)
    assert result.exchange == "BSE"


def test_exact_symbol_match_case_insensitive(table):
    result = resolve_ticker("tcs", table)
    assert isinstance(result, TickerInfo)
    assert result.symbol == "TCS"


def test_alias_table_resolves_flagship(table):
    result = resolve_ticker("Reliance", table)
    assert isinstance(result, TickerInfo)
    assert result.symbol == "RELIANCE"


def test_exact_normalized_name_match(table):
    result = resolve_ticker("Infosys", table)
    assert isinstance(result, TickerInfo)
    assert result.symbol == "INFY"


def test_genuinely_ambiguous_query(table):
    result = resolve_ticker("HDFC", table)
    assert isinstance(result, AmbiguousMatch)
    symbols = {c.symbol for c in result.candidates}
    assert symbols == {"HDFCAMC", "HDFCBANK", "HDFCLIFE"}


def test_unknown_input_returns_none(table):
    assert resolve_ticker("asdfgh", table) is None


def test_blank_input_returns_none(table):
    assert resolve_ticker("   ", table) is None


def test_short_prefix_fragment_surfaces_matching_company():
    # Regression test: "rel" scored only ~27 against "reliance industries"
    # under token_set_ratio (a token-overlap scorer with no shared whole
    # token), so it never crossed the 60-cutoff and Reliance never appeared
    # as a candidate at all — found live from a real user typing "rel"
    # expecting Reliance to be suggested. Needs the real multi-company NSE
    # table (RELIANCE/RPOWER/RCOM/RINFRA all sharing "reliance" as their
    # first word) to be meaningful, so build a slightly bigger fixture here
    # rather than reusing the small module-level one.
    rows = [
        ("RELIANCE", "Reliance Industries Limited", "INE002A01018"),
        ("RPOWER", "Reliance Power Limited", "INE614G01033"),
        ("TCS", "Tata Consultancy Services Limited", "INE467B01029"),
        ("INFY", "Infosys Limited", "INE009A01021"),
    ]
    big_table = pd.DataFrame(
        [
            {
                "symbol": s,
                "exchange": "NSE",
                "company_name": n,
                "isin": i,
                "normalized_name": normalize_company_name(n),
            }
            for s, n, i in rows
        ]
    )
    result = resolve_ticker("rel", big_table)
    assert isinstance(result, AmbiguousMatch)
    symbols = {c.symbol for c in result.candidates}
    assert "RELIANCE" in symbols
    assert "RPOWER" in symbols
    assert "TCS" not in symbols  # unrelated company must not be pulled in


def test_prefix_fragment_resolves_uniquely_when_only_one_match():
    rows = [
        ("TCS", "Tata Consultancy Services Limited", "INE467B01029"),
        ("RELIANCE", "Reliance Industries Limited", "INE002A01018"),
    ]
    small_table = pd.DataFrame(
        [
            {
                "symbol": s,
                "exchange": "NSE",
                "company_name": n,
                "isin": i,
                "normalized_name": normalize_company_name(n),
            }
            for s, n, i in rows
        ]
    )
    result = resolve_ticker("tat", small_table)
    assert isinstance(result, TickerInfo)
    assert result.symbol == "TCS"


def test_full_company_name_resolves_uniquely(table):
    result = resolve_ticker("Reliance Power Limited", table)
    assert isinstance(result, TickerInfo)
    assert result.symbol == "RPOWER"


def test_normalize_company_name_strips_suffix_and_punctuation():
    assert normalize_company_name("Reliance Industries Limited") == "reliance industries"
    assert normalize_company_name("HDFC Bank Ltd.") == "hdfc bank"


def test_suggest_tickers_prefix_and_fuzzy(table):
    from stockbot.fetch.tickers import suggest_tickers

    hits = suggest_tickers("tcs", table=table, limit=5)
    assert len(hits) >= 1
    assert hits[0].symbol == "TCS"

    hero = suggest_tickers("hcl", table=table, limit=5)
    symbols = {t.symbol for t in hero}
    assert "HCL-INSYS" in symbols
