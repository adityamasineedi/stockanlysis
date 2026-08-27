"""Module 3 (fundamentals) unit tests against synthetic Screener-shaped
HTML — no network. Live behaviour (real basis fallback, real 404 handling,
real negative-number formatting) was already verified by hand against
RELIANCE, IDEA, and JYOTHYLAB during development."""

from unittest.mock import MagicMock

import pandas as pd
from bs4 import BeautifulSoup

from stockbot.fetch import fundamentals
from stockbot.fetch.fundamentals import (
    FundamentalsSchemaError,
    _validate_schema,
    detect_basis,
    fetch_business_description,
    fetch_cash_equivalents_row,
    parse_number,
    parse_screener_table,
)


def _section_html(section_id: str, headers: list[str], rows: list[tuple[str, list[str]]]) -> str:
    header_cells = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = ""
    for label, values in rows:
        cells = "".join(f"<td>{v}</td>" for v in values)
        body_rows += f"<tr><td>{label}</td>{cells}</tr>"
    return f"""
    <section id="{section_id}">
        <table>
            <thead><tr><th></th>{header_cells}</tr></thead>
            <tbody>{body_rows}</tbody>
        </table>
    </section>
    """


def testparse_number_strips_commas_and_percent():
    assert parse_number("1,23,456") == 123456.0
    assert parse_number("16%") == 16.0
    assert parse_number("-6,499") == -6499.0
    assert parse_number("-") is None
    assert parse_number("") is None


def testdetect_basis_consolidated():
    soup = BeautifulSoup("<div>Consolidated Figures in Rs. Crores</div>", "lxml")
    assert detect_basis(soup) == "consolidated"


def testdetect_basis_standalone():
    soup = BeautifulSoup("<div>Standalone Figures in Rs. Crores</div>", "lxml")
    assert detect_basis(soup) == "standalone"


def testdetect_basis_unknown_page_returns_none():
    soup = BeautifulSoup("<div>Page not found</div>", "lxml")
    assert detect_basis(soup) is None


def testparse_screener_table_strips_plus_suffix_and_negative_numbers():
    html = _section_html(
        "profit-loss",
        ["Mar 2023", "Mar 2024"],
        [("Sales+", ["1,000", "1,200"]), ("Net Profit+", ["-50", "80"])],
    )
    soup = BeautifulSoup(html, "lxml")
    df = parse_screener_table(soup, "profit-loss")

    assert list(df.index) == ["Sales", "Net Profit"]
    assert list(df.columns) == ["Mar 2023", "Mar 2024"]
    assert df.loc["Sales", "Mar 2024"] == 1200.0
    assert df.loc["Net Profit", "Mar 2023"] == -50.0
    assert df.dtypes.unique().tolist() == [df["Mar 2023"].dtype]  # all numeric


def testparse_screener_table_missing_section_raises():
    soup = BeautifulSoup("<div>nothing here</div>", "lxml")
    try:
        parse_screener_table(soup, "profit-loss")
        raise AssertionError("expected FundamentalsSchemaError")
    except FundamentalsSchemaError:
        pass


def _minimal_frames():
    pnl_html = _section_html(
        "profit-loss",
        ["Mar 2022", "Mar 2023", "Mar 2024"],
        [("Sales", ["100", "110", "120"]), ("Net Profit", ["10", "12", "14"])],
    )
    bs_html = _section_html(
        "balance-sheet",
        ["Mar 2022", "Mar 2023", "Mar 2024"],
        [("Total Assets", ["500", "550", "600"])],
    )
    cf_html = _section_html(
        "cash-flow",
        ["Mar 2022", "Mar 2023", "Mar 2024"],
        [("Net Cash Flow", ["5", "6", "7"])],
    )
    ratios_html = _section_html(
        "ratios", ["Mar 2022", "Mar 2023", "Mar 2024"], [("ROCE %", ["10%", "11%", "12%"])]
    )
    soup = BeautifulSoup(pnl_html + bs_html + cf_html + ratios_html, "lxml")
    return (
        parse_screener_table(soup, "profit-loss"),
        parse_screener_table(soup, "balance-sheet"),
        parse_screener_table(soup, "cash-flow"),
        parse_screener_table(soup, "ratios"),
    )


def test_validate_schema_passes_on_well_formed_tables():
    pnl, bs, cf, ratios = _minimal_frames()
    _validate_schema(pnl, bs, cf, ratios)  # should not raise


def test_validate_schema_raises_on_missing_net_profit_row():
    pnl, bs, cf, ratios = _minimal_frames()
    pnl = pnl.drop(index="Net Profit")
    try:
        _validate_schema(pnl, bs, cf, ratios)
        raise AssertionError("expected FundamentalsSchemaError")
    except FundamentalsSchemaError:
        pass


def test_validate_schema_accepts_revenue_in_place_of_sales_for_banks():
    # verified live against HDFCBANK's real Screener page: banks/NBFCs use
    # "Revenue" instead of "Sales" as the P&L top line
    pnl, bs, cf, ratios = _minimal_frames()
    pnl = pnl.rename(index={"Sales": "Revenue"})
    _validate_schema(pnl, bs, cf, ratios)  # should not raise


def test_validate_schema_raises_when_neither_sales_nor_revenue_present():
    pnl, bs, cf, ratios = _minimal_frames()
    pnl = pnl.drop(index="Sales")
    try:
        _validate_schema(pnl, bs, cf, ratios)
        raise AssertionError("expected FundamentalsSchemaError")
    except FundamentalsSchemaError:
        pass


def test_validate_schema_raises_on_empty_ratios():
    pnl, bs, cf, ratios = _minimal_frames()
    ratios = ratios.iloc[0:0]
    try:
        _validate_schema(pnl, bs, cf, ratios)
        raise AssertionError("expected FundamentalsSchemaError")
    except FundamentalsSchemaError:
        pass


def test_fetch_business_description_extracts_about_block_text():
    soup = BeautifulSoup(
        '<div class="sub show-more-box about"><p>KPIT is a global technology '
        'company.<sup><a href="https://x">[1]</a></sup></p></div>',
        "lxml",
    )
    assert fetch_business_description(soup) == "KPIT is a global technology company."


def test_fetch_business_description_returns_none_when_about_block_absent():
    soup = BeautifulSoup("<div>no about block here</div>", "lxml")
    assert fetch_business_description(soup) is None


def _balance_sheet_with_other_assets() -> pd.DataFrame:
    return pd.DataFrame(
        {"Mar 2023": [500.0], "Mar 2024": [600.0]}, index=["Other Assets"]
    )


def _company_info_soup(company_id: str = "123", consolidated: str = "true") -> BeautifulSoup:
    return BeautifulSoup(
        f'<div id="company-info" data-company-id="{company_id}" '
        f'data-consolidated="{consolidated}"></div>',
        "lxml",
    )


def test_fetch_cash_equivalents_row_returns_none_without_other_assets_row(monkeypatch):
    # No "Other Assets" row -> nothing to expand, no HTTP call should even
    # be attempted (banks/NBFCs have a different schema entirely).
    called = MagicMock()
    monkeypatch.setattr(fundamentals.httpx, "Client", called)
    balance_sheet = pd.DataFrame({"Mar 2024": [1.0]}, index=["Total Assets"])
    result = fetch_cash_equivalents_row(_company_info_soup(), balance_sheet)
    assert result is None
    called.assert_not_called()


def test_fetch_cash_equivalents_row_returns_none_without_company_info():
    soup = BeautifulSoup("<div>no company-info div here</div>", "lxml")
    result = fetch_cash_equivalents_row(soup, _balance_sheet_with_other_assets())
    assert result is None


def test_fetch_cash_equivalents_row_parses_real_schedule_response(monkeypatch):
    # Real shape confirmed live against Screener's schedules endpoint for
    # KPITTECH: {"Inventories": {...}, "Cash Equivalents": {...}, ...}.
    schedule_json = {
        "Inventories": {"Mar 2023": "59", "Mar 2024": "90"},
        "Cash Equivalents": {"Mar 2023": "549", "Mar 2024": "771"},
    }
    fake_response = MagicMock()
    fake_response.json.return_value = schedule_json
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.return_value = fake_response
    monkeypatch.setattr(fundamentals.httpx, "Client", MagicMock(return_value=fake_client))
    monkeypatch.setattr(fundamentals, "_rate_limit", lambda: None)

    result = fetch_cash_equivalents_row(
        _company_info_soup(), _balance_sheet_with_other_assets()
    )

    assert result is not None
    assert result.name == "Cash Equivalents"
    assert result["Mar 2023"] == 549.0
    assert result["Mar 2024"] == 771.0
    fake_response.raise_for_status.assert_called_once()


def test_fetch_cash_equivalents_row_returns_none_when_schedule_has_no_cash_key(monkeypatch):
    # Confirmed live: an inapplicable parent (e.g. a bank's schema) returns
    # HTTP 200 with an empty {} body, not an error.
    fake_response = MagicMock()
    fake_response.json.return_value = {}
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.return_value = fake_response
    monkeypatch.setattr(fundamentals.httpx, "Client", MagicMock(return_value=fake_client))
    monkeypatch.setattr(fundamentals, "_rate_limit", lambda: None)

    result = fetch_cash_equivalents_row(
        _company_info_soup(), _balance_sheet_with_other_assets()
    )
    assert result is None


def test_fetch_cash_equivalents_row_uses_consolidated_flag_from_company_info(monkeypatch):
    fake_response = MagicMock()
    fake_response.json.return_value = {"Cash Equivalents": {"Mar 2024": "100"}}
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.return_value = fake_response
    monkeypatch.setattr(fundamentals.httpx, "Client", MagicMock(return_value=fake_client))
    monkeypatch.setattr(fundamentals, "_rate_limit", lambda: None)

    fetch_cash_equivalents_row(
        _company_info_soup(consolidated="false"), _balance_sheet_with_other_assets()
    )

    _, kwargs = fake_client.__enter__.return_value.get.call_args
    assert kwargs["params"]["consolidated"] == "false"
