"""Module 3 — fundamentals. Parses the public Screener.in company page.

Verified structure (inspected real pages for RELIANCE and IDEA before
writing this parser):
  - Each statement lives in <section id="..."><table>, with a <thead> row
    of period labels ("Mar 2024" for years, "Jun 2024" for quarters) and a
    <tbody> of label/value rows. Row labels can carry a trailing "+" for
    Screener's expandable sub-items — stripped on parse.
  - Section ids used here: profit-loss, balance-sheet, cash-flow, ratios,
    quarters. #shareholding is Module 4's territory (fetch/shareholding.py),
    which reuses fetch_screener_page/parse_screener_table/detect_basis
    below rather than re-fetching or re-parsing the page.
  - The page states "Consolidated Figures in Rs. Crores" or "Standalone
    Figures in Rs. Crores" near the top — this is both the basis signal
    and confirmation the values are already in ₹ crore, matching this
    project's units convention with no conversion needed.
  - Negative values use a plain leading '-', not parentheses.
  - The #quarters table has a trailing "Raw PDF" column that's a button,
    not data — its header doesn't match the period-label pattern so it's
    naturally excluded rather than needing a special case.
"""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from stockbot.config import HTTP_USER_AGENT, SCREENER_CACHE_DIR
from stockbot.models import Financials

logger = logging.getLogger(__name__)

USER_AGENT = HTTP_USER_AGENT
CACHE_MAX_AGE_HOURS = 24
MIN_REQUEST_INTERVAL_SECONDS = 1.0

_ROW_LABEL_SUFFIX_RE = re.compile(r"\+\s*$")
_PERIOD_RE = re.compile(r"^[A-Za-z]{3} \d{4}$")  # e.g. "Mar 2024", "Jun 2024"

_last_request_time: float = 0.0


class FundamentalsSchemaError(Exception):
    pass


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def _cache_path(symbol: str, basis: str) -> Path:
    return SCREENER_CACHE_DIR / f"{symbol}_{basis}.html"


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    reraise=True,
    retry=retry_if_exception_type(httpx.HTTPError),
)
def _http_get_screener(url: str) -> httpx.Response:
    """Network fetch with one retry on transient HTTP failures."""
    _rate_limit()
    with httpx.Client(
        timeout=30.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        return client.get(url)


def fetch_screener_page(symbol: str, basis: str) -> str:
    cache_path = _cache_path(symbol, basis)
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < CACHE_MAX_AGE_HOURS:
            return cache_path.read_text(encoding="utf-8")

    url = f"https://www.screener.in/company/{symbol}/{basis}/"
    response = _http_get_screener(url)
    if response.status_code == 404:
        raise FundamentalsSchemaError(f"Screener has no {basis!r} page for {symbol!r} (404)")
    response.raise_for_status()

    html = response.text
    SCREENER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


def parse_number(raw: str) -> float | None:
    cleaned = raw.strip().replace(",", "").replace("%", "")
    if cleaned in ("", "-"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_screener_table(soup: BeautifulSoup, section_id: str) -> pd.DataFrame:
    section = soup.find("section", id=section_id)
    if section is None:
        raise FundamentalsSchemaError(f"Screener page has no #{section_id} section")
    table = section.find("table")
    if table is None:
        raise FundamentalsSchemaError(f"#{section_id} section has no table")

    thead = table.find("thead")
    header_cells = thead.find_all("th") if thead else []
    columns = [c.get_text(strip=True) for c in header_cells[1:]]

    tbody = table.find("tbody")
    if tbody is None:
        raise FundamentalsSchemaError(f"#{section_id} table has no tbody")

    rows: dict[str, list[float | None]] = {}
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        label = _ROW_LABEL_SUFFIX_RE.sub("", cells[0].get_text(strip=True)).strip()
        if not label:
            continue
        values = [parse_number(c.get_text(strip=True)) for c in cells[1 : 1 + len(columns)]]
        values += [None] * (len(columns) - len(values))
        rows[label] = values

    # from_dict(orient="index") rather than pd.DataFrame(rows, index=columns).T —
    # transposing a DataFrame built the other way around upcasts every column to
    # object dtype the moment any two source rows have differing dtypes, which
    # silently violates the "numeric dtypes" contract.
    df = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    return df.apply(pd.to_numeric, errors="coerce")


def detect_basis(soup: BeautifulSoup) -> str | None:
    text = soup.get_text()
    if "Consolidated Figures" in text:
        return "consolidated"
    if "Standalone Figures" in text:
        return "standalone"
    return None


def fetch_business_description(soup: BeautifulSoup) -> str | None:
    """Screener's own company page carries a short "About" blurb (verified
    live: KPITTECH's reads "KPIT is a global technology company with
    software solutions...") — free to extract from the same page fetch
    fundamentals already makes, no separate request. This exists because
    v3's §2 ("COMPANY IN 60 SECONDS") requires the reader understand what
    the company does, but the annual report's own text (auditor's report,
    key audit matters, contingent liabilities, related party) never
    describes the business itself — that gap showed up live as §2 going
    empty ("the supplied context does not include a business-description
    block") on a real report. None if Screener has no About block for this
    company — a genuine MISSING, not an empty string.
    """
    container = soup.select_one("div.sub.show-more-box.about")
    if container is None:
        return None
    for sup in container.find_all("sup"):  # citation markers ("[1]"), not content
        sup.decompose()
    text = " ".join(container.get_text(separator=" ", strip=True).split())
    return text or None


_CASH_ROW_RE = re.compile(r"cash", re.IGNORECASE)


def fetch_cash_equivalents_row(soup: BeautifulSoup, balance_sheet: pd.DataFrame) -> pd.Series | None:
    """Screener's condensed balance sheet has no direct "Cash" row for most
    non-financial companies — it's nested one level down under the
    expandable "Other Assets+" row, loaded client-side via
    Company.showSchedule() and never present in the static HTML at all
    (confirmed live: KPITTECH's #balance-sheet tbody has no cash row of any
    name). A report built from the condensed table alone can state
    borrowings but never cash, making net debt uncomputable — found live on
    a real KPITTECH report that built its entire "debt-funded acquisition"
    framing on borrowings alone.

    The schedule is a real, separate JSON endpoint
    (/api/company/<id>/schedules/?parent=<row>&section=<section>&consolidated=<bool>),
    confirmed live: returns the sub-line items (Inventories, Trade
    receivables, Cash Equivalents, ...) for whichever row is passed as
    `parent`, or an empty `{}` (HTTP 200, not an error) when that parent
    doesn't apply to this company's schema — e.g. banks/NBFCs, which don't
    have an "Other Assets" breakdown at all. That empty case is a genuine
    "not applicable", not a fetch failure, so it returns None rather than
    raising.
    """
    if "Other Assets" not in balance_sheet.index:
        return None

    info = soup.find(id="company-info")
    if info is None or not info.get("data-company-id"):
        return None
    company_id = info["data-company-id"]
    consolidated = info.get("data-consolidated") == "true"

    _rate_limit()
    with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(
            f"https://www.screener.in/api/company/{company_id}/schedules/",
            params={
                "parent": "Other Assets",
                "section": "balance-sheet",
                "consolidated": "true" if consolidated else "false",
            },
        )
    response.raise_for_status()
    schedule = response.json()

    cash_key = next((key for key in schedule if _CASH_ROW_RE.search(key)), None)
    if cash_key is None:
        return None

    values = {period: parse_number(raw) for period, raw in schedule[cash_key].items()}
    return pd.Series(values, name="Cash Equivalents").reindex(balance_sheet.columns)


def _validate_schema(
    pnl: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
    ratios: pd.DataFrame,
) -> None:
    # "Sales" is the top-line label for non-financial companies; banks/NBFCs
    # use "Revenue" instead (verified live: HDFCBANK's P&L has "Revenue" and
    # "Net Profit" but no "Sales" row at all — confirmed via its Screener
    # page directly, not a fetch failure). Accept either.
    has_top_line = "Sales" in pnl.index or "Revenue" in pnl.index
    if pnl.empty or not has_top_line or "Net Profit" not in pnl.index:
        raise FundamentalsSchemaError(
            "P&L table missing expected rows (Sales or Revenue, and Net Profit)"
        )
    if balance_sheet.empty or "Total Assets" not in balance_sheet.index:
        raise FundamentalsSchemaError("Balance sheet table missing expected row (Total Assets)")
    if cash_flow.empty or "Net Cash Flow" not in cash_flow.index:
        raise FundamentalsSchemaError("Cash flow table missing expected row (Net Cash Flow)")
    if ratios.empty:
        raise FundamentalsSchemaError("Ratios table is empty")

    year_columns = sum(1 for c in pnl.columns if _PERIOD_RE.match(c))
    if year_columns < 3:
        raise FundamentalsSchemaError(
            f"P&L table has only {year_columns} year columns — implausibly little history"
        )


_CRORE = 1e7

_YF_PNL_ROWS: dict[str, tuple[str, ...]] = {
    "Sales": ("Total Revenue", "Operating Revenue"),
    "Operating Profit": ("Operating Income",),
    "Depreciation": ("Reconciled Depreciation", "Depreciation Income Statement"),
    "Interest": ("Interest Expense", "Interest Expense Non Operating"),
    "Net Profit": ("Net Income", "Net Income Common Stockholders"),
    "EPS in Rs": ("Basic EPS", "Diluted EPS"),
}

_YF_BS_ROWS: dict[str, tuple[str, ...]] = {
    "Total Assets": ("Total Assets",),
    "Borrowings": ("Total Debt",),
    "Cash Equivalents": (
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ),
    "Shareholders Funds": (
        "Stockholders Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
    ),
}

_YF_CF_ROWS: dict[str, tuple[str, ...]] = {
    "Cash from Operating Activity": ("Operating Cash Flow",),
    "Cash from Investing Activity": ("Investing Cash Flow",),
    "Cash from Financing Activity": ("Financing Cash Flow",),
    "Free Cash Flow": ("Free Cash Flow",),
}


def _yf_first_row(df: pd.DataFrame, *substrings: str) -> pd.Series | None:
    for sub in substrings:
        needle = sub.lower()
        for idx in df.index:
            if str(idx).lower() == needle:
                return df.loc[idx]
        matches = [idx for idx in df.index if needle in str(idx).lower()]
        if matches:
            # Prefer the tightest label — avoids "Other Non Operating Income Expenses"
            # winning over "Operating Income", etc.
            best = min(matches, key=lambda label: len(str(label)))
            return df.loc[best]
    return None


def _yf_sorted_columns(df: pd.DataFrame) -> list[object]:
    return sorted(df.columns, key=lambda c: pd.Timestamp(c))


def _yf_period_label(column: object) -> str:
    return pd.Timestamp(column).strftime("%b %Y")


def _to_crore(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value) / _CRORE
    except (TypeError, ValueError):
        return None


def _build_yf_statement(
    yf_df: pd.DataFrame,
    row_map: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    if yf_df is None or yf_df.empty:
        return pd.DataFrame()
    col_order = _yf_sorted_columns(yf_df)
    labels = [_yf_period_label(c) for c in col_order]
    rows: dict[str, list[float | None]] = {}
    for screener_label, yf_names in row_map.items():
        series = _yf_first_row(yf_df, *yf_names)
        if series is None:
            continue
        rows[screener_label] = [_to_crore(series[c]) for c in col_order]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index", columns=labels)


def _append_net_cash_flow(cf: pd.DataFrame) -> pd.DataFrame:
    if cf.empty:
        return cf
    parts = (
        "Cash from Operating Activity",
        "Cash from Investing Activity",
        "Cash from Financing Activity",
    )
    if not all(row in cf.index for row in parts):
        return cf
    net: list[float | None] = []
    for col in cf.columns:
        values = [cf.loc[row, col] for row in parts]
        if any(v is None for v in values):
            net.append(None)
        else:
            net.append(float(sum(values)))
    cf = cf.copy()
    cf.loc["Net Cash Flow"] = net
    return cf


def _build_yf_ratios(pnl: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.DataFrame:
    cols = list(pnl.columns)
    roe_vals: list[float | None] = []
    roce_vals: list[float | None] = []
    for col in cols:
        pat = pnl.loc["Net Profit", col] if "Net Profit" in pnl.index else None
        equity = (
            balance_sheet.loc["Shareholders Funds", col]
            if "Shareholders Funds" in balance_sheet.index
            else None
        )
        op = pnl.loc["Operating Profit", col] if "Operating Profit" in pnl.index else None
        debt = balance_sheet.loc["Borrowings", col] if "Borrowings" in balance_sheet.index else None
        if pat is not None and equity is not None and equity > 0:
            roe_vals.append((pat / equity) * 100.0)
        else:
            roe_vals.append(None)
        capital = None
        if equity is not None and debt is not None:
            capital = equity + debt
        if op is not None and capital is not None and capital > 0:
            roce_vals.append((op / capital) * 100.0)
        else:
            roce_vals.append(None)
    return pd.DataFrame({"ROE %": roe_vals, "ROCE %": roce_vals}, index=cols).T


def _fetch_yfinance_fundamentals(
    symbol: str,
    *,
    business_description: str | None = None,
) -> Financials:
    """Build Screener-shaped statements from yfinance when Screener is stale or empty."""
    last_error: str | None = None
    for suffix in (".NS", ".BO"):
        yf_symbol = f"{symbol}{suffix}"
        try:
            ticker = yf.Ticker(yf_symbol)
            pnl_raw = ticker.financials
            bs_raw = ticker.balance_sheet
            cf_raw = ticker.cashflow
            if pnl_raw is None or pnl_raw.empty:
                last_error = f"{yf_symbol}: empty financials"
                continue
            if bs_raw is None or bs_raw.empty:
                last_error = f"{yf_symbol}: empty balance sheet"
                continue
            if cf_raw is None or cf_raw.empty:
                last_error = f"{yf_symbol}: empty cash flow"
                continue

            pnl = _build_yf_statement(pnl_raw, _YF_PNL_ROWS)
            balance_sheet = _build_yf_statement(bs_raw, _YF_BS_ROWS)
            cash_flow = _append_net_cash_flow(_build_yf_statement(cf_raw, _YF_CF_ROWS))
            ratios = _build_yf_ratios(pnl, balance_sheet)
            quarterly_raw = ticker.quarterly_financials
            quarterly = (
                _build_yf_statement(quarterly_raw, _YF_PNL_ROWS)
                if quarterly_raw is not None and not quarterly_raw.empty
                else pnl.copy()
            )
            _validate_schema(pnl, balance_sheet, cash_flow, ratios)
        except FundamentalsSchemaError as exc:
            last_error = f"{yf_symbol}: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001 - try next suffix
            last_error = f"{yf_symbol}: {type(exc).__name__}: {exc}"
            continue

        years_available = sum(1 for c in pnl.columns if _PERIOD_RE.match(c))
        logger.info(
            "Fundamentals fallback to yfinance for %s via %s (%d fiscal years)",
            symbol,
            yf_symbol,
            years_available,
        )
        return Financials(
            pnl=pnl,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            ratios=ratios,
            quarterly=quarterly,
            basis="consolidated",
            years_available=years_available,
            source=f"yfinance:{suffix.removeprefix('.').lower()}",
            fetched_at=datetime.now(UTC),
            business_description=business_description,
        )

    raise FundamentalsSchemaError(
        f"yfinance fallback failed for {symbol!r}"
        + (f" ({last_error})" if last_error else "")
    )


def fetch_fundamentals(symbol: str) -> Financials:
    basis_tried: list[str] = []
    business_description: str | None = None
    for basis in ("consolidated", "standalone"):
        basis_tried.append(basis)
        try:
            html = fetch_screener_page(symbol, basis)
        except FundamentalsSchemaError:
            continue

        soup = BeautifulSoup(html, "lxml")
        if business_description is None:
            business_description = fetch_business_description(soup)
        detected_basis = detect_basis(soup)
        if detected_basis is None:
            continue

        try:
            pnl = parse_screener_table(soup, "profit-loss")
            balance_sheet = parse_screener_table(soup, "balance-sheet")
            cash_flow = parse_screener_table(soup, "cash-flow")
            ratios = parse_screener_table(soup, "ratios")
            quarterly = parse_screener_table(soup, "quarters")
            _validate_schema(pnl, balance_sheet, cash_flow, ratios)
        except FundamentalsSchemaError:
            continue

        years_available = sum(1 for c in pnl.columns if _PERIOD_RE.match(c))

        cash_row = fetch_cash_equivalents_row(soup, balance_sheet)
        if cash_row is not None:
            balance_sheet = pd.concat([balance_sheet, cash_row.to_frame().T])

        return Financials(
            pnl=pnl,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            ratios=ratios,
            quarterly=quarterly,
            basis=detected_basis,
            years_available=years_available,
            source=f"screener:{basis}",
            fetched_at=datetime.now(UTC),
            business_description=business_description,
        )

    try:
        return _fetch_yfinance_fundamentals(
            symbol,
            business_description=business_description,
        )
    except FundamentalsSchemaError as exc:
        raise FundamentalsSchemaError(
            f"Could not fetch usable fundamentals for {symbol!r} from Screener "
            f"(tried: {', '.join(basis_tried)}) or yfinance ({exc})"
        ) from exc
