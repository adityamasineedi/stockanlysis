"""Module 7 — brief assembly. The one document everything downstream reads.

Runs modules 3-6 in parallel via ThreadPoolExecutor (not asyncio — every
fetcher in this project is synchronous/blocking: yfinance, httpx-sync,
pdfplumber). Price and the technicals computed from it (Module 2) are
fetched first and are fatal: if that fails, assemble_brief raises rather
than returning a degraded brief, since nothing else here is trustworthy
without a live price. Financials, shareholding, and news degrade to None
on total failure, with a MISSING entry recording why — see PROJECT.md for
why those three are Optional on Brief while annual_report is not (its own
fetcher already returns an empty ReportText for "not found", so it never
needs Brief-level None).

Per-fetcher timeout is 120s, uniform across all four parallel fetchers —
generous margin above the ~55s observed for annual-report ingestion
against a real, dense filing during Module 6 development. They run
concurrently, so the brief's wall-clock time is bounded by the slowest
fetcher, not the sum.

Red-flag news is capped per adversarial query (not globally) when
rendered to markdown — a heavily-covered company can return 300+ deduped
candidates, mostly noise from one broad query, which would otherwise
crowd out a real hit from a quieter query.
"""

from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime

from tabulate import tabulate

from stockbot.analysis.technicals import compute_technicals
from stockbot.brief_enrichment import enrich_brief
from stockbot.fetch.annual_report import business_narrative_gap, fetch_annual_report
from stockbot.fetch.fundamentals import fetch_fundamentals
from stockbot.fetch.news import fetch_news
from stockbot.fetch.prices import fetch_price_data
from stockbot.fetch.shareholding import fetch_shareholding
from stockbot.models import (
    Brief,
    Financials,
    NewsItems,
    PriceData,
    RedFlag,
    ReportText,
    Shareholding,
    Technicals,
    TickerInfo,
)

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 120
ANNUAL_REPORT_TIMEOUT_SECONDS = 180
CHARS_PER_TOKEN_ESTIMATE = 4
RED_FLAGS_PER_QUERY_LIMIT = 8
GENERAL_NEWS_DISPLAY_LIMIT = 15


def _resolve(
    future: Future,
    label: str,
    missing: list[str],
    *,
    timeout_seconds: int = FETCH_TIMEOUT_SECONDS,
):
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        message = f"MISSING: {label} — timed out after {timeout_seconds}s"
        logger.warning(message)
        missing.append(message)
        return None
    except Exception as exc:  # noqa: BLE001 - this is the module's resilience boundary
        message = f"MISSING: {label} — {exc}"
        logger.warning(message)
        missing.append(message)
        return None


def assemble_brief(ticker: TickerInfo) -> Brief:
    price = fetch_price_data(ticker.symbol)  # fatal — let it raise
    technicals = compute_technicals(price)  # fatal — deterministic given price

    missing: list[str] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        financials_future = executor.submit(fetch_fundamentals, ticker.symbol)
        shareholding_future = executor.submit(
            fetch_shareholding, ticker.symbol, exchange=ticker.exchange
        )
        news_future = executor.submit(fetch_news, ticker.company_name)
        annual_report_future = executor.submit(fetch_annual_report, ticker.symbol)

        financials = _resolve(financials_future, "financials", missing)
        shareholding = _resolve(shareholding_future, "shareholding", missing)
        news = _resolve(news_future, "news", missing)
        annual_report_result = _resolve(
            annual_report_future,
            "annual report",
            missing,
            timeout_seconds=ANNUAL_REPORT_TIMEOUT_SECONDS,
        )

    if annual_report_result is None:
        # the future itself failed/timed out — _resolve already recorded why
        annual_report = ReportText(
            sections={},
            report_year=None,
            source_url=None,
            truncated=False,
            dropped_sections=[],
            source="nse_annual_reports",
            fetched_at=datetime.now(UTC),
        )
    else:
        annual_report = annual_report_result
        if not annual_report.sections:
            reason = (
                "not found on NSE"
                if annual_report.source_url is None
                else "found but no usable text extracted "
                "(scanned/image-only, or no target headings matched)"
            )
            missing.append(f"MISSING: annual report — {reason}")
        else:
            narrative_gap = business_narrative_gap(
                annual_report.sections,
                annual_report.dropped_sections,
            )
            if narrative_gap:
                missing.append(narrative_gap)

    confidence_ceiling = 10
    if financials is None:
        confidence_ceiling = min(confidence_ceiling, 4)
    if not annual_report.sections:
        confidence_ceiling = min(confidence_ceiling, 5)

    brief = Brief(
        ticker=ticker,
        price=price,
        technicals=technicals,
        financials=financials,
        shareholding=shareholding,
        news=news,
        annual_report=annual_report,
        missing=missing,
        token_count=0,
        confidence_ceiling=confidence_ceiling,
        generated_at=datetime.now(UTC),
    )

    brief = enrich_brief(brief)
    markdown = to_markdown(brief)
    token_count = len(markdown) // CHARS_PER_TOKEN_ESTIMATE
    return dataclasses.replace(brief, token_count=token_count)


def _fmt_or_missing(value: float | None, why: str) -> str:
    return f"MISSING: {why}" if value is None else f"{value:.2f}"


def _pct_or_missing(value: float | None, why: str = "not available from this source") -> str:
    return f"MISSING: {why}" if value is None else f"{value:.2f}%"


def format_price_section(price: PriceData, technicals: Technicals) -> str:
    support = ", ".join(f"{v:.2f}" for v in technicals.support_abs) or "none detected"
    resistance = ", ".join(f"{v:.2f}" for v in technicals.resistance_abs) or "none detected"
    return "\n".join(
        [
            "### Price & Technicals",
            f"*Source: {price.source}, as of {price.price_date.isoformat()}*",
            "",
            f"- Current Price: ₹{price.current_price_abs:.2f}",
            f"- 52-Week Range: ₹{price.week52_low_abs:.2f} – ₹{price.week52_high_abs:.2f}",
            f"- SMA50: {_fmt_or_missing(technicals.sma50, 'insufficient price history')}",
            f"- SMA200: {_fmt_or_missing(technicals.sma200, 'insufficient price history')}",
            f"- RSI(14): {_fmt_or_missing(technicals.rsi14, 'insufficient price history')}",
            f"- Support levels: {support}",
            f"- Resistance levels: {resistance}",
        ]
    )


def format_financials_section(financials: Financials | None) -> str:
    if financials is None:
        return (
            "### Financials\n\n"
            "MISSING: financials — could not fetch from Screener "
            "(consolidated or standalone)\n"
        )

    basis_label = (
        "CONSOLIDATED"
        if financials.basis == "consolidated"
        else "STANDALONE — consolidated unavailable"
    )
    lines = [
        "### Company Description",
        financials.business_description
        or "MISSING: business description — Screener had no About block for this company",
        "",
        f"### Financials — {basis_label}",
        (
            f"*Source: {financials.source}, fetched {financials.fetched_at.date().isoformat()}, "
            f"{financials.years_available} years available*"
        ),
        "",
        "#### Profit & Loss (₹ crore)",
        tabulate(financials.pnl, headers="keys", tablefmt="pipe"),
        "",
        "#### Balance Sheet (₹ crore)",
        tabulate(financials.balance_sheet, headers="keys", tablefmt="pipe"),
        "",
        "#### Cash Flow (₹ crore)",
        tabulate(financials.cash_flow, headers="keys", tablefmt="pipe"),
        "",
        "#### Ratios",
        tabulate(financials.ratios, headers="keys", tablefmt="pipe"),
        "",
        "#### Quarterly Results (₹ crore)",
        tabulate(financials.quarterly, headers="keys", tablefmt="pipe"),
    ]
    return "\n".join(lines)


def format_shareholding_section(shareholding: Shareholding | None) -> str:
    if shareholding is None:
        return (
            "### Shareholding\n\n"
            "MISSING: shareholding — could not fetch from NSE or Screener\n"
        )

    lines = [
        "### Shareholding",
        (
            f"*Source: {shareholding.source}, quarter: {shareholding.quarter or 'unknown'}, "
            f"fetched {shareholding.fetched_at.date().isoformat()}*"
        ),
        "",
        f"- Promoter holding: {_pct_or_missing(shareholding.promoter_pct)}",
        (
            f"- Promoter pledge (% of promoter holding): "
            f"{_pct_or_missing(shareholding.pledge_pct_of_promoter_holding, 'unconfirmed from an exchange source — not necessarily zero')}"
        ),
        f"- FII holding: {_pct_or_missing(shareholding.fii_pct)}",
        f"- DII holding: {_pct_or_missing(shareholding.dii_pct)}",
    ]
    return "\n".join(lines)


def cap_red_flags_per_query(red_flags: list[RedFlag], queries_run: list[str]) -> list[RedFlag]:
    selected: dict[tuple[str, str], RedFlag] = {}
    for query in queries_run:
        matching = [item for item in red_flags if query in item.found_by_query.split(", ")]
        matching.sort(key=lambda item: item.published_date, reverse=True)
        for item in matching[:RED_FLAGS_PER_QUERY_LIMIT]:
            selected[(item.headline, item.url)] = item
    return sorted(selected.values(), key=lambda item: item.published_date, reverse=True)


def _format_news_section(news: NewsItems | None) -> str:
    if news is None:
        return "### News\n\nMISSING: news — could not fetch from Google News RSS\n"

    lines = [
        "### News",
        f"*Source: {news.source}, fetched {news.fetched_at.date().isoformat()}*",
        "",
        "#### General (last 12 months)",
    ]
    if news.general:
        for item in news.general[:GENERAL_NEWS_DISPLAY_LIMIT]:
            lines.append(f"- {item.published_date.isoformat()} — {item.headline} ({item.url})")
    else:
        lines.append("- none found")

    lines.append("")
    lines.append("#### Red-flag search (adversarial disconfirmation)")
    lines.append(f"Queries run: {', '.join(news.queries_run)}")
    if news.queries_empty:
        lines.append(f"Queries with zero results: {', '.join(news.queries_empty)}")

    capped = cap_red_flags_per_query(news.red_flags, news.queries_run)
    if capped:
        if len(capped) < len(news.red_flags):
            lines.append(
                f"(showing {len(capped)} of {len(news.red_flags)} deduped results, "
                f"capped at {RED_FLAGS_PER_QUERY_LIMIT} most recent per query)"
            )
        for item in capped:
            lines.append(
                f"- [{item.found_by_query}] {item.published_date.isoformat()} — "
                f"{item.headline} ({item.url})"
            )
    else:
        lines.append("- none found")

    return "\n".join(lines)


def _format_annual_report_section(report: ReportText) -> str:
    if not report.sections:
        reason = (
            "not found on NSE"
            if report.source_url is None
            else "found but no usable text extracted "
            "(scanned/image-only, or no target headings matched)"
        )
        return f"### Annual Report\n\nMISSING: annual report — {reason}\n"

    lines = [
        "### Annual Report",
        (
            f"*Source: {report.source} ({report.source_url}), "
            f"report year: {report.report_year or 'unknown'}, "
            f"fetched {report.fetched_at.date().isoformat()}*"
        ),
    ]
    if report.truncated:
        lines.append(
            f"*Truncated to fit the token budget — "
            f"incomplete/dropped: {', '.join(report.dropped_sections)}*"
        )
    lines.append("")
    for heading, text in report.sections.items():
        lines.append(f"#### {heading}")
        lines.append(text)
        lines.append("")
    if report.business_summary and report.business_summary.order_book_cr is not None:
        lines.append(
            f"*Parsed order book (AR): ₹{report.business_summary.order_book_cr:.0f} cr "
            f"(rule-based, verify in filing)*"
        )
        lines.append("")

    return "\n".join(lines)


def to_markdown(brief: Brief) -> str:
    header = [
        f"# {brief.ticker.company_name} ({brief.ticker.symbol}, {brief.ticker.exchange})",
        (
            f"*Brief generated {brief.generated_at.date().isoformat()}, "
            f"confidence ceiling: {brief.confidence_ceiling}/10*"
        ),
        "",
    ]
    sections = [
        format_price_section(brief.price, brief.technicals),
        "",
        format_financials_section(brief.financials),
        "",
        format_shareholding_section(brief.shareholding),
        "",
        _format_news_section(brief.news),
        "",
        _format_annual_report_section(brief.annual_report),
    ]
    return "\n".join(header + sections)
