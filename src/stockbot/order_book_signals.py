"""Structured order-book / backlog signals for Stage 2 context injection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from stockbot.models import Brief, NewsItems

_ORDER_BOOK_HEADLINE_RE = re.compile(
    r"order\s*book|orderbook|backlog",
    re.IGNORECASE,
)
_ORDER_BOOK_SECTION_RE = re.compile(
    r"order\s*book|orderbook|backlog|unexecuted\s+order",
    re.IGNORECASE,
)
_AMOUNT_CR_RE = re.compile(
    r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(crore|cr\.?|lakh|lac|bn|billion)",
    re.IGNORECASE,
)
_SNIPPET_RADIUS = 220


@dataclass(frozen=True)
class OrderBookSignal:
    source: str  # "news" | "annual_report"
    text: str
    amount_cr: float | None = None
    as_of: date | None = None


def _parse_amount_cr(text: str) -> float | None:
    match = _AMOUNT_CR_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = match.group(2).lower()
    if unit.startswith(("lakh", "lac")):
        return value / 100.0
    if unit.startswith(("bn", "billion")):
        return value * 100.0
    return value


def _snippet_around_match(text: str, match: re.Match[str]) -> str:
    start = max(0, match.start() - _SNIPPET_RADIUS)
    end = min(len(text), match.end() + _SNIPPET_RADIUS)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def extract_order_book_news_signals(
    news: NewsItems | None,
    *,
    limit: int = 5,
) -> list[OrderBookSignal]:
    if news is None:
        return []
    signals: list[OrderBookSignal] = []
    seen: set[str] = set()
    for item in news.general + news.red_flags:
        if not _ORDER_BOOK_HEADLINE_RE.search(item.headline):
            continue
        key = item.headline.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        signals.append(
            OrderBookSignal(
                source="news",
                text=item.headline.strip(),
                amount_cr=_parse_amount_cr(item.headline),
                as_of=item.published_date,
            )
        )
        if len(signals) >= limit:
            break
    return signals


def extract_order_book_annual_report_signals(
    brief: Brief,
    *,
    limit: int = 3,
) -> list[OrderBookSignal]:
    sections = brief.annual_report.sections
    if not sections:
        return []
    signals: list[OrderBookSignal] = []
    seen: set[str] = set()
    for heading, body in sections.items():
        for match in _ORDER_BOOK_SECTION_RE.finditer(body):
            snippet = _snippet_around_match(body, match)
            key = snippet.lower()[:120]
            if key in seen:
                continue
            seen.add(key)
            signals.append(
                OrderBookSignal(
                    source="annual_report",
                    text=f"[{heading}] {snippet}",
                    amount_cr=_parse_amount_cr(snippet),
                    as_of=None,
                )
            )
            if len(signals) >= limit:
                return signals
    return signals


def collect_order_book_signals(brief: Brief) -> list[OrderBookSignal]:
    """News headlines first, then annual-report snippets (primary filings)."""
    news_signals = extract_order_book_news_signals(brief.news)
    report_signals = extract_order_book_annual_report_signals(brief)
    return news_signals + report_signals


def format_order_book_signals_for_stage2(signals: list[OrderBookSignal]) -> list[str]:
    lines: list[str] = []
    for signal in signals:
        prefix = "UNVERIFIED NEWS" if signal.source == "news" else "ANNUAL REPORT"
        date_part = f" ({signal.as_of.isoformat()})" if signal.as_of else ""
        amount_part = (
            f" [parsed ~₹{signal.amount_cr:,.0f} cr]" if signal.amount_cr is not None else ""
        )
        lines.append(f"[{prefix}{date_part}]{amount_part} {signal.text}")
    return lines


def order_book_wc_billing_hint(brief: Brief, signals: list[OrderBookSignal]) -> str | None:
    """Suggest TEMPORARY_BILLING_CYCLE review when backlog dwarfs revenue."""
    amounts = [s.amount_cr for s in signals if s.amount_cr is not None]
    if not amounts or brief.financials is None:
        return None
    pnl = brief.financials.pnl
    if "Sales" not in pnl.index and "Revenue" not in pnl.index:
        return None
    label = "Sales" if "Sales" in pnl.index else "Revenue"
    row = pnl.loc[label]
    raw = row["TTM"] if "TTM" in pnl.columns else row.iloc[-1]
    try:
        revenue_cr = float(raw)
    except (TypeError, ValueError):
        return None
    if revenue_cr <= 0:
        return None
    largest = max(amounts)
    ratio = largest / revenue_cr
    if ratio < 2.0:
        return None
    return (
        f"ORDER-BOOK NOTE: largest cited backlog/order book (~₹{largest:,.0f} cr) is "
        f"{ratio:.1f}× TTM revenue (₹{revenue_cr:,.0f} cr). If reported cash conversion "
        f"is weak, evaluate wc_gap_classification=TEMPORARY_BILLING_CYCLE before "
        f"blocking on OCF/PAT alone."
    )


# Back-compat for tests and thin imports.
def extract_order_book_news_claims(news: NewsItems | None, *, limit: int = 5) -> list[str]:
    return format_order_book_signals_for_stage2(extract_order_book_news_signals(news, limit=limit))
