"""Pre-LLM data readiness — verify sources, run fallbacks, block token spend.

Institution-style rule: never bill Stage 1/2 until the free data layer has
exhausted its documented fallback chain. Each field records primary →
secondary → tertiary sources attempted.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Literal

from stockbot.brief import assemble_brief, enrich_brief, to_markdown
from stockbot.fetch.annual_report import BUSINESS_HEADING_PRIORITY
from stockbot.fetch.fundamentals import fetch_fundamentals
from stockbot.fetch.news import fetch_news
from stockbot.fetch.shareholding import fetch_shareholding
from stockbot.models import Brief, Financials, NewsItems, Shareholding, TickerInfo
from stockbot.trade_policy import business_context_blocks_preflight

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 4
MIN_FINANCIAL_YEARS = 3
PREFERRED_FINANCIAL_YEARS = 5

FieldState = Literal["ok", "degraded", "missing"]

# Documented free-data fallback chains (primary → secondary → tertiary).
FALLBACK_CHAINS: dict[str, tuple[str, ...]] = {
    "price": ("yfinance .NS", "yfinance .BO"),
    "financials": (
        "screener.in consolidated",
        "screener.in standalone",
        "yfinance .NS statements",
        "yfinance .BO statements",
    ),
    "business_description": (
        "screener About block",
        "annual report MD&A excerpt",
        "yfinance longBusinessSummary",
    ),
    "shareholding": ("NSE shareholding API + XBRL pledge", "screener FII/DII merge"),
    "annual_report": ("NSE annual-reports PDF extract",),
    "news": ("Google News RSS company name", "Google News RSS symbol+NSE"),
}


@dataclass(frozen=True)
class FallbackAttempt:
    field: str
    tier: int
    source: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class FieldStatus:
    name: str
    state: FieldState
    source: str | None
    chain: tuple[str, ...]
    attempts: tuple[FallbackAttempt, ...]
    note: str | None = None


@dataclass(frozen=True)
class DataReadinessReport:
    symbol: str
    ready_for_llm: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    fields: tuple[FieldStatus, ...]
    confidence_ceiling: int

    def telegram_summary(self) -> str:
        lines = [
            f"<b>Data preflight — {self.symbol}</b>",
            f"Ready for paid analysis: <b>{'YES' if self.ready_for_llm else 'NO'}</b>",
            "",
        ]
        if self.blockers:
            lines.append("<b>Blockers (no LLM spend):</b>")
            for item in self.blockers:
                lines.append(f"• {item}")
            lines.append("")
        if self.warnings:
            lines.append("<b>Warnings (analysis proceeds with caps):</b>")
            for item in self.warnings:
                lines.append(f"• {item}")
            lines.append("")
        lines.append("<b>Sources gathered:</b>")
        for field in self.fields:
            src = field.source or "—"
            flag = {"ok": "✓", "degraded": "~", "missing": "✗"}[field.state]
            lines.append(f"{flag} {field.name}: {src}")
            if field.note:
                lines.append(f"   <i>{field.note}</i>")
        lines.append("")
        lines.append(f"Confidence ceiling after fetch: {self.confidence_ceiling}/10")
        return "\n".join(lines)

    def markdown_summary(self) -> str:
        lines = [
            f"### Data readiness — {self.symbol}",
            f"- **Ready for LLM:** {'yes' if self.ready_for_llm else 'no'}",
            f"- **Confidence ceiling:** {self.confidence_ceiling}/10",
            "",
        ]
        if self.blockers:
            lines.append("**Blockers**")
            lines.extend(f"- {b}" for b in self.blockers)
            lines.append("")
        if self.warnings:
            lines.append("**Warnings**")
            lines.extend(f"- {w}" for w in self.warnings)
            lines.append("")
        lines.append("| Field | State | Source | Fallback chain |")
        lines.append("|-------|-------|--------|----------------|")
        for field in self.fields:
            chain = " → ".join(field.chain)
            lines.append(
                f"| {field.name} | {field.state} | {field.source or '—'} | {chain} |"
            )
        return "\n".join(lines)


def _yfinance_business_summary(symbol: str, exchange: str) -> str | None:
    try:
        import yfinance as yf

        suffix = ".NS" if exchange == "NSE" else ".BO"
        info = yf.Ticker(f"{symbol}{suffix}").info or {}
        text = info.get("longBusinessSummary") or info.get("description")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception as exc:  # noqa: BLE001 - best-effort tertiary fallback
        logger.warning("yfinance business summary failed for %s: %s", symbol, exc)
    return None


def _ar_business_excerpt(brief: Brief, *, max_chars: int = 900) -> str | None:
    ar = brief.annual_report
    parts: list[str] = []
    for heading in BUSINESS_HEADING_PRIORITY:
        text = ar.sections.get(heading)
        if text:
            parts.append(text[:500])
    if not parts:
        return None
    combined = "\n\n".join(parts)
    return combined[:max_chars].strip()


def _has_business_context(brief: Brief) -> bool:
    if brief.financials and brief.financials.business_description:
        return True
    if brief.annual_report.business_summary is not None:
        return True
    return any(h in brief.annual_report.sections for h in BUSINESS_HEADING_PRIORITY)


def _attempt(
    attempts: list[FallbackAttempt],
    *,
    field: str,
    tier: int,
    source: str,
    ok: bool,
    detail: str,
) -> None:
    attempts.append(
        FallbackAttempt(field=field, tier=tier, source=source, ok=ok, detail=detail)
    )


def apply_data_fallbacks(brief: Brief, ticker: TickerInfo) -> tuple[Brief, list[FallbackAttempt]]:
    """Run tier-2/3 fetches for gaps the primary parallel fetch left open."""
    attempts: list[FallbackAttempt] = []
    updated = brief
    financials = brief.financials
    shareholding = brief.shareholding
    news = brief.news

    # --- business description ---
    if financials is not None and not financials.business_description:
        excerpt = _ar_business_excerpt(brief)
        if excerpt:
            _attempt(
                attempts,
                field="business_description",
                tier=2,
                source="annual report MD&A excerpt",
                ok=True,
                detail=f"{len(excerpt)} chars",
            )
            financials = dataclasses.replace(
                financials,
                business_description=f"[AR excerpt] {excerpt}",
            )
        else:
            _attempt(
                attempts,
                field="business_description",
                tier=2,
                source="annual report MD&A excerpt",
                ok=False,
                detail="no business headings in AR sections",
            )
            yf_text = _yfinance_business_summary(ticker.symbol, ticker.exchange)
            if yf_text:
                _attempt(
                    attempts,
                    field="business_description",
                    tier=3,
                    source="yfinance longBusinessSummary",
                    ok=True,
                    detail=f"{len(yf_text)} chars",
                )
                financials = dataclasses.replace(financials, business_description=yf_text)
            else:
                _attempt(
                    attempts,
                    field="business_description",
                    tier=3,
                    source="yfinance longBusinessSummary",
                    ok=False,
                    detail="empty or unavailable",
                )

    # --- shareholding retry ---
    if shareholding is None:
        try:
            shareholding = fetch_shareholding(ticker.symbol, exchange=ticker.exchange)
            _attempt(
                attempts,
                field="shareholding",
                tier=2,
                source="NSE shareholding API (retry)",
                ok=shareholding is not None,
                detail="ok" if shareholding else "still empty",
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            _attempt(
                attempts,
                field="shareholding",
                tier=2,
                source="NSE shareholding API (retry)",
                ok=False,
                detail=str(exc)[:120],
            )

    # --- news retry with symbol-qualified query ---
    if news is None:
        try:
            news = fetch_news(f"{ticker.company_name} {ticker.symbol} NSE")
            _attempt(
                attempts,
                field="news",
                tier=2,
                source="Google News RSS symbol+NSE",
                ok=news is not None,
                detail="ok" if news else "empty",
            )
        except Exception as exc:  # noqa: BLE001
            _attempt(
                attempts,
                field="news",
                tier=2,
                source="Google News RSS symbol+NSE",
                ok=False,
                detail=str(exc)[:120],
            )

    # --- financials explicit retry (tertiary yfinance if primary assemble missed) ---
    if financials is None:
        try:
            financials = fetch_fundamentals(ticker.symbol)
            _attempt(
                attempts,
                field="financials",
                tier=2,
                source="fetch_fundamentals retry",
                ok=True,
                detail=f"{financials.years_available} years via {financials.source}",
            )
        except Exception as exc:  # noqa: BLE001
            _attempt(
                attempts,
                field="financials",
                tier=2,
                source="fetch_fundamentals retry",
                ok=False,
                detail=str(exc)[:160],
            )

    missing = _rebuild_missing(updated, financials, shareholding, news)
    confidence_ceiling = _confidence_ceiling(financials, updated.annual_report.sections)

    updated = dataclasses.replace(
        updated,
        financials=financials,
        shareholding=shareholding,
        news=news,
        missing=missing,
        confidence_ceiling=confidence_ceiling,
    )
    updated = enrich_brief(updated)
    token_count = len(to_markdown(updated)) // CHARS_PER_TOKEN_ESTIMATE
    return dataclasses.replace(updated, token_count=token_count), attempts


def _rebuild_missing(
    brief: Brief,
    financials: Financials | None,
    shareholding: Shareholding | None,
    news: NewsItems | None,
) -> list[str]:
    """Refresh brief.missing after fallback passes — keep only still-open gaps."""
    missing: list[str] = []
    if financials is None:
        missing.append("MISSING: financials — all Screener/yfinance fallbacks failed")
    if shareholding is None:
        missing.append("MISSING: shareholding — NSE/Screener fallbacks failed")
    if news is None:
        missing.append("MISSING: news — RSS fetch failed")
    ar = brief.annual_report
    if not ar.sections:
        reason = (
            "not found on NSE"
            if ar.source_url is None
            else "found but no usable text extracted"
        )
        missing.append(f"MISSING: annual report — {reason}")
    elif financials is not None and not financials.business_description:
        from stockbot.fetch.annual_report import business_narrative_gap

        gap = business_narrative_gap(ar.sections, ar.dropped_sections)
        if gap:
            missing.append(gap)
    return missing


def _confidence_ceiling(financials: Financials | None, ar_sections: dict[str, str]) -> int:
    ceiling = 10
    if financials is None:
        ceiling = min(ceiling, 4)
    if not ar_sections:
        ceiling = min(ceiling, 5)
    return ceiling


def assess_data_readiness(
    brief: Brief,
    *,
    attempts: list[FallbackAttempt] | None = None,
) -> DataReadinessReport:
    attempts = attempts or []
    blockers: list[str] = []
    warnings: list[str] = []
    fields: list[FieldStatus] = []

    # Price — fatal if assemble_brief returned; always ok here.
    fields.append(
        FieldStatus(
            name="price",
            state="ok",
            source=brief.price.source,
            chain=FALLBACK_CHAINS["price"],
            attempts=tuple(a for a in attempts if a.field == "price"),
        )
    )

    # Financials
    fin = brief.financials
    if fin is None:
        blockers.append(
            "Financial statements missing after screener.in (consolidated → standalone) "
            "and yfinance (.NS → .BO) fallbacks."
        )
        fields.append(
            FieldStatus(
                name="financials",
                state="missing",
                source=None,
                chain=FALLBACK_CHAINS["financials"],
                attempts=tuple(a for a in attempts if a.field == "financials"),
            )
        )
    elif fin.years_available < MIN_FINANCIAL_YEARS:
        blockers.append(
            f"Only {fin.years_available} fiscal years available — need ≥{MIN_FINANCIAL_YEARS} "
            f"for institution-grade analysis."
        )
        fields.append(
            FieldStatus(
                name="financials",
                state="degraded",
                source=fin.source,
                chain=FALLBACK_CHAINS["financials"],
                attempts=tuple(a for a in attempts if a.field == "financials"),
                note=f"{fin.years_available} years",
            )
        )
    else:
        note = None
        if fin.years_available < PREFERRED_FINANCIAL_YEARS:
            warnings.append(
                f"Financial history is {fin.years_available} years "
                f"(prefer ≥{PREFERRED_FINANCIAL_YEARS})."
            )
            note = f"{fin.years_available} years"
        fields.append(
            FieldStatus(
                name="financials",
                state="ok",
                source=fin.source,
                chain=FALLBACK_CHAINS["financials"],
                attempts=tuple(a for a in attempts if a.field == "financials"),
                note=note,
            )
        )

    # Business context
    if _has_business_context(brief):
        src = "screener About"
        if fin and fin.business_description and fin.business_description.startswith("[AR excerpt]"):
            src = "annual report MD&A excerpt"
        elif fin and fin.business_description:
            if "yfinance" in (fin.source or ""):
                src = "yfinance summary"
            else:
                src = "screener About"
        elif brief.annual_report.business_summary:
            src = "AR business_summary parse"
        fields.append(
            FieldStatus(
                name="business_description",
                state="ok",
                source=src,
                chain=FALLBACK_CHAINS["business_description"],
                attempts=tuple(a for a in attempts if a.field == "business_description"),
            )
        )
    else:
        years = fin.years_available if fin is not None else None
        if business_context_blocks_preflight(financial_years=years):
            blockers.append(
                "No business description after screener About → AR MD&A → yfinance fallbacks, "
                f"and financial history is below {PREFERRED_FINANCIAL_YEARS} years."
            )
        else:
            warnings.append(
                "Business description still thin — analysis must cite FINANCIALS only in §2."
            )
        fields.append(
            FieldStatus(
                name="business_description",
                state="missing",
                source=None,
                chain=FALLBACK_CHAINS["business_description"],
                attempts=tuple(a for a in attempts if a.field == "business_description"),
            )
        )

    # Annual report
    ar = brief.annual_report
    if not ar.sections:
        blockers.append(
            "Annual report governance text missing — NSE PDF not found or image-only/scanned."
        )
        fields.append(
            FieldStatus(
                name="annual_report",
                state="missing",
                source=None,
                chain=FALLBACK_CHAINS["annual_report"],
                attempts=tuple(a for a in attempts if a.field == "annual_report"),
            )
        )
    else:
        note = None
        if ar.truncated or ar.dropped_sections:
            warnings.append(
                f"Annual report truncated — dropped sections: {ar.dropped_sections[:4]}"
            )
            note = "truncated"
        fields.append(
            FieldStatus(
                name="annual_report",
                state="degraded" if ar.truncated else "ok",
                source=ar.source,
                chain=FALLBACK_CHAINS["annual_report"],
                attempts=tuple(a for a in attempts if a.field == "annual_report"),
                note=note,
            )
        )

    # Shareholding
    sh = brief.shareholding
    if sh is None:
        warnings.append(
            "Shareholding/pledge unconfirmed — promoter % missing after NSE/Screener fallbacks."
        )
        fields.append(
            FieldStatus(
                name="shareholding",
                state="missing",
                source=None,
                chain=FALLBACK_CHAINS["shareholding"],
                attempts=tuple(a for a in attempts if a.field == "shareholding"),
            )
        )
    else:
        note = None
        if sh.pledge_pct_of_promoter_holding is None:
            warnings.append("Promoter pledge % unconfirmed — do not state a pledge figure.")
            note = "pledge unconfirmed"
        fields.append(
            FieldStatus(
                name="shareholding",
                state="degraded" if note else "ok",
                source=sh.source,
                chain=FALLBACK_CHAINS["shareholding"],
                attempts=tuple(a for a in attempts if a.field == "shareholding"),
                note=note,
            )
        )

    # News
    if brief.news is None:
        warnings.append("News/RSS unavailable — red-flag scan degraded.")
        fields.append(
            FieldStatus(
                name="news",
                state="missing",
                source=None,
                chain=FALLBACK_CHAINS["news"],
                attempts=tuple(a for a in attempts if a.field == "news"),
            )
        )
    else:
        fields.append(
            FieldStatus(
                name="news",
                state="ok",
                source=brief.news.source,
                chain=FALLBACK_CHAINS["news"],
                attempts=tuple(a for a in attempts if a.field == "news"),
            )
        )

    return DataReadinessReport(
        symbol=brief.ticker.symbol,
        ready_for_llm=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        fields=tuple(fields),
        confidence_ceiling=brief.confidence_ceiling,
    )


def assemble_brief_for_analysis(ticker: TickerInfo) -> tuple[Brief, DataReadinessReport]:
    """Fetch → fallback → assess. Call before any paid LLM stage."""
    brief = assemble_brief(ticker)
    brief, attempts = apply_data_fallbacks(brief, ticker)
    report = assess_data_readiness(brief, attempts=attempts)
    return brief, report
