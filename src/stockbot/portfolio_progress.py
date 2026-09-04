"""Telegram portfolio-build progress — shortlist → analyze → hold toward 12–18.

Read-only snapshot over the unified product universe, prescan pick log, stored
analyses, SIP membership, and optional holdings.

Funnel symbols from /prescan, /pick, /analyze, and /hold are included even when
they are not yet on the watchlist ∪ SIP universe — otherwise /progress looked
empty while soft picks existed off-list.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as html_escape
from typing import Any, Literal

from stockbot.portfolio_screener.outcome_log import load_prescan_outcomes
from stockbot.portfolio_screener.pick_policy import (
    is_pick_eligible,
    pick_tier,
    query_pick_outcomes,
)
from stockbot.product_universe import (
    ProductUniverse,
    UniverseSymbol,
    format_universe_summary,
    load_product_universe,
)
from stockbot.storage import list_holdings, list_latest_analyses

Stage = Literal[
    "not_prescanned",
    "prescanned",
    "soft_pick",
    "analyzed",
    "buy_range",
    "held",
]

TARGET_PORTFOLIO_MIN = 12
TARGET_PORTFOLIO_MAX = 18


@dataclass(frozen=True)
class SymbolProgress:
    symbol: str
    in_sip: bool
    sip_bucket: str | None
    prescanned: bool
    soft_pick: bool
    pick_tier: str | None
    analyzed: bool
    verdict: str | None
    buy_range_issued: bool
    held: bool
    stage: Stage
    in_universe: bool = True


@dataclass(frozen=True)
class PortfolioProgressReport:
    universe_size: int
    rows: tuple[SymbolProgress, ...]
    target_min: int = TARGET_PORTFOLIO_MIN
    target_max: int = TARGET_PORTFOLIO_MAX

    @property
    def buy_range_count(self) -> int:
        return sum(1 for r in self.rows if r.buy_range_issued)

    @property
    def held_count(self) -> int:
        return sum(1 for r in self.rows if r.held)

    @property
    def soft_pick_count(self) -> int:
        return sum(1 for r in self.rows if r.soft_pick)

    @property
    def analyzed_count(self) -> int:
        return sum(1 for r in self.rows if r.analyzed)

    @property
    def off_universe_count(self) -> int:
        return sum(1 for r in self.rows if not r.in_universe)


def _buy_range_issued(verdict_json: dict[str, Any]) -> bool:
    if verdict_json.get("buy_range_allowed") is not True:
        return False
    zone = verdict_json.get("buy_zone_abs")
    return isinstance(zone, (list, tuple)) and len(zone) >= 2


def _stage_for(row: SymbolProgress) -> Stage:
    if row.held:
        return "held"
    if row.buy_range_issued:
        return "buy_range"
    if row.analyzed:
        return "analyzed"
    if row.soft_pick:
        return "soft_pick"
    if row.prescanned:
        return "prescanned"
    return "not_prescanned"


def _row_for_symbol(
    symbol: str,
    *,
    uni_item: UniverseSymbol | None,
    prescanned_row: dict[str, Any] | None,
    pick_set: set[str],
    analyzed_v: dict[str, Any] | None,
    held: set[str],
) -> SymbolProgress:
    soft = symbol in pick_set
    tier: str | None = None
    if prescanned_row is not None and is_pick_eligible(prescanned_row):
        soft = True
        tier = pick_tier(prescanned_row)
    elif soft and prescanned_row is not None:
        tier = pick_tier(prescanned_row)

    draft = SymbolProgress(
        symbol=symbol,
        in_sip=bool(uni_item is not None and "sip" in uni_item.sources),
        sip_bucket=uni_item.sip_bucket_label if uni_item is not None else None,
        prescanned=prescanned_row is not None,
        soft_pick=soft,
        pick_tier=tier,
        analyzed=analyzed_v is not None,
        verdict=(
            str(analyzed_v.get("verdict"))
            if isinstance(analyzed_v, dict) and analyzed_v.get("verdict")
            else None
        ),
        buy_range_issued=(
            _buy_range_issued(analyzed_v) if isinstance(analyzed_v, dict) else False
        ),
        held=symbol in held,
        stage="not_prescanned",
        in_universe=uni_item is not None,
    )
    return SymbolProgress(
        symbol=draft.symbol,
        in_sip=draft.in_sip,
        sip_bucket=draft.sip_bucket,
        prescanned=draft.prescanned,
        soft_pick=draft.soft_pick,
        pick_tier=draft.pick_tier,
        analyzed=draft.analyzed,
        verdict=draft.verdict,
        buy_range_issued=draft.buy_range_issued,
        held=draft.held,
        stage=_stage_for(draft),
        in_universe=draft.in_universe,
    )


def build_portfolio_progress(
    chat_id: int | None = None,
    *,
    universe: ProductUniverse | None = None,
) -> PortfolioProgressReport:
    uni = universe or load_product_universe()
    uni_by_symbol = {item.symbol: item for item in uni.symbols}

    prescan_rows = load_prescan_outcomes()
    by_ticker = {
        str(r.get("ticker") or "").strip().upper(): r
        for r in prescan_rows
        if str(r.get("ticker") or "").strip()
    }
    pick_rows = query_pick_outcomes(prescan_rows)
    pick_set = {
        str(r.get("ticker") or "").strip().upper()
        for r in pick_rows
        if str(r.get("ticker") or "").strip()
    }
    analyses = {
        ticker.upper(): verdict
        for ticker, verdict, _ts in list_latest_analyses()
    }
    held: set[str] = set()
    if chat_id is not None:
        held = {h.ticker.upper() for h in list_holdings(chat_id)}

    # Universe first (stable order), then funnel-only symbols (prescan / pick /
    # analyze / hold) that are not yet on the watchlist ∪ SIP list.
    ordered: list[str] = [item.symbol for item in uni.symbols]
    seen = set(ordered)
    extras = sorted(
        (pick_set | set(by_ticker) | set(analyses) | held) - seen
    )
    ordered.extend(extras)

    rows: list[SymbolProgress] = []
    for symbol in ordered:
        rows.append(
            _row_for_symbol(
                symbol,
                uni_item=uni_by_symbol.get(symbol),
                prescanned_row=by_ticker.get(symbol),
                pick_set=pick_set,
                analyzed_v=analyses.get(symbol),
                held=held,
            )
        )

    return PortfolioProgressReport(universe_size=len(uni.symbols), rows=tuple(rows))


def select_daily_tips(
    *,
    limit: int = 2,
    universe: ProductUniverse | None = None,
    pick_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Curate ≤``limit`` tip names: analyze_now first, then if_interested.

    Prefers unified-universe symbols; falls back to any soft pick if needed.
    """
    uni = universe or load_product_universe()
    uni_set = set(uni.tickers)
    picks = pick_rows if pick_rows is not None else query_pick_outcomes(load_prescan_outcomes())
    in_uni = [r for r in picks if str(r.get("ticker") or "").upper() in uni_set]
    pool = in_uni or picks
    now = [r for r in pool if pick_tier(r) == "analyze_now"]
    later = [r for r in pool if pick_tier(r) == "analyze_if_interested"]
    ordered = now + later
    return ordered[: max(1, limit)]


def format_daily_tips_html(rows: list[dict[str, Any]], *, limit: int = 2) -> str:
    """Beginner-friendly HTML for today's 1–2 tips."""
    tips = rows[: max(1, limit)]
    lines = [
        "<b>🎯 Today’s tips (max 2)</b>",
        "These are the next names to deep-dive — not automatic buys.",
        "",
    ]
    if not tips:
        lines.extend(
            [
                "📭 No soft picks ready yet.",
                (
                    "💡 Run <code>/prescan SYMBOL</code> or <code>/sip prescan</code>, "
                    "then retry <code>/pick daily</code>."
                ),
            ]
        )
        return "\n".join(lines)

    for i, row in enumerate(tips, start=1):
        ticker = html_escape(str(row.get("ticker") or "?"))
        tier = pick_tier(row)
        quant = row.get("quant_score")
        score = f"{float(quant):.0f}" if isinstance(quant, (int, float)) else "?"
        if tier == "analyze_now":
            cue = "✅ Run /analyze now"
        else:
            cue = "🔎 Worth /analyze if interested"
        lines.append(f"{i}. <b>{ticker}</b> — quant {score} · {cue}")
        lines.append(f"   Next: <code>/analyze lite {ticker}</code> (cheaper) or <code>/analyze {ticker}</code>")

    lines.extend(
        [
            "",
            "Buy only when <code>/analyze</code> issues a buy range and price fits.",
            "Full soft list: <code>/pick</code> · Progress: <code>/progress</code>",
        ]
    )
    return "\n".join(lines)


def format_portfolio_progress_html(
    report: PortfolioProgressReport,
    *,
    universe: ProductUniverse | None = None,
) -> str:
    uni = universe or load_product_universe()
    lines = [
        "<b>🏗 Portfolio build progress</b>",
        format_universe_summary(uni),
        (
            f"Toward {report.target_min}–{report.target_max} quality holdings: "
            f"<b>{report.held_count}</b> held · "
            f"<b>{report.buy_range_count}</b> with buy range · "
            f"<b>{report.analyzed_count}</b> analyzed · "
            f"<b>{report.soft_pick_count}</b> soft picks"
        ),
    ]
    if report.off_universe_count:
        lines.append(
            f"📌 <b>{report.off_universe_count}</b> funnel name(s) not yet on "
            "watchlist/SIP — still counted below."
        )
    lines.append("")

    by_stage: dict[Stage, list[SymbolProgress]] = {
        "buy_range": [],
        "held": [],
        "analyzed": [],
        "soft_pick": [],
        "prescanned": [],
        "not_prescanned": [],
    }
    for row in report.rows:
        by_stage[row.stage].append(row)

    def _show(title: str, items: list[SymbolProgress], *, limit: int = 12) -> None:
        if not items:
            return
        lines.append(f"<b>{title}</b> ({len(items)})")
        for row in items[:limit]:
            bits = [html_escape(row.symbol)]
            if row.verdict:
                bits.append(html_escape(row.verdict))
            if row.in_sip:
                bits.append("SIP")
            if row.held:
                bits.append("held")
            if not row.in_universe:
                bits.append("off-list")
            lines.append("· " + " · ".join(bits))
        if len(items) > limit:
            lines.append(f"· … +{len(items) - limit} more")
        lines.append("")

    _show("✅ Buy range ready (count toward book)", by_stage["buy_range"])
    _show("📁 Already held", by_stage["held"])
    _show("🧠 Analyzed — no buy range yet", by_stage["analyzed"])
    _show("🎯 Soft pick — needs /analyze", by_stage["soft_pick"])

    # Off-list prescans that are not soft picks yet — surface briefly.
    off_prescan = [r for r in by_stage["prescanned"] if not r.in_universe]
    _show("📋 Prescanned (off-list, not soft pick yet)", off_prescan, limit=8)

    not_ready = sum(1 for r in by_stage["prescanned"] if r.in_universe) + len(
        by_stage["not_prescanned"]
    )
    lines.append(
        f"⏳ Universe not ready yet: {not_ready} (prescan or wait). "
        f"Batch: <code>/sip prescan</code>"
    )
    lines.append("")
    if report.held_count < report.target_min:
        need = report.target_min - report.held_count
        lines.append(
            f"💡 Need ~{need} more holdings for the {report.target_min}–{report.target_max} book. "
            f"Try <code>/pick daily</code> → <code>/analyze</code> → <code>/hold</code>."
        )
    elif report.held_count <= report.target_max:
        lines.append(
            f"✅ Holding count is inside the {report.target_min}–{report.target_max} target band."
        )
    else:
        lines.append(
            f"⚠️ Held {report.held_count} names — above the {report.target_max} target; "
            "prefer adds only on high-conviction dips."
        )
    return "\n".join(lines)
