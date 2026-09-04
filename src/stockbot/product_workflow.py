"""Product workflows — daily tips vs 3y portfolio build.

Actionable Telegram playbooks that tie existing commands together. Read-only:
no auto-trading, no config mutation.
"""

from __future__ import annotations

from html import escape as html_escape
from typing import Any

from stockbot.portfolio_progress import format_daily_tips_html, select_daily_tips
from stockbot.portfolio_screener.outcome_log import load_prescan_outcomes
from stockbot.portfolio_screener.pick_policy import (
    pick_tier,
    query_pick_outcomes,
)
from stockbot.product_universe import format_universe_summary, load_product_universe


def _pick_snapshot() -> dict[str, Any]:
    uni = load_product_universe()
    uni_set = set(uni.tickers)
    rows = load_prescan_outcomes()
    picks = query_pick_outcomes(rows)
    analyze_now = [r for r in picks if pick_tier(r) == "analyze_now"]
    if_interested = [r for r in picks if pick_tier(r) == "analyze_if_interested"]
    off_list = [
        r
        for r in picks
        if str(r.get("ticker") or "").strip().upper() not in uni_set
    ]
    return {
        "total_logged": len(rows),
        "pick_count": len(picks),
        "off_list_count": len(off_list),
        "analyze_now": analyze_now[:3],
        "if_interested": if_interested[:3],
        "universe_tickers": uni_set,
    }


def _format_pick_lines(snapshot: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    uni_set: set[str] = snapshot.get("universe_tickers") or set()
    if snapshot["total_logged"] == 0:
        lines.append(
            "No prescan history yet — run <code>/prescan SYMBOL</code> "
            "(watchlist or any NSE name you care about)."
        )
        return lines
    lines.append(
        f"From {snapshot['pick_count']} soft pick(s) in {snapshot['total_logged']} logged name(s):"
    )
    if snapshot["off_list_count"]:
        lines.append(
            f"📌 {snapshot['off_list_count']} soft pick(s) are off watchlist/SIP — "
            "still tippable; add to watchlist when you want them in the 12–18 book."
        )

    def _tickers(rows: list[dict[str, Any]]) -> str:
        bits: list[str] = []
        for r in rows:
            raw = str(r.get("ticker") or "?").strip().upper()
            label = html_escape(raw or "?")
            if raw and raw not in uni_set:
                label = f"{label} (off-list)"
            bits.append(label)
        return ", ".join(bits)

    if snapshot["analyze_now"]:
        lines.append(f"• Run /analyze first: {_tickers(snapshot['analyze_now'])}")
    if snapshot["if_interested"]:
        lines.append(
            f"• Worth /analyze if interested: {_tickers(snapshot['if_interested'])}"
        )
    if not snapshot["analyze_now"] and not snapshot["if_interested"]:
        lines.append(
            "• No names pass <code>/pick</code> right now — "
            "prescan more names (on or off the watchlist)."
        )
    return lines


def format_daily_workflow() -> str:
    """1–2 daily tip workflow — fast funnel, minimal over-filtering."""
    uni = load_product_universe()
    snap = _pick_snapshot()
    tips = select_daily_tips(limit=2, universe=uni)
    lines = [
        "<b>📅 Daily tip workflow (1–2 names)</b>",
        "Goal: one actionable buy/add idea per day without over-filtering.",
        format_universe_summary(uni),
        "",
        "<b>Step 1 — Refresh the list (weekly or when stale)</b>",
        "<code>/prescan SYMBOL</code> on watchlist names <b>or any NSE name</b>, or",
        "<code>/sip prescan</code> for the full SIP + watchlist batch (quant-only).",
        "Off-list soft picks still appear in tips and <code>/progress</code>.",
        "",
        "<b>Step 2 — Today’s curated tips</b>",
        "<code>/pick daily</code> — max 2 names (analyze_now first).",
        "Full soft list anytime: <code>/pick</code>. MONITOR is not a sell.",
        "",
        format_daily_tips_html(tips, limit=2, universe=uni),
        "",
        "<b>Step 3 — Deep dive on those 1–2 names only</b>",
        "<code>/analyze lite SYMBOL</code> — cheaper/faster Stage 2 for daily tips.",
        "Use plain <code>/analyze SYMBOL</code> when you want the full Sonnet report.",
        "Pick only when buy range issued + base 3y CAGR OK.",
        "After a few analyses: <code>/rank</code> — order by expected long-term return.",
        "Send <code>/stop</code> to cancel a long analysis.",
        "",
        "<b>Step 4 — Execute & record</b>",
        "<code>/hold SYMBOL qty avg_price</code> after you buy.",
        "Track the 12–18 book: <code>/progress</code>.",
        "",
        "<b>Do not</b>",
        "• Treat MONITOR as sell for holdings",
        "• Require score≥65 for every tip (/pick is enough to shortlist)",
        "• Skip /analyze because /candidates filtered a name out",
        "• Ignore off-list soft picks — they are part of the same funnel",
        "",
        "<i>Review: <code>/track analyze</code> monthly — did BUY calls work?</i>",
        "",
        "<i>Soft-pick snapshot</i>",
    ]
    lines.extend(_format_pick_lines(snap))
    return "\n".join(lines)


def format_portfolio_workflow() -> str:
    """12–18 name 3y portfolio build — slower, sector-aware funnel."""
    uni = load_product_universe()
    snap = _pick_snapshot()
    lines = [
        "<b>🏗 Portfolio build workflow (12–18 names, 3y horizon)</b>",
        "Goal: quality portfolio with sector caps, DCA tranches, and analyze-backed ranges.",
        format_universe_summary(uni),
        "",
        "<b>Step 1 — One funnel (universe + off-list research)</b>",
        (
            "Watchlist + SIP names share the same funnel "
            "(<code>data/portfolio/watchlist.txt</code> ∪ "
            "<code>sip_portfolios.json</code>)."
        ),
        (
            "You can still <code>/prescan</code> / <code>/analyze</code> names "
            "not on that list — they show in <code>/pick</code>, "
            "<code>/progress</code>, and tips as off-list."
        ),
        "SIP buckets still drive monthly DCA amounts via <code>/sip plan</code>.",
        "",
        "<b>Step 2 — Batch prescan (quant-only first)</b>",
        "<code>/sip prescan</code> — writes history for SIP symbols.",
        "<code>/prescan SYMBOL</code> — any NSE name you want researched.",
        "",
        "<b>Step 3 — Shortlist without over-filtering</b>",
        "<code>/pick daily</code> for 1–2 tips, or <code>/pick</code> for the full soft list.",
        "Target 12–18 survivors — check <code>/progress</code>.",
        "",
    ]
    lines.extend(_format_pick_lines(snap))
    lines.extend(
        [
            "",
            "<b>Step 4 — Deep analyze survivors</b>",
            "<code>/analyze SYMBOL</code> on each shortlisted name.",
            "Reject for <b>new</b> capital only — not automatic sell if already held.",
            "Then <code>/rank</code> (or <code>/rank entry</code>) to order by expected base CAGR.",
            "",
            "<b>Step 5 — Size & sector limits</b>",
            "<code>/capital TOTAL max N sector 25</code> — capital, per-stock cap, sector cap.",
            "<code>/hold</code> shows per-name and sector concentration breaches.",
            "",
            "<b>Step 6 — DCA execution</b>",
            "<code>/sip plan</code> — bucket tables with live prices.",
            "<code>/sip track</code> — planned vs logged this month.",
            "Use 4 tranches / 70-20-10 DCA from your SIP plan — bot surfaces ranges, you execute.",
            "",
            "<b>Step 7 — Tune the pick floor (monthly)</b>",
            "<code>/track pick</code> — did soft picks beat rejects?",
            "<code>/track pick tune</code> — threshold suggestions from your history.",
            "",
            "<i>Progress: <code>/progress</code> · Holdings: <code>/hold</code></i>",
        ]
    )
    return "\n".join(lines)
