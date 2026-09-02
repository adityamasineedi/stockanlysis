"""Product workflows — daily tips vs 3y portfolio build.

Actionable Telegram playbooks that tie existing commands together. Read-only:
no auto-trading, no config mutation.
"""

from __future__ import annotations

from html import escape as html_escape
from typing import Any

from stockbot.portfolio_screener.outcome_log import load_prescan_outcomes
from stockbot.portfolio_screener.pick_policy import (
    pick_tier,
    query_pick_outcomes,
)


def _pick_snapshot() -> dict[str, Any]:
    rows = load_prescan_outcomes()
    picks = query_pick_outcomes(rows)
    analyze_now = [r for r in picks if pick_tier(r) == "analyze_now"]
    if_interested = [r for r in picks if pick_tier(r) == "analyze_if_interested"]
    return {
        "total_logged": len(rows),
        "pick_count": len(picks),
        "analyze_now": analyze_now[:3],
        "if_interested": if_interested[:3],
    }


def _format_pick_lines(snapshot: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if snapshot["total_logged"] == 0:
        lines.append(
            "No prescan history yet — run <code>/prescan SYMBOL</code> on your watchlist first."
        )
        return lines
    lines.append(
        f"From {snapshot['pick_count']} soft pick(s) in {snapshot['total_logged']} logged name(s):"
    )
    if snapshot["analyze_now"]:
        tickers = ", ".join(
            html_escape(str(r.get("ticker") or "?")) for r in snapshot["analyze_now"]
        )
        lines.append(f"• Run /analyze first: {tickers}")
    if snapshot["if_interested"]:
        tickers = ", ".join(
            html_escape(str(r.get("ticker") or "?")) for r in snapshot["if_interested"]
        )
        lines.append(f"• Worth /analyze if interested: {tickers}")
    if not snapshot["analyze_now"] and not snapshot["if_interested"]:
        lines.append("• No names pass <code>/pick</code> right now — widen watchlist or prescan more.")
    return lines


def format_daily_workflow() -> str:
    """1–2 daily tip workflow — fast funnel, minimal over-filtering."""
    snap = _pick_snapshot()
    lines = [
        "<b>📅 Daily tip workflow (1–2 names)</b>",
        "Goal: one actionable buy/add idea per day without over-filtering.",
        "",
        "<b>Step 1 — Refresh the list (weekly or when stale)</b>",
        "<code>/prescan SYMBOL</code> on new watchlist names, or",
        "<code>/sip prescan</code> for the full portfolio batch (quant-only).",
        "",
        "<b>Step 2 — Soft pick (do not use /candidates alone)</b>",
        "<code>/pick</code> — quant≥50 or any Q/G/S pillar≥70; MONITOR is not a sell.",
        "",
    ]
    lines.extend(_format_pick_lines(snap))
    lines.extend(
        [
            "",
            "<b>Step 3 — Deep dive on 1–2 names only</b>",
            "<code>/analyze SYMBOL</code> on ✅ tier first, then 🔎 if you care.",
            "Pick only when: buy range issued + base 3y CAGR acceptable.",
            "After a few analyses: <code>/rank</code> — order by expected long-term return.",
            "Send <code>/stop</code> to cancel a long analysis.",
            "",
            "<b>Step 4 — Execute & record</b>",
            "<code>/hold SYMBOL qty avg_price</code> after you buy.",
            "Use analyze buy/add ranges — not prescan score alone.",
            "",
            "<b>Do not</b>",
            "• Treat MONITOR as sell for holdings",
            "• Require score≥65 for every tip (/pick is enough to shortlist)",
            "• Skip /analyze because /candidates filtered a name out",
            "",
            "<i>Review: <code>/track analyze</code> monthly — did BUY calls work?</i>",
        ]
    )
    return "\n".join(lines)


def format_portfolio_workflow() -> str:
    """12–18 name 3y portfolio build — slower, sector-aware funnel."""
    snap = _pick_snapshot()
    lines = [
        "<b>🏗 Portfolio build workflow (12–18 names, 3y horizon)</b>",
        "Goal: quality portfolio with sector caps, DCA tranches, and analyze-backed ranges.",
        "",
        "<b>Step 1 — Universe (~50 watchlist names)</b>",
        "Keep names in <code>sip_portfolios.json</code> buckets (core / satellite / ETF).",
        "",
        "<b>Step 2 — Batch prescan (quant-only first)</b>",
        "<code>/sip prescan</code> — writes prescan history for all portfolio symbols.",
        "<code>/sip prescan full</code> — adds AI eligibility (costs more).",
        "",
        "<b>Step 3 — Shortlist without over-filtering</b>",
        "<code>/pick</code> — soft floor; take top scores with sector diversity.",
        "Target 12–18 survivors — not every checklist box must pass as hard gate.",
        "",
    ]
    lines.extend(_format_pick_lines(snap))
    lines.extend(
        [
            "",
            "<b>Step 4 — Deep analyze survivors</b>",
            "<code>/analyze SYMBOL</code> on each shortlisted name (trade-friendly skips prescan gate).",
            "Reject for <b>new</b> capital only — not automatic sell if already held.",
            "Then <code>/rank</code> (or <code>/rank entry</code>) to order by expected base CAGR.",
            "",
            "<b>Step 5 — Size & sector limits</b>",
            "<code>/capital TOTAL max N</code> — set total capital and per-stock cap.",
            "Default sector cap 25% — check <code>/hold</code> list for concentration.",
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
            "<i>Holdings: <code>/hold</code> · Past calls: <code>/track prescan</code></i>",
        ]
    )
    return "\n".join(lines)
