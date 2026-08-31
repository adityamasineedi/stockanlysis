"""Telegram HTML for multi-stock portfolio SIP plans."""

from __future__ import annotations

from html import escape

from stockbot.portfolio_sip import (
    AllocationLine,
    PortfolioAllocation,
    PortfolioSipPlan,
)
from stockbot.sip_messages import TOPUP_RISK_NOTE

TELEGRAM_CHUNK_LIMIT = 4000


def _money(value: float) -> str:
    return f"₹{value:,.0f}"


def _price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"₹{value:,.2f}"


def format_allocation_line_html(line: AllocationLine) -> str:
    if line.prescan_skip:
        rank = f"P{line.priority_rank} " if line.priority_rank else ""
        note = line.note or "prescan skip"
        return f"{rank}<code>{escape(line.symbol):8}</code> — <i>{escape(note)}</i>"
    if line.rotation_skip:
        rank = f"P{line.priority_rank} " if line.priority_rank else ""
        return f"{rank}<code>{escape(line.symbol):8}</code> — <i>rotation skip</i>"
    if line.error:
        rank = f"P{line.priority_rank} " if line.priority_rank else ""
        return f"{rank}{escape(line.symbol)} — <i>{escape(line.error)}</i>"
    dip_hint = ""
    if line.dip_label and line.topup_range and line.dip_pct is not None:
        low, high = line.topup_range
        dip_hint = f" · dip {line.dip_pct:.1f}% → top-up {_money(low)}–{_money(high)}"
    rank = f"P{line.priority_rank} " if line.priority_rank else ""
    return (
        f"{rank}<code>{escape(line.symbol):8}</code> "
        f"{_price(line.price):>12} "
        f"{line.shares:>3} sh "
        f"{_money(line.invested):>8}{dip_hint}"
    )


def _allocation_mode_label(allocation: PortfolioAllocation) -> str:
    mode = allocation.portfolio.allocation_mode
    if mode == "priority":
        return "priority order (P1 first)"
    if mode == "prescan_rank":
        return "prescan rank (higher Q first)"
    if mode in {"equal_split", "equal"}:
        return "target ₹ per name"
    return "equal ₹ per name"


def format_portfolio_allocation_html(allocation: PortfolioAllocation) -> str:
    mode = _allocation_mode_label(allocation)
    thesis_line = ""
    if allocation.portfolio.thesis:
        thesis_line = f"\n<i>{escape(allocation.portfolio.thesis)}</i>"
    header = (
        f"<b>{escape(allocation.portfolio.label)}</b> "
        f"({_money(allocation.portfolio.monthly_budget)}/mo · {mode})"
        f"{thesis_line}\n"
        f"<pre>{'':3}{'Symbol':8} {'Price':>12} {'Qty':>6} {'Invest':>8}</pre>"
    )
    body_lines = [format_allocation_line_html(line) for line in allocation.lines]
    footer = (
        f"\nInvested {_money(allocation.invested)} · "
        f"cash aside {_money(allocation.cash_aside)}"
    )
    return header + "\n".join(body_lines) + footer


def format_portfolio_plan_html(plan: PortfolioSipPlan) -> str:
    mode_key = plan.config.default_allocation_mode
    if mode_key == "priority":
        mode = "priority order (P1 first)"
    elif mode_key == "prescan_rank":
        mode = "prescan rank (higher Q first)"
    elif mode_key in {"equal_split", "equal"}:
        mode = "target ₹ per name"
    else:
        mode = "equal ₹ per name"
    parts = [
        f"<b>Portfolio SIP plan</b> (whole shares · {mode} · live prices)",
        f"Total budget {_money(plan.config.total_monthly_budget)}/mo",
        "",
    ]
    for allocation in plan.allocations:
        parts.append(format_portfolio_allocation_html(allocation))
        parts.append("")
    parts.append(
        f"<b>Monthly total:</b> {_money(plan.total_invested)} into shares · "
        f"{_money(plan.total_cash_aside)} cash aside"
    )
    parts.append(
        "<i>Whole shares only — leftover rupees stay in the bucket until the "
        "next full share fits.</i>"
    )
    return "\n".join(parts)


def split_telegram_chunks(text: str, *, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
    """Split long plan text on portfolio section boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(block) <= limit:
            current = block
        else:
            chunks.append(block[: limit - 1] + "…")
            current = ""
    if current:
        chunks.append(current)
    return chunks


def format_portfolio_track_html(
    plan: PortfolioSipPlan,
    paid_by_symbol: dict[str, float],
    *,
    month_label: str,
) -> str:
    lines = [
        f"<b>SIP track — {escape(month_label)}</b>",
        "Planned (this month's whole-share table) vs logged via <code>/sip paid</code>.",
        "",
    ]
    for allocation in plan.allocations:
        lines.append(f"<b>{escape(allocation.portfolio.label)}</b>")
        lines.append(f"<pre>{'Symbol':8} {'Plan':>8} {'Paid':>8} {'Gap':>8}</pre>")
        for row in allocation.lines:
            if row.error or row.price is None:
                lines.append(f"<code>{escape(row.symbol):8}</code> — skipped")
                continue
            paid = paid_by_symbol.get(row.symbol, 0.0)
            gap = round(row.invested - paid, 2)
            status = "✅" if gap <= 0 else "⏳"
            gap_label = "₹0" if gap <= 0 else _money(gap)
            lines.append(
                f"<code>{escape(row.symbol):8}</code> "
                f"{_money(row.invested):>8} "
                f"{_money(paid):>8} "
                f"{gap_label:>8} {status}"
            )
        lines.append("")
    total_planned = plan.total_invested
    total_paid = round(sum(paid_by_symbol.values()), 2)
    lines.append(
        f"<b>Total:</b> planned {_money(total_planned)} · "
        f"logged {_money(total_paid)} · "
        f"remaining {_money(max(total_planned - total_paid, 0))}"
    )
    if any(
        line.dip_label
        for allocation in plan.allocations
        for line in allocation.lines
        if line.dip_label
    ):
        lines.extend(["", f"<i>{TOPUP_RISK_NOTE}</i>"])
    return "\n".join(lines)
