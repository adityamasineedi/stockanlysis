"""Generate a markdown-style review doc for portfolio SIP Telegram UI."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from html import unescape

from stockbot.portfolio_sip import build_portfolio_sip_plan
from stockbot.portfolio_sip_messages import (
    format_portfolio_plan_html,
    format_portfolio_track_html,
    split_telegram_chunks,
)

IST = timezone(timedelta(hours=5, minutes=30))


def html_to_review(text: str) -> str:
    t = text.replace("<b>", "**").replace("</b>", "**")
    t = t.replace("<i>", "_").replace("</i>", "_")
    t = t.replace("<code>", "`").replace("</code>", "`")
    t = re.sub(r"</?pre>", "", t)
    return unescape(t)


def main() -> None:
    plan = build_portfolio_sip_plan()
    plan_html = format_portfolio_plan_html(plan)

    sample_paid: dict[str, float] = {}
    for allocation in plan.allocations:
        for index, line in enumerate(allocation.lines):
            if line.error or line.price is None or line.invested <= 0:
                continue
            if index % 2 == 0:
                sample_paid[line.symbol] = line.invested

    month_label = datetime.now(IST).strftime("%B %Y")
    track_html = format_portfolio_track_html(plan, sample_paid, month_label=month_label)

    lines: list[str] = []
    stamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    lines.extend(
        [
            "=" * 72,
            "STOCKBOT PORTFOLIO SIP — REVIEW DOC",
            f"Generated: {stamp}",
            "Config: data/portfolio/sip_portfolios.json",
            "Prices: live yfinance (same as /sip plan in Telegram)",
            "=" * 72,
            "",
            "## Commands",
            "",
            "| Command | Purpose |",
            "|---------|---------|",
            "| `/sip plan` | Monthly whole-share split (target ₹ per name) |",
            "| `/sip track` | Planned vs logged this month |",
            "| `/sip paid KAYNES 3685` | Log a buy after broker execution |",
            "| `/sip paid BEL 2500 topup` | Log optional dip top-up |",
            "",
            "## Bucket thesis",
            "",
        ]
    )
    for allocation in plan.allocations:
        thesis = allocation.portfolio.thesis or "_No thesis set._"
        lines.append(f"- **{allocation.portfolio.label}** — {thesis}")
    lines.extend(
        [
            "",
            "## TELEGRAM preview: `/sip plan`",
            "",
        ]
    )
    for index, chunk in enumerate(split_telegram_chunks(plan_html), 1):
        lines.append(f"### Message {index}")
        lines.append("")
        lines.append(html_to_review(chunk))
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## TELEGRAM preview: `/sip track`",
            "",
            "_Demo below: every other stock marked as paid for illustration._",
            "",
        ]
    )
    for index, chunk in enumerate(split_telegram_chunks(track_html), 1):
        lines.append(f"### Message {index}")
        lines.append("")
        lines.append(html_to_review(chunk))
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Numbers summary",
            "",
            f"- **Total budget:** Rs {plan.config.total_monthly_budget:,.0f}/month",
            f"- **Into shares:** Rs {plan.total_invested:,.0f}",
            f"- **Cash aside:** Rs {plan.total_cash_aside:,.0f}",
            "",
            "### Per bucket",
            "",
        ]
    )
    for allocation in plan.allocations:
        dip_count = sum(1 for line in allocation.lines if line.dip_label)
        lines.append(
            f"- **{allocation.portfolio.label}** — "
            f"invest Rs {allocation.invested:,.0f}, "
            f"aside Rs {allocation.cash_aside:,.0f}, "
            f"{dip_count} dip alert(s) today"
        )

    lines.extend(
        [
            "",
            "## Config file (edit to change lists/budgets)",
            "",
            "```json",
            "data/portfolio/sip_portfolios.json",
            "  existing  -> 9 names  @ Rs 20,000  (P1 MAZDOCK ... P9 NETWEB)",
            "  growth    -> 4 names  @ Rs 20,000",
            "  metals    -> 6 names  @ Rs 20,000",
            "```",
            "",
            "_Educational research only — not investment advice._",
            "",
        ]
    )

    output = "\n".join(lines)
    out_path = "logs/sip_review_doc.md"
    from pathlib import Path

    Path("logs").mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(output, encoding="utf-8")
    print(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
