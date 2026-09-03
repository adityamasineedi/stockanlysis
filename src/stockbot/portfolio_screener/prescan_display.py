"""Beginner-friendly labels for prescan Telegram output."""

from __future__ import annotations

from typing import Any

BAND_ICONS: dict[str, str] = {
    "STRONG_CANDIDATE": "🏆",
    "CANDIDATE": "👍",
    "WATCHLIST": "📌",
    "REMOVE": "⬇️",
}

BAND_LABELS: dict[str, str] = {
    "STRONG_CANDIDATE": "Top tier (80+)",
    "CANDIDATE": "Good pick (70–79)",
    "WATCHLIST": "Watchlist (60–69)",
    "REMOVE": "Below threshold",
}

VERDICT_ICONS: dict[str, str] = {
    "AUTO_DEEP_ANALYSIS": "✅",
    "SECTOR_SPECIFIC_REVIEW": "🔎",
    "HOLDING_MONITOR_ONLY": "👀",
    "NOT_SUITABLE_FOR_3Y_RESEARCH": "❌",
    "DATA_UNAVAILABLE_RETRY": "📭",
}

VERDICT_LABELS: dict[str, str] = {
    "AUTO_DEEP_ANALYSIS": "Ready for /analyze",
    "SECTOR_SPECIFIC_REVIEW": "Sector review first — then /analyze",
    "HOLDING_MONITOR_ONLY": "Monitor only — skip /analyze",
    "NOT_SUITABLE_FOR_3Y_RESEARCH": "Not suitable for 3Y research",
    "DATA_UNAVAILABLE_RETRY": "Data missing — retry /prescan",
}

VERDICT_SUMMARY: dict[str, tuple[str, str]] = {
    "AUTO_DEEP_ANALYSIS": (
        "Passes the 3-year quality screen — good candidate for full research.",
        "Run /analyze for the detailed report.",
    ),
    "SECTOR_SPECIFIC_REVIEW": (
        "Promising score, but this business type needs a sector-specific lens first.",
        "Run /analyze — the bot will use a sector-aware framework.",
    ),
    "HOLDING_MONITOR_ONLY": (
        "Not eligible for fresh 3-year research or new capital right now.",
        "If you already hold it: monitor only — this is not a sell signal.",
    ),
    "DATA_UNAVAILABLE_RETRY": (
        "Could not fetch enough data — no quality conclusion yet.",
        "Retry /prescan later or check the symbol spelling.",
    ),
    "NOT_SUITABLE_FOR_3Y_RESEARCH": (
        "Outside the profitable-compounder 3-year screen.",
        "Skip /analyze for now — if already held, this is not a sell signal.",
    ),
}

CASH_ICONS: dict[str, str] = {
    "PASS": "💚",
    "WATCH": "💛",
    "ESCALATED_WATCH": "🧡",
    "CRITICAL": "❤️",
    "NOT_APPLICABLE": "➖",
    "NOT_APPLICABLE_WHILE_LOSS_MAKING": "🔥",
    "DATA_INSUFFICIENT_FOR_TREND": "❔",
}

CASH_LABELS: dict[str, str] = {
    "PASS": "Cash flow OK",
    "WATCH": "Cash flow — needs a closer look",
    "ESCALATED_WATCH": "Cash flow — elevated watch (explain before buying)",
    "CRITICAL": "Cash flow — weak / concerning",
    "NOT_APPLICABLE": "Cash check not applicable (e.g. bank/financial)",
    "NOT_APPLICABLE_WHILE_LOSS_MAKING": "Loss-maker — cash check skipped; watch cash burn",
    "DATA_INSUFFICIENT_FOR_TREND": "Not enough history for a cash-flow trend",
}

ISSUER_ICONS: dict[str, str] = {
    "BANK": "🏦",
    "NBFC_HFC": "🏦",
    "INSURER": "🛡️",
    "RATING_ANALYTICS": "📊",
    "MARKET_INFRA": "🏛️",
    "FINTECH_PLATFORM": "💳",
    "UTILITY": "⚡",
    "DEFENCE_EPC_PROJECT": "🪖",
    "EPC_PROJECT_BUSINESS": "🏗️",
    "AUTO_OEM": "🚗",
    "CONGLOMERATE": "🏢",
    "LOSS_MAKING_GROWTH": "🌱",
    "NON_FINANCIAL": "🏭",
    "OTHER": "📦",
}

ISSUER_LABELS: dict[str, str] = {
    "BANK": "Bank",
    "NBFC_HFC": "NBFC / housing finance",
    "INSURER": "Insurance company",
    "RATING_ANALYTICS": "Rating / analytics firm",
    "MARKET_INFRA": "Market infrastructure (exchange, clearing)",
    "FINTECH_PLATFORM": "Fintech platform",
    "UTILITY": "Utility (power, gas, etc.)",
    "DEFENCE_EPC_PROJECT": "Defence / project EPC company",
    "EPC_PROJECT_BUSINESS": "Engineering / EPC project company (non-defence)",
    "AUTO_OEM": "Auto manufacturer",
    "CONGLOMERATE": "Conglomerate",
    "LOSS_MAKING_GROWTH": "Loss-making growth company",
    "NON_FINANCIAL": "Regular industrial / consumer company",
    "OTHER": "Other",
}

ROUTE_LABELS: dict[str, str] = {
    "AUTO_DEEP": "Standard profitable-compounder screen",
    "SECTOR_SPECIFIC_REVIEW": "Sector-specific review",
    "BANK_SCORECARD": "Bank scorecard (not generic ratios)",
    "LOSS_MAKING_GROWTH_FRAMEWORK": "Loss-making growth framework",
    "UTILITY_DEEP_REVIEW": "Utility-sector deep review",
    "CONGLOMERATE_SOTP_REVIEW": "Conglomerate sum-of-parts review",
    "DEFENCE_WC_REVIEW": "Defence EPC — working-capital review",
    "EPC_WC_REVIEW": "EPC project — working-capital review",
    "EXCEPTION_DEEP_REVIEW": "Exception review (quality override)",
    "HOLDING_MONITOR": "Monitor only",
    "REJECT": "Rejected by hard filters",
    "DATA_RETRY": "Retry when data is available",
}

# "BANK_SCORECARD" is one shared route for every FINANCIAL_SCORECARD_ISSUERS member
# (see issuer_routing.py) — this refines the displayed label by the stock's actual
# issuer class so a rating agency or exchange isn't shown "Bank scorecard".
SCORECARD_ROUTE_LABELS_BY_ISSUER: dict[str, str] = {
    "BANK": "Bank scorecard (not generic ratios)",
    "NBFC_HFC": "NBFC scorecard (not generic ratios)",
    "INSURER": "Insurer scorecard (not generic ratios)",
    "RATING_ANALYTICS": "Rating/analytics scorecard (not generic ratios)",
    "MARKET_INFRA": "Market-infrastructure scorecard (not generic ratios)",
    "FINTECH_PLATFORM": "Fintech/platform scorecard (not generic ratios)",
}

NEXT_ACTION_LABELS: dict[str, str] = {
    "FULL_DEEP_ANALYSIS": "Run /analyze for full research",
    "CHEAP_WC_RECONCILIATION_FIRST": "Explain working capital first (cheap check)",
    "SECTOR_SCORECARD_FIRST": "Run /analyze with sector/bank scorecard",
    "HOLDING_MONITOR": "Monitor only — no new research",
    "DATA_RETRY": "Retry /prescan when data improves",
    "NO_RESEARCH": "Skip research for now",
}

# Compact single-ticker card. Telegram HTML has no font-size tag; blockquote +
# <code> is the densest native rendering (monospace, inset, typically smaller).
ENTRY_HEADLINES: dict[str, tuple[str, str]] = {
    "AUTO_DEEP_ANALYSIS": ("🟢", "RESEARCH ENTRY OPEN"),
    "SECTOR_SPECIFIC_REVIEW": ("🔎", "SECTOR REVIEW BEFORE RESEARCH"),
    "HOLDING_MONITOR_ONLY": ("🔴", "RESEARCH ENTRY REJECTED"),
    "NOT_SUITABLE_FOR_3Y_RESEARCH": ("🔴", "RESEARCH ENTRY REJECTED"),
    "DATA_UNAVAILABLE_RETRY": ("📭", "DATA MISSING — RETRY"),
}


def _pillar_light(score: float | None) -> str:
    if score is None:
        return "⚪"
    if score >= 60:
        return "🟢"
    if score >= 40:
        return "🟡"
    return "🔴"


def telegram_small_block(lines: list[str]) -> str:
    """Inset compact block — closest Telegram gets to a smaller font."""
    body = "\n".join(lines).strip("\n")
    if not body:
        return ""
    return f"<blockquote>{body}</blockquote>"


def telegram_code_line(text: str) -> str:
    return f"<code>{text}</code>"


def format_qgs_compact(
    *,
    quality: float | None,
    growth: float | None,
    strength: float | None,
) -> str | None:
    parts: list[str] = []
    if quality is not None:
        parts.append(f"Q {quality:.1f} {_pillar_light(quality)}")
    if growth is not None:
        parts.append(f"G {growth:.1f} {_pillar_light(growth)}")
    if strength is not None:
        parts.append(f"S {strength:.1f} {_pillar_light(strength)}")
    if not parts:
        return None
    return telegram_code_line("📊 " + " | ".join(parts))


def _de_light(value: float) -> str:
    if value <= 0.5:
        return "🟢"
    if value <= 1.0:
        return "🟡"
    return "🔴"


def _roe_light(value: float) -> str:
    if value >= 15:
        return "🟢"
    if value >= 8:
        return "🟡"
    return "🔴"


def _ocf_pat_light(value: float) -> str:
    if value >= 0.80:
        return "🟢"
    if value >= 0.50:
        return "🟡"
    return "🔴"


def _nd_ebitda_light(value: float) -> str:
    # Negative = net cash — healthy for non-financials.
    if value <= 1.0:
        return "🟢"
    if value <= 3.0:
        return "🟡"
    return "🔴"


def _ic_light(value: float) -> str:
    if value >= 5.0:
        return "🟢"
    if value >= 2.0:
        return "🟡"
    return "🔴"


def format_metric_lines(
    *,
    debt_equity: float | None = None,
    roe: float | None = None,
    ocf_pat: float | None = None,
    net_debt_ebitda: float | None = None,
    interest_coverage: float | None = None,
) -> list[str]:
    lines: list[str] = []
    if debt_equity is not None:
        lines.append(telegram_code_line(f"💰 D/E {debt_equity:.2f}× {_de_light(debt_equity)}"))
    if roe is not None:
        lines.append(telegram_code_line(f"📈 ROE {roe:.1f}% {_roe_light(roe)}"))
    if ocf_pat is not None:
        lines.append(telegram_code_line(f"💵 OCF/PAT {ocf_pat:.2f}× {_ocf_pat_light(ocf_pat)}"))
    if net_debt_ebitda is not None:
        lines.append(
            telegram_code_line(
                f"🏦 Net Debt/EBITDA {net_debt_ebitda:.2f}× {_nd_ebitda_light(net_debt_ebitda)}"
            )
        )
    if interest_coverage is not None:
        lines.append(
            telegram_code_line(
                f"💳 Interest Coverage {interest_coverage:.2f}× {_ic_light(interest_coverage)}"
            )
        )
    return lines


def format_action_lines(*, verdict: str, suitable_for_deep_analysis: bool) -> list[str]:
    if verdict == "AUTO_DEEP_ANALYSIS":
        research = "✅ /analyze"
        holding = "✅ Hold OK to research"
    elif verdict == "SECTOR_SPECIFIC_REVIEW":
        research = "🔎 Sector lens → /analyze"
        holding = "👀 Monitor"
    elif verdict == "DATA_UNAVAILABLE_RETRY":
        research = "📭 Retry /prescan"
        holding = "👀 Wait for data"
    elif verdict == "HOLDING_MONITOR_ONLY":
        research = "❌"
        holding = "👀 Monitor"
    else:
        research = "❌"
        holding = "👀 Monitor"
    if not suitable_for_deep_analysis and verdict == "SECTOR_SPECIFIC_REVIEW":
        research = "❌"
    return [
        f"➡️ New research: {research}",
        f"➡️ Existing holding: {holding}",
        "➡️ Sell signal: ❌ No",
    ]


_JARGON_WHY_MARKERS = (
    "quant_score",
    "quant ",
    "research floor",
    "cash pass",
    "cash watch",
    "cash critical",
    "auto-eligible",
    "3y research",
    "deep spend",
    "hard_exclude",
)


def _is_jargon_why(reason: str) -> bool:
    low = reason.lower()
    return any(marker in low for marker in _JARGON_WHY_MARKERS)


def synthesize_why(
    *,
    key_reason: str,
    quality: float | None,
    growth: float | None,
    strength: float | None,
) -> str:
    strong: list[str] = []
    weak: list[str] = []
    for label, score in (
        ("quality/returns", quality),
        ("growth", growth),
        ("financial strength", strength),
    ):
        if score is None:
            continue
        if score >= 60:
            strong.append(label)
        elif score < 40:
            weak.append(label)
    if strong and weak:
        return f"{' + '.join(strong).capitalize()} good, but {'/'.join(weak)} weak."
    if weak:
        return f"{'/'.join(weak).capitalize()} weak for a 3-year compounder screen."

    reason = (key_reason or "").strip()
    if reason and not _is_jargon_why(reason):
        if len(reason) > 140:
            return reason[:137] + "…"
        return reason
    if strong:
        return f"{' + '.join(strong).capitalize()} look fine — see gate above."
    return "Score is below the 3-year research bar."


def format_quality_growth_strength(
    *,
    quality: float | None,
    growth: float | None,
    strength: float | None,
) -> str | None:
    parts: list[str] = []
    if quality is not None:
        parts.append(f"Quality {quality:.1f}")
    if growth is not None:
        parts.append(f"Growth {growth:.1f}")
    if strength is not None:
        parts.append(f"Strength {strength:.1f}")
    if not parts:
        return None
    return " · ".join(parts)


def format_qgs_from_row(row: dict[str, Any]) -> str:
    text = format_quality_growth_strength(
        quality=row.get("quality_score")
        if isinstance(row.get("quality_score"), (int, float))
        else None,
        growth=row.get("growth_score")
        if isinstance(row.get("growth_score"), (int, float))
        else None,
        strength=row.get("strength_score")
        if isinstance(row.get("strength_score"), (int, float))
        else None,
    )
    if text:
        return text
    return "Pillar scores unavailable"
