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
    "EXCEPTION_DEEP_REVIEW": "Exception review (quality override)",
    "HOLDING_MONITOR": "Monitor only",
    "REJECT": "Rejected by hard filters",
    "DATA_RETRY": "Retry when data is available",
}

NEXT_ACTION_LABELS: dict[str, str] = {
    "FULL_DEEP_ANALYSIS": "Run /analyze for full research",
    "CHEAP_WC_RECONCILIATION_FIRST": "Explain working capital first (cheap check)",
    "SECTOR_SCORECARD_FIRST": "Run /analyze with sector/bank scorecard",
    "HOLDING_MONITOR": "Monitor only — no new research",
    "DATA_RETRY": "Retry /prescan when data improves",
    "NO_RESEARCH": "Skip research for now",
}


def format_quality_growth_strength(
    *,
    quality: float | None,
    growth: float | None,
    strength: float | None,
) -> str | None:
    parts: list[str] = []
    if quality is not None:
        parts.append(f"Quality {quality:.0f}")
    if growth is not None:
        parts.append(f"Growth {growth:.0f}")
    if strength is not None:
        parts.append(f"Strength {strength:.0f}")
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
    return "Quality/Growth/Strength not logged — re-run /prescan"
