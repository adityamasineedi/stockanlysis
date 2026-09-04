"""Red-flag penalty system — for signals not already priced into pillars.

Promoter *holding* % is never a red flag by itself.
Only *pledged* share of promoter holding uses pledge thresholds
(aligned with eligibility prompt: Prefer ≤10, Borderline 10–25, Critical >25).

OCF/PAT gaps are recorded as flags for routing, but carry **zero score
penalty** — cash_flow_quality already embeds that weakness. Stacking a
second −10 crushed mixed names into the low 30s on a 0–100 scale.
"""

from __future__ import annotations

from stockbot.portfolio_screener.models import RedFlag, StockMetrics
from stockbot.portfolio_screener.score_utils import series_present
from stockbot.portfolio_screener.scoring_config import RedFlagPenalties

# Eligibility / gatekeeper bands for pledged_promoter_holding_pct only.
_PLEDGE_PREFER_MAX = 10.0
_PLEDGE_BORDERLINE_MAX = 25.0
_PLEDGE_SEVERE_AT = 40.0


def collect_red_flags(
    metrics: StockMetrics,
    penalties: RedFlagPenalties | None = None,
) -> list[RedFlag]:
    penalties = penalties or RedFlagPenalties()
    flags: list[RedFlag] = []

    def add(
        severity: str,
        code: str,
        message: str,
    ) -> None:
        penalty = getattr(penalties, severity)
        flags.append(
            RedFlag(
                severity=severity,  # type: ignore[arg-type]
                code=code,
                message=message,
                penalty=penalty,
            )
        )

    # --- Pledge only (never promoter_holding_pct) ---
    pledge = metrics.pledged_promoter_holding_pct
    if pledge is not None:
        if pledge >= _PLEDGE_SEVERE_AT:
            add(
                "severe",
                "PLEDGE_HIGH",
                f"pledged_promoter_holding_pct {pledge:.1f}% (Critical)",
            )
        elif pledge > _PLEDGE_BORDERLINE_MAX:
            add(
                "major",
                "PLEDGE_ELEVATED",
                f"pledged_promoter_holding_pct {pledge:.1f}% (Critical)",
            )
        elif pledge > _PLEDGE_PREFER_MAX:
            add(
                "minor",
                "PLEDGE_BORDERLINE",
                f"pledged_promoter_holding_pct {pledge:.1f}% (Borderline)",
            )
        # Prefer (≤10%): no red flag

    if metrics.share_dilution_pct is not None and metrics.share_dilution_pct > 25:
        add(
            "moderate",
            "DILUTION",
            f"Equity capital up {metrics.share_dilution_pct:.1f}% over history",
        )

    if (
        metrics.ocf_to_pat is not None
        and metrics.net_income is not None
        and metrics.net_income > 0
        and metrics.ocf_to_pat < 0.5
    ):
        from stockbot.portfolio_screener.issuer_routing import (
            assess_cash_conversion,
            classify_issuer,
        )

        issuer = classify_issuer(metrics)
        cash = assess_cash_conversion(metrics, issuer)
        # OCF/PAT is already scored inside cash_flow_quality (and was double-hit
        # via earnings_quality). Keep the flag for routing / Telegram context,
        # but do NOT subtract another −10/−5 from the composite — that triple
        # count pushed mixed names into the low 30s on a 0–100 scale.
        if cash.status == "CRITICAL":
            flags.append(
                RedFlag(
                    severity="major",
                    code="OCF_PAT_GAP",
                    message=(
                        f"OCF/PAT current={cash.ocf_pat_current} "
                        f"3y_cum={cash.ocf_pat_3y} (Critical)"
                    ),
                    penalty=0.0,
                )
            )
        elif cash.status == "ESCALATED_WATCH":
            flags.append(
                RedFlag(
                    severity="moderate",
                    code="OCF_PAT_ESCALATED",
                    message=(
                        f"OCF/PAT ESCALATED_WATCH: current={cash.ocf_pat_current} "
                        f"3y_cum={cash.ocf_pat_3y}"
                    ),
                    penalty=0.0,
                )
            )
        elif cash.status == "WATCH":
            flags.append(
                RedFlag(
                    severity="minor",
                    code="OCF_PAT_WATCH",
                    message=f"OCF/PAT WATCH: {cash.reason}",
                    penalty=0.0,
                )
            )
        elif cash.status in {
            "NOT_APPLICABLE",
            "NOT_APPLICABLE_WHILE_LOSS_MAKING",
            "DATA_INSUFFICIENT_FOR_TREND",
        }:
            pass
        else:
            flags.append(
                RedFlag(
                    severity="minor",
                    code="OCF_PAT_WATCH",
                    message=f"OCF/PAT={metrics.ocf_to_pat:.2f}",
                    penalty=0.0,
                )
            )

    margins = series_present(metrics.operating_margin_series)
    if len(margins) >= 3 and margins[-1] < margins[0] - 0.05:
        add(
            "moderate",
            "MARGIN_DOWN",
            "Operating margin deteriorated over available history",
        )

    if metrics.market_cap_cr is not None and metrics.market_cap_cr < 500:
        add("moderate", "SMALL_CAP", f"Market cap ₹{metrics.market_cap_cr:.0f} Cr")
    elif metrics.market_cap_cr is not None and metrics.market_cap_cr < 2000:
        add("minor", "MID_SMALL_CAP", f"Market cap ₹{metrics.market_cap_cr:.0f} Cr")

    if metrics.debt_equity is not None and metrics.debt_equity > 2.0:
        add("major", "HIGH_LEVERAGE", f"D/E={metrics.debt_equity:.2f}")
    elif metrics.debt_equity is not None and metrics.debt_equity > 1.0:
        add("minor", "ELEVATED_LEVERAGE", f"D/E={metrics.debt_equity:.2f}")

    if metrics.pe is not None and metrics.pe > 80:
        add("moderate", "RICH_VALUATION", f"P/E={metrics.pe:.1f}")

    return flags


def total_penalty(flags: list[RedFlag]) -> float:
    return sum(f.penalty for f in flags)


def governance_notes(metrics: StockMetrics) -> list[str]:
    """Informational holding/pledge lines for AI payload and Telegram Data:."""
    notes: list[str] = []
    holding = metrics.promoter_holding_pct
    pledge = metrics.pledged_promoter_holding_pct
    if holding is not None:
        notes.append(
            f"promoter_holding_pct {holding:.2f}% (ownership concentration; informational — not Critical)"
        )
    else:
        notes.append("promoter_holding_pct null")
    if pledge is None:
        notes.append("pledged_promoter_holding_pct null")
    elif pledge > _PLEDGE_BORDERLINE_MAX:
        notes.append(f"pledged_promoter_holding_pct {pledge:.2f}% (Critical)")
    elif pledge > _PLEDGE_PREFER_MAX:
        notes.append(f"pledged_promoter_holding_pct {pledge:.2f}% (Borderline)")
    else:
        notes.append(f"pledged_promoter_holding_pct {pledge:.2f}% (Prefer)")
    return notes
