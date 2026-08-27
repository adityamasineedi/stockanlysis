"""Red-flag penalty system — separate from base metric scores."""

from __future__ import annotations

from stockbot.portfolio_screener.models import RedFlag, StockMetrics
from stockbot.portfolio_screener.score_utils import series_present
from stockbot.portfolio_screener.scoring_config import RedFlagPenalties


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

    if metrics.promoter_pledge_pct is not None:
        if metrics.promoter_pledge_pct >= 40:
            add(
                "severe",
                "PLEDGE_HIGH",
                f"Promoter pledge {metrics.promoter_pledge_pct:.1f}%",
            )
        elif metrics.promoter_pledge_pct >= 20:
            add(
                "major",
                "PLEDGE_ELEVATED",
                f"Promoter pledge {metrics.promoter_pledge_pct:.1f}%",
            )
        elif metrics.promoter_pledge_pct >= 5:
            add(
                "minor",
                "PLEDGE_LOW",
                f"Promoter pledge {metrics.promoter_pledge_pct:.1f}%",
            )

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
        add(
            "major",
            "OCF_PAT_GAP",
            f"OCF/PAT={metrics.ocf_to_pat:.2f}",
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
