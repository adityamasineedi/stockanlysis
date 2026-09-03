"""Calibrate /pick thresholds from prescan history — read-only tuning hints.

Uses the same forward-return scoring as /track prescan. Does not change config;
suggests when history supports tightening or loosening pick_min_quant_score /
pick_min_pillar_score.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from stockbot.calibration import (
    MIN_SAMPLE,
    ScoredCall,
    build_bucket,
    score_calls,
    tier_spread,
)
from stockbot.calibration_report import _current_prices, _load_prescan_rows
from stockbot.config import settings
from stockbot.portfolio_screener.pick_policy import (
    is_pick_eligible,
    pick_min_pillar_score,
    pick_min_quant_score,
)


def _quant_band(quant: object) -> str:
    if not isinstance(quant, (int, float)):
        return "quant unknown"
    q = float(quant)
    if q < 50:
        return "quant <50"
    if q < 55:
        return "quant 50–54"
    if q < 65:
        return "quant 55–64"
    return "quant 65+"


def _pillar_override_row(row: dict) -> bool:
    if row.get("quality_override"):
        return True
    quant = row.get("quant_score")
    if isinstance(quant, (int, float)) and float(quant) >= pick_min_quant_score():
        return False
    min_pillar = pick_min_pillar_score()
    for key in ("quality_score", "growth_score", "strength_score"):
        value = row.get(key)
        if isinstance(value, (int, float)) and float(value) >= min_pillar:
            return True
    return False


@dataclass(frozen=True)
class PickCalibrationReport:
    total_rows: int
    scored: int
    spread: object | None
    pick_median: float | None
    no_pick_median: float | None
    n_pick: int
    n_no_pick: int
    by_quant_band: list[tuple[str, int, float | None]]
    n_pillar_only: int
    pillar_only_median: float | None


def _median_returns(calls: list[ScoredCall]) -> float | None:
    if len(calls) < MIN_SAMPLE:
        return None
    return round(statistics.median([c.return_pct for c in calls]), 2)


def build_pick_calibration(
    *,
    prices: dict[str, float | None] | None = None,
) -> PickCalibrationReport:
    rows = _load_prescan_rows()
    tickers = {str(r.get("ticker") or "").upper() for r in rows if r.get("ticker")}
    resolved = prices if prices is not None else _current_prices(tickers)
    # score_calls now reports why rows were left out, not just the survivors.
    scored = score_calls(rows, resolved).calls

    by_ticker_row: dict[str, dict] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            by_ticker_row[ticker] = row

    pick_calls: list[ScoredCall] = []
    no_pick_calls: list[ScoredCall] = []
    pillar_calls: list[ScoredCall] = []
    band_groups: dict[str, list[ScoredCall]] = {}

    for call in scored:
        row = by_ticker_row.get(call.ticker, {})
        band = _quant_band(row.get("quant_score"))
        band_groups.setdefault(band, []).append(call)
        if is_pick_eligible(row):
            pick_calls.append(call)
            if (
                _pillar_override_row(row)
                and isinstance(row.get("quant_score"), (int, float))
                and float(row["quant_score"]) < pick_min_quant_score()
            ):
                pillar_calls.append(call)
        else:
            no_pick_calls.append(call)

    spread = tier_spread(
        [
            ScoredCall(c.ticker, "pick" if is_pick_eligible(by_ticker_row.get(c.ticker, {})) else "no_pick", c.return_pct, c.days)
            for c in scored
        ],
        positive_labels={"pick"},
        negative_labels={"no_pick"},
    )

    by_quant = []
    for label in ("quant <50", "quant 50–54", "quant 55–64", "quant 65+", "quant unknown"):
        group = band_groups.get(label, [])
        bucket = build_bucket(label, group)
        by_quant.append((label, bucket.n, bucket.median_return_pct))

    return PickCalibrationReport(
        total_rows=len(rows),
        scored=len(scored),
        spread=spread,
        pick_median=_median_returns(pick_calls),
        no_pick_median=_median_returns(no_pick_calls),
        n_pick=len(pick_calls),
        n_no_pick=len(no_pick_calls),
        by_quant_band=by_quant,
        n_pillar_only=len(pillar_calls),
        pillar_only_median=_median_returns(pillar_calls),
    )


@dataclass(frozen=True)
class PickTuneAdvice:
    current_quant: float
    current_pillar: float
    lines: tuple[str, ...]


def build_pick_tune_advice(report: PickCalibrationReport) -> PickTuneAdvice:
    """Conservative threshold suggestions from scored history."""
    q_floor = pick_min_quant_score()
    p_floor = pick_min_pillar_score()
    lines: list[str] = []

    if report.scored < MIN_SAMPLE:
        lines.append(
            f"Only {report.scored} scoreable prescan row(s) — need at least {MIN_SAMPLE} "
            "before tuning advice is meaningful. Keep current floors and run /sip prescan."
        )
        return PickTuneAdvice(q_floor, p_floor, tuple(lines))

    if report.spread is not None:
        sp = report.spread
        if sp.discriminates:
            lines.append(
                f"Soft picks median {sp.positive_median_pct:+.1f}% vs "
                f"rejects {sp.negative_median_pct:+.1f}% "
                f"(spread {sp.spread_pct:+.1f} pts) — current floors look useful."
            )
        else:
            lines.append(
                f"Soft picks ({sp.n_positive}) median {sp.positive_median_pct:+.1f}% vs "
                f"rejects ({sp.n_negative}) {sp.negative_median_pct:+.1f}% — "
                "<b>pick floor may be too loose</b> or prescan scores need time to discriminate."
            )
    else:
        lines.append(
            "Not enough pick vs reject calls on both sides yet — keep defaults until /track pick fills in."
        )

    band_medians = {label: med for label, n, med in report.by_quant_band if n >= MIN_SAMPLE and med is not None}
    low_band = band_medians.get("quant 50–54")
    high_band = band_medians.get("quant 65+")
    if low_band is not None and high_band is not None:
        if low_band > high_band + 3:
            lines.append(
                f"Quant 50–54 band median {low_band:+.1f}% beats 65+ ({high_band:+.1f}%) — "
                f"keeping <code>pick_min_quant_score={q_floor:.0f}</code> is reasonable."
            )
        elif high_band > low_band + 5:
            lines.append(
                f"Quant 65+ median {high_band:+.1f}% beats 50–54 ({low_band:+.1f}%) — "
                f"consider raising pick_min_quant_score toward 55 via env (currently {q_floor:.0f})."
            )

    if report.n_pillar_only >= MIN_SAMPLE and report.pillar_only_median is not None:
        lines.append(
            f"Pillar-only picks (quant&lt;{q_floor:.0f}, pillar≥{p_floor:.0f}): "
            f"n={report.n_pillar_only}, median {report.pillar_only_median:+.1f}% — "
            "strength/quality override path is doing work; keep pillar floor."
        )
    elif report.n_pillar_only > 0:
        lines.append(
            f"Pillar-only picks: {report.n_pillar_only} scored — too few for median; keep pillar≥{p_floor:.0f}."
        )

    under_50 = next((med for label, n, med in report.by_quant_band if label == "quant <50" and n >= MIN_SAMPLE), None)
    if under_50 is not None and under_50 < -5:
        lines.append(
            f"Quant &lt;50 band median {under_50:+.1f}% — do not lower pick_min_quant_score below {q_floor:.0f}."
        )

    if not any("consider" in line.lower() or "too loose" in line.lower() for line in lines):
        lines.append(
            f"No strong signal to change env defaults "
            f"(pick_min_quant_score={settings.pick_min_quant_score}, "
            f"pick_min_pillar_score={settings.pick_min_pillar_score}). Re-run monthly."
        )

    return PickTuneAdvice(q_floor, p_floor, tuple(lines))


def format_pick_calibration_report(report: PickCalibrationReport) -> str:
    lines = [
        "🎯 <b>Track record — /pick policy</b>",
        f"{report.scored} scoreable of {report.total_rows} prescan row(s).",
        f"Floors: quant≥{pick_min_quant_score():.0f} or Q/G/S≥{pick_min_pillar_score():.0f}.",
        "",
    ]
    if report.scored < MIN_SAMPLE:
        lines.append(
            f"Not enough history yet (need {MIN_SAMPLE}+ scoreable). "
            "Keep using /pick and /sip prescan — this fills automatically."
        )
        return "\n".join(lines)

    if report.spread is not None:
        sp = report.spread
        verdict = "discriminating" if sp.discriminates else "<b>not discriminating</b>"
        lines.extend(
            [
                "<b>Pick vs reject</b>",
                f"Pick ({sp.n_positive}): {_pct(sp.positive_median_pct)} median",
                f"Reject ({sp.n_negative}): {_pct(sp.negative_median_pct)} median",
                f"Spread: <b>{sp.spread_pct:+.1f} pts</b> — {verdict}.",
                "",
            ]
        )

    lines.append("<b>By quant band</b>")
    for label, n, med in report.by_quant_band:
        if n == 0:
            continue
        if med is None:
            lines.append(f"{label} — n={n}, too few for median")
        else:
            lines.append(f"{label} — n={n} · median {_pct(med)}")

    lines.extend(
        [
            "",
            "<i>Run <code>/track pick tune</code> for env threshold suggestions.</i>",
        ]
    )
    return "\n".join(lines)


def format_pick_tune_report(advice: PickTuneAdvice) -> str:
    lines = [
        "🔧 <b>Pick threshold tune (read-only)</b>",
        (
            f"Current: pick_min_quant_score={advice.current_quant:.0f}, "
            f"pick_min_pillar_score={advice.current_pillar:.0f}"
        ),
        "",
        *advice.lines,
        "",
        "Set via Railway/env — bot does not auto-change these.",
    ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1f}%"
