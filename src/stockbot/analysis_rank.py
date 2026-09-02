"""Rank stored /analyze results for long-term hold and expected return.

Read-only: uses latest SQLite analysis per ticker. Primary signal is base-case
3y CAGR from expected_return (Python-computed FV scenarios), gated by verdict
and five-year test. Does not fetch live prices (fast, no LLM).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from typing import Any, Literal

RankMode = Literal["hold", "entry"]

_VERDICT_RANK: dict[str, int] = {
    "BUY": 4,
    "BUY ON CORRECTION": 3,
    "WATCH": 2,
    "SKIP": 0,
    "AVOID": 0,
}


def _midpoint(pair: object) -> float | None:
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return None
    try:
        low, high = float(pair[0]), float(pair[1])
    except (TypeError, ValueError):
        return None
    return round((low + high) / 2.0, 2)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class RankedAnalysis:
    ticker: str
    verdict: str
    base_cagr_mid: float | None
    bull_cagr_mid: float | None
    bear_cagr_mid: float | None
    confidence: int | None
    five_year: str
    thesis: str
    buy_range_allowed: bool
    entry_ready: bool
    wait_dip: bool
    risk: str
    price: float | None
    buy_high: float | None
    analyzed_at: datetime | None
    score: float
    skip_reason: str | None = None


def _entry_ready(verdict_json: dict) -> bool:
    if verdict_json.get("buy_range_allowed") is not True:
        return False
    price = _float_or_none(verdict_json.get("current_price_abs"))
    buy = verdict_json.get("buy_zone_abs")
    if price is None or not isinstance(buy, (list, tuple)) or len(buy) < 2:
        return False
    try:
        buy_high = float(buy[1])
    except (TypeError, ValueError):
        return False
    return price <= buy_high * 1.02  # 2% tolerance for stale cache sync


def _compute_score(row: RankedAnalysis) -> float:
    """Higher = better long-term hold candidate for new capital.

    Base CAGR midpoint dominates (expected profit). Verdict quality and 5y YES
    gate soft penalties; entry readiness is a small bonus (not the main sort).
    """
    if row.skip_reason:
        return -1000.0
    base = row.base_cagr_mid if row.base_cagr_mid is not None else -50.0
    bull = row.bull_cagr_mid if row.bull_cagr_mid is not None else base
    verdict_pts = float(_VERDICT_RANK.get(row.verdict.upper(), 1)) * 2.0
    five_y_pts = 5.0 if row.five_year == "YES" else (-8.0 if row.five_year == "NO" else -2.0)
    conf = float(row.confidence or 0) * 0.3
    entry_pts = 3.0 if row.entry_ready else (1.0 if row.wait_dip else 0.0)
    # Slight bull upside weight so two similar base cases separate.
    upside = max(0.0, bull - base) * 0.15
    return round(base + verdict_pts + five_y_pts + conf + entry_pts + upside, 2)


def rank_from_verdict(
    ticker: str,
    verdict_json: dict[str, Any],
    *,
    analyzed_at: datetime | None = None,
) -> RankedAnalysis:
    er = verdict_json.get("expected_return") or {}
    if not isinstance(er, dict):
        er = {}
    five = verdict_json.get("five_year_business_test") or {}
    five_answer = str(five.get("answer") or "").strip().upper() if isinstance(five, dict) else ""
    verdict = str(verdict_json.get("verdict") or "UNKNOWN").strip().upper()
    buy = verdict_json.get("buy_zone_abs")
    buy_high = None
    if isinstance(buy, (list, tuple)) and len(buy) >= 2:
        try:
            buy_high = float(buy[1])
        except (TypeError, ValueError):
            buy_high = None

    skip: str | None = None
    if verdict in {"SKIP", "AVOID"}:
        skip = "verdict SKIP/AVOID"
    elif five_answer == "NO":
        skip = "five-year NO"

    row = RankedAnalysis(
        ticker=ticker.upper(),
        verdict=verdict,
        base_cagr_mid=_midpoint(er.get("base_cagr_range_pct")),
        bull_cagr_mid=_midpoint(er.get("bull_cagr_range_pct")),
        bear_cagr_mid=_midpoint(er.get("bear_cagr_range_pct")),
        confidence=(
            int(verdict_json["confidence"])
            if isinstance(verdict_json.get("confidence"), (int, float))
            else None
        ),
        five_year=five_answer or "—",
        thesis=str(verdict_json.get("thesis_status") or "—"),
        buy_range_allowed=verdict_json.get("buy_range_allowed") is True,
        entry_ready=_entry_ready(verdict_json),
        wait_dip=verdict == "BUY ON CORRECTION",
        risk=str(verdict_json.get("risk") or "—"),
        price=_float_or_none(verdict_json.get("current_price_abs")),
        buy_high=buy_high,
        analyzed_at=analyzed_at,
        score=0.0,
        skip_reason=skip,
    )
    return RankedAnalysis(
        **{**row.__dict__, "score": _compute_score(row)},
    )


def rank_analyses(
    rows: list[tuple[str, dict[str, Any], datetime | None]],
    *,
    mode: RankMode = "hold",
    limit: int = 18,
) -> list[RankedAnalysis]:
    ranked = [rank_from_verdict(t, v, analyzed_at=ts) for t, v, ts in rows]
    if mode == "entry":
        # Entry-first: ready names first, then wait-dip, then by score.
        ranked.sort(
            key=lambda r: (
                0 if r.entry_ready else (1 if r.wait_dip and not r.skip_reason else 2),
                -r.score,
                r.ticker,
            )
        )
    else:
        ranked.sort(key=lambda r: (-r.score, r.ticker))
    return ranked[: max(1, limit)]


def format_rank_telegram(
    ranked: list[RankedAnalysis],
    *,
    mode: RankMode = "hold",
    total_analyzed: int,
) -> str:
    title = (
        "📈 <b>Rank — long-term hold (expected base CAGR)</b>"
        if mode == "hold"
        else "🎯 <b>Rank — entry-ready first</b>"
    )
    if not ranked:
        return (
            f"{title}\n\n"
            "No stored /analyze reports yet.\n"
            "Run <code>/analyze SYMBOL</code> on /pick names, then retry <code>/rank</code>."
        )

    lines = [
        title,
        (
            f"{len(ranked)} of {total_analyzed} analyzed name(s). "
            "Score ≈ base 3y CAGR + verdict/5y gates (not a guarantee)."
        ),
        "Pick only when buy range fits your capital plan.",
        "",
    ]

    shown = 0
    for i, row in enumerate(ranked, start=1):
        if row.skip_reason and mode == "hold" and i > 5:
            continue
        shown += 1
        if shown > 18:
            break
        base = (
            f"{row.base_cagr_mid:+.1f}%"
            if row.base_cagr_mid is not None
            else "base —"
        )
        bull = (
            f"{row.bull_cagr_mid:+.1f}%"
            if row.bull_cagr_mid is not None
            else "—"
        )
        tags: list[str] = []
        if row.entry_ready:
            tags.append("entry✓")
        elif row.wait_dip:
            tags.append("wait dip")
        if row.skip_reason:
            tags.append(f"skip:{row.skip_reason}")
        tag_txt = f" · {' · '.join(tags)}" if tags else ""
        lines.append(
            f"{i}. <b>{html_escape(row.ticker)}</b> — {html_escape(row.verdict)} · "
            f"base {base} (bull {bull}) · score {row.score:.0f}{tag_txt}"
        )

    lines.extend(
        [
            "",
            (
                "<i>HEROMOTOCO-style BUY ON CORRECTION with weak base CAGR ranks below "
                "BUY names with stronger base expected return. Re-run after new /analyze.</i>"
            ),
            "Detail: <code>/analyze SYMBOL</code> · Tune list: <code>/pick</code>",
        ]
    )
    return "\n".join(lines)
