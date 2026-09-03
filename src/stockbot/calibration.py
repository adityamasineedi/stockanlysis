"""Did the bot's own calls work? — pure maths, no I/O.

Every card prints ``Confidence: N/10``, which is the model's self-assessment
with nothing behind it. A senior analyst's confidence is different in kind: it
is backed by a known hit rate. This module turns the prescan history the bot
has been accumulating — each row a dated verdict with the price it saw — into
that hit rate.

Three rules about honesty are enforced here rather than in the presentation
layer, because a caller that wants a flattering number should not be able to
get one by formatting differently:

1. **Small samples get no median.** Three data points have no central tendency
   worth printing.
2. **Short windows are never annualized.** A +8% move over three weeks
   annualizes to roughly +250% — noise wearing a suit.
3. **Absolute return is not skill.** +11% in a +15% market is
   underperformance, so the headline is the *spread between the bot's own
   tiers*, which is immune to market direction because every tier rode the
   same market.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

# Below this, report the count and decline to summarise.
MIN_SAMPLE = 5

# Annualizing anything shorter turns noise into an impressive-looking rate.
MIN_DAYS_TO_ANNUALIZE = 180

# A tier spread inside this band is noise between two medians, not evidence
# that the ranking works. Without it a 0.04-point gap reported success while
# the display rounded that same gap to "+0.0 points".
MIN_MATERIAL_SPREAD_PCT = 1.0

# A verdict needs time to be right or wrong about a business. Scoring a
# same-day call measures that morning's market, not the judgement — the live
# report showed "~0d" against every bucket and presented it as a track record.
MIN_DAYS_TO_SCORE = 30

# Verdicts that record "we could not assess", not "we assessed and concluded".
# They carry no opinion about a business, so they cannot be right or wrong and
# do not belong in a report about judgement quality. Routing decisions
# (SECTOR_SPECIFIC_REVIEW, REVIEW_EXCEPTION) are judgements and stay.
NON_JUDGMENT_VERDICTS = frozenset({"DATA_UNAVAILABLE_RETRY", "MODEL_NOT_APPLICABLE"})

SpreadFinding = Literal["DISCRIMINATING", "NO_DIFFERENCE", "INVERTED"]

# Calls of different ages are not the same measurement, so they are compared
# within a band rather than pooled.
AGE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("under 3 months", 0, 90),
    ("3-12 months", 90, 365),
    ("over 12 months", 365, 10**6),
)


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def forward_return_pct(entry_price: object, current_price: object) -> float | None:
    """Percent change since the call. None when either price is unusable."""
    entry = _finite_positive(entry_price)
    current = _finite_positive(current_price)
    if entry is None or current is None:
        return None
    return round((current / entry - 1.0) * 100.0, 2)


def days_since(timestamp: object, now: datetime | None = None) -> int | None:
    """Whole days since an ISO timestamp. None when unparseable."""
    if not isinstance(timestamp, str) or not timestamp.strip():
        return None
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (now or datetime.now(UTC)) - when
    return max(int(delta.days), 0)


def age_band(days: object) -> str | None:
    """Which comparison window a call belongs to."""
    if not isinstance(days, int) or isinstance(days, bool) or days < 0:
        return None
    for label, low, high in AGE_BANDS:
        if low <= days < high:
            return label
    return None


def annualized_pct(return_pct: float, days: int) -> float | None:
    """CAGR, but only where the window is long enough to mean anything.

    Returns None under MIN_DAYS_TO_ANNUALIZE rather than a number the caller
    might print. A total loss (-100%) has no defined rate, so it is refused
    too.
    """
    if days < MIN_DAYS_TO_ANNUALIZE or days <= 0:
        return None
    if return_pct <= -100.0:
        return None
    years = days / 365.0
    return round(((1.0 + return_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0, 2)


@dataclass(frozen=True)
class CalibrationBucket:
    """One group of calls — a verdict, a band, an age window."""

    label: str
    n: int
    median_return_pct: float | None
    mean_return_pct: float | None
    best_pct: float | None
    worst_pct: float | None
    median_days: int | None

    @property
    def is_reportable(self) -> bool:
        """Whether this bucket has enough data to summarise at all."""
        return self.n >= MIN_SAMPLE and self.median_return_pct is not None

    @property
    def annualized_median_pct(self) -> float | None:
        if not self.is_reportable or self.median_days is None:
            return None
        return annualized_pct(self.median_return_pct, self.median_days)


@dataclass(frozen=True)
class ScoredCall:
    """One historical call with its outcome resolved."""

    ticker: str
    label: str
    return_pct: float
    days: int

    @property
    def band(self) -> str | None:
        return age_band(self.days)


@dataclass(frozen=True)
class ScoringResult:
    """Scored calls, plus why the rest were left out.

    The counts are not bookkeeping: a report that silently shrinks its sample
    looks as confident on twelve rows as on twelve hundred. Surfacing them lets
    the reader see that most history is pending rather than damning.
    """

    calls: list[ScoredCall]
    too_recent: int
    not_a_judgment: int
    unscoreable: int


def score_calls(
    rows: list[dict],
    current_prices: dict[str, float | None],
    *,
    label_key: str = "verdict",
    entry_key: str = "price_at_scan",
    time_key: str = "logged_at",
    now: datetime | None = None,
) -> ScoringResult:
    """Resolve each logged call into a return, saying what it had to leave out.

    Rows are dropped, never counted as flat: a call with an unknown outcome is
    not a zero-return call, and treating it as one would quietly drag every
    median toward zero. Three separate reasons, counted separately:

    - **not a judgement** — the verdict records that the bot could not assess
      (see NON_JUDGMENT_VERDICTS); it cannot be right or wrong.
    - **too recent** — younger than MIN_DAYS_TO_SCORE, so the number measures
      this week's market rather than the verdict.
    - **unscoreable** — no entry price, no usable timestamp, or no current
      price for that ticker.
    """
    scored: list[ScoredCall] = []
    too_recent = 0
    not_a_judgment = 0
    unscoreable = 0

    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        label = str(row.get(label_key) or "").strip()
        if not ticker or not label:
            unscoreable += 1
            continue
        if label.upper() in NON_JUDGMENT_VERDICTS:
            not_a_judgment += 1
            continue
        days = days_since(row.get(time_key), now)
        if days is None:
            unscoreable += 1
            continue
        change = forward_return_pct(row.get(entry_key), current_prices.get(ticker))
        if change is None:
            unscoreable += 1
            continue
        # Age is checked last so a too-recent row is reported as pending rather
        # than lumped in with rows that can never be scored at all.
        if days < MIN_DAYS_TO_SCORE:
            too_recent += 1
            continue
        scored.append(ScoredCall(ticker=ticker, label=label, return_pct=change, days=days))

    return ScoringResult(
        calls=scored,
        too_recent=too_recent,
        not_a_judgment=not_a_judgment,
        unscoreable=unscoreable,
    )


def build_bucket(label: str, calls: list[ScoredCall]) -> CalibrationBucket:
    """Summarise a group, withholding statistics below MIN_SAMPLE."""
    n = len(calls)
    if n == 0:
        return CalibrationBucket(label, 0, None, None, None, None, None)
    returns = [c.return_pct for c in calls]
    if n < MIN_SAMPLE:
        # Count is a fact; a median of three is not.
        return CalibrationBucket(label, n, None, None, None, None, None)
    return CalibrationBucket(
        label=label,
        n=n,
        median_return_pct=round(statistics.median(returns), 2),
        mean_return_pct=round(statistics.fmean(returns), 2),
        best_pct=round(max(returns), 2),
        worst_pct=round(min(returns), 2),
        median_days=int(statistics.median([c.days for c in calls])),
    )


def bucket_by_label(calls: list[ScoredCall]) -> list[CalibrationBucket]:
    """One bucket per verdict/band, largest sample first."""
    groups: dict[str, list[ScoredCall]] = {}
    for call in calls:
        groups.setdefault(call.label, []).append(call)
    buckets = [build_bucket(label, group) for label, group in groups.items()]
    buckets.sort(key=lambda b: (-b.n, b.label))
    return buckets


def bucket_by_age(calls: list[ScoredCall]) -> list[CalibrationBucket]:
    """Buckets per age window, in chronological band order."""
    groups: dict[str, list[ScoredCall]] = {}
    for call in calls:
        band = call.band
        if band is not None:
            groups.setdefault(band, []).append(call)
    order = [label for label, _, _ in AGE_BANDS]
    return [build_bucket(label, groups[label]) for label in order if label in groups]


@dataclass(frozen=True)
class TierSpread:
    """The headline: do the calls the bot liked beat the ones it rejected?"""

    positive_label: str
    negative_label: str
    positive_median_pct: float
    negative_median_pct: float
    n_positive: int
    n_negative: int

    @property
    def spread_pct(self) -> float:
        return round(self.positive_median_pct - self.negative_median_pct, 2)

    @property
    def finding(self) -> SpreadFinding:
        """Three states, because "positive" is not the same as "meaningful".

        A bare ``spread_pct > 0`` called a 0.04-point gap success, and the
        renderer's ``:+.1f`` printed that same gap as "+0.0 points" — the
        sentence contradicted its own number. Anything inside
        ±MIN_MATERIAL_SPREAD_PCT is noise between two medians, not evidence of
        ranking skill.
        """
        if self.spread_pct >= MIN_MATERIAL_SPREAD_PCT:
            return "DISCRIMINATING"
        if self.spread_pct <= -MIN_MATERIAL_SPREAD_PCT:
            return "INVERTED"
        return "NO_DIFFERENCE"

    @property
    def discriminates(self) -> bool:
        return self.finding == "DISCRIMINATING"


def tier_spread(
    calls: list[ScoredCall],
    positive_labels: set[str],
    negative_labels: set[str],
) -> TierSpread | None:
    """Median of liked calls minus median of rejected ones, same window.

    This is the headline rather than absolute return because it needs no
    benchmark and survives any market: both sides rode the same one. None when
    either side is under MIN_SAMPLE — a spread computed off three rejects is
    not evidence of anything.

    A spread at or below zero is a real, reportable finding: the score is not
    discriminating. Callers must not suppress it.
    """
    positive = [c for c in calls if c.label in positive_labels]
    negative = [c for c in calls if c.label in negative_labels]
    if len(positive) < MIN_SAMPLE or len(negative) < MIN_SAMPLE:
        return None
    return TierSpread(
        positive_label="/".join(sorted(positive_labels)),
        negative_label="/".join(sorted(negative_labels)),
        positive_median_pct=round(statistics.median([c.return_pct for c in positive]), 2),
        negative_median_pct=round(statistics.median([c.return_pct for c in negative]), 2),
        n_positive=len(positive),
        n_negative=len(negative),
    )
