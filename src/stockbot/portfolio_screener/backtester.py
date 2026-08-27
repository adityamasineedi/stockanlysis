"""Point-in-time validation harness for historical screening selections.

Avoids look-ahead bias by requiring the caller to supply metrics / prices
as-of the screening date. Does not fetch "future" data itself beyond the
explicit forward return windows provided by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardPath:
    """Price path after a screening date. All returns are total simple returns."""

    ticker: str
    return_6m: float | None
    return_12m: float | None
    return_24m: float | None
    max_drawdown: float | None
    volatility: float | None


@dataclass(frozen=True)
class BenchmarkPath:
    name: str
    return_6m: float | None
    return_12m: float | None
    return_24m: float | None


@dataclass(frozen=True)
class BacktestReport:
    n_selected: int
    avg_return_6m: float | None
    avg_return_12m: float | None
    avg_return_24m: float | None
    avg_max_drawdown: float | None
    avg_volatility: float | None
    hit_rate_12m: float | None
    excess_vs_benchmark_12m: float | None
    sharpe_proxy_12m: float | None
    notes: tuple[str, ...]


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def compute_path_stats(prices: list[float]) -> tuple[float | None, float | None]:
    """Return (max_drawdown, annualised_vol) from a post-screen price series."""
    if len(prices) < 5:
        return None, None
    arr = np.asarray(prices, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    max_dd = float(dd.min())
    rets = np.diff(arr) / arr[:-1]
    vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else None
    return max_dd, vol


def evaluate_selection(
    selected_paths: list[ForwardPath],
    *,
    benchmark: BenchmarkPath | None = None,
) -> BacktestReport:
    """Compare historical selections to forward outcomes.

    Callers must ensure paths use only information available after the
    screening date (no look-ahead into the screening features themselves).
    """
    notes: list[str] = [
        "Validation only — screening scores are not return guarantees.",
        "Caller must supply point-in-time features as-of screen date.",
    ]
    r6 = _mean([p.return_6m for p in selected_paths])
    r12 = _mean([p.return_12m for p in selected_paths])
    r24 = _mean([p.return_24m for p in selected_paths])
    mdd = _mean([p.max_drawdown for p in selected_paths])
    vol = _mean([p.volatility for p in selected_paths])

    hits = [p.return_12m for p in selected_paths if p.return_12m is not None]
    hit_rate = (sum(1 for h in hits if h > 0) / len(hits)) if hits else None

    excess = None
    if r12 is not None and benchmark is not None and benchmark.return_12m is not None:
        excess = r12 - benchmark.return_12m
        notes.append(f"Benchmark: {benchmark.name}")

    sharpe = None
    if r12 is not None and vol is not None and vol > 0:
        sharpe = r12 / vol

    return BacktestReport(
        n_selected=len(selected_paths),
        avg_return_6m=r6,
        avg_return_12m=r12,
        avg_return_24m=r24,
        avg_max_drawdown=mdd,
        avg_volatility=vol,
        hit_rate_12m=hit_rate,
        excess_vs_benchmark_12m=excess,
        sharpe_proxy_12m=sharpe,
        notes=tuple(notes),
    )
