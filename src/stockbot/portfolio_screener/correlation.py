"""Pairwise return correlation clustering for diversification awareness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.scoring_config import PortfolioConstraints


@dataclass(frozen=True)
class CorrelationInfo:
    ticker: str
    correlation_risk: str  # LOW / MEDIUM / HIGH
    correlation_cluster: str | None
    max_peer_correlation: float | None
    high_corr_peers: tuple[str, ...]


def _aligned_corr(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 60:
        return None
    x = np.asarray(a[-n:], dtype=float)
    y = np.asarray(b[-n:], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_correlation_infos(
    universe: list[StockMetrics],
    constraints: PortfolioConstraints | None = None,
) -> dict[str, CorrelationInfo]:
    constraints = constraints or PortfolioConstraints()
    threshold = constraints.correlation_cluster_threshold

    by_ticker = {m.ticker: m for m in universe}
    tickers = [m.ticker for m in universe if m.price_returns and len(m.price_returns) >= 60]

    # Union-find style clustering for pairs above threshold
    parent: dict[str, str] = {t: t for t in tickers}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_corr: dict[tuple[str, str], float] = {}
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1 :]:
            corr = _aligned_corr(
                by_ticker[t1].price_returns or [],
                by_ticker[t2].price_returns or [],
            )
            if corr is None:
                continue
            pair_corr[(t1, t2)] = corr
            if corr >= threshold:
                union(t1, t2)

    # Assign cluster labels
    cluster_members: dict[str, list[str]] = {}
    for t in tickers:
        root = find(t)
        cluster_members.setdefault(root, []).append(t)

    cluster_id: dict[str, str] = {}
    for idx, (_root, members) in enumerate(sorted(cluster_members.items()), start=1):
        if len(members) < 2:
            continue
        label = f"C{idx}"
        for m in members:
            cluster_id[m] = label

    results: dict[str, CorrelationInfo] = {}
    for m in universe:
        t = m.ticker
        if t not in tickers:
            results[t] = CorrelationInfo(
                ticker=t,
                correlation_risk="LOW",
                correlation_cluster=None,
                max_peer_correlation=None,
                high_corr_peers=(),
            )
            continue

        peers: list[tuple[str, float]] = []
        for (a, b), corr in pair_corr.items():
            if a == t:
                peers.append((b, corr))
            elif b == t:
                peers.append((a, corr))
        max_corr = max((c for _, c in peers), default=None)
        high_peers = tuple(sorted(p for p, c in peers if c >= threshold))

        if max_corr is None:
            risk = "LOW"
        elif max_corr >= threshold:
            risk = "HIGH"
        elif max_corr >= 0.65:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        results[t] = CorrelationInfo(
            ticker=t,
            correlation_risk=risk,
            correlation_cluster=cluster_id.get(t),
            max_peer_correlation=max_corr,
            high_corr_peers=high_peers,
        )
    return results
