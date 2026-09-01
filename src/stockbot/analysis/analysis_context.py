"""Peer fundamentals, sector scorecards, and portfolio execution context for /analyze.

Deterministic enrichment — no LLM. Keeps single-stock analysis aligned with
portfolio screener lenses without running the full prescan universe pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

import pandas as pd

from stockbot.config import WATCHLIST_PATH
from stockbot.models import (
    Brief,
    BriefMetadata,
    Financials,
    PeerRow,
    PeerSnapshot,
    PortfolioExecutionContext,
    PrescanSummary,
    ReportText,
    SectorScorecardContext,
)
from stockbot.portfolio_screener.metrics import fetch_market_metadata
from stockbot.portfolio_screener.scoring_config import DEFAULT_SECTOR_BENCHMARKS
from stockbot.portfolio_screener.score_utils import percentile_rank
from stockbot.portfolio_sip_schema import load_portfolio_sip_config
from stockbot.portfolio_state import DEFAULT_MAX_POSITION_PCT, TRANCHE_COUNT

logger = logging.getLogger(__name__)

MAX_PEER_METADATA_FETCHES = 15
_AR_SNIPPET_CHARS = 280

_SCORECARD_LENSES: dict[str, str] = {
    "BANK": "Bank scorecard — NIM, GNPA, PCR, CAR, P/B (not OCF/PAT).",
    "NBFC_HFC": "NBFC / HFC scorecard — NIM, GNPA, leverage, ALM (not OCF/PAT).",
    "INSURER": "Insurer scorecard — combined ratio, solvency, embedded value (not OCF/PAT).",
    "RATING_ANALYTICS": "Rating / analytics — fee growth, margins, ROE.",
    "MARKET_INFRA": "Market infrastructure — volumes, pricing, ROE.",
    "FINTECH_PLATFORM": "Fintech platform — TPV, unit economics, burn.",
    "CONGLOMERATE": "Conglomerate — SOTP / segment ROE; generic quant is weak alone.",
    "UTILITY": "Utility — leverage and regulated ROE; cyclical cash cliffs are softer.",
    "DEFENCE_EPC_PROJECT": "Defence EPC — order book + WC reconciliation before cash gates.",
    "EPC_PROJECT_BUSINESS": "EPC / project — WC billing cycle and backlog quality.",
}

_RATIO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "NIM": ("nim", "net interest margin"),
    "GNPA": ("gnpa", "gross npa", "gross non-performing"),
    "Net NPA": ("net npa", "nnpa"),
    "PCR": ("provision coverage", "pcr"),
    "CAR": ("capital adequacy", "car"),
    "P/B": ("price to book", "p/b", "pb ratio"),
    "Combined ratio": ("combined ratio", "claims ratio"),
    "Solvency": ("solvency ratio", "solvency margin"),
    "ROE": ("roe", "return on equity"),
    "ROCE": ("roce", "return on capital"),
    "Debt/Equity": ("debt to equity", "debt/equity", "d/e"),
    "Current ratio": ("current ratio"),
}

_AR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "combined ratio": ("combined ratio", "claims ratio", "loss ratio"),
    "solvency": ("solvency ratio", "solvency margin", "solvency level"),
    "gnpa": ("gross npa", "gnpa", "non-performing asset"),
    "nim": ("net interest margin", "nim "),
    "embedded value": ("embedded value", "ev per share"),
    "capital adequacy": ("capital adequacy", "car "),
}


def _load_universe_symbols() -> set[str]:
    symbols: set[str] = set()
    if WATCHLIST_PATH.is_file():
        for line in WATCHLIST_PATH.read_text(encoding="utf-8").splitlines():
            token = line.strip().upper()
            if token and not token.startswith("#"):
                symbols.add(token)
    try:
        cfg = load_portfolio_sip_config()
        for bucket in cfg.portfolios:
            for sym in bucket.symbols:
                if sym.enabled:
                    symbols.add(sym.symbol.upper())
    except Exception as exc:  # noqa: BLE001 — optional portfolio file
        logger.info("sip portfolio symbols unavailable for peer universe: %s", exc)
    return symbols


def _sector_benchmark_pe(sector: str | None) -> float | None:
    if not sector:
        return None
    bench = DEFAULT_SECTOR_BENCHMARKS.get(sector)
    if bench is None:
        bench = DEFAULT_SECTOR_BENCHMARKS.get("Unknown")
    return bench.pe_fair if bench else None


def build_peer_snapshot(
    symbol: str,
    metadata: BriefMetadata | None,
) -> PeerSnapshot | None:
    if metadata is None:
        return None
    sector = metadata.sector
    target_pe = metadata.pe_price_eps or metadata.ttm_pe
    target_meta = fetch_market_metadata(symbol.upper())
    target_roe = target_meta.get("roe_pct")
    universe = _load_universe_symbols()
    universe.discard(symbol.upper())

    peer_rows: list[PeerRow] = []
    if sector and universe:
        candidates = sorted(universe)[:MAX_PEER_METADATA_FETCHES * 2]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_market_metadata, sym): sym for sym in candidates}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    meta = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.info("peer metadata failed for %s: %s", sym, exc)
                    continue
                if meta.get("sector") != sector:
                    continue
                pe = meta.get("trailing_pe")
                if isinstance(pe, (int, float)) and pe > 0:
                    peer_rows.append(
                        PeerRow(
                            symbol=sym,
                            pe=round(float(pe), 2),
                            roe_pct=(
                                round(float(meta["roe_pct"]), 2)
                                if isinstance(meta.get("roe_pct"), (int, float))
                                else None
                            ),
                            market_cap_cr=(
                                round(float(meta["market_cap_cr"]), 2)
                                if isinstance(meta.get("market_cap_cr"), (int, float))
                                else None
                            ),
                        )
                    )
                if len(peer_rows) >= MAX_PEER_METADATA_FETCHES:
                    break

    peer_pes = [p.pe for p in peer_rows if p.pe is not None]
    peer_median = round(median(peer_pes), 2) if peer_pes else None
    pe_pct = percentile_rank(float(target_pe), peer_pes) if target_pe is not None else None
    bench_pe = _sector_benchmark_pe(sector)

    note: str | None = None
    if not peer_pes:
        note = (
            "No same-sector peers with P/E found in watchlist/SIP universe — "
            "use sector benchmark reference only."
        )
    elif len(peer_pes) < 3:
        note = f"Thin peer set ({len(peer_pes)} names) — treat percentile as indicative."

    return PeerSnapshot(
        target_symbol=symbol.upper(),
        sector=sector,
        target_pe=round(float(target_pe), 2) if target_pe is not None else None,
        target_roe_pct=(
            round(float(target_roe), 2) if isinstance(target_roe, (int, float)) else None
        ),
        peer_median_pe=peer_median,
        peer_count=len(peer_pes),
        pe_percentile=pe_pct,
        sector_benchmark_pe_fair=bench_pe,
        peers=tuple(peer_rows),
        note=note,
    )


def _latest_ratio_value(financials: Financials, row_label: str) -> str | None:
    if row_label not in financials.ratios.index:
        return None
    row = financials.ratios.loc[row_label]
    for val in reversed(row.tolist()):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            continue
    return None


def _scan_ratios_for_scorecard(financials: Financials | None) -> list[tuple[str, str]]:
    if financials is None or financials.ratios.empty:
        return []
    found: list[tuple[str, str]] = []
    index_blob = " ".join(str(i).lower() for i in financials.ratios.index)
    for label, keywords in _RATIO_KEYWORDS.items():
        if any(k in index_blob for k in keywords):
            for row_name in financials.ratios.index:
                row_lower = str(row_name).lower()
                if any(k in row_lower for k in keywords):
                    value = _latest_ratio_value(financials, str(row_name))
                    if value is not None:
                        found.append((label, value))
                    break
    return found


def _ar_snippets(report: ReportText, keywords: tuple[str, ...]) -> list[str]:
    snippets: list[str] = []
    for heading, text in report.sections.items():
        lowered = text.lower()
        for kw in keywords:
            if kw in lowered:
                match = re.search(re.escape(kw), text, flags=re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 80)
                    snippet = text[start : start + _AR_SNIPPET_CHARS].replace("\n", " ").strip()
                    snippets.append(f"[{heading}] …{snippet}…")
                break
        if len(snippets) >= 4:
            break
    return snippets


def build_sector_scorecard(
    brief: Brief,
    prescan_summary: PrescanSummary | None,
) -> SectorScorecardContext | None:
    issuer = prescan_summary.issuer_class if prescan_summary else None
    lens = _SCORECARD_LENSES.get(
        issuer or "",
        "Standard equity — ROE/ROCE, margins, FCF, leverage, growth vs history.",
    )
    supplied = _scan_ratios_for_scorecard(brief.financials)
    ar_keys = tuple(k for keys in _AR_KEYWORDS.values() for k in keys)
    ar_snippets = tuple(_ar_snippets(brief.annual_report, ar_keys))

    generic_note: str | None = None
    if prescan_summary and prescan_summary.quant_score is not None:
        if issuer and issuer not in {"NON_FINANCIAL", "OTHER", "AUTO_OEM", "LOSS_MAKING_GROWTH"}:
            generic_note = (
                f"Generic prescan quant ({prescan_summary.quant_score:.1f}/100) is not decisive "
                f"for issuer_class={issuer} — lead with the sector scorecard."
            )

    if not supplied and not ar_snippets and generic_note is None:
        return None

    return SectorScorecardContext(
        issuer_class=issuer,
        scorecard_lens=lens,
        supplied_metrics=tuple(supplied),
        ar_snippets=ar_snippets,
        generic_quant_note=generic_note,
    )


def build_portfolio_execution(
    symbol: str,
    metadata: BriefMetadata | None,
) -> PortfolioExecutionContext:
    sym = symbol.upper()
    bucket_name: str | None = None
    monthly_inr: float | None = None
    same_sector: int | None = None
    diversification: str | None = None
    in_sip = False

    try:
        cfg = load_portfolio_sip_config()
        sector = metadata.sector if metadata else None
        for bucket in cfg.portfolios:
            bucket_symbols = [s for s in bucket.symbols if s.enabled]
            if any(s.symbol.upper() == sym for s in bucket_symbols):
                in_sip = True
                bucket_name = bucket.label
                for s in bucket_symbols:
                    if s.symbol.upper() == sym and s.target_amount_monthly > 0:
                        monthly_inr = s.target_amount_monthly
                if sector:
                    sector_count = 0
                    for s in bucket_symbols:
                        if s.symbol.upper() == sym:
                            continue
                        meta = fetch_market_metadata(s.symbol)
                        if meta.get("sector") == sector:
                            sector_count += 1
                    same_sector = sector_count
                    if sector_count >= 2:
                        diversification = (
                            f"{sector_count} other names in bucket '{bucket.label}' "
                            f"share sector {sector} — watch concentration."
                        )
                break
    except Exception as exc:  # noqa: BLE001
        logger.info("portfolio execution context skipped: %s", exc)

    tranche_inr = (
        round(monthly_inr / TRANCHE_COUNT, 2) if monthly_inr is not None else None
    )

    return PortfolioExecutionContext(
        in_sip_portfolio=in_sip,
        sip_bucket=bucket_name,
        suggested_monthly_inr=monthly_inr,
        suggested_tranche_inr=tranche_inr,
        max_position_pct=DEFAULT_MAX_POSITION_PCT,
        same_sector_count_in_bucket=same_sector,
        review_cadence=(
            "Quarterly results + semiannual portfolio review; "
            "revisit thesis on governance/accounting flags."
        ),
        delivery_note=(
            "Delivery-only positioning — no intraday execution guidance. "
            "You place orders; bot supplies research, ranges, and gates."
        ),
        diversification_note=diversification,
    )


def format_peer_snapshot_json(peer: PeerSnapshot | None) -> str:
    if peer is None:
        return "MISSING: peer snapshot not built"
    payload: dict[str, object] = {
        "target_symbol": peer.target_symbol,
        "sector": peer.sector,
        "target_pe": peer.target_pe,
        "target_roe_pct": peer.target_roe_pct,
        "peer_median_pe": peer.peer_median_pe,
        "peer_count": peer.peer_count,
        "pe_percentile": peer.pe_percentile,
        "sector_benchmark_pe_fair": peer.sector_benchmark_pe_fair,
        "note": peer.note,
        "peers": [
            {
                "symbol": row.symbol,
                "pe": row.pe,
                "roe_pct": row.roe_pct,
                "market_cap_cr": row.market_cap_cr,
            }
            for row in peer.peers
        ],
    }
    return json.dumps(payload, indent=2)


def format_sector_scorecard_json(scorecard: SectorScorecardContext | None) -> str:
    if scorecard is None:
        return "MISSING: sector scorecard not built"
    payload = {
        "issuer_class": scorecard.issuer_class,
        "scorecard_lens": scorecard.scorecard_lens,
        "supplied_metrics": {k: v for k, v in scorecard.supplied_metrics},
        "ar_snippets": list(scorecard.ar_snippets),
        "generic_quant_note": scorecard.generic_quant_note,
    }
    return json.dumps(payload, indent=2)


def format_portfolio_execution_json(ctx: PortfolioExecutionContext | None) -> str:
    if ctx is None:
        return "MISSING: portfolio execution context not built"
    payload = {
        "in_sip_portfolio": ctx.in_sip_portfolio,
        "sip_bucket": ctx.sip_bucket,
        "suggested_monthly_inr": ctx.suggested_monthly_inr,
        "suggested_tranche_inr": ctx.suggested_tranche_inr,
        "max_position_pct": ctx.max_position_pct,
        "same_sector_count_in_bucket": ctx.same_sector_count_in_bucket,
        "review_cadence": ctx.review_cadence,
        "delivery_note": ctx.delivery_note,
        "diversification_note": ctx.diversification_note,
    }
    return json.dumps(payload, indent=2)


def execution_pm_for_verdict(
    peer: PeerSnapshot | None,
    scorecard: SectorScorecardContext | None,
    portfolio: PortfolioExecutionContext | None,
    technicals_trend: str | None,
    technicals_bb: str | None,
) -> dict[str, object]:
    """Compact dict stored on verdict_json for Telegram cards."""
    out: dict[str, object] = {}
    if peer and peer.pe_percentile is not None:
        out["peer_pe_percentile"] = round(peer.pe_percentile, 1)
        out["peer_count"] = peer.peer_count
    if scorecard and scorecard.issuer_class:
        out["sector_scorecard_issuer"] = scorecard.issuer_class
    if portfolio:
        out["sip_bucket"] = portfolio.sip_bucket
        out["suggested_tranche_inr"] = portfolio.suggested_tranche_inr
        out["max_position_pct"] = portfolio.max_position_pct
        out["review_cadence"] = portfolio.review_cadence
        if portfolio.diversification_note:
            out["diversification_note"] = portfolio.diversification_note
    if technicals_bb:
        out["bollinger_position"] = technicals_bb
    if technicals_trend:
        out["trend_label"] = technicals_trend
    return out
