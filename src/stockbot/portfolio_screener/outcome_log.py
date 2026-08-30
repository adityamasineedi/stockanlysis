"""Append-only pre-scan outcome log for fetch/hard-filter monitoring.

Tracks how often DATA_INSUFFICIENT vs weak-fundamentals rejects occur after
ratio compute fallbacks (e.g. BBOX-style missing Screener ROE %).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any

from stockbot.config import PORTFOLIO_DIR

logger = logging.getLogger(__name__)

OUTCOMES_PATH = PORTFOLIO_DIR / "prescan_outcomes.jsonl"

_USER_LABELS = {
    "roe": "ROE",
    "roce": "ROCE",
    "debt_equity": "D/E",
    "ocf_to_pat": "OCF/PAT",
    "interest_coverage": "Interest coverage",
    "net_debt_ebitda": "Net debt/EBITDA",
    "current_ratio": "Current ratio",
}

# Only surface high-stakes derived ratios to users (not every margin/turnover).
_TELEGRAM_FIELDS = frozenset(_USER_LABELS)

_SOURCE_HINT = {
    "computed": "derived from statements — cross-check Screener.in",
    "yfinance": "from Yahoo Finance — cross-check Screener.in",
}

# Always statement-derived (never a Screener "ratios" row) — softer wording.
_ALWAYS_DERIVED = frozenset({"ocf_to_pat", "net_debt_ebitda", "interest_coverage", "debt_equity"})
_ALWAYS_DERIVED_HINT = "derived from P&L/BS/CF (not a Screener ratio row)"


def format_computed_metric_warnings(
    metric_sources: dict[str, str],
    metrics_values: dict[str, float | None] | None = None,
) -> list[str]:
    """Human-readable lines for Telegram when a metric was not Screener-fetched."""
    lines: list[str] = []
    values = metrics_values or {}
    for field, source in sorted(metric_sources.items()):
        if source not in ("computed", "yfinance"):
            continue
        if field not in _TELEGRAM_FIELDS:
            continue
        label = _USER_LABELS[field]
        if field in _ALWAYS_DERIVED and source == "computed":
            hint = _ALWAYS_DERIVED_HINT
        else:
            hint = _SOURCE_HINT[source]
        raw = values.get(field)
        if isinstance(raw, (int, float)):
            if field in ("roe", "roce"):
                shown = f"{raw:.1f}%"
            else:
                shown = f"{raw:.2f}"
            lines.append(f"{label} {shown} — {hint}")
        else:
            lines.append(f"{label} — {hint}")
    return lines


def log_prescan_outcome(record: dict[str, Any]) -> None:
    """Structured log + JSONL append for later aggregation."""
    payload = {
        **record,
        "logged_at": datetime.now(UTC).isoformat(),
    }
    logger.info(
        "prescan_outcome ticker=%s verdict=%s hard=%s quant=%s band=%s "
        "reject_class=%s computed=%s missing_key=%s",
        payload.get("ticker"),
        payload.get("verdict"),
        payload.get("hard_filter_status"),
        payload.get("quant_score"),
        payload.get("candidate_band"),
        payload.get("reject_class"),
        payload.get("computed_metrics"),
        payload.get("missing_key_trio"),
    )
    try:
        PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
        with OUTCOMES_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Failed to append pre-scan outcome to %s", OUTCOMES_PATH)


def classify_reject(
    *,
    hard_status: str | None,
    verdict: str,
    quant_score: float | None,
) -> str:
    """Bucket for monitoring: data_gap vs weak_fundamentals vs other."""
    if hard_status == "DATA_UNAVAILABLE" or verdict in {
        "DATA_UNAVAILABLE",
        "DATA_UNAVAILABLE_RETRY",
    }:
        return "data_unavailable"
    if verdict in {"MODEL_NOT_APPLICABLE", "SECTOR_SPECIFIC_REVIEW"}:
        return "sector_specific_review"
    if verdict in {"REVIEW_EXCEPTION"}:
        return "review_exception"
    if verdict == "HOLDING_MONITOR_ONLY":
        return "holding_monitor"
    if hard_status == "DATA_INSUFFICIENT":
        return "data_gap"
    if hard_status == "HARD_EXCLUDE":
        return "hard_exclude"
    if verdict in {"NOT_SUITABLE", "NOT_SUITABLE_FOR_3Y_RESEARCH"} and (
        quant_score is not None and quant_score < 60
    ):
        return "weak_fundamentals"
    if verdict in {"MARGINAL", "SECTOR_SPECIFIC_REVIEW"}:
        return "sector_specific_review"
    if verdict in {"SUITABLE_FOR_DEEP_ANALYSIS", "AUTO_DEEP_ANALYSIS"}:
        return "auto_deep"
    return "other"


def load_prescan_outcomes(
    path: Path | None = None,
    *,
    limit: int | None = None,
    latest_per_ticker: bool = True,
) -> list[dict[str, Any]]:
    """Load JSONL rows; optionally keep only the latest row per ticker."""
    target = path or OUTCOMES_PATH
    if not target.exists():
        return []
    raw = target.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    if limit is not None:
        lines = lines[-limit:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(normalize_prescan_row(json.loads(line)))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid prescan JSONL line in %s", target)
    if not latest_per_ticker:
        return rows
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        prev = latest.get(ticker)
        if prev is None or str(row.get("logged_at") or "") >= str(prev.get("logged_at") or ""):
            latest[ticker] = row
    return sorted(latest.values(), key=lambda r: str(r.get("ticker") or ""))


def _row_has_qgs(row: dict[str, Any]) -> bool:
    return all(
        isinstance(row.get(key), (int, float))
        for key in ("quality_score", "growth_score", "strength_score")
    )


def normalize_prescan_row(row: dict[str, Any]) -> dict[str, Any]:
    """Unify legacy JSONL field names."""
    out = dict(row)
    if out.get("strength_score") is None and isinstance(
        out.get("financial_strength_score"), (int, float)
    ):
        out["strength_score"] = out["financial_strength_score"]
    return out


def backfill_row_qgs(
    row: dict[str, Any],
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """Recompute Quality/Growth/Strength when older prescan logs lack pillar scores."""
    row = normalize_prescan_row(row)
    if _row_has_qgs(row):
        return row

    ticker = str(row.get("ticker") or "").strip().upper()
    if not ticker:
        return row

    from stockbot.fetch.tickers import AmbiguousMatch, load_symbol_table, resolve_ticker
    from stockbot.portfolio_screener.data_loader import fetch_universe_metrics
    from stockbot.portfolio_screener.quant_engine import compute_quant_score
    from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig

    resolved = resolve_ticker(ticker, load_symbol_table())
    if resolved is None or isinstance(resolved, AmbiguousMatch):
        return row

    metrics = fetch_universe_metrics([resolved])
    if not metrics:
        return row

    quant = compute_quant_score(metrics[0], ScreenerRunConfig())
    enriched = {
        **row,
        "quality_score": round(quant.components.business_quality, 1),
        "growth_score": round(quant.components.growth, 1),
        "strength_score": round(quant.components.financial_strength, 1),
    }
    if persist:
        log_prescan_outcome(
            {key: value for key, value in enriched.items() if key != "logged_at"}
        )
        logger.info(
            "backfilled prescan Q/G/S for %s (Q=%.1f G=%.1f S=%.1f)",
            ticker,
            enriched["quality_score"],
            enriched["growth_score"],
            enriched["strength_score"],
        )
    return enriched


def backfill_rows_qgs(
    rows: list[dict[str, Any]],
    *,
    persist: bool = True,
) -> list[dict[str, Any]]:
    return [backfill_row_qgs(row, persist=persist) for row in rows]


def query_prescan_outcomes(
    rows: list[dict[str, Any]],
    *,
    min_quality: float | None = None,
    min_growth: float | None = None,
    min_strength: float | None = None,
    min_quant: float | None = None,
    bands: set[str] | None = None,
    verdicts: set[str] | None = None,
    cash_statuses: set[str] | None = None,
    exclude_hard_exclude: bool = True,
    analyze_ready_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter prescan rows using quality-first keys (Q/G/S + quant + gates)."""
    analyze_verdicts = {"AUTO_DEEP_ANALYSIS", "SECTOR_SPECIFIC_REVIEW"}
    ok_cash = {"PASS", "WATCH", "NOT_APPLICABLE"}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if min_quality is not None:
            q = row.get("quality_score")
            if not isinstance(q, (int, float)) or q < min_quality:
                continue
        if min_growth is not None:
            g = row.get("growth_score")
            if not isinstance(g, (int, float)) or g < min_growth:
                continue
        if min_strength is not None:
            s = row.get("strength_score")
            if not isinstance(s, (int, float)) or s < min_strength:
                continue
        if min_quant is not None:
            quant = row.get("quant_score")
            if not isinstance(quant, (int, float)) or quant < min_quant:
                continue
        if bands is not None and str(row.get("candidate_band") or "") not in bands:
            continue
        verdict = str(row.get("verdict") or "")
        if verdicts is not None and verdict not in verdicts:
            continue
        if analyze_ready_only and verdict not in analyze_verdicts:
            continue
        cash = str(row.get("cash_conversion_status") or "")
        if cash_statuses is not None and cash not in cash_statuses:
            continue
        if analyze_ready_only and cash not in ok_cash:
            continue
        if exclude_hard_exclude and str(row.get("hard_filter_status") or "") == "HARD_EXCLUDE":
            continue
        filtered.append(row)
    # Sort by Overall (quant_score) first — that's the number shown first and
    # bold on each line, so the visible order must match it or the list reads
    # as unsorted (e.g. a Quality-first sort put an Overall-82 name below an
    # Overall-56 one). Quality stays as the tiebreaker.
    filtered.sort(
        key=lambda r: (
            -(float(r["quant_score"]) if isinstance(r.get("quant_score"), (int, float)) else -1),
            -(float(r["quality_score"]) if isinstance(r.get("quality_score"), (int, float)) else -1),
            str(r.get("ticker") or ""),
        ),
    )
    return filtered


def format_prescan_table(rows: list[dict[str, Any]]) -> str:
    """Human-readable table for CLI / ops."""
    if not rows:
        return "(no rows)"
    headers = [
        "Ticker",
        "Q",
        "G",
        "S",
        "Quant",
        "Band",
        "Cash",
        "Verdict",
    ]

    def _cell(row: dict[str, Any], key: str) -> str:
        val = row.get(key)
        if isinstance(val, float):
            return f"{val:.1f}" if key.endswith("_score") else f"{val:.2f}"
        return str(val or "")

    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("ticker") or ""),
                    _cell(row, "quality_score") if row.get("quality_score") is not None else "—",
                    _cell(row, "growth_score") if row.get("growth_score") is not None else "—",
                    _cell(row, "strength_score") if row.get("strength_score") is not None else "—",
                    _cell(row, "quant_score"),
                    str(row.get("candidate_band") or ""),
                    str(row.get("cash_conversion_status") or ""),
                    str(row.get("verdict") or ""),
                ]
            )
        )
    return "\n".join(lines)


def pull_prescan_from_railway(
    *,
    service: str = "stockanlysis",
    remote_path: str = "/app/data/portfolio/prescan_outcomes.jsonl",
    dest: Path | None = None,
) -> Path:
    """Fetch prescan JSONL from a Railway deployment via `railway ssh`."""
    import shutil
    import subprocess

    dest = dest or OUTCOMES_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    railway_bin = shutil.which("railway") or shutil.which("railway.exe")
    if railway_bin is None:
        raise RuntimeError(
            "Railway CLI not found on PATH. Install it or pull manually: "
            "railway ssh -s stockanlysis -- cat /app/data/portfolio/prescan_outcomes.jsonl"
        )
    cmd = [railway_bin, "ssh", "-s", service, "--", "cat", remote_path]
    logger.info("Pulling prescan outcomes from Railway service=%s path=%s", service, remote_path)
    if railway_bin.lower().endswith(".cmd"):
        cmd_str = (
            f'"{railway_bin}" ssh -s {service} -- cat "{remote_path}"'
        )
        proc = subprocess.run(cmd_str, capture_output=True, text=True, check=False, shell=True)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"railway ssh failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    body = proc.stdout
    if not body.strip():
        raise RuntimeError(f"Remote prescan log empty at {remote_path}")
    dest.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
    logger.info("Wrote %s rows to %s", len(body.splitlines()), dest)
    return dest


def summarize_outcomes(path: Path | None = None, *, limit: int = 500) -> dict[str, Any]:
    """Read recent JSONL rows and count reject classes (ops helper)."""
    target = path or OUTCOMES_PATH
    counts: dict[str, int] = {}
    n = 0
    if not target.exists():
        return {"rows": 0, "by_reject_class": counts, "path": str(target)}
    lines = target.read_text(encoding="utf-8").splitlines()[-limit:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        key = str(row.get("reject_class") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"rows": n, "by_reject_class": counts, "path": str(target)}


BAND_ALIASES: dict[str, str] = {
    "strong": "STRONG_CANDIDATE",
    "candidate": "CANDIDATE",
    "candidates": "CANDIDATE",
    "watchlist": "WATCHLIST",
    "watch": "WATCHLIST",
}

from stockbot.portfolio_screener.prescan_display import (
    BAND_ICONS,
    BAND_LABELS,
    CASH_ICONS,
    CASH_LABELS,
    VERDICT_ICONS,
    VERDICT_LABELS,
    format_qgs_from_row,
)

CANDIDATES_USAGE = (
    "Usage:\n"
    "/candidates — all analyze-ready names from your /prescan history\n"
    "/candidates strong — top tier (overall score 80+)\n"
    "/candidates candidate — good picks (score 70–79)\n"
    "/candidates watchlist — watchlist (score 60–69)\n"
    "/candidates quality 65 — Quality score ≥65 and analyze-ready\n"
    "/candidates all — every logged prescan (latest per symbol)\n\n"
    "Each line shows: tier · overall score · Quality/Growth/Strength · "
    "cash-flow gate · whether /analyze is OK.\n"
    "Run <code>/prescan SYMBOL</code> first to build the list."
)

TELEGRAM_PSCAN_CHUNK = 3800


@dataclass(frozen=True)
class CandidatesFilter:
    bands: set[str] | None = None
    min_quality: float | None = None
    analyze_ready_only: bool = True
    label: str = "Analyze-ready"


def parse_candidates_filter(args: list[str]) -> CandidatesFilter | str:
    """Parse /candidates args. Returns usage text when input is invalid."""
    if not args:
        return CandidatesFilter(
            analyze_ready_only=True,
            label="Analyze-ready (all bands)",
        )

    lowered = [a.lower() for a in args]
    if lowered[0] in {"help", "?"}:
        return CANDIDATES_USAGE

    if lowered[0] == "all":
        return CandidatesFilter(
            analyze_ready_only=False,
            label="All logged prescans",
        )

    if lowered[0] == "quality":
        if len(args) < 2:
            return "Usage: /candidates quality 65"
        try:
            min_q = float(args[1])
        except ValueError:
            return "Usage: /candidates quality 65 — quality must be a number"
        return CandidatesFilter(
            min_quality=min_q,
            analyze_ready_only=True,
            label=f"Q≥{min_q:.0f} analyze-ready",
        )

    band_key = lowered[0]
    if band_key not in BAND_ALIASES:
        return CANDIDATES_USAGE

    band = BAND_ALIASES[band_key]
    labels = {
        "STRONG_CANDIDATE": "Strong (score ≥80)",
        "CANDIDATE": "Candidate (score 70–79)",
        "WATCHLIST": "Watchlist (score 60–69)",
    }
    return CandidatesFilter(
        bands={band},
        analyze_ready_only=True,
        label=labels.get(band, band),
    )


def _format_qgs(row: dict[str, Any]) -> str:
    text = format_qgs_from_row(row)
    if text == "Pillar scores unavailable":
        return text
    labels = (
        ("quality_score", "Quality"),
        ("growth_score", "Growth"),
        ("strength_score", "Strength"),
    )
    present = sum(
        1 for key, _ in labels if isinstance(row.get(key), (int, float))
    )
    if present < 3:
        return text + " (partial — re-run /prescan for full scores)"
    return text


def _format_prescan_row_html(row: dict[str, Any]) -> str:
    ticker = html_escape(str(row.get("ticker") or "?"))
    band = str(row.get("candidate_band") or "")
    band_icon = BAND_ICONS.get(band, "📊")
    band_label = html_escape(BAND_LABELS.get(band, "Unranked"))
    qgs = html_escape(_format_qgs(row))
    quant = row.get("quant_score")
    quant_txt = f"{quant:.1f}" if isinstance(quant, (int, float)) else "?"
    cash = str(row.get("cash_conversion_status") or "")
    cash_icon = CASH_ICONS.get(cash, "💵")
    cash_label = html_escape(CASH_LABELS.get(cash, cash or "Cash unknown"))
    verdict = str(row.get("verdict") or "")
    verdict_icon = VERDICT_ICONS.get(verdict, "📋")
    verdict_label = html_escape(VERDICT_LABELS.get(verdict, verdict or "Unknown route"))
    headline = (
        f"{band_icon} <b>{ticker}</b> — {band_label} · Overall {quant_txt}/100"
    )
    detail = (
        f"   {qgs}\n"
        f"   {cash_icon} {cash_label} · {verdict_icon} {verdict_label}"
    )
    return f"{headline}\n{detail}"


def format_prescan_telegram_chunks(
    rows: list[dict[str, Any]],
    *,
    title: str,
    max_len: int = TELEGRAM_PSCAN_CHUNK,
) -> list[str]:
    """HTML message chunks for Telegram (≤4096 chars each)."""
    if not rows:
        return [
            (
                "<b>📋 Prescan list</b>\n\n"
                f"No names match <i>{html_escape(title)}</i>.\n"
                "Run <code>/prescan SYMBOL</code> first — the list builds from prescan history."
            )
        ]

    header = (
        f"<b>📋 Prescan list — {html_escape(title)}</b>\n"
        f"{len(rows)} name(s). Full detail: <code>/prescan SYMBOL</code>\n"
        "<i>Overall</i> = combined score (0–100). "
        "<i>Quality / Growth / Strength</i> = the three pillar scores.\n\n"
    )
    chunks: list[str] = []
    current = header
    for row in rows:
        line = _format_prescan_row_html(row) + "\n"
        if len(current) + len(line) > max_len and current.strip():
            chunks.append(current.rstrip())
            current = line
        else:
            current += line
    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def build_candidates_messages(
    args: list[str],
    *,
    path: Path | None = None,
) -> tuple[list[str], str | None]:
    """Return Telegram HTML chunks and optional error/usage text."""
    parsed = parse_candidates_filter(args)
    if isinstance(parsed, str):
        return [], parsed

    target = path or OUTCOMES_PATH
    if not target.exists():
        return [], (
            "No prescan log yet on this bot.\n"
            "Run <code>/prescan SYMBOL</code> on names you care about — "
            "then retry <code>/candidates</code>."
        )

    rows = load_prescan_outcomes(target)
    matched = query_prescan_outcomes(
        rows,
        bands=parsed.bands,
        min_quality=parsed.min_quality,
        analyze_ready_only=parsed.analyze_ready_only,
    )
    stale = sum(1 for row in matched if not _row_has_qgs(row))
    if stale:
        logger.info("Backfilling Q/G/S for %d prescan row(s) missing pillar scores", stale)
    matched = backfill_rows_qgs(matched, persist=True)
    return format_prescan_telegram_chunks(matched, title=parsed.label), None
