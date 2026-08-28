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
            rows.append(json.loads(line))
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
    filtered.sort(
        key=lambda r: (
            -(float(r["quality_score"]) if isinstance(r.get("quality_score"), (int, float)) else -1),
            -(float(r["quant_score"]) if isinstance(r.get("quant_score"), (int, float)) else -1),
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

_BAND_ICONS: dict[str, str] = {
    "STRONG_CANDIDATE": "🏆",
    "CANDIDATE": "👍",
    "WATCHLIST": "📌",
    "REMOVE": "⬇️",
}

_VERDICT_ICONS: dict[str, str] = {
    "AUTO_DEEP_ANALYSIS": "✅",
    "SECTOR_SPECIFIC_REVIEW": "🔎",
    "HOLDING_MONITOR_ONLY": "👀",
    "NOT_SUITABLE_FOR_3Y_RESEARCH": "❌",
    "DATA_UNAVAILABLE_RETRY": "📭",
}

_CASH_ICONS: dict[str, str] = {
    "PASS": "💚",
    "WATCH": "💛",
    "ESCALATED_WATCH": "🧡",
    "CRITICAL": "❤️",
    "NOT_APPLICABLE": "➖",
    "NOT_APPLICABLE_WHILE_LOSS_MAKING": "🔥",
}

CANDIDATES_USAGE = (
    "Usage:\n"
    "/candidates — analyze-ready names (✅/🔎 gate, cash OK)\n"
    "/candidates strong — score ≥80 (STRONG_CANDIDATE)\n"
    "/candidates candidate — score 70–79\n"
    "/candidates watchlist — score 60–69\n"
    "/candidates quality 65 — Q≥65 and analyze-ready\n"
    "/candidates all — every logged prescan (latest per symbol)\n\n"
    "List builds from /prescan history on this bot."
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
    parts: list[str] = []
    for key, letter in (
        ("quality_score", "Q"),
        ("growth_score", "G"),
        ("strength_score", "S"),
    ):
        val = row.get(key)
        if isinstance(val, (int, float)):
            parts.append(f"{letter}{val:.0f}")
    return " ".join(parts) if parts else "Q/G/S —"


def _format_prescan_row_html(row: dict[str, Any]) -> str:
    ticker = html_escape(str(row.get("ticker") or "?"))
    band = str(row.get("candidate_band") or "")
    band_icon = _BAND_ICONS.get(band, "📊")
    qgs = html_escape(_format_qgs(row))
    quant = row.get("quant_score")
    quant_txt = f"{quant:.0f}" if isinstance(quant, (int, float)) else "?"
    cash = str(row.get("cash_conversion_status") or "")
    cash_icon = _CASH_ICONS.get(cash, "💵")
    verdict = str(row.get("verdict") or "")
    verdict_icon = _VERDICT_ICONS.get(verdict, "📋")
    verdict_short = {
        "AUTO_DEEP_ANALYSIS": "AUTO_DEEP",
        "SECTOR_SPECIFIC_REVIEW": "SECTOR",
        "HOLDING_MONITOR_ONLY": "MONITOR",
        "NOT_SUITABLE_FOR_3Y_RESEARCH": "NOT_SUITABLE",
        "DATA_UNAVAILABLE_RETRY": "RETRY",
    }.get(verdict, verdict[:12])
    return (
        f"{band_icon} <b>{ticker}</b> · {qgs} · score {quant_txt} · "
        f"{cash_icon} · {verdict_icon} {html_escape(verdict_short)}"
    )


def format_prescan_telegram_chunks(
    rows: list[dict[str, Any]],
    *,
    title: str,
    max_len: int = TELEGRAM_PSCAN_CHUNK,
) -> list[str]:
    """HTML message chunks for Telegram (≤4096 chars each)."""
    if not rows:
        return [
            "<b>📋 Prescan list</b>\n\n"
            f"No names match <i>{html_escape(title)}</i>.\n"
            "Run <code>/prescan SYMBOL</code> first — the list builds from prescan history."
        ]

    header = (
        f"<b>📋 Prescan list — {html_escape(title)}</b>\n"
        f"{len(rows)} name(s) · detail: <code>/prescan SYMBOL</code>\n\n"
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
    return format_prescan_telegram_chunks(matched, title=parsed.label), None
