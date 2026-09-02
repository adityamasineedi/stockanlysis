"""Deterministic health audit — cost leaks, token waste, analysis quality.

Reads SQLite (`analyses`, `llm_calls`), optional `logs/stockbot.log`, and
LLM fixtures. No LLM call required; safe to run after every session or on a
schedule.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from html import escape as html_escape
from typing import Literal

from stockbot.config import (  # noqa: F401 - DB_PATH re-exported for tests to monkeypatch
    DATA_DIR,
    DB_PATH,
    LOGS_DIR,
    settings,
)
from stockbot.costs import _connect as connect_costs_db
from stockbot.costs import month_to_date_spend
from stockbot.monitor.health_audit_state import (
    FindingDiff,
    HealthAuditState,
    StoredFinding,
    clear_health_audit_state,
    diff_findings,
    finding_key,
    load_health_audit_state,
    log_cutoff,
    save_health_audit_state,
)
from stockbot.storage import _connect as connect_analyses_db

Severity = Literal["critical", "warning", "info"]
TRACKED_SEVERITIES = frozenset({"critical", "warning"})

STAGE1_INPUT_WARN = 50_000
STAGE1_INPUT_CRITICAL = 80_000
THINKING_RATIO_WARN = 0.55
THINKING_RATIO_CRITICAL = 0.70
ANALYSIS_COST_WARN_INR = 55.0
ANALYSIS_COST_CRITICAL_INR = 75.0
ORPHAN_SESSION_MIN_INR = 25.0
BRIEF_BLOAT_BYTES = 250_000
# Prompt cache TTL is 1h — only flag a retry cache miss when the prior Stage 2
# call for the same ticker fell inside that window.
CACHE_RETRY_WINDOW = timedelta(hours=1)


@dataclass(frozen=True)
class Finding:
    severity: Severity
    category: str
    title: str
    detail: str
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass
class HealthAuditReport:
    generated_at: datetime
    days: int
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)
    diff: FindingDiff | None = None
    log_since: datetime | None = None

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    def _status_for(self, finding: Finding) -> str:
        if self.diff is None:
            return ""
        key = finding_key(finding.category, finding.title)
        if key in self.diff.new_keys:
            return "new"
        if key in self.diff.open_keys:
            return "open"
        return ""

    def to_markdown(self) -> str:
        lines = [
            f"# stockbot health audit ({self.generated_at.date().isoformat()})",
            "",
            (
                f"Window: last **{self.days}** days · "
                f"**{self.critical_count}** critical · **{self.warning_count}** warnings · "
                f"**{len(self.findings)}** total findings"
            ),
            "",
        ]
        if self.log_since is not None:
            lines.append(
                f"Log patterns counted since **{self.log_since.isoformat()}** "
                "(max of window start, last green, and `/health clear`)."
            )
            lines.append("")
        if self.summary:
            lines.append("## Summary")
            for key, value in self.summary.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if self.diff is not None and self.diff.resolved:
            lines.append("## RESOLVED since last audit")
            for item in self.diff.resolved:
                lines.append(f"- [{item.category}] {item.title}")
                if item.detail:
                    lines.append(f"  - {item.detail}")
            lines.append("")

        by_sev: dict[Severity, list[Finding]] = {"critical": [], "warning": [], "info": []}
        for item in self.findings:
            by_sev[item.severity].append(item)

        for sev in ("critical", "warning", "info"):
            items = by_sev[sev]
            if not items:
                continue
            lines.append(f"## {sev.upper()}")
            for item in items:
                status = self._status_for(item)
                prefix = f"[{status}] " if status else ""
                lines.append(f"### {prefix}[{item.category}] {item.title}")
                lines.append(item.detail)
                if item.evidence:
                    lines.append("")
                    lines.append("```json")
                    lines.append(json.dumps(item.evidence, indent=2, default=str))
                    lines.append("```")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_telegram_html(self, *, max_findings_per_severity: int = 5) -> str:
        """Compact HTML summary for Telegram (≤4096 chars)."""
        lines = [
            f"<b>Health audit</b> ({self.generated_at.date().isoformat()})",
            (
                f"Last {self.days} days · "
                f"{self.critical_count} critical · {self.warning_count} warnings"
            ),
            "",
        ]
        mtd = self.summary.get("mtd_spend_inr")
        budget = self.summary.get("monthly_budget_inr")
        if mtd is not None and budget is not None:
            lines.append(
                f"MTD spend: ₹{float(mtd):.2f} / ₹{float(budget):.0f}"
            )
        analyses = self.summary.get("analyses")
        llm_calls = self.summary.get("llm_calls")
        if analyses is not None or llm_calls is not None:
            parts: list[str] = []
            if llm_calls is not None:
                parts.append(f"{llm_calls} LLM calls")
            if analyses is not None:
                parts.append(f"{analyses} saved analyses")
            lines.append(" · ".join(parts))
        if self.log_since is not None:
            lines.append(
                f"Log window from {html_escape(self.log_since.date().isoformat())}"
            )
        lines.append("")

        for sev, emoji in (("critical", "🔴"), ("warning", "🟡")):
            items = [f for f in self.findings if f.severity == sev]
            if not items:
                continue
            lines.append(f"<b>{emoji} {sev.upper()}</b>")
            shown = items[:max_findings_per_severity]
            for item in shown:
                status = self._status_for(item)
                tag = f"[{status}] " if status else ""
                lines.append(
                    f"• {tag}[{html_escape(item.category)}] {html_escape(item.title)}"
                )
                if item.detail:
                    detail = html_escape(item.detail)
                    if len(detail) > 140:
                        detail = detail[:137] + "…"
                    lines.append(f"  {detail}")
            remaining = len(items) - len(shown)
            if remaining > 0:
                lines.append(f"  … +{remaining} more (see attached report)")
            lines.append("")

        if self.diff is not None and self.diff.resolved:
            lines.append("<b>✅ Resolved since last audit</b>")
            for item in self.diff.resolved[:max_findings_per_severity]:
                lines.append(
                    f"• [{html_escape(item.category)}] {html_escape(item.title)}"
                )
            remaining = len(self.diff.resolved) - min(
                len(self.diff.resolved), max_findings_per_severity
            )
            if remaining > 0:
                lines.append(f"  … +{remaining} more")
            lines.append("")

        if not self.critical_count and not self.warning_count:
            if self.diff is not None and self.diff.resolved:
                lines.append("✅ No open critical/warning findings (prior issues cleared).")
            else:
                lines.append("✅ No critical or warning findings.")

        text = "\n".join(lines).rstrip()
        if len(text) > 4096:
            return text[:4090] + "…"
        return text


def _since_iso(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def _connect_analyses() -> sqlite3.Connection:
    return connect_analyses_db()


def _connect_llm_calls() -> sqlite3.Connection:
    conn = connect_costs_db()
    conn.row_factory = sqlite3.Row
    return conn


def _audit_budget(findings: list[Finding]) -> None:
    spent = month_to_date_spend()
    cap = settings.monthly_budget_inr
    pct = spent / cap * 100 if cap else 0
    if pct >= 100:
        findings.append(
            Finding(
                "critical",
                "cost_leak",
                "Monthly budget exhausted",
                f"MTD spend ₹{spent:.2f} ≥ cap ₹{cap:.0f} — new paid runs blocked.",
                {"spent_inr": spent, "cap_inr": cap},
            )
        )
    elif pct >= 80:
        findings.append(
            Finding(
                "warning",
                "cost_leak",
                "Monthly budget nearly exhausted",
                f"MTD spend ₹{spent:.2f} is {pct:.0f}% of ₹{cap:.0f} cap.",
                {"spent_inr": spent, "cap_inr": cap, "pct": round(pct, 1)},
            )
        )


def _audit_llm_calls(findings: list[Finding], days: int) -> dict[str, object]:
    since = _since_iso(days)
    with _connect_llm_calls() as conn:
        rows = conn.execute(
            "SELECT * FROM llm_calls WHERE called_at >= ? ORDER BY called_at",
            (since,),
        ).fetchall()

    total_cost = sum(float(r["cost_inr"]) for r in rows)
    by_stage: dict[str, float] = defaultdict(float)
    for row in rows:
        stage = row["stage"] or "unknown"
        by_stage[stage] += float(row["cost_inr"])

    by_ticker: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        if row["ticker"]:
            by_ticker[str(row["ticker"]).upper()].append(row)

    stage2_prior_counts: dict[str, int] = defaultdict(int)
    stage2_last_called_at: dict[str, datetime] = {}

    for row in rows:
        inp = int(row["input_tokens"])
        out = int(row["output_tokens"])
        think = int(row["thinking_tokens"] or 0)
        stage = (row["stage"] or "").lower()
        model = row["model"]
        cost = float(row["cost_inr"])

        if stage == "stage1" or (stage == "unknown" and inp > 30_000 and "haiku" not in model):
            if inp >= STAGE1_INPUT_CRITICAL:
                findings.append(
                    Finding(
                        "critical",
                        "token_waste",
                        "Stage 1 input extremely large",
                        f"{inp:,} input tokens (₹{cost:.2f}) — trim annual-report sections further.",
                        {"called_at": row["called_at"], "ticker": row["ticker"], "input_tokens": inp},
                    )
                )
            elif inp >= STAGE1_INPUT_WARN:
                findings.append(
                    Finding(
                        "warning",
                        "token_waste",
                        "Stage 1 input larger than trimmed target",
                        f"{inp:,} input tokens — expected ≤~15k after audit/governance trim.",
                        {"called_at": row["called_at"], "ticker": row["ticker"], "input_tokens": inp},
                    )
                )

        if out > 0 and think > 0 and stage.startswith("stage2"):
            ratio = think / out
            if ratio >= THINKING_RATIO_CRITICAL:
                findings.append(
                    Finding(
                        "critical",
                        "token_waste",
                        "Stage 2 thinking dominates output",
                        f"{ratio:.0%} of output tokens are thinking ({think:,}/{out:,}) — "
                        f"consider LITE path or shorter prompt.",
                        {
                            "called_at": row["called_at"],
                            "ticker": row["ticker"],
                            "stage": row["stage"],
                            "thinking_ratio": round(ratio, 3),
                            "cost_inr": cost,
                        },
                    )
                )
            elif ratio >= THINKING_RATIO_WARN:
                findings.append(
                    Finding(
                        "warning",
                        "token_waste",
                        "High Stage 2 thinking ratio",
                        f"{ratio:.0%} thinking tokens on {row['stage'] or 'stage2'}.",
                        {"called_at": row["called_at"], "ticker": row["ticker"], "thinking_ratio": round(ratio, 3)},
                    )
                )

        cache_create = int(row["cache_creation_tokens"] or 0)
        cache_read = int(row["cached_tokens"] or 0)
        if stage.startswith("stage2"):
            ticker_key = str(row["ticker"] or "").upper()
            prior_stage2 = stage2_prior_counts[ticker_key]
            stage2_prior_counts[ticker_key] += 1
            called_at = _parse_ts(str(row["called_at"]))
            prior_at = stage2_last_called_at.get(ticker_key)
            # First Stage 2 call per ticker always writes cache; reads appear
            # on retries inside the 1h TTL. Calls spaced >1h apart are a new
            # session, not a cache-reuse failure.
            within_cache_ttl = (
                prior_at is not None and (called_at - prior_at) <= CACHE_RETRY_WINDOW
            )
            if (
                cache_create > 5000
                and cache_read == 0
                and prior_stage2 >= 1
                and within_cache_ttl
            ):
                findings.append(
                    Finding(
                        "warning",
                        "token_waste",
                        "Stage 2 retry did not read prompt cache",
                        f"Call #{prior_stage2 + 1} for {ticker_key or '?'} wrote {cache_create:,} "
                        f"cache tokens (₹{cost:.2f}) without a cache read — prior Stage 2 was "
                        f"within 1h; retries should reuse the prompt cache.",
                        {
                            "called_at": row["called_at"],
                            "ticker": row["ticker"],
                            "cache_creation_tokens": cache_create,
                            "stage2_call_index": prior_stage2 + 1,
                            "hours_since_prior": round(
                                (called_at - prior_at).total_seconds() / 3600.0, 2
                            ),
                        },
                    )
                )
            stage2_last_called_at[ticker_key] = called_at

    with _connect_analyses() as conn:
        analysis_rows = conn.execute(
            "SELECT ticker, cost_inr, created_at FROM analyses WHERE created_at >= ?",
            (since,),
        ).fetchall()
    analysis_by_ticker: dict[str, list[tuple[float, datetime]]] = defaultdict(list)
    for ar in analysis_rows:
        analysis_by_ticker[str(ar["ticker"]).upper()].append(
            (float(ar["cost_inr"]), _parse_ts(str(ar["created_at"])))
        )

    for ticker, calls in by_ticker.items():
        deep_calls = [c for c in calls if (c["stage"] or "").startswith("stage")]
        if len(deep_calls) < 2:
            continue
        session_cost = sum(float(c["cost_inr"]) for c in deep_calls)
        if session_cost < ORPHAN_SESSION_MIN_INR:
            continue
        first = _parse_ts(str(deep_calls[0]["called_at"]))
        last = _parse_ts(str(deep_calls[-1]["called_at"]))
        saved = analysis_by_ticker.get(ticker, [])
        matched = any(first - timedelta(hours=1) <= ts <= last + timedelta(hours=2) for _, ts in saved)
        if not matched and session_cost >= ORPHAN_SESSION_MIN_INR:
            findings.append(
                Finding(
                    "critical",
                    "cost_leak",
                    "Likely abandoned analysis session",
                    f"{ticker}: {len(deep_calls)} Stage 1/2 calls totalling ₹{session_cost:.2f} "
                    f"with no saved analysis in the window (bot restart / failed validation?).",
                    {
                        "ticker": ticker,
                        "calls": len(deep_calls),
                        "cost_inr": round(session_cost, 2),
                        "first": first.isoformat(),
                        "last": last.isoformat(),
                    },
                )
            )
        elif matched and len(deep_calls) >= 3:
            saved_cost = max(
                (c for c, ts in saved if first - timedelta(hours=1) <= ts <= last + timedelta(hours=2)),
                default=0.0,
            )
            excess = session_cost - saved_cost
            if excess >= 20:
                findings.append(
                    Finding(
                        "warning",
                        "cost_leak",
                        "Retries exceeded saved analysis cost",
                        f"{ticker}: logged ₹{session_cost:.2f} across {len(deep_calls)} calls but "
                        f"saved analysis cost ₹{saved_cost:.2f} (₹{excess:.2f} likely retries/truncation).",
                        {
                            "ticker": ticker,
                            "session_cost_inr": round(session_cost, 2),
                            "saved_cost_inr": saved_cost,
                            "excess_inr": round(excess, 2),
                        },
                    )
                )

    return {
        "llm_calls": len(rows),
        "llm_cost_inr": round(total_cost, 2),
        "cost_by_stage": {k: round(v, 2) for k, v in sorted(by_stage.items())},
    }


def _audit_analyses(findings: list[Finding], days: int) -> dict[str, object]:
    since = _since_iso(days)
    with _connect_analyses() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE created_at >= ? ORDER BY created_at DESC",
            (since,),
        ).fetchall()

    if not rows:
        findings.append(
            Finding(
                "info",
                "quality",
                "No completed analyses in window",
                f"No rows in `analyses` since {since[:10]}.",
            )
        )
        return {"analyses": 0}

    for row in rows:
        cost = float(row["cost_inr"])
        ticker = row["ticker"]
        if cost >= ANALYSIS_COST_CRITICAL_INR:
            findings.append(
                Finding(
                    "critical",
                    "cost_leak",
                    "Single analysis near per-run cap",
                    f"{ticker} cost ₹{cost:.2f} (cap ₹{settings.per_analysis_cost_cap_inr:.0f}) — "
                    "check retries/truncation/Stage 2 FULL path.",
                    {"ticker": ticker, "cost_inr": cost, "created_at": row["created_at"]},
                )
            )
        elif cost >= ANALYSIS_COST_WARN_INR:
            findings.append(
                Finding(
                    "warning",
                    "cost_leak",
                    "Expensive analysis run",
                    f"{ticker} cost ₹{cost:.2f}.",
                    {"ticker": ticker, "cost_inr": cost},
                )
            )

        brief_len = len(row["brief_text"] or "")
        if brief_len >= BRIEF_BLOAT_BYTES:
            findings.append(
                Finding(
                    "warning",
                    "token_waste",
                    "Large brief stored in DB",
                    f"{ticker} brief_text {brief_len // 1024} KB — inflates SQLite; consider trimming stored brief.",
                    {"ticker": ticker, "brief_bytes": brief_len},
                )
            )

        try:
            verdict = json.loads(row["verdict_json"])
        except json.JSONDecodeError:
            findings.append(
                Finding(
                    "critical",
                    "quality",
                    "Invalid verdict_json in DB",
                    f"{ticker} row id={row['id']} has corrupt JSON.",
                    {"ticker": ticker, "id": row["id"]},
                )
            )
            continue

        if not verdict.get("expected_return"):
            findings.append(
                Finding(
                    "info",
                    "quality",
                    "Missing expected_return block",
                    f"{ticker} ({row['created_at'][:10]}) — re-run or backfill for scenario CAGR card.",
                    {"ticker": ticker},
                )
            )

        mode = verdict.get("stage2_mode")
        reasons = verdict.get("stage2_routing_reasons") or []
        if mode == "FULL" and any("clean AUTO_DEEP" in str(r) for r in reasons):
            findings.append(
                Finding(
                    "info",
                    "token_waste",
                    "FULL Stage 2 despite clean prescan routing",
                    f"{ticker} used FULL — check extraction red flags or FORCE_STAGE2_FULL.",
                    {"ticker": ticker, "reasons": reasons},
                )
            )

        if not bool(row["validation_passed"]):
            findings.append(
                Finding(
                    "warning",
                    "quality",
                    "Stored analysis failed validation flag",
                    f"{ticker} has validation_passed=0 in DB (unexpected for delivered reports).",
                    {"ticker": ticker},
                )
            )

    return {
        "analyses": len(rows),
        "analysis_cost_inr": round(sum(float(r["cost_inr"]) for r in rows), 2),
    }


def _audit_logs(findings: list[Finding], *, since: datetime) -> None:
    log_path = LOGS_DIR / "stockbot.log"
    if not log_path.exists():
        return
    patterns = {
        "analysis_cost_exceeded": re.compile(r"analysis_cost_exceeded|AnalysisCostExceeded", re.IGNORECASE),
        "analysis_truncated": re.compile(
            r"analysis_truncated|Stage2TruncationExhausted|truncated.*without completing",
            re.IGNORECASE,
        ),
        "runtime_exceeded": re.compile(r"analysis_runtime_exceeded|AnalysisRuntimeExceeded", re.IGNORECASE),
        "truncated": re.compile(r"Stage 2 response truncated|TruncatedResponseError", re.IGNORECASE),
        "validation_failed": re.compile(r"validation failed", re.IGNORECASE),
        "render_failed": re.compile(r"render_failed|PlaceholderError", re.IGNORECASE),
    }
    counts = {key: 0 for key in patterns}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        if " ERROR " not in line and " WARNING " not in line:
            continue
        try:
            ts_str = line.split(" ", 1)[0]
            if _parse_ts(ts_str.replace(",", ".")) < since:
                continue
        except (ValueError, IndexError):
            pass
        for key, pat in patterns.items():
            if pat.search(line):
                counts[key] += 1

    for key, count in counts.items():
        if count == 0:
            continue
        sev: Severity = "warning" if key in {"validation_failed", "truncated", "analysis_truncated"} else "info"
        if key in {"analysis_cost_exceeded", "runtime_exceeded"}:
            sev = "critical"
        findings.append(
            Finding(
                sev,
                "cost_leak" if "cost" in key or "runtime" in key else "quality",
                f"Log pattern: {key}",
                (
                    f"{count} matching line(s) in logs/stockbot.log since "
                    f"{since.isoformat()}."
                ),
                {"count": count, "pattern": key, "since": since.isoformat()},
            )
        )


def _audit_fixtures(findings: list[Finding], days: int) -> None:
    fixtures_dir = DATA_DIR / "llm_fixtures"
    if not fixtures_dir.exists():
        return
    since = datetime.now(UTC) - timedelta(days=days)
    truncated: list[str] = []
    for path in fixtures_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("stop_reason") != "max_tokens":
            continue
        m = re.search(r"(\d{8}T\d+)", path.name)
        if m:
            try:
                ts = datetime.strptime(m.group(1)[:15], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
                if ts < since:
                    continue
            except ValueError:
                pass
        truncated.append(path.name)

    if truncated:
        findings.append(
            Finding(
                "warning",
                "token_waste",
                "Truncated LLM responses saved as fixtures",
                f"{len(truncated)} fixture(s) with stop_reason=max_tokens — "
                "raise Stage 2 max_tokens (LITE base is now 32k; prefer /analyze lite "
                "over repeated FULL retries).",
                {"files": truncated[:10], "total": len(truncated)},
            )
        )


def run_health_audit(*, days: int = 14, persist: bool = True) -> HealthAuditReport:
    """Run all deterministic checks and return a structured report.

    When ``persist`` is True (default), compares against the previous snapshot,
    records resolved findings, and updates the ledger. A clean run advances
    ``last_green_at`` so old log lines stop counting.
    """
    previous = load_health_audit_state()
    now = datetime.now(UTC)
    log_since = log_cutoff(days, previous, now=now)

    findings: list[Finding] = []
    _audit_budget(findings)
    llm_stats = _audit_llm_calls(findings, days)
    analysis_stats = _audit_analyses(findings, days)
    _audit_logs(findings, since=log_since)
    _audit_fixtures(findings, days)

    findings.sort(key=lambda f: ({"critical": 0, "warning": 1, "info": 2}[f.severity], f.category, f.title))

    tracked = [f for f in findings if f.severity in TRACKED_SEVERITIES]
    current_stored = [
        StoredFinding(
            severity=f.severity,
            category=f.category,
            title=f.title,
            detail=f.detail,
        )
        for f in tracked
    ]
    current_keys = {finding_key(f.category, f.title) for f in current_stored}
    finding_diff = diff_findings(previous, current_keys)

    report = HealthAuditReport(
        generated_at=now,
        days=days,
        findings=findings,
        summary={
            **llm_stats,
            **analysis_stats,
            "mtd_spend_inr": round(month_to_date_spend(), 2),
            "monthly_budget_inr": settings.monthly_budget_inr,
            "resolved_since_last": len(finding_diff.resolved),
            "new_findings": len(finding_diff.new_keys),
            "open_findings": len(finding_diff.open_keys),
        },
        diff=finding_diff,
        log_since=log_since,
    )

    if persist:
        last_green = previous.last_green_at
        ignore_before = previous.ignore_log_before
        if report.critical_count == 0 and report.warning_count == 0:
            last_green = now
            # Once clean, stop replaying older log noise on the next run.
            ignore_before = now
        save_health_audit_state(
            HealthAuditState(
                updated_at=now,
                days=days,
                findings=current_stored,
                last_green_at=last_green,
                ignore_log_before=ignore_before,
            )
        )

    return report


@dataclass(frozen=True)
class VerifyAndClearResult:
    report: HealthAuditReport
    cleared: bool
    reason: str
    clear_meta: dict[str, object] | None = None


def verify_and_clear_health_audit(
    *,
    days: int = 14,
    fail_on: Literal["critical", "warning"] = "warning",
) -> VerifyAndClearResult:
    """Run a full audit; clear baseline only when verification passes.

    Never clears when open critical (or warning, depending on ``fail_on``)
    findings remain.
    """
    report = run_health_audit(days=days, persist=True)
    blocked = report.critical_count > 0
    if fail_on == "warning":
        blocked = blocked or report.warning_count > 0
    if blocked:
        return VerifyAndClearResult(
            report=report,
            cleared=False,
            reason=(
                f"verification failed: {report.critical_count} critical, "
                f"{report.warning_count} warning — baseline NOT cleared"
            ),
        )
    meta = clear_health_audit_state(prune_reports=True)
    return VerifyAndClearResult(
        report=report,
        cleared=True,
        reason="verification passed — baseline cleared",
        clear_meta=meta,
    )


# Re-export clear for bot/CLI callers.
__all__ = [
    "Finding",
    "HealthAuditReport",
    "VerifyAndClearResult",
    "clear_health_audit_state",
    "run_health_audit",
    "verify_and_clear_health_audit",
]
