"""One-shot sequential /analyze all queue from prescanned candidates.

Runs one ticker at a time (same lock as single /analyze). Default scope is
TOP+GOOD (overall ≥70), LITE Stage 2, skip names that already have a stored
analysis. Hard max protects monthly LLM budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

from stockbot.portfolio_screener.outcome_log import (
    OUTCOMES_PATH,
    load_prescan_outcomes,
    query_prescan_outcomes,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_QUANT = 70.0
DEFAULT_MAX_NAMES = 12
ABSOLUTE_MAX_NAMES = 30

BatchScope = Literal["good", "top", "ready"]


@dataclass(frozen=True)
class AnalyzeBatchRequest:
    """Parsed /analyze all options."""

    scope: BatchScope = "good"
    min_quant: float = DEFAULT_MIN_QUANT
    max_names: int = DEFAULT_MAX_NAMES
    force_lite: bool = True
    full_report: bool = False
    skip_cache: bool = False
    force_gate: bool = False
    skip_already_analyzed: bool = True


@dataclass(frozen=True)
class AnalyzeBatchPlan:
    """Resolved ticker queue ready to run."""

    tickers: list[str]
    skipped_analyzed: list[str]
    request: AnalyzeBatchRequest
    label: str


ANALYZE_ALL_USAGE = (
    "<b>/analyze all — one-shot queue</b>\n"
    "Runs deep analysis <b>one name at a time</b> (not parallel).\n\n"
    "<code>/analyze all</code> — overall ≥70, LITE, skip already analyzed, max 12\n"
    "<code>/analyze all top</code> — overall ≥80 only\n"
    "<code>/analyze all ready</code> — every analyze-ready name (all bands)\n"
    "<code>/analyze all 5</code> — limit to 5 names\n"
    "<code>/analyze all full</code> — Sonnet Stage 2 (costlier)\n"
    "<code>/analyze all fresh</code> — re-run even if already analyzed\n"
    "<code>/analyze all force</code> — bypass eligibility gate\n\n"
    "Send <code>/stop</code> to cancel the current name and remaining queue."
)


def parse_analyze_all_args(args: list[str] | None) -> AnalyzeBatchRequest | str:
    """Parse tokens after ``/analyze all``. Returns usage text when invalid."""
    scope: BatchScope = "good"
    min_quant = DEFAULT_MIN_QUANT
    max_names = DEFAULT_MAX_NAMES
    force_lite = True
    full_report = False
    skip_cache = False
    force_gate = False
    skip_already_analyzed = True

    for token in args or []:
        low = token.lower().strip()
        if low in {"help", "?"}:
            return ANALYZE_ALL_USAGE
        if low == "lite":
            force_lite = True
            continue
        if low in {"full", "sonnet"}:
            # full Stage 2 model (not digest attachment)
            force_lite = False
            continue
        if low == "fresh":
            skip_cache = True
            skip_already_analyzed = False
            continue
        if low == "force":
            force_gate = True
            continue
        if low == "top":
            scope = "top"
            min_quant = 80.0
            continue
        if low in {"ready", "candidates", "allbands", "all-bands"}:
            scope = "ready"
            min_quant = 0.0
            continue
        if low in {"good", "strong"}:
            scope = "good"
            min_quant = 70.0
            continue
        if low.isdigit():
            n = int(low)
            if n < 1:
                return "Limit must be at least 1 — e.g. <code>/analyze all 5</code>"
            max_names = min(n, ABSOLUTE_MAX_NAMES)
            continue
        return f"Unknown option <code>{_esc(token)}</code>.\n\n{ANALYZE_ALL_USAGE}"

    if scope == "ready":
        max_names = min(max_names, ABSOLUTE_MAX_NAMES)

    return AnalyzeBatchRequest(
        scope=scope,
        min_quant=min_quant,
        max_names=max_names,
        force_lite=force_lite,
        full_report=full_report,
        skip_cache=skip_cache,
        force_gate=force_gate,
        skip_already_analyzed=skip_already_analyzed,
    )


def _esc(value: object) -> str:
    return escape(str(value), quote=False)


def _scope_label(request: AnalyzeBatchRequest) -> str:
    if request.scope == "top":
        return "TOP TIER (overall ≥80)"
    if request.scope == "ready":
        return "Analyze-ready (all bands)"
    return "TOP + GOOD (overall ≥70)"


def already_analyzed_tickers() -> set[str]:
    """Tickers with at least one stored /analyze row."""
    from stockbot.storage import list_latest_analyses

    return {ticker for ticker, _verdict, _when in list_latest_analyses()}


def plan_analyze_batch(
    request: AnalyzeBatchRequest,
    *,
    path: Path | None = None,
    already: set[str] | None = None,
) -> AnalyzeBatchPlan | str:
    """Build the ticker queue from the latest prescan outcomes."""
    target = path or OUTCOMES_PATH
    if not target.exists():
        return (
            "📭 No prescan log yet.\n"
            "Run <code>/prescan SYMBOL</code> (or <code>/sip prescan</code>), "
            "then retry <code>/analyze all</code>."
        )

    rows = load_prescan_outcomes(target)
    matched = query_prescan_outcomes(
        rows,
        min_quant=request.min_quant if request.min_quant > 0 else None,
        analyze_ready_only=True,
    )
    if not matched:
        return (
            f"📭 No analyze-ready names in scope ({_esc(_scope_label(request))}).\n"
            "Try <code>/candidates</code> or widen with "
            "<code>/analyze all ready</code>."
        )

    if already is not None:
        analyzed = already
    elif request.skip_already_analyzed:
        analyzed = already_analyzed_tickers()
    else:
        analyzed = set()

    queue: list[str] = []
    skipped: list[str] = []
    for row in matched:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        if request.skip_already_analyzed and ticker in analyzed:
            skipped.append(ticker)
            continue
        queue.append(ticker)
        if len(queue) >= request.max_names:
            break

    if not queue:
        skipped_note = (
            f" Skipped already analyzed: {', '.join(skipped[:8])}"
            f"{'…' if len(skipped) > 8 else ''}."
            if skipped
            else ""
        )
        return (
            "📭 Nothing left to queue in this scope."
            f"{skipped_note}\n"
            "Use <code>/analyze all fresh</code> to re-run, or "
            "<code>/analyze all ready</code> for a wider list."
        )

    logger.info(
        "analyze_batch planned=%d skipped_analyzed=%d scope=%s min_quant=%s lite=%s",
        len(queue),
        len(skipped),
        request.scope,
        request.min_quant,
        request.force_lite,
    )
    return AnalyzeBatchPlan(
        tickers=queue,
        skipped_analyzed=skipped,
        request=request,
        label=_scope_label(request),
    )


def format_batch_start_html(plan: AnalyzeBatchPlan) -> str:
    """Kickoff message before the first ticker runs."""
    req = plan.request
    mode = "lite (cheaper)" if req.force_lite else "full Sonnet"
    preview = ", ".join(plan.tickers[:8])
    if len(plan.tickers) > 8:
        preview += f" … +{len(plan.tickers) - 8} more"
    lines = [
        f"🚀 <b>Batch analyze</b> — {len(plan.tickers)} name(s)",
        f"Scope: {_esc(plan.label)}",
        f"Mode: {mode} · one at a time",
        f"Queue: <code>{_esc(preview)}</code>",
    ]
    if plan.skipped_analyzed:
        lines.append(f"Skipped (already analyzed): {len(plan.skipped_analyzed)}")
    if req.skip_cache:
        lines.append("Fresh: ignore cache / re-run paid analysis")
    lines.append("Send <code>/stop</code> to cancel current + remaining.")
    return "\n".join(lines)


def format_batch_summary_html(
    *,
    planned: list[str],
    completed: list[str],
    failed: list[tuple[str, str]],
    skipped_gate: list[str],
    stopped: bool,
    budget_stopped: bool,
) -> str:
    """Final roll-up after the queue finishes or stops."""
    lines = ["📋 <b>Batch analyze summary</b>"]
    lines.append(f"Planned: {len(planned)} · Done: {len(completed)}")
    if completed:
        lines.append(
            "Completed: " + ", ".join(f"<code>{_esc(t)}</code>" for t in completed)
        )
    if skipped_gate:
        lines.append(
            "Gate blocked: "
            + ", ".join(f"<code>{_esc(t)}</code>" for t in skipped_gate)
        )
    if failed:
        bits = [
            f"<code>{_esc(t)}</code> ({_esc(reason)})" for t, reason in failed[:12]
        ]
        lines.append("Failed/skipped: " + "; ".join(bits))
    if budget_stopped:
        lines.append("🚫 Stopped — monthly budget reached.")
    elif stopped:
        lines.append("⏹ Stopped early by /stop.")
    remaining = [
        t
        for t in planned
        if t not in completed
        and t not in skipped_gate
        and t not in {f[0] for f in failed}
    ]
    if remaining and (stopped or budget_stopped):
        lines.append(
            "Not run: " + ", ".join(f"<code>{_esc(t)}</code>" for t in remaining)
        )
    lines.append("Next: <code>/rank</code> · <code>/progress</code>")
    return "\n".join(lines)


def row_tickers_for_tests(
    rows: list[dict[str, Any]], request: AnalyzeBatchRequest
) -> list[str]:
    """Test helper — plan tickers from in-memory rows (no disk / storage)."""
    matched = query_prescan_outcomes(
        rows,
        min_quant=request.min_quant if request.min_quant > 0 else None,
        analyze_ready_only=True,
    )
    out: list[str] = []
    for row in matched:
        ticker = str(row.get("ticker") or "").upper().strip()
        if ticker:
            out.append(ticker)
        if len(out) >= request.max_names:
            break
    return out
