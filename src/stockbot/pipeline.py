"""pipeline.py — orchestrates resolve → cache → budget → brief → extract
→ verdict → validate → store, end to end.

This is where Prompt 11's retry-with-feedback loop actually lives: Stage
2 runs, gets validated, and on failure gets ONE retry with the specific
failures fed back as extra_instruction. A second failure is a legitimate
terminal outcome ("insufficient data for a confident view"), not an error
to hide — see PipelineResult.status == "insufficient_data".

Built as its own module (ahead of Prompt 14, which was originally framed
as "wire everything into pipeline.py") because Prompt 13's Telegram
/analyze handler cannot function without this orchestration — there is no
sensible way to build the bot first and the pipeline after when the bot's
entire job is calling the pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Literal

from stockbot import storage
from stockbot.analysis_routing import (
    Stage2Mode,
    analysis_routing_from_brief,
    resolve_stage2_mode,
)
from stockbot.brief import assemble_brief, to_markdown
from stockbot.config import settings
from stockbot.constitution_gates import (
    apply_constitution_overrides,
    sync_live_price_into_verdict,
)
from stockbot.costs import check_budget
from stockbot.expected_return import merge_expected_return_into_verdict_json
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.llm.extract import run_stage1
from stockbot.llm.verdict import (
    TruncatedResponseError,
    compute_valuation,
    extract_verdict_json,
    run_stage2,
)
from stockbot.models import AmbiguousMatch, Analysis, TickerInfo, ValidationResult
from stockbot.render import PlaceholderError, render_report
from stockbot.storage import CacheHit, build_staleness_banner
from stockbot.validate import (
    classify_retry_mode,
    format_validation_errors,
    try_auto_fix_report,
    validate_report,
)

logger = logging.getLogger(__name__)

MAX_STAGE2_RETRIES = 1
# Truncation is an infrastructure failure, not a validation failure, and
# must not consume a validation retry (v3 migration note) — but it still
# needs its own bound so a persistently-truncating call can't loop forever
# independent of the cost cap below (which would eventually stop it anyway,
# but a call could burn several truncations before crossing ₹80).
MAX_TRUNCATION_RETRIES = 2

# Hard per-analysis kill switch. Separate from costs.check_budget()'s
# monthly cap — that one is checked BEFORE an analysis starts; this one
# stops a single analysis mid-flight. Added after a real run spent ₹243
# across repeated Stage 1 + Stage 2 (+ retries) failures on one ticker —
# nothing was stopping it. ₹80 is a hard ceiling meant to bound the worst
# case (Stage 1 + Stage 2 + one retry), not a number any single healthy
# analysis should get near.
PER_ANALYSIS_COST_CAP_INR = 80.0
# Stop starting new paid Stage 2 attempts once wall-clock exceeds this —
# prevents retry loops from burning budget after an abandoned session.
ANALYSIS_RUNTIME_CAP_SECONDS = 600

# Paid-path concurrency. Cache hits skip this. Recreated when settings change
# only at process start — config is read once at import of settings.
_ANALYSIS_SLOTS = threading.BoundedSemaphore(max(1, settings.max_concurrent_analyses))


@dataclass(frozen=True)
class PipelineResult:
    status: Literal[
        "ok",
        "ambiguous",
        "not_found",
        "insufficient_data",
        "budget_exceeded",
        "analysis_cost_exceeded",
        "analysis_runtime_exceeded",
        "render_failed",
        "busy",
    ]
    analysis: Analysis | None = None
    candidates: AmbiguousMatch | None = None
    validation_failures: list[str] | None = None
    spent_inr: float | None = None
    render_error: str | None = None
    from_cache: bool = False
    staleness_banner: str | None = None


class AnalysisCostExceeded(Exception):
    def __init__(self, spent_inr: float):
        self.spent_inr = spent_inr
        super().__init__(
            f"Per-analysis cost cap (₹{PER_ANALYSIS_COST_CAP_INR:.0f}) exceeded: "
            f"₹{spent_inr:.2f} spent so far on this analysis"
        )


class AnalysisRuntimeExceeded(Exception):
    def __init__(self, elapsed_seconds: float):
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"Analysis runtime cap ({ANALYSIS_RUNTIME_CAP_SECONDS}s) exceeded "
            f"after {elapsed_seconds:.0f}s — stopping before further LLM calls"
        )


def _runtime_exceeded(started_at: float) -> bool:
    return time.monotonic() - started_at > ANALYSIS_RUNTIME_CAP_SECONDS


def _call_stage2_absorbing_truncation(
    brief,
    extraction,
    extra_instruction: str | None,
    running_cost_inr: float,
    stage2_mode: Stage2Mode,
    started_at: float,
) -> tuple[str, dict, float]:
    """Runs one Stage 2 call, transparently re-issuing it (same
    extra_instruction, i.e. not a validation attempt) if truncated. Returns
    (report_text, usage, updated running_cost_inr). Truncation is an
    infrastructure failure, not a validation failure — it must not consume
    one of the scarce MAX_STAGE2_RETRIES slots."""
    if _runtime_exceeded(started_at):
        raise AnalysisRuntimeExceeded(time.monotonic() - started_at)
    for truncation_attempt in range(MAX_TRUNCATION_RETRIES + 1):
        try:
            report_text, _verdict, usage = run_stage2(
                brief,
                extraction,
                extra_instruction=extra_instruction,
                mode=stage2_mode,
            )
        except TruncatedResponseError as exc:
            running_cost_inr += exc.cost_inr
            if running_cost_inr > PER_ANALYSIS_COST_CAP_INR:
                raise AnalysisCostExceeded(running_cost_inr) from exc
            logger.warning(
                "Stage 2 response truncated (infra failure %d/%d, not a validation attempt): %s",
                truncation_attempt + 1,
                MAX_TRUNCATION_RETRIES,
                exc,
            )
            continue
        running_cost_inr += usage["cost_inr"]
        if running_cost_inr > PER_ANALYSIS_COST_CAP_INR:
            raise AnalysisCostExceeded(running_cost_inr)
        return report_text, usage, running_cost_inr
    raise AnalysisCostExceeded(running_cost_inr)  # truncation retries exhausted without ever completing


def _run_stage2_with_validation(
    brief,
    extraction,
    running_cost_inr: float,
    stage2_mode: Stage2Mode,
    started_at: float,
) -> tuple[str, dict, ValidationResult]:
    report_text, usage, running_cost_inr = _call_stage2_absorbing_truncation(
        brief, extraction, None, running_cost_inr, stage2_mode, started_at
    )
    validation = validate_report(report_text, brief, stage2_mode=stage2_mode)

    fixed = try_auto_fix_report(report_text, validation, brief, stage2_mode=stage2_mode)
    if fixed is not None:
        report_text, validation = fixed

    attempt = 1
    while not validation.passed and attempt <= MAX_STAGE2_RETRIES:
        if _runtime_exceeded(started_at):
            raise AnalysisRuntimeExceeded(time.monotonic() - started_at)
        retry_mode = classify_retry_mode(validation)
        feedback = format_validation_errors(validation, retry_mode=retry_mode)
        logger.warning(
            "Stage 2 validation failed (attempt %d, %s retry): %s",
            attempt,
            retry_mode,
            feedback,
        )
        report_text, retry_usage, running_cost_inr = _call_stage2_absorbing_truncation(
            brief, extraction, feedback, running_cost_inr, stage2_mode, started_at
        )
        usage = {
            "input_tokens": usage["input_tokens"] + retry_usage["input_tokens"],
            "output_tokens": usage["output_tokens"] + retry_usage["output_tokens"],
            "cost_inr": usage["cost_inr"] + retry_usage["cost_inr"],
        }
        validation = validate_report(report_text, brief, stage2_mode=stage2_mode)
        fixed = try_auto_fix_report(report_text, validation, brief, stage2_mode=stage2_mode)
        if fixed is not None:
            report_text, validation = fixed
        attempt += 1

    return report_text, usage, validation


def _run_paid_analysis(ticker: TickerInfo) -> PipelineResult:
    started_at = time.monotonic()
    budget_ok, spent = check_budget()
    if not budget_ok:
        return PipelineResult(status="budget_exceeded", spent_inr=spent)

    brief = assemble_brief(ticker)
    prescan_routing = analysis_routing_from_brief(brief)
    extraction, stage1_usage = run_stage1(brief)
    stage2_mode = resolve_stage2_mode(ticker, extraction, prescan=prescan_routing)
    logger.info(
        "%s Stage 2 mode=%s (prescan: %s)",
        ticker.symbol,
        stage2_mode,
        "; ".join(prescan_routing.reasons),
    )

    if stage1_usage["cost_inr"] > PER_ANALYSIS_COST_CAP_INR:
        return PipelineResult(status="analysis_cost_exceeded", spent_inr=stage1_usage["cost_inr"])

    # Re-check monthly budget after Stage 1 so a concurrent sibling process
    # (or a slow Stage 1) that filled the cap does not proceed into Stage 2.
    budget_ok, spent = check_budget()
    if not budget_ok:
        return PipelineResult(status="budget_exceeded", spent_inr=spent)

    try:
        report_text, stage2_usage, validation = _run_stage2_with_validation(
            brief,
            extraction,
            stage1_usage["cost_inr"],
            stage2_mode,
            started_at,
        )
    except AnalysisCostExceeded as exc:
        return PipelineResult(status="analysis_cost_exceeded", spent_inr=exc.spent_inr)
    except AnalysisRuntimeExceeded:
        spent_so_far = stage1_usage["cost_inr"]
        return PipelineResult(status="analysis_runtime_exceeded", spent_inr=spent_so_far)

    if not validation.passed:
        return PipelineResult(status="insufficient_data", validation_failures=validation.failures)

    # extract_verdict_json already succeeded inside validate_report — re-parse
    # here rather than threading a fourth return value through the retry loop
    verdict = extract_verdict_json(report_text)
    valuation = compute_valuation(verdict.valuation_inputs)
    verdict = apply_constitution_overrides(verdict, valuation, brief)
    total_cost_inr = stage1_usage["cost_inr"] + stage2_usage["cost_inr"]

    # v3's placeholder-token contract: a report that passed every other
    # check can still fail to render, e.g. the model used {{pledge_pct}}
    # when pledge was unconfirmed. That's a real, billed analysis that
    # can't be delivered as-is — surfaced distinctly rather than silently
    # storing/sending a report with literal "{{...}}" left in it.
    try:
        rendered_report = render_report(
            report_text, brief.price, brief.technicals, verdict, valuation, brief.shareholding
        )
    except PlaceholderError as exc:
        return PipelineResult(status="render_failed", spent_inr=total_cost_inr, render_error=str(exc))

    # Merge the Python-computed fair-value ranges into the stored dict
    # alongside the model's raw valuation_inputs — bot.py's "format from
    # verdict_json only, never report_md" rule (Prompt 13) needs these
    # available directly rather than recomputing compute_valuation() again
    # at display time.
    verdict_json = {**verdict.model_dump(mode="json"), **valuation.model_dump(mode="json")}
    verdict_json["analysis_price_abs"] = verdict.current_price_abs
    verdict_json["analysis_price_date"] = verdict.price_date.isoformat()
    verdict_json["stage2_mode"] = stage2_mode
    verdict_json["stage2_routing_reasons"] = list(prescan_routing.reasons)
    if settings.force_stage2_full:
        verdict_json["stage2_mode_forced"] = True
    verdict_json = merge_expected_return_into_verdict_json(verdict_json)

    storage.save_analysis(
        ticker=ticker.symbol,
        verdict_json=verdict_json,
        report_md=rendered_report,
        brief_text=to_markdown(brief),
        stage1_tokens=stage1_usage["input_tokens"] + stage1_usage["output_tokens"],
        stage2_tokens=stage2_usage["input_tokens"] + stage2_usage["output_tokens"],
        cost_inr=total_cost_inr,
        validation_passed=True,
        missing=brief.missing,
    )

    analysis = Analysis(
        ticker=ticker.symbol,
        run_date=brief.generated_at.date(),
        verdict_json=verdict_json,
        report_md=rendered_report,
        costs=total_cost_inr,
        validation=validation,
        missing=brief.missing,
    )
    return PipelineResult(status="ok", analysis=analysis)


def run_full_analysis(query: str, max_cache_age_days: int = 7) -> PipelineResult:
    symbol_table = load_symbol_table()
    resolved = resolve_ticker(query, symbol_table)

    if resolved is None:
        return PipelineResult(status="not_found")
    if isinstance(resolved, AmbiguousMatch):
        return PipelineResult(status="ambiguous", candidates=resolved)

    ticker: TickerInfo = resolved

    cached: CacheHit | None = storage.get_cached(ticker.symbol, max_age_days=max_cache_age_days)
    if cached is not None:
        synced_json = sync_live_price_into_verdict(
            cached.analysis.verdict_json,
            live_price_abs=cached.current_price_abs,
            live_price_date=cached.price_date,
        )
        analysis = Analysis(
            ticker=cached.analysis.ticker,
            run_date=cached.analysis.run_date,
            verdict_json=synced_json,
            report_md=cached.analysis.report_md,
            costs=cached.analysis.costs,
            validation=cached.analysis.validation,
            missing=cached.analysis.missing,
        )
        banner = build_staleness_banner(analysis, cached.current_price_abs)
        return PipelineResult(
            status="ok",
            analysis=analysis,
            from_cache=True,
            staleness_banner=banner or None,
        )

    acquired = _ANALYSIS_SLOTS.acquire(blocking=False)
    if not acquired:
        return PipelineResult(status="busy")
    try:
        return _run_paid_analysis(ticker)
    finally:
        _ANALYSIS_SLOTS.release()


def run_portfolio_prescreen_then_analyze(
    symbols: list[str] | None = None,
    *,
    dry_run: bool = False,
    skip_ai: bool = False,
    run_deep: bool = False,
    max_deep: int | None = None,
):
    """Pre-screen the portfolio watchlist, then optionally deep-analyze
    only the 10–18 survivors. See stockbot.portfolio_screener.
    """
    from stockbot.portfolio_screener import (
        ScreenerRunConfig,
        run_prescreen_then_analyze,
    )

    return run_prescreen_then_analyze(
        symbols,
        config=ScreenerRunConfig(
            dry_run=dry_run,
            skip_ai=skip_ai or dry_run,
            run_deep_analysis=run_deep and not dry_run,
            max_deep_analyses=max_deep,
        ),
    )
