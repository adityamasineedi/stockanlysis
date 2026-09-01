"""Module 12 (part 2) — Telegram bot. Polling mode only — no webhook, no
web server, no public IP. python-telegram-bot v22 (confirmed installed;
CommandHandler/MessageHandler/Application.run_polling signatures verified
against the installed package before writing this).

Reply formatting reads verdict_json for the in-chat card (compact by
default). The attached ``.md`` is a reading digest extracted from the full
Stage 2 report — generation and DB storage are unchanged (no token savings). HTML parse mode (not
MarkdownV2 — ₹ and decimals break MarkdownV2's escaping rules), so any
LLM-generated text interpolated into a tag is escaped via html.escape
first, or a stray '<'/'>'/'&' in the model's own output breaks the whole
message.

run_full_analysis is a blocking, synchronous function by design (this
project's own rule: the fetch layer is threads, not asyncio). It's
offloaded to a worker thread via asyncio.to_thread so it doesn't block
the bot's event loop for the several minutes a real analysis takes,
without turning the pipeline itself async.

Real timing, measured live: a full run (brief fetch + Stage 1 + Stage 2)
takes roughly 8–12 minutes end to end, and a Stage 2 validation retry adds
another 5–8 minutes on top of that — Sonnet's adaptive thinking plus a
full 16-section report is genuinely slow to generate, not stuck. The runtime
cap defaults to 15 minutes (ANALYSIS_RUNTIME_CAP_SECONDS) so one retry can finish.
The original "~45s" estimate in ANALYZING_TEMPLATE was never measured
against a real run and badly undersold this; a restart mid-analysis
abandons the in-flight background thread with no recovery — the API call
still gets billed, but the result is never saved or sent. See the
"analysis already running" guard in handle_analyze below, added for the
same reason: a second /analyze for a ticker already in flight would
otherwise silently burn another full analysis's cost while the first one
is still working.
"""

from __future__ import annotations

import asyncio
import html
import logging
import math
from datetime import UTC, datetime, time, timedelta, timezone
from types import SimpleNamespace

from anthropic import APIConnectionError, APITimeoutError, RateLimitError
from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from stockbot.action_ranges import (
    add_more_range_blocked_reason,
    buy_zone_price_ceiling,
    capital_range_blocked_reason,
    format_bear_entry_reference_line,
    resolve_add_more_zone_abs,
)
from stockbot.bot_suggestions import (
    build_inline_query_results,
    build_symbol_pick_keyboard,
    default_pick_tickers,
    parse_pick_callback,
    suggestion_hint_markdown,
)
from stockbot.config import (
    LOGS_DIR,
    REPORTS_DIR,
    parse_telegram_allowed_chat_ids,
    settings,
    setup_logging,
)
from stockbot.constitution_gates import (
    should_anti_chase_from_dict,
    wc_gap_blocks_buy_zone,
)
from stockbot.costs import month_to_date_spend
from stockbot.data_readiness import assemble_brief_for_analysis
from stockbot.expected_return import format_expected_return_telegram
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.models import AmbiguousMatch, Analysis, TickerInfo
from stockbot.monitor.health_audit import run_health_audit
from stockbot.pipeline import (
    ANALYSIS_RUNTIME_CAP_SECONDS,
    MAX_TRUNCATION_RETRIES,
    PER_ANALYSIS_COST_CAP_INR,
    PipelineResult,
    run_full_analysis,
)
from stockbot.portfolio_screener.eligibility import (
    check_deep_analysis_eligibility,
    format_analyze_gate_block,
)
from stockbot.portfolio_screener.outcome_log import build_candidates_messages
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig
from stockbot.portfolio_state import DEFAULT_MAX_POSITION_PCT
from stockbot.report_digest import (
    TELEGRAM_MAX_MISSING,
    TELEGRAM_MAX_REASON_CHARS,
    TELEGRAM_MAX_REASONS,
    TELEGRAM_MAX_WATCH_CHARS,
    _clip,
    _compact_context_flags_line,
    build_compact_attachment_md,
    compact_delivery_note,
)
from stockbot.storage import (
    backfill_cached_verdicts,
    delete_holding,
    get_holding,
    get_latest_verdict_json,
    get_risk_policy,
    get_sip_plan,
    invalidate_cached_analyses,
    list_active_sip_plans,
    list_holdings,
    record_sip_contribution,
    save_holding,
    save_risk_policy,
    save_sip_plan,
    seed_holding_from_sip,
    set_sip_plan_active,
    summarize_sip_contributions,
)
from stockbot.trade_policy import prescan_required_for_analyze

logger = logging.getLogger(__name__)

HEALTH_AUDIT_DAYS = 14

# SIP reminders fire on IST, not the container's UTC — a 10:00 reminder must
# land at 10:00 for the user, and Railway runs in UTC.
_IST = timezone(timedelta(hours=5, minutes=30))

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
ANALYSIS_RUNTIME_CAP_MINUTES = ANALYSIS_RUNTIME_CAP_SECONDS // 60
ANALYZING_TEMPLATE = (
    "⏳ Analyzing {company}... (usually 8–{cap_min} min — please don't restart the bot)"
)
DISCLAIMER = (
    "<i>Educational research only — not investment advice. "
    "Do your own due diligence before any trade.</i>"
)

# Set when user taps /prescan or /analyze from Telegram's menu without a symbol.
AWAITING_PRESCAN_SYMBOL = "awaiting_prescan_symbol"
AWAITING_ANALYZE_SYMBOL = "awaiting_analyze_symbol"


def _clear_awaiting_symbol(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop(AWAITING_PRESCAN_SYMBOL, None)
    context.user_data.pop(AWAITING_ANALYZE_SYMBOL, None)


def _consume_awaiting(context: ContextTypes.DEFAULT_TYPE, key: str) -> bool:
    return bool(context.user_data.pop(key, False))


async def _prompt_for_symbol(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    awaiting_key: str,
    headline: str,
    example: str,
    pick_action: str,
) -> None:
    context.user_data[awaiting_key] = True
    picks = await asyncio.to_thread(default_pick_tickers)
    keyboard = build_symbol_pick_keyboard(picks, action=pick_action)  # type: ignore[arg-type]
    bot_user = context.bot.username if context.bot else None
    inline_tip = suggestion_hint_markdown(bot_user)
    await update.message.reply_text(
        f"{headline}\n\n"
        "Send the <b>NSE symbol or company name</b> or tap a match below.\n"
        f"{inline_tip}\n"
        f"Example: <code>{esc(example)}</code>",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


async def _offer_symbol_picks(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    *,
    action: str,
    awaiting_key: str,
) -> bool:
    """If the fragment matches several names, show tap-to-pick buttons."""
    from stockbot.fetch.tickers import suggest_tickers

    tickers = await asyncio.to_thread(suggest_tickers, text, limit=8)
    if not tickers:
        return False

    query = text.strip()
    if len(tickers) == 1:
        only = tickers[0]
        if only.symbol.upper() == query.upper():
            return False
        resolved = await asyncio.to_thread(resolve_ticker, query)
        if isinstance(resolved, TickerInfo):
            return False

    keyboard = build_symbol_pick_keyboard(tickers, action=action)  # type: ignore[arg-type]
    if keyboard is None:
        return False

    context.user_data[awaiting_key] = True
    await update.message.reply_text(
        f"Matches for <b>{esc(query)}</b> — tap one or send a clearer name:",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
    return True


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


# One deep analysis at a time for the whole bot. While run_full_analysis is
# in flight, every other command/plain-text action is rejected so a second
# prescan or /analyze cannot overlap or burn extra LLM budget.
_analysis_lock = asyncio.Lock()
_analysis_in_progress: str | None = None

BUSY_ANALYSIS_REPLY = (
    "Deep analysis in progress for {company} (usually 8–{cap_min} min). "
    "Please wait for the result — other bot commands are paused until it finishes."
)


async def _active_analysis_query() -> str | None:
    async with _analysis_lock:
        return _analysis_in_progress


async def _try_begin_analysis(query: str) -> bool:
    global _analysis_in_progress
    normalized = query.strip()
    async with _analysis_lock:
        if _analysis_in_progress is not None:
            return False
        _analysis_in_progress = normalized
        return True


async def _end_analysis() -> None:
    global _analysis_in_progress
    async with _analysis_lock:
        _analysis_in_progress = None


async def _reject_if_analysis_busy(update: Update) -> bool:
    """Return True when input was rejected because an analysis is running."""
    active = await _active_analysis_query()
    if active is None:
        return False
    reply = BUSY_ANALYSIS_REPLY.format(
        company=esc(active), cap_min=ANALYSIS_RUNTIME_CAP_MINUTES
    )
    if update.message is not None:
        await update.message.reply_text(reply)
    elif update.callback_query is not None:
        await update.callback_query.answer(reply[:200], show_alert=True)
    return True


async def _reject_if_unauthorized(update: Update) -> bool:
    """Return True when the chat is blocked (message already sent)."""
    allowed = parse_telegram_allowed_chat_ids()
    if not allowed:
        return False
    chat = update.effective_chat
    if chat is not None and chat.id in allowed:
        return False
    if update.message is not None:
        await update.message.reply_text("This bot is restricted to authorized users only.")
    return True


def _money_pair(pair: object) -> tuple[str, str] | None:
    """Format a [low, high] money pair to 2dp, or None if unusable."""
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return None
    low, high = pair[0], pair[1]
    if low is None or high is None:
        return None
    try:
        return f"{float(low):.2f}", f"{float(high):.2f}"
    except (TypeError, ValueError):
        return None


def _format_price_abs(value: object) -> str:
    """Format a price for Telegram display (2dp, no yfinance float noise)."""
    if value is None:
        return "?"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _resolve_scenario_fair_value(
    verdict_json: dict,
    scenario: str,
) -> tuple[str, str] | None:
    """Fair-value band for bear / base / bull — never bear-low–bull-high combined."""
    key = f"fair_value_{scenario}_abs"
    formatted = _money_pair(verdict_json.get(key))
    if formatted is not None:
        return formatted

    raw_inputs = verdict_json.get("valuation_inputs")
    if isinstance(raw_inputs, dict):
        try:
            from stockbot.llm.verdict import ValuationInputs, compute_valuation

            valuation = compute_valuation(ValuationInputs.model_validate(raw_inputs))
            return _money_pair(getattr(valuation, key))
        except Exception:
            logger.exception("Could not recompute %s for Telegram card", key)

    if scenario == "base":
        return _money_pair(verdict_json.get("fair_value_abs"))
    return None


def _resolve_base_fair_value(verdict_json: dict) -> tuple[str, str] | None:
    """Headline FV is always the BASE range — never bear-low–bull-high."""
    return _resolve_scenario_fair_value(verdict_json, "base")


def _capital_range_gate_context(verdict_json: dict) -> tuple[bool, bool, str | None]:
    """Shared constitution gates for buy/add range display."""
    wc_gap = verdict_json.get("wc_gap_classification")
    wc_gap_norm = str(wc_gap).strip().upper() if wc_gap else None
    cash_gap_blocks = wc_gap_blocks_buy_zone(wc_gap)
    anti_chase = bool(verdict_json.get("anti_chase_flag")) or should_anti_chase_from_dict(
        verdict_json
    )[0]
    return anti_chase, cash_gap_blocks, wc_gap_norm


def _range_block_label(reason: str | None) -> str:
    """Card label for a blocked capital range.

    Shared by the buy and add-more lines so the two cannot describe the same
    gate differently — or, as before, so one cannot stay silent about a gate
    the other names.
    """
    if not reason:
        return ""
    if reason.startswith("anti-chase"):
        return "anti-chase: pause new capital"
    for prefix, label in (
        ("WC:", "WC"),
        ("five-year test:", "five-year"),
        ("thesis:", "thesis"),
    ):
        if reason.startswith(prefix):
            return f"{label}: {esc(reason.removeprefix(prefix).strip())}"
    return ""


def _format_buy_range_line(verdict_json: dict) -> str:
    buy_zone = _money_pair(verdict_json.get("buy_zone_abs"))
    buy_range_allowed = verdict_json.get("buy_range_allowed")
    anti_chase, cash_gap_blocks, _wc_gap_norm = _capital_range_gate_context(verdict_json)
    buy_zone_ok = (
        buy_zone is not None
        and buy_range_allowed is True
        and not cash_gap_blocks
        and not anti_chase
    )
    if buy_zone_ok:
        return f"Buy range: ₹{buy_zone[0]}–₹{buy_zone[1]}"
    # Suppression itself is unchanged above; this only explains it — the gate
    # that fired (including the five-year test the old branch chain missed)
    # and the price bar a buy zone has to clear, which was never shown at all.
    parts = [p for p in (_range_block_label(capital_range_blocked_reason(verdict_json)),) if p]
    ceiling = buy_zone_price_ceiling(verdict_json)
    # Only worth saying when the price is the thing in the way. A stock at ₹150
    # told it "needs ≤₹176.40" reads as nonsense — it already clears the bar,
    # and the named gate above is the whole story. Absent price: stay quiet
    # rather than guess which side of the bar it sits on.
    price = verdict_json.get("current_price_abs")
    if ceiling is not None and isinstance(price, (int, float)) and float(price) > ceiling[0]:
        parts.append(f"needs ≤₹{ceiling[0]:.2f} at {esc(ceiling[1])} risk")
    if parts:
        return f"Buy range: not issued ({' · '.join(parts)})"
    return "Buy range: not issued"


def _format_sell_range_line(verdict_json: dict) -> str:
    base_fv = _resolve_scenario_fair_value(verdict_json, "base")
    if base_fv is not None:
        return f"Sell range: ₹{base_fv[0]}–₹{base_fv[1]}"
    return "Sell range: unavailable"


def _format_add_more_range_line(verdict_json: dict) -> str:
    block_reason = add_more_range_blocked_reason(verdict_json)
    if block_reason:
        label = _range_block_label(block_reason)
        return f"Add-more range: not issued ({label})" if label else "Add-more range: not issued"

    add_zone = resolve_add_more_zone_abs(verdict_json)
    if add_zone is None:
        return "Add-more range: unavailable"
    low, high = f"{add_zone[0]:.2f}", f"{add_zone[1]:.2f}"
    return f"Add-more range: ₹{low}–₹{high} (on-dip · bear FV)"


def _format_take_profit_targets_line(verdict_json: dict) -> str:
    bull_fv = _resolve_scenario_fair_value(verdict_json, "bull")
    if bull_fv is not None:
        return f"Take-profit targets: ₹{bull_fv[0]}–₹{bull_fv[1]}"
    return "Take-profit targets: unavailable"


def _format_execution_pm_line(verdict_json: dict) -> str | None:
    pm = verdict_json.get("execution_pm")
    if not isinstance(pm, dict):
        return None
    parts: list[str] = []
    bucket = pm.get("sip_bucket")
    if bucket:
        tranche = pm.get("suggested_tranche_inr")
        if isinstance(tranche, (int, float)) and tranche > 0:
            parts.append(f"SIP bucket {bucket}: ~₹{tranche:.0f}/tranche (25% of monthly)")
        else:
            parts.append(f"SIP bucket: {bucket}")
    peer_pct = pm.get("peer_pe_percentile")
    peer_n = pm.get("peer_count")
    if isinstance(peer_pct, (int, float)) and isinstance(peer_n, int) and peer_n > 0:
        parts.append(f"P/E vs peers: {peer_pct:.0f}th pctile (n={peer_n})")
    bb = pm.get("bollinger_position")
    trend = pm.get("trend_label")
    if bb or trend:
        tech_bits = [b for b in (bb, trend) if b]
        parts.append("Technicals: " + ", ".join(str(b) for b in tech_bits))
    div = pm.get("diversification_note")
    if div:
        parts.append(str(div))
    if not parts:
        return None
    return "PM: " + " · ".join(parts)


def _format_profit_review_line(verdict_json: dict) -> str | None:
    profit_review = verdict_json.get("profit_review")
    if not isinstance(profit_review, dict):
        return None
    status = str(profit_review.get("status") or "").strip().upper()
    if status != "REVIEW_FOR_REBALANCING":
        return None
    return "Profit review: rebalance review triggered (not an automatic sell)"


def _format_stage2_mode_line(verdict_json: dict) -> str | None:
    """Telegram card line for lite vs full Stage 2 (omitted on older cache rows)."""
    raw_mode = verdict_json.get("stage2_mode")
    if raw_mode not in ("LITE", "FULL"):
        return None
    detail = "Haiku compact report" if raw_mode == "LITE" else "Sonnet deep report"
    if verdict_json.get("stage2_mode_forced"):
        detail += " (config override)"
    return f"Stage 2: <b>{esc(str(raw_mode))}</b> · {esc(detail)}"


def format_verdict_reply(
    analysis: Analysis,
    *,
    staleness_banner: str | None = None,
    compact: bool = True,
    full_attachment: bool = False,
    from_cache: bool = False,
    position_line: str | None = None,
) -> str:
    v = analysis.verdict_json
    # cash_gap_blocks is applied inside _format_buy_range_line/_format_add_more_range_line,
    # not here — this scope only needs the anti-chase and WC-gap labels.
    anti_chase, _cash_gap_blocks, wc_gap_norm = _capital_range_gate_context(v)

    lines: list[str] = []
    if staleness_banner:
        lines.append(esc(staleness_banner))
        lines.append("")

    five_year = v.get("five_year_business_test") or {}
    five_year_answer = five_year.get("answer") if isinstance(five_year, dict) else None

    action_range_lines = [
        _format_buy_range_line(v),
        _format_sell_range_line(v),
        _format_add_more_range_line(v),
        _format_take_profit_targets_line(v),
    ]
    entry_ref = format_bear_entry_reference_line(v)
    if entry_ref and capital_range_blocked_reason(v) is not None:
        action_range_lines.append(entry_ref)
    profit_review_line = _format_profit_review_line(v)
    if profit_review_line:
        action_range_lines.append(profit_review_line)
    execution_pm_line = _format_execution_pm_line(v)
    if execution_pm_line:
        action_range_lines.append(execution_pm_line)
    # Sits with the action ranges because it is the same kind of information:
    # what to do with money. Passed in rather than looked up here so this stays
    # a pure formatter — the caller knows the chat, this does not. Absent when
    # no capital is declared; never a percentage of a guessed denominator.
    if position_line:
        action_range_lines.append(position_line)

    lines.extend(
        [
            f"<b>{esc(v.get('verdict', '?'))}</b> — {esc(analysis.ticker)}",
            *action_range_lines,
            f"Price: ₹{_format_price_abs(v.get('current_price_abs'))} (as of {esc(v.get('price_date', '?'))})",
            f"Risk: {esc(v.get('risk', '?'))} · Confidence: {v.get('confidence', '?')}/10",
            f"Holding Period: {esc(v.get('holding_period', '?'))}",
        ]
    )
    tension = v.get("external_valuation_tension")
    stage2_line = _format_stage2_mode_line(v)
    if compact:
        flags_line = _compact_context_flags_line(
            five_year_answer=str(five_year_answer) if five_year_answer else None,
            wc_gap_norm=wc_gap_norm,
            anti_chase=anti_chase,
            tension=tension,
            thesis_status=v.get("thesis_status"),
        )
        if flags_line:
            lines.append(esc(flags_line))
    else:
        if stage2_line:
            lines.append(stage2_line)
        if five_year_answer:
            lines.append(f"5y business test: {esc(str(five_year_answer))}")
        if wc_gap_norm:
            lines.append(f"WC gap: {esc(wc_gap_norm)}")
        if anti_chase:
            lines.append("Anti-chase: pause new capital — valuation recheck")
        if tension and str(tension).upper() not in ("NONE", ""):
            lines.append(f"Valuation tension: {esc(str(tension))}")
        if v.get("thesis_status"):
            lines.append(f"Thesis: {esc(str(v.get('thesis_status')))}")
    for er_line in format_expected_return_telegram(
        v.get("expected_return") or {},
        compact=compact,
    ):
        lines.append(esc(er_line))
    lines.extend(
        [
            "",
            "<b>Why buy</b>",
        ]
    )
    reasons_buy = v.get("reasons_buy") or []
    if compact:
        reasons_buy = reasons_buy[:TELEGRAM_MAX_REASONS]
    for reason in reasons_buy:
        text = _clip(reason, TELEGRAM_MAX_REASON_CHARS) if compact else str(reason)
        lines.append(f"• {esc(text)}")
    lines.append("")
    lines.append("<b>Why avoid</b>")
    reasons_avoid = v.get("reasons_avoid") or []
    if compact:
        reasons_avoid = reasons_avoid[:TELEGRAM_MAX_REASONS]
    for reason in reasons_avoid:
        text = _clip(reason, TELEGRAM_MAX_REASON_CHARS) if compact else str(reason)
        lines.append(f"• {esc(text)}")
    lines.append("")
    watch = v.get("biggest_watch", "?")
    if compact:
        watch = _clip(watch, TELEGRAM_MAX_WATCH_CHARS)
    lines.append(f"<b>Biggest watch:</b> {esc(watch)}")

    missing = analysis.missing
    if compact and len(missing) > TELEGRAM_MAX_MISSING:
        for item in missing[:TELEGRAM_MAX_MISSING]:
            lines.append(f"⚠️ {esc(item)}")
        lines.append(f"⚠️ … +{len(missing) - TELEGRAM_MAX_MISSING} more data gaps")
    else:
        for item in missing:
            lines.append(f"⚠️ {esc(item)}")

    lines.append("")
    lines.append(DISCLAIMER)
    if compact:
        note = compact_delivery_note(from_cache=from_cache, full_attachment=full_attachment)
        lines.append(f"<i>{esc(note)}</i>")

    text = "\n".join(lines)
    if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
        text = text[: TELEGRAM_MAX_MESSAGE_LENGTH - 20].rstrip() + "\n(truncated)"
    return text


def format_ambiguous_reply(candidates: AmbiguousMatch) -> str:
    lines = ["Multiple companies match — reply with the exact symbol:"]
    for i, candidate in enumerate(candidates.candidates, start=1):
        lines.append(f"{i}. <code>{esc(candidate.symbol)}</code> — {esc(candidate.company_name)}")
    return "\n".join(lines)


async def _deliver_result(
    update: Update,
    result: PipelineResult,
    status_message,
    *,
    full_report: bool = False,
) -> None:
    if result.status == "not_found":
        await status_message.edit_text(
            "Couldn't find that company. Check the spelling, or try the exact NSE symbol."
        )
        return

    if result.status == "ambiguous":
        await status_message.edit_text(
            format_ambiguous_reply(result.candidates), parse_mode=ParseMode.HTML
        )
        return

    if result.status == "busy":
        await status_message.edit_text(
            
                "Another analysis is already running. Please wait for it to finish "
                f"(usually 8–{ANALYSIS_RUNTIME_CAP_MINUTES} min), then try again."
            
        )
        return

    if result.status == "budget_exceeded":
        extra = ""
        if result.validation_failures:
            extra = "\n\n" + "\n".join(esc(f) for f in result.validation_failures)
        await status_message.edit_text(
            f"Monthly budget reached (₹{result.spent_inr:.0f} spent this month). "
            f"No new analyses until next month — the cap is real, not advisory."
            f"{extra}"
        )
        return

    if result.status == "data_unready":
        failures = "\n".join(f"- {esc(f)}" for f in (result.validation_failures or []))
        await status_message.edit_text(
            "Data preflight failed — no LLM tokens spent.\n"
            "Free sources and fallbacks did not gather enough for a full report.\n\n"
            f"{failures}\n\n"
            "Run /preflight SYMBOL for the full source checklist, then retry /analyze.",
            parse_mode=ParseMode.HTML,
        )
        return

    if result.status == "insufficient_data":
        failures = "\n".join(f"- {esc(f)}" for f in (result.validation_failures or []))
        spent_note = (
            f"\n\n(₹{result.spent_inr:.2f} spent on this attempt — logged, not saved as a report.)"
            if result.spent_inr
            else ""
        )
        await status_message.edit_text(
            "Insufficient data for a confident view after validation. "
            f"This is an honest answer, not a bug.{spent_note}\n\n{failures}"
        )
        return

    if result.status == "analysis_cost_exceeded":
        await status_message.edit_text(
            f"This analysis hit its per-run cost cap (₹{PER_ANALYSIS_COST_CAP_INR:.0f}) before "
            f"producing a validated verdict (₹{result.spent_inr:.2f} spent on this attempt). "
            f"Try again later, or send /spend to check the monthly total."
        )
        return

    if result.status == "analysis_truncated":
        attempts = result.truncation_attempts or (MAX_TRUNCATION_RETRIES + 1)
        await status_message.edit_text(
            f"Stage 2 output was cut off {attempts} time(s) (each try used a higher token "
            f"budget) before the report could finish (₹{result.spent_inr:.2f} spent — "
            f"not the ₹{PER_ANALYSIS_COST_CAP_INR:.0f} cap). Long FULL analyses with fat "
            f"annual reports hit this rarest now. Send /spend to check the monthly total, "
            f"or retry later."
        )
        return

    if result.status == "analysis_runtime_exceeded":
        cap_min = ANALYSIS_RUNTIME_CAP_MINUTES
        lines = [
            (
                f"Analysis hit the {cap_min}-minute time cap "
                f"(₹{result.spent_inr:.2f} billed on this attempt — all stages included)."
            ),
        ]
        if result.validation_failures:
            lines.append(
                "Stage 2 ran but validation did not pass — a retry was blocked by the time cap."
            )
            failures = "\n".join(f"- {esc(f)}" for f in result.validation_failures[:6])
            lines.append(f"\nValidation issues:\n{failures}")
        else:
            lines.append(
                "Brief fetch or Stage 2 may still have been in progress when the cap hit."
            )
        lines.append(
            f"\nRetry /analyze when you can wait up to {cap_min} min — do not restart the bot mid-run."
        )
        await status_message.edit_text("\n".join(lines), parse_mode=ParseMode.HTML)
        return

    if result.status == "render_failed":
        await status_message.edit_text(
            f"The analysis passed validation but couldn't be rendered "
            f"(₹{result.spent_inr:.2f} spent, not wasted — logged either way): "
            f"{esc(result.render_error)}"
        )
        return

    analysis = result.analysis
    banner_parts: list[str] = []
    if result.cache_miss_reason:
        banner_parts.append(result.cache_miss_reason)
    if result.staleness_banner:
        banner_parts.append(result.staleness_banner)
    combined_banner = "\n".join(banner_parts) if banner_parts else None
    chat = update.effective_chat
    position_line = (
        await asyncio.to_thread(_build_position_line, chat.id, analysis.ticker)
        if chat is not None
        else None
    )
    await status_message.edit_text(
        format_verdict_reply(
            analysis,
            staleness_banner=combined_banner,
            full_attachment=full_report,
            from_cache=result.from_cache,
            position_line=position_line,
        ),
        parse_mode=ParseMode.HTML,
    )
    await _send_report_attachment(
        update,
        analysis,
        compact=not full_report,
        from_cache=result.from_cache,
    )


def _write_report_file(analysis: Analysis, *, compact: bool = True, from_cache: bool = False):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "digest" if compact else "full"
    report_path = REPORTS_DIR / f"{analysis.ticker}_{analysis.run_date.isoformat()}_{suffix}.md"
    body = (
        build_compact_attachment_md(analysis, from_cache=from_cache)
        if compact
        else analysis.report_md
    )
    report_path.write_text(body, encoding="utf-8")
    return report_path


async def _send_report_attachment(
    update: Update,
    analysis: Analysis,
    *,
    compact: bool = True,
    from_cache: bool = False,
) -> None:
    report_path = await asyncio.to_thread(
        _write_report_file, analysis, compact=compact, from_cache=from_cache
    )
    document_bytes = await asyncio.to_thread(report_path.read_bytes)
    await update.message.reply_document(document=document_bytes, filename=report_path.name)


async def _send_progress_updates(status_message, query: str) -> None:
    minutes = 0
    try:
        while True:
            await asyncio.sleep(60)
            minutes += 1
            try:
                elapsed_note = ""
                if minutes >= ANALYSIS_RUNTIME_CAP_MINUTES:
                    elapsed_note = (
                        f" — over {ANALYSIS_RUNTIME_CAP_MINUTES} min; "
                        f"new LLM retries are blocked to save budget"
                    )
                await status_message.edit_text(
                    f"⏳ Still analyzing {esc(query)}... ({minutes} min elapsed{elapsed_note} — "
                    f"please don't restart the bot)"
                )
            except Exception:
                # Cosmetic progress ping — a transient Telegram/network hiccup here (we've seen
                # httpx.ReadError on this exact polling connection live) must not take down the
                # analysis itself, but it's still logged rather than silently swallowed.
                logger.warning("Progress update edit failed for %r", query, exc_info=True)
    except asyncio.CancelledError:
        pass


async def _check_analyze_eligibility_gate(update: Update, query: str) -> bool:
    """Quant-only prescan gate — no eligibility AI spend on every /analyze."""
    try:
        result = await asyncio.to_thread(
            check_deep_analysis_eligibility,
            query,
            config=ScreenerRunConfig(ai_provider="auto", skip_ai=True),
        )
    except Exception as exc:
        logger.exception("analyze eligibility gate failed for %r", query)
        await update.message.reply_text(
            f"Could not run eligibility check: {esc(exc)}. "
            "Try /prescan first, or /analyze force if you accept the risk."
        )
        return False

    if result.suitable_for_deep_analysis:
        logger.info(
            "Analyze gate passed for %r: verdict=%s ticker=%s",
            query,
            result.verdict,
            result.ticker,
        )
        return True

    await update.message.reply_text(
        format_analyze_gate_block(result),
        parse_mode=ParseMode.HTML,
    )
    return False


async def _run_and_reply(
    update: Update,
    query: str,
    *,
    force: bool = False,
    full_report: bool = False,
    skip_cache: bool = False,
) -> None:
    if not await _try_begin_analysis(query):
        active = await _active_analysis_query()
        await update.message.reply_text(
            BUSY_ANALYSIS_REPLY.format(
                company=esc(active or query), cap_min=ANALYSIS_RUNTIME_CAP_MINUTES
            )
        )
        return

    try:
        if prescan_required_for_analyze() and not force:
            allowed = await _check_analyze_eligibility_gate(update, query)
            if not allowed:
                return

        status_message = await update.message.reply_text(
            ANALYZING_TEMPLATE.format(
                company=esc(query), cap_min=ANALYSIS_RUNTIME_CAP_MINUTES
            )
        )
        progress_task = asyncio.create_task(_send_progress_updates(status_message, query))
        try:
            try:
                result = await asyncio.to_thread(
                    run_full_analysis, query, skip_cache=skip_cache
                )
            except Exception as exc:
                logger.exception("run_full_analysis failed for %r", query)
                if isinstance(exc, (APIConnectionError, APITimeoutError)):
                    await status_message.edit_text(
                        "Network error reaching the LLM API — nothing was saved. "
                        "Wait a minute and retry /analyze (transient errors retry automatically). "
                        f"Detail: {esc(str(exc))}"
                    )
                elif isinstance(exc, RateLimitError):
                    await status_message.edit_text(
                        "LLM rate limit hit — nothing was saved. "
                        "Wait a few minutes and retry /analyze. "
                        f"Detail: {esc(str(exc))}"
                    )
                else:
                    await status_message.edit_text(f"Something went wrong: {esc(exc)}")
                return
        finally:
            progress_task.cancel()

        await _deliver_result(update, result, status_message, full_report=full_report)
    finally:
        await _end_analysis()


async def handle_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    query, force, full_report, fresh = _parse_analyze_command_args(context.args)
    if not query:
        await _prompt_for_symbol(
            update,
            context,
            awaiting_key=AWAITING_ANALYZE_SYMBOL,
            headline="Which stock should I analyze?",
            example="BEL",
            pick_action="analyze",
        )
        return
    _clear_awaiting_symbol(context)
    await _run_and_reply(
        update, query, force=force, full_report=full_report, skip_cache=fresh
    )


def _parse_analyze_modifiers(args: list[str] | None) -> tuple[bool, bool, bool]:
    parts = list(args or [])
    force = False
    full_report = False
    fresh = False
    for token in parts:
        low = token.lower()
        if low == "force":
            force = True
        elif low == "full":
            full_report = True
        elif low == "fresh":
            fresh = True
    return force, full_report, fresh


def _parse_analyze_command_args(args: list[str] | None) -> tuple[str, bool, bool, bool]:
    parts = list(args or [])
    force, full_report, fresh = _parse_analyze_modifiers(parts)
    cleaned: list[str] = []
    for token in parts:
        if token.lower() in {"force", "full", "fresh"}:
            continue
        cleaned.append(token)
    return " ".join(cleaned).strip(), force, full_report, fresh


async def handle_prescan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    query = " ".join(context.args) if context.args else ""
    if not query:
        await _prompt_for_symbol(
            update,
            context,
            awaiting_key=AWAITING_PRESCAN_SYMBOL,
            headline="Which stock should I pre-scan?",
            example="BEL",
            pick_action="prescan",
        )
        return
    _clear_awaiting_symbol(context)
    await _run_prescan_and_reply(update, query)


async def _run_prescan_and_reply(update: Update, query: str) -> None:
    status = await update.message.reply_text(
        f"🔎 Pre-scanning {esc(query)} (cheap eligibility — not full analysis)…"
    )
    try:
        result = await asyncio.to_thread(
            check_deep_analysis_eligibility,
            query,
            config=ScreenerRunConfig(ai_provider="auto"),
        )
    except Exception as exc:  # resilience boundary — always reply
        logger.exception("prescan failed for %r", query)
        await status.edit_text(f"Pre-scan failed: {esc(exc)}")
        return
    await status.edit_text(result.telegram_html(), parse_mode=ParseMode.HTML)


def _parse_prescan_plain_text(text: str) -> str | None:
    """Accept 'prescan BEL' / 'pre-scan BEL' as plain text shortcuts."""
    lowered = text.strip()
    for prefix in ("prescan ", "pre-scan ", "prescreen "):
        if lowered.lower().startswith(prefix):
            rest = text.strip()[len(prefix) :].strip()
            return rest or None
    return None


def _parse_analyze_plain_text(text: str) -> tuple[str, bool, bool, bool] | None:
    """Accept 'force BEL', 'full CAMS', 'fresh BEL', or 'force full BEL' shortcuts."""
    stripped = text.strip()
    if not stripped:
        return None
    parts = stripped.split()
    force, full_report, fresh = _parse_analyze_modifiers(parts)
    cleaned = [p for p in parts if p.lower() not in {"force", "full", "fresh"}]
    if not cleaned:
        return None
    if not force and not full_report and not fresh:
        return None
    return " ".join(cleaned), force, full_report, fresh


async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    text = (update.message.text or "").strip()
    if not text:
        return
    prescan_query = _parse_prescan_plain_text(text)
    if prescan_query is not None:
        _clear_awaiting_symbol(context)
        await _run_prescan_and_reply(update, prescan_query)
        return
    analyze_query = _parse_analyze_plain_text(text)
    if analyze_query is not None:
        _clear_awaiting_symbol(context)
        query, force, full_report, fresh = analyze_query
        await _run_and_reply(
            update, query, force=force, full_report=full_report, skip_cache=fresh
        )
        return
    if _consume_awaiting(context, AWAITING_PRESCAN_SYMBOL):
        if await _offer_symbol_picks(
            update,
            context,
            text,
            action="prescan",
            awaiting_key=AWAITING_PRESCAN_SYMBOL,
        ):
            return
        _clear_awaiting_symbol(context)
        await _run_prescan_and_reply(update, text)
        return
    if _consume_awaiting(context, AWAITING_ANALYZE_SYMBOL):
        if await _offer_symbol_picks(
            update,
            context,
            text,
            action="analyze",
            awaiting_key=AWAITING_ANALYZE_SYMBOL,
        ):
            return
        _clear_awaiting_symbol(context)
        await _run_and_reply(update, text, force=False)
        return
    await _run_and_reply(update, text)


async def handle_candidates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    args = list(context.args or [])
    chunks, err = await asyncio.to_thread(build_candidates_messages, args)
    if err:
        await update.message.reply_text(err, parse_mode=ParseMode.HTML)
        return
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)


async def handle_preflight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Free data-layer audit — no LLM spend."""
    if await _reject_if_unauthorized(update):
        return
    _clear_awaiting_symbol(context)
    query = " ".join(context.args or []).strip()
    if not query:
        await update.message.reply_text(
            "Send a symbol to check data readiness before /analyze, e.g.\n"
            "<code>/preflight ADVENZYMES</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    status = await update.message.reply_text(f"Preflight {esc(query)} — fetching free sources…")
    try:
        table = await asyncio.to_thread(load_symbol_table)
        resolved = await asyncio.to_thread(resolve_ticker, query, table)
    except Exception as exc:  # noqa: BLE001 - preflight resilience; always reply
        await status.edit_text(f"Preflight failed: {esc(exc)}")
        return
    if resolved is None:
        await status.edit_text("Couldn't find that company. Try the exact NSE symbol.")
        return
    if isinstance(resolved, AmbiguousMatch):
        await status.edit_text(format_ambiguous_reply(resolved), parse_mode=ParseMode.HTML)
        return
    ticker: TickerInfo = resolved
    try:
        _brief, report = await asyncio.to_thread(assemble_brief_for_analysis, ticker)
    except Exception as exc:
        logger.exception("preflight failed for %r", query)
        await status.edit_text(f"Preflight fetch failed: {esc(exc)}")
        return
    await status.edit_text(report.telegram_summary(), parse_mode=ParseMode.HTML)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    bot_user = context.bot.username if context.bot else None
    inline_tip = suggestion_hint_markdown(bot_user)
    await update.message.reply_text(
        "Commands:\n"
        "/prescan — tap from menu, then send the stock name\n"
        "/prescan <symbol> — one-step example: /prescan BEL\n"
        "/candidates — list analyze-ready names from prescan history\n"
        "/candidates strong|candidate|watchlist — filter by score tier\n"
        "/candidates quality 65 — Quality ≥65 and analyze-ready\n"
        "/analyze — tap from menu, then send the stock name\n"
        "/analyze <company> — deep analysis (digest .md attached by default)\n"
        "/analyze full <symbol> — attach full §1–§16 report .md\n"
        "/analyze fresh <symbol> — skip cache, run new paid analysis\n"
        "/analyze force <symbol> — bypass gate (not recommended)\n"
        "/preflight <symbol> — free data audit before /analyze (no LLM spend)\n"
        "Trade-friendly mode ON: easier buy ranges on UNCERTAIN+evidence; prescan skipped for /analyze\n"
        "/refresh SYMBOL — clear cached analysis for symbol\n"
        "/refresh backfill — recompute gates + expected_return on cached rows\n"
        "/capital 500000 max 10 — total capital and per-stock cap\n"
        "/hold BEL 25 412.50 — record a position (/hold to list)\n"
        "/track — score the bot's past calls (/track analyze for verdicts)\n"
        "/sip BEL 5000 — monthly plan with dip alerts and projections\n"
        "/sip plan — ₹60k portfolio tables (3 buckets, live prices)\n"
        "/sip track — planned vs logged this month\n"
        "/sip prescan — batch prescan all portfolio names (quant-only)\n"
        "/sip paid BEL 3213 — log a portfolio buy\n"
        "/sip status|paid|pause|resume — single-stock plan\n"
        "/spend — month-to-date cost\n"
        "/health — cost/token/quality audit (no LLM spend)\n\n"
        "Tip: after /prescan, just reply with BEL (no need to type prescan again).\n"
        f"{inline_tip}\n"
        "(One-time in BotFather: /setinline — enables @-mention suggestions.)\n\n"
        "Educational research only — not investment advice."
    )


async def handle_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    args = list(context.args or [])
    if not args:
        await update.message.reply_text(
            "Usage:\n"
            "/refresh SYMBOL — drop cached analysis for that symbol\n"
            "/refresh backfill — recompute constitution gates + expected_return "
            "on all cached analyses (no LLM spend)"
        )
        return
    if args[0].lower() == "backfill":
        result = await asyncio.to_thread(backfill_cached_verdicts)
        await update.message.reply_text(
            f"Backfill complete: {result.rows_updated} updated, "
            f"{result.rows_skipped} unchanged "
            f"(scanned {result.rows_scanned} rows)."
        )
        return
    symbol = args[0].upper()
    deleted = await asyncio.to_thread(invalidate_cached_analyses, symbol)
    await update.message.reply_text(
        f"Cleared {deleted} cached row(s) for {esc(symbol)}. "
        f"Next /analyze will run fresh."
        )


async def _reject_inline_if_unauthorized(update: Update) -> bool:
    allowed = parse_telegram_allowed_chat_ids()
    if not allowed:
        return False
    user = update.inline_query.from_user if update.inline_query else None
    if user is not None and user.id in allowed:
        return False
    await update.inline_query.answer([], cache_time=1)
    return True


async def handle_inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_inline_if_unauthorized(update):
        return
    query = (update.inline_query.query or "").strip()
    if len(query) < 1:
        tickers = await asyncio.to_thread(default_pick_tickers, 12)
    else:
        from stockbot.fetch.tickers import suggest_tickers

        tickers = await asyncio.to_thread(suggest_tickers, query, limit=12)
    results = build_inline_query_results(tickers, action="prescan")
    await update.inline_query.answer(results, cache_time=30, is_personal=True)


async def _reject_callback_if_unauthorized(update: Update) -> bool:
    cq = update.callback_query
    if cq is None or cq.message is None:
        return True
    allowed = parse_telegram_allowed_chat_ids()
    if not allowed:
        return False
    if cq.message.chat_id in allowed:
        return False
    await cq.answer("Not authorized.", show_alert=True)
    return True


async def handle_symbol_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cq = update.callback_query
    if cq is None:
        return
    if await _reject_callback_if_unauthorized(update):
        return
    parsed = parse_pick_callback(cq.data)
    if parsed is None:
        await cq.answer("Unknown selection")
        return
    action, symbol = parsed
    await cq.answer()
    _clear_awaiting_symbol(context)
    if cq.message is None:
        return
    shim = SimpleNamespace(message=cq.message)
    if action == "prescan":
        await _run_prescan_and_reply(shim, symbol)
    else:
        await _run_and_reply(shim, symbol, force=False)


def _build_track_report(source: str) -> str:
    """Calibration text for one source. Fetches prices, so runs off-thread."""
    from stockbot.calibration_report import (
        build_analyze_calibration,
        build_prescan_calibration,
        format_calibration_report,
    )

    builder = build_analyze_calibration if source == "analyze" else build_prescan_calibration
    return format_calibration_report(builder())


async def handle_track(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Has the bot been right? Reads history it has already been logging."""
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    args = list(context.args or [])
    source = "analyze" if args and args[0].lower() in {"analyze", "analyse"} else "prescan"

    # One price fetch per distinct ticker, so this is slow enough to warrant a
    # status message but costs nothing in LLM spend.
    status = await update.message.reply_text("⏳ Scoring past calls against live prices…")
    try:
        text = await asyncio.to_thread(_build_track_report, source)
    except Exception as exc:
        logger.exception("Calibration report failed")
        await status.edit_text(f"Could not build the track record: {esc(exc)}")
        return
    await status.edit_text(f"{text}\n\n{DISCLAIMER}", parse_mode=ParseMode.HTML)


async def handle_spend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    spent = await asyncio.to_thread(month_to_date_spend)
    await update.message.reply_text(
        f"Month-to-date spend: ₹{spent:.2f} of ₹{settings.monthly_budget_inr:.0f}"
    )


def _write_health_audit_file(markdown: str):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).date().isoformat()
    path = LOGS_DIR / f"health_audit_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


async def handle_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    status = await update.message.reply_text("⏳ Running health audit…")
    try:
        report = await asyncio.to_thread(run_health_audit, days=HEALTH_AUDIT_DAYS)
    except Exception as exc:
        logger.exception("health audit failed")
        await status.edit_text(f"Health audit failed: {esc(exc)}")
        return

    await status.edit_text(report.to_telegram_html(), parse_mode=ParseMode.HTML)

    if report.critical_count or report.warning_count:
        audit_path = await asyncio.to_thread(_write_health_audit_file, report.to_markdown())
        document_bytes = await asyncio.to_thread(audit_path.read_bytes)
        await update.message.reply_document(
            document=document_bytes,
            filename=audit_path.name,
        )


BOT_COMMANDS = [
    BotCommand("prescan", "Tap, then send stock name (or /prescan BEL)"),
    BotCommand("candidates", "Prescan picks with plain-English labels"),
    BotCommand("analyze", "Tap, then send stock name (or /analyze BEL)"),
    BotCommand("refresh", "Clear cache or backfill stored verdicts"),
    BotCommand("help", "Usage instructions"),
    BotCommand("spend", "Month-to-date cost"),
    BotCommand("health", "Cost/token/quality audit"),
    BotCommand("sip", "Portfolio plan, track, or single-stock SIP"),
    BotCommand("capital", "Set total capital and per-stock cap"),
    BotCommand("hold", "Record and review what you own"),
    BotCommand("track", "Has the bot been right? Past calls scored"),
]


CAPITAL_USAGE = (
    "<b>/capital — your risk policy</b>\n"
    "<code>/capital 500000</code> — total investable capital\n"
    "<code>/capital 500000 max 8</code> — and cap any one stock at 8%\n"
    "<code>/capital</code> — show the current policy\n\n"
    "<i>Without this the bot cannot say what fraction of your money a position "
    "is, so it won't guess one.</i>"
)

HOLD_USAGE = (
    "<b>/hold — what you own</b>\n"
    "<code>/hold BEL 25 412.50</code> — 25 shares at ₹412.50 average\n"
    "<code>/hold</code> — list positions with live value and headroom\n"
    "<code>/hold seed BEL</code> — build it from your logged SIP contributions\n"
    "<code>/hold remove BEL</code>\n\n"
    "<i>Set <code>/capital</code> first so percentages mean something.</i>"
)


def _money_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def _build_position_line(chat_id: int, ticker: str) -> str | None:
    """"Your position" line for the /analyze card, or None.

    Returns None whenever the answer would be invented: no declared capital, no
    holding, or no live price. A position percentage against a guessed
    denominator looks authoritative and means nothing.
    """
    from stockbot.portfolio_state import size_position

    policy = get_risk_policy(chat_id)
    holding = get_holding(chat_id, ticker)
    if policy is None or holding is None:
        return None

    price, _ = _sip_price_and_high(ticker)
    if price is None:
        return None

    sizing = size_position(
        holding.quantity, price, policy.total_capital_inr, policy.max_position_pct
    )
    if sizing is None:
        return None

    room = (
        f"room for {_money_inr(sizing.headroom_inr)} more"
        if sizing.headroom_inr > 0
        else "<b>at or over your cap</b>"
    )
    return (
        f"Your position: {_money_inr(sizing.value_inr)} · "
        f"{sizing.pct_of_capital:.1f}% of capital · {room} "
        f"(cap {sizing.max_position_pct:.0f}%)"
    )


def _parse_capital_args(args: list[str]) -> tuple[float, float | None] | str:
    """(total_capital, max_position_pct or None) or an error string."""
    if not args:
        return "Send an amount, e.g. <code>/capital 500000</code>."
    try:
        capital = float(args[0].replace(",", "").replace("₹", ""))
    except ValueError:
        return f"<code>{esc(args[0])}</code> is not a number."
    if not math.isfinite(capital) or capital <= 0:
        return "Capital must be a real number greater than zero."

    cap_pct: float | None = None
    extra = args[1:]
    if extra:
        if extra[0].lower() not in {"max", "cap"} or len(extra) < 2:
            return (
                f"Did not understand <code>{esc(' '.join(extra))}</code>. "
                "To set a per-stock cap use <code>/capital 500000 max 8</code>."
            )
        try:
            cap_pct = float(extra[1].replace("%", ""))
        except ValueError:
            return f"<code>{esc(extra[1])}</code> is not a valid percent."
        if not math.isfinite(cap_pct) or not (0 < cap_pct <= 100):
            return "The per-stock cap must be between 0 and 100 percent."
    return (capital, cap_pct)


async def handle_capital(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    chat_id = update.effective_chat.id
    args = list(context.args or [])

    if not args:
        policy = await asyncio.to_thread(get_risk_policy, chat_id)
        if policy is None:
            await update.message.reply_text(CAPITAL_USAGE, parse_mode=ParseMode.HTML)
            return
        await update.message.reply_text(
            f"<b>Risk policy</b>\n"
            f"Capital: {_money_inr(policy.total_capital_inr)}\n"
            f"Max per stock: {policy.max_position_pct:.0f}% "
            f"({_money_inr(policy.total_capital_inr * policy.max_position_pct / 100)})",
            parse_mode=ParseMode.HTML,
        )
        return

    parsed = await asyncio.to_thread(_parse_capital_args, args)
    if isinstance(parsed, str):
        await update.message.reply_text(
            f"{parsed}\n\n{CAPITAL_USAGE}", parse_mode=ParseMode.HTML
        )
        return

    capital, cap_pct = parsed
    existing = await asyncio.to_thread(get_risk_policy, chat_id)
    resolved_cap = cap_pct if cap_pct is not None else (
        existing.max_position_pct if existing else DEFAULT_MAX_POSITION_PCT
    )
    policy = await asyncio.to_thread(
        save_risk_policy, chat_id, capital, max_position_pct=resolved_cap
    )
    await update.message.reply_text(
        f"✅ Capital {_money_inr(policy.total_capital_inr)}, "
        f"max {policy.max_position_pct:.0f}% per stock "
        f"({_money_inr(policy.total_capital_inr * policy.max_position_pct / 100)}).\n\n"
        f"{DISCLAIMER}",
        parse_mode=ParseMode.HTML,
    )


def _format_holdings_list(chat_id: int) -> str:
    """Every position with live value, share of capital, and headroom."""
    from stockbot.portfolio_state import concentration_breaches, size_position

    holdings = list_holdings(chat_id)
    if not holdings:
        return "No positions recorded.\n\n" + HOLD_USAGE

    policy = get_risk_policy(chat_id)
    lines = ["<b>Your positions</b>"]
    values: dict[str, float] = {}
    for holding in holdings:
        price, _ = _sip_price_and_high(holding.ticker)
        if price is None:
            lines.append(
                f"{esc(holding.ticker)} — {holding.quantity:g} @ "
                f"{_money_inr(holding.avg_cost)} · <i>price unavailable</i>"
            )
            continue
        sizing = size_position(
            holding.quantity,
            price,
            policy.total_capital_inr if policy else None,
            policy.max_position_pct if policy else DEFAULT_MAX_POSITION_PCT,
        )
        if sizing is None:
            # No capital declared — report value, withhold the percentage.
            value = holding.quantity * price
            values[holding.ticker] = value
            lines.append(
                f"{esc(holding.ticker)} — {holding.quantity:g} @ "
                f"{_money_inr(holding.avg_cost)} · now {_money_inr(value)}"
            )
            continue
        values[holding.ticker] = sizing.value_inr
        flag = " ⚠️" if sizing.over_cap else ""
        lines.append(
            f"{esc(holding.ticker)} — {holding.quantity:g} @ "
            f"{_money_inr(holding.avg_cost)} · now {_money_inr(sizing.value_inr)} · "
            f"{sizing.pct_of_capital:.1f}%{flag}"
        )

    if policy is None:
        lines.append("")
        lines.append("<i>Set <code>/capital</code> to see these as % of capital.</i>")
        return "\n".join(lines)

    breaches = concentration_breaches(values, policy.total_capital_inr, policy.max_position_pct)
    if breaches:
        named = ", ".join(f"{t} {p:.1f}%" for t, p in breaches)
        lines.append("")
        lines.append(f"⚠️ Over your {policy.max_position_pct:.0f}% cap: {esc(named)}")
    return "\n".join(lines)


async def handle_hold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    sub = args[0].lower() if args else ""

    if not args:
        text = await asyncio.to_thread(_format_holdings_list, chat_id)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    if sub in {"seed", "remove", "delete"} and len(args) >= 2:
        resolved = await asyncio.to_thread(resolve_ticker, args[1])
        if isinstance(resolved, AmbiguousMatch) or resolved is None:
            await update.message.reply_text(
                f"Could not resolve <b>{esc(args[1])}</b> to one NSE symbol.",
                parse_mode=ParseMode.HTML,
            )
            return
        if sub == "seed":
            holding = await asyncio.to_thread(seed_holding_from_sip, chat_id, resolved.symbol)
            if holding is None:
                await update.message.reply_text(
                    f"Cannot seed <b>{esc(resolved.symbol)}</b> — no SIP contributions "
                    "logged, or some were logged without a price, so the share count "
                    "is unknowable. Enter it directly instead: "
                    f"<code>/hold {esc(resolved.symbol)} 25 412.50</code>",
                    parse_mode=ParseMode.HTML,
                )
                return
            await update.message.reply_text(
                f"✅ Seeded {esc(holding.ticker)} from your SIP ledger — "
                f"{holding.quantity:.2f} shares at {_money_inr(holding.avg_cost)} average.",
                parse_mode=ParseMode.HTML,
            )
            return
        removed = await asyncio.to_thread(delete_holding, chat_id, resolved.symbol)
        await update.message.reply_text(
            f"✅ Removed {esc(resolved.symbol)}." if removed
            else f"No position in {esc(resolved.symbol)} to remove."
        )
        return

    parsed = await asyncio.to_thread(_parse_hold_args, args)
    if isinstance(parsed, str):
        await update.message.reply_text(f"{parsed}\n\n{HOLD_USAGE}", parse_mode=ParseMode.HTML)
        return

    symbol, quantity, avg_cost = parsed
    resolved = await asyncio.to_thread(resolve_ticker, symbol)
    if isinstance(resolved, AmbiguousMatch) or resolved is None:
        await update.message.reply_text(
            f"Could not resolve <b>{esc(symbol)}</b> to one NSE symbol.",
            parse_mode=ParseMode.HTML,
        )
        return

    holding = await asyncio.to_thread(
        save_holding, chat_id, resolved.symbol, quantity, avg_cost
    )
    line = await asyncio.to_thread(_build_position_line, chat_id, holding.ticker)
    reply = (
        f"✅ {esc(holding.ticker)}: {holding.quantity:g} @ "
        f"{_money_inr(holding.avg_cost)} (cost {_money_inr(holding.cost_basis_inr)})"
    )
    if line:
        reply += f"\n{line}"
    await update.message.reply_text(reply, parse_mode=ParseMode.HTML)


def _parse_hold_args(args: list[str]) -> tuple[str, float, float] | str:
    """(symbol, quantity, avg_cost) or an error string."""
    if len(args) < 3:
        return "Send symbol, quantity and average cost, e.g. <code>/hold BEL 25 412.50</code>."
    numbers = []
    for token in args[1:3]:
        try:
            numbers.append(float(token.replace(",", "").replace("₹", "")))
        except ValueError:
            return f"<code>{esc(token)}</code> is not a number."
    quantity, avg_cost = numbers
    if not math.isfinite(quantity) or quantity <= 0:
        return "Quantity must be a real number greater than zero."
    if not math.isfinite(avg_cost) or avg_cost <= 0:
        return "Average cost must be a real number greater than zero."
    return (args[0], quantity, avg_cost)


SIP_USAGE = (
    "<b>/sip — monthly investing</b>\n\n"
    "<b>Portfolio (₹60k / 3 buckets)</b>\n"
    "<code>/sip plan</code> — whole-share split table, live prices\n"
    "<code>/sip track</code> — planned vs logged this month\n"
    "<code>/sip prescan</code> — batch prescan portfolio names (quant-only)\n"
    "<code>/sip prescan full</code> — prescan with AI eligibility\n"
    "<code>/sip paid KAYNES 3685</code> — log a buy\n"
    "<code>/sip paid BEL 2500 topup</code> — log dip top-up\n\n"
    "<b>Single stock</b>\n"
    "<code>/sip BEL 5000</code> — ₹5,000/month into one name\n"
    "<code>/sip BEL 5000 stepup 10</code> — raise instalment 10%/yr\n"
    "<code>/sip status</code> · <code>/sip paid</code> · "
    "<code>/sip pause</code> / <code>/sip resume</code>\n\n"
    "<i>Portfolio lists live in <code>sip_portfolios.json</code>. "
    "Whole shares only — cash aside accumulates in each bucket.</i>"
)

# The monthly nudge. Day-of-month is deliberate rather than "every 30 days":
# SIPs are anchored to a date, and a drifting reminder stops matching the
# user's bank mandate.
SIP_REMINDER_DAY = 1
SIP_REMINDER_HOUR_IST = 10


def _sip_price_and_high(ticker: str) -> tuple[float | None, float | None]:
    """Live price and 3-month high, or (None, None) when the fetch fails.

    A dead price feed must not kill the whole reminder — the instalment is due
    regardless of whether we can check for a dip.
    """
    from stockbot.fetch.prices import fetch_price_data
    from stockbot.sip import three_month_high

    try:
        price_data = fetch_price_data(ticker)
    except Exception:
        logger.exception("SIP price fetch failed for %s", ticker)
        return (None, None)
    # Both numbers must come from the same series. current_price_abs is taken
    # from the *unadjusted* closes (fetch/prices.py), so pairing it with an
    # adjusted high understates the high after any split or dividend inside the
    # window — a real drawdown would then report "No dip right now", exactly
    # when the dip check matters. Unadjusted also matches the price the user
    # sees in their broker.
    return (price_data.current_price_abs, three_month_high(price_data.ohlcv_unadjusted))


def _sip_scenario_rates(ticker: str):
    """Scenario CAGRs, preferring the stock's own stored analysis.

    Reads the stored verdict directly rather than via get_cached: that helper
    fetches the live price and refuses on a >10% move, so over a SIP's life it
    would reject every past analysis and silently fall back to generic rates —
    and it would cost a second price fetch per reminder to do it.
    """
    from stockbot.sip_messages import resolve_scenario_rates

    try:
        stored = get_latest_verdict_json(ticker)
    except Exception:
        logger.exception("SIP could not read stored analysis for %s", ticker)
        stored = None
    if stored is None:
        return resolve_scenario_rates(None)
    verdict_json, computed_at = stored
    return resolve_scenario_rates(verdict_json, computed_at=computed_at)


def _build_sip_status(chat_id: int, ticker: str) -> str:
    from stockbot.sip_messages import format_status

    plan = get_sip_plan(chat_id)
    assert plan is not None  # caller checked
    ledger = summarize_sip_contributions(chat_id, ticker)
    price, _ = _sip_price_and_high(ticker)
    return format_status(plan, ledger, _sip_scenario_rates(ticker), current_price=price)


def _build_sip_reminder(chat_id: int, ticker: str) -> str:
    from stockbot.sip_messages import format_monthly_reminder

    plan = get_sip_plan(chat_id)
    assert plan is not None  # caller iterates active plans
    ledger = summarize_sip_contributions(chat_id, ticker)
    price, high = _sip_price_and_high(ticker)
    return format_monthly_reminder(
        plan,
        ledger,
        _sip_scenario_rates(ticker),
        current_price=price,
        high_3m=high,
    )


def _parse_sip_setup(args: list[str]) -> tuple[str, float, float] | str:
    """(symbol, monthly, step_up_pct) or an error string."""
    if len(args) < 2:
        return "Send a symbol and a monthly amount, e.g. <code>/sip BEL 5000</code>."
    symbol = args[0]
    try:
        monthly = float(args[1].replace(",", "").replace("₹", ""))
    except ValueError:
        return f"<code>{esc(args[1])}</code> is not a number — try <code>/sip BEL 5000</code>."
    # float() happily parses "nan" and "inf", and `nan <= 0` is False, so
    # without an explicit finite check a NaN plan saves and every later
    # message renders "₹nan".
    if not math.isfinite(monthly) or monthly <= 0:
        return "The monthly amount must be a real number greater than zero."

    step_up = 0.0
    extra = args[2:]
    if extra:
        keyword = extra[0].lower()
        if keyword not in {"stepup", "step-up", "step_up"}:
            return (
                f"Did not understand <code>{esc(' '.join(extra))}</code>. "
                "For a yearly increase use <code>/sip BEL 5000 stepup 10</code>."
            )
        # A bare "stepup" with no value used to be dropped in silence, so the
        # plan saved with no step-up and still replied "Plan saved".
        if len(extra) < 2:
            return "Say how much to step up, e.g. <code>/sip BEL 5000 stepup 10</code>."
        try:
            step_up = float(extra[1].replace("%", ""))
        except ValueError:
            return f"<code>{esc(extra[1])}</code> is not a valid step-up percent."
        if not math.isfinite(step_up) or step_up < 0:
            return "Step-up must be a real number of percent, and cannot be negative."
    return (symbol, monthly, step_up)


def _parse_sip_amount_token(token: str) -> float | None:
    try:
        value = float(token.replace(",", "").replace("₹", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _parse_portfolio_paid_args(args: list[str]) -> tuple[str, float, bool] | str:
    """Parse ``/sip paid SYMBOL amount [topup]``."""
    if len(args) < 2:
        return "Send symbol and amount, e.g. <code>/sip paid BEL 3213</code>."
    symbol = args[1].upper()
    was_topup = any(a.lower() in {"topup", "top-up"} for a in args[2:])
    amount: float | None = None
    for token in args[2:]:
        if token.lower() in {"topup", "top-up"}:
            continue
        parsed = _parse_sip_amount_token(token)
        if parsed is not None:
            amount = parsed
            break
    if amount is None or not math.isfinite(amount) or amount <= 0:
        return f"Could not read amount in <code>{esc(' '.join(args))}</code>."
    return (symbol, amount, was_topup)


async def _reply_portfolio_prescan(update: Update, *, skip_ai: bool = True) -> None:
    from stockbot.portfolio_sip import load_portfolio_sip_config
    from stockbot.portfolio_sip_prescan import (
        all_portfolio_symbols,
        batch_prescan_symbols,
        format_prescan_batch_summary_html,
    )

    mode = "quant-only" if skip_ai else "with AI"
    status = await update.message.reply_text(
        f"⏳ Portfolio prescan ({mode}) — this may take several minutes…"
    )
    try:
        cfg = await asyncio.to_thread(load_portfolio_sip_config)
        symbols = all_portfolio_symbols(cfg)
        items = await asyncio.to_thread(
            batch_prescan_symbols,
            symbols,
            skip_ai=skip_ai,
            delay_seconds=2.0,
        )
        html = format_prescan_batch_summary_html(items)
    except FileNotFoundError as exc:
        await status.edit_text(esc(str(exc)))
        return
    except Exception as exc:
        logger.exception("portfolio prescan failed")
        await status.edit_text(f"Portfolio prescan failed: {esc(exc)}")
        return

    await status.delete()
    await update.message.reply_text(
        f"{html}\n\n{DISCLAIMER}",
        parse_mode=ParseMode.HTML,
    )


def _build_portfolio_plan_chunks_sync(chat_id: int) -> list[str]:
    from stockbot.portfolio_sip import build_portfolio_sip_plan
    from stockbot.portfolio_sip_messages import (
        format_portfolio_plan_html,
        split_telegram_chunks,
    )
    from stockbot.storage import summarize_average_cost_by_symbol

    avg_costs = summarize_average_cost_by_symbol(chat_id)
    plan = build_portfolio_sip_plan(avg_costs=avg_costs or None)
    html = format_portfolio_plan_html(plan)
    return split_telegram_chunks(html)


async def _build_portfolio_plan_chunks(chat_id: int) -> list[str]:
    return await asyncio.to_thread(_build_portfolio_plan_chunks_sync, chat_id)


async def _reply_portfolio_sip_plan(update: Update, chat_id: int) -> None:
    from stockbot.portfolio_sip import build_portfolio_sip_plan
    from stockbot.portfolio_sip_messages import (
        format_portfolio_plan_html,
        split_telegram_chunks,
    )
    from stockbot.storage import summarize_average_cost_by_symbol

    status = await update.message.reply_text(
        "⏳ Building portfolio SIP plan (live prices for all buckets)…"
    )
    try:
        avg_costs = await asyncio.to_thread(summarize_average_cost_by_symbol, chat_id)
        plan = await asyncio.to_thread(
            build_portfolio_sip_plan,
            avg_costs=avg_costs or None,
        )
        html = format_portfolio_plan_html(plan)
        chunks = split_telegram_chunks(html)
    except FileNotFoundError as exc:
        await status.edit_text(esc(str(exc)))
        return
    except Exception as exc:
        logger.exception("portfolio SIP plan failed")
        await status.edit_text(f"Portfolio SIP plan failed: {esc(exc)}")
        return

    await status.delete()
    for index, chunk in enumerate(chunks):
        suffix = f"\n\n{DISCLAIMER}" if index == len(chunks) - 1 else ""
        await update.message.reply_text(chunk + suffix, parse_mode=ParseMode.HTML)


async def _reply_portfolio_sip_track(update: Update, chat_id: int) -> None:
    from stockbot.portfolio_sip import build_portfolio_sip_plan
    from stockbot.portfolio_sip_messages import (
        format_portfolio_track_html,
        split_telegram_chunks,
    )
    from stockbot.storage import (
        summarize_average_cost_by_symbol,
        summarize_sip_contributions_by_symbol_for_month,
    )

    now = datetime.now(_IST)
    month_label = now.strftime("%B %Y")
    status = await update.message.reply_text(
        f"⏳ Loading SIP track for {esc(month_label)}…"
    )
    try:
        avg_costs = await asyncio.to_thread(summarize_average_cost_by_symbol, chat_id)
        plan = await asyncio.to_thread(
            build_portfolio_sip_plan,
            avg_costs=avg_costs or None,
        )
        paid = await asyncio.to_thread(
            summarize_sip_contributions_by_symbol_for_month,
            chat_id,
            year=now.year,
            month=now.month,
        )
        html = format_portfolio_track_html(plan, paid, month_label=month_label)
        chunks = split_telegram_chunks(html)
    except FileNotFoundError as exc:
        await status.edit_text(esc(str(exc)))
        return
    except Exception as exc:
        logger.exception("portfolio SIP track failed")
        await status.edit_text(f"Portfolio SIP track failed: {esc(exc)}")
        return

    await status.delete()
    for index, chunk in enumerate(chunks):
        suffix = f"\n\n{DISCLAIMER}" if index == len(chunks) - 1 else ""
        await update.message.reply_text(chunk + suffix, parse_mode=ParseMode.HTML)


async def _handle_portfolio_paid(
    update: Update,
    chat_id: int,
    args: list[str],
) -> None:
    from stockbot.sip_messages import TOPUP_RISK_NOTE

    parsed = _parse_portfolio_paid_args(args)
    if isinstance(parsed, str):
        await update.message.reply_text(f"{parsed}\n\n{SIP_USAGE}", parse_mode=ParseMode.HTML)
        return

    symbol, amount, was_topup = parsed
    resolved = await asyncio.to_thread(resolve_ticker, symbol)
    if isinstance(resolved, AmbiguousMatch) or resolved is None:
        await update.message.reply_text(
            f"Could not resolve <b>{esc(symbol)}</b>. "
            "Use the NSE symbol, e.g. <code>/sip paid KAYNES 3685</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    price, _ = await asyncio.to_thread(_sip_price_and_high, resolved.symbol)
    await asyncio.to_thread(
        record_sip_contribution,
        chat_id,
        resolved.symbol,
        amount,
        price_at_contribution=price,
        was_topup=was_topup,
    )
    kind = "top-up" if was_topup else "buy"
    lines = [
        f"✅ Logged ₹{amount:,.0f} {kind} for <b>{esc(resolved.symbol)}</b>"
        + (f" at ₹{price:,.2f}." if price else " (price unavailable)."),
        "",
        "Send <code>/sip track</code> to see planned vs logged this month.",
    ]
    if was_topup:
        lines.extend(["", f"<i>{TOPUP_RISK_NOTE}</i>"])
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def handle_sip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    # Every other command carries this guard. /sip was added after the lock
    # landed, so it silently escaped a policy that says "all bot commands".
    if await _reject_if_analysis_busy(update):
        return
    _clear_awaiting_symbol(context)
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    sub = args[0].lower() if args else ""

    if sub == "plan":
        await _reply_portfolio_sip_plan(update, chat_id)
        return

    if sub == "track":
        await _reply_portfolio_sip_track(update, chat_id)
        return

    if sub == "prescan":
        skip_ai = not (len(args) > 1 and args[1].lower() == "full")
        await _reply_portfolio_prescan(update, skip_ai=skip_ai)
        return

    if sub == "paid" and len(args) >= 2 and _parse_sip_amount_token(args[1]) is None:
        await _handle_portfolio_paid(update, chat_id, args)
        return

    if not args:
        plan = await asyncio.to_thread(get_sip_plan, chat_id)
        if plan is None:
            await update.message.reply_text(SIP_USAGE, parse_mode=ParseMode.HTML)
            return
        text = await asyncio.to_thread(_build_sip_status, chat_id, plan.ticker)
        await update.message.reply_text(f"{text}\n\n{DISCLAIMER}", parse_mode=ParseMode.HTML)
        return

    if sub in {"status", "pause", "resume", "paid"}:
        plan = await asyncio.to_thread(get_sip_plan, chat_id)
        if plan is None and sub == "paid":
            await update.message.reply_text(
                "No single-stock SIP plan.\n"
                "For portfolio buys use "
                "<code>/sip paid SYMBOL amount</code>.\n\n" + SIP_USAGE,
                parse_mode=ParseMode.HTML,
            )
            return
        if plan is None:
            await update.message.reply_text(
                "No single-stock SIP plan yet.\n\n" + SIP_USAGE, parse_mode=ParseMode.HTML
            )
            return
        await _handle_sip_subcommand(update, sub, args, plan)
        return

    parsed = await asyncio.to_thread(_parse_sip_setup, args)
    if isinstance(parsed, str):
        await update.message.reply_text(
            f"{parsed}\n\n{SIP_USAGE}", parse_mode=ParseMode.HTML
        )
        return

    symbol, monthly, step_up = parsed
    resolved = await asyncio.to_thread(resolve_ticker, symbol)
    if isinstance(resolved, AmbiguousMatch) or resolved is None:
        await update.message.reply_text(
            f"Could not resolve <b>{esc(symbol)}</b> to one NSE symbol. "
            "Send the exact symbol, e.g. <code>/sip BEL 5000</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    plan = await asyncio.to_thread(
        save_sip_plan, chat_id, resolved.symbol, monthly, step_up_pct=step_up
    )
    text = await asyncio.to_thread(_build_sip_status, chat_id, plan.ticker)
    await update.message.reply_text(
        f"✅ Plan saved.\n\n{text}\n\n"
        f"I'll remind you on day {SIP_REMINDER_DAY} of each month.\n\n{DISCLAIMER}",
        parse_mode=ParseMode.HTML,
    )


async def _handle_sip_subcommand(update: Update, sub: str, args: list[str], plan) -> None:
    from stockbot.sip import current_instalment
    from stockbot.sip_messages import TOPUP_RISK_NOTE, format_plan_summary

    chat_id = plan.chat_id
    if sub == "status":
        text = await asyncio.to_thread(_build_sip_status, chat_id, plan.ticker)
        await update.message.reply_text(f"{text}\n\n{DISCLAIMER}", parse_mode=ParseMode.HTML)
        return

    if sub in {"pause", "resume"}:
        active = sub == "resume"
        await asyncio.to_thread(set_sip_plan_active, chat_id, active)
        if active:
            # Spec: never suggest stopping, and encourage restarting.
            note = "▶️ Resumed. Falling markets are when averaging does its work."
        else:
            note = (
                "⏸ Paused — your plan and history are kept. "
                "Send <code>/sip resume</code> when you want it back."
            )
        await update.message.reply_text(note, parse_mode=ParseMode.HTML)
        return

    # /sip paid [amount] [topup]
    # Default to the instalment actually due, step-up included — using the
    # original amount meant a step-up plan silently never stepped up.
    amount = current_instalment(plan.monthly_amount, plan.step_up_pct, plan.started_at)
    was_topup = any(a.lower() in {"topup", "top-up"} for a in args[1:])
    for token in args[1:]:
        try:
            parsed = float(token.replace(",", "").replace("₹", ""))
        except ValueError:
            continue
        if math.isfinite(parsed) and parsed > 0:
            amount = parsed
            break
    price, _ = await asyncio.to_thread(_sip_price_and_high, plan.ticker)
    await asyncio.to_thread(
        record_sip_contribution,
        chat_id,
        plan.ticker,
        amount,
        price_at_contribution=price,
        was_topup=was_topup,
    )
    ledger = await asyncio.to_thread(summarize_sip_contributions, chat_id, plan.ticker)
    kind = "top-up" if was_topup else "instalment"
    lines = [
        f"✅ Logged ₹{amount:,.0f} {kind} for {esc(plan.ticker)}"
        + (f" at ₹{price:,.2f}." if price else " (price unavailable)."),
        "",
        format_plan_summary(plan),
        (
            f"Invested so far: ₹{ledger.total_invested:,.0f} "
            f"across {ledger.contributions} contribution(s)."
        ),
    ]
    if was_topup:
        lines.extend(["", f"<i>{TOPUP_RISK_NOTE}</i>"])
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def _portfolio_prescan_monthly_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    from stockbot.portfolio_sip import load_portfolio_sip_config
    from stockbot.portfolio_sip_prescan import (
        all_portfolio_symbols,
        batch_prescan_symbols,
    )

    try:
        cfg = await asyncio.to_thread(load_portfolio_sip_config)
    except FileNotFoundError:
        logger.warning("portfolio prescan monthly job skipped — no sip_portfolios.json")
        return
    if not cfg.prescan_gate.monthly_auto_prescan:
        return
    symbols = all_portfolio_symbols(cfg)
    logger.info("Monthly portfolio prescan starting for %d symbols", len(symbols))
    await asyncio.to_thread(
        batch_prescan_symbols,
        symbols,
        skip_ai=True,
        delay_seconds=2.0,
    )


async def _portfolio_sip_plan_monthly_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    from stockbot.portfolio_sip_prescan import portfolio_reminder_chat_ids

    chat_ids = await asyncio.to_thread(portfolio_reminder_chat_ids)
    if not chat_ids:
        logger.info("Portfolio SIP monthly reminder skipped — no target chat IDs")
        return
    for chat_id in chat_ids:
        try:
            chunks = await _build_portfolio_plan_chunks(chat_id)
            for index, chunk in enumerate(chunks):
                suffix = f"\n\n{DISCLAIMER}" if index == len(chunks) - 1 else ""
                await context.bot.send_message(
                    chat_id,
                    f"<b>Monthly portfolio SIP plan</b>\n\n{chunk}{suffix}",
                    parse_mode=ParseMode.HTML,
                )
        except Exception:
            logger.exception("Portfolio SIP monthly plan failed for chat %s", chat_id)


async def _sip_monthly_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Message every active plan. One failure must not stop the rest.

    Deliberately not gated on the analysis lock, unlike the /sip command: this
    fires on a date, so refusing it because an analysis happens to be running
    would drop that month's reminder entirely rather than defer it. The job
    spends nothing and makes one price call per plan.
    """
    plans = await asyncio.to_thread(list_active_sip_plans)
    for plan in plans:
        try:
            text = await asyncio.to_thread(_build_sip_reminder, plan.chat_id, plan.ticker)
            await context.bot.send_message(
                plan.chat_id, f"{text}\n\n{DISCLAIMER}", parse_mode=ParseMode.HTML
            )
        except Exception:
            logger.exception("SIP reminder failed for chat %s", plan.chat_id)


PORTFOLIO_PRESCAN_HOUR_IST = 8


def schedule_sip_reminder(application: Application) -> bool:
    """Arm the monthly job. Returns False when no JobQueue is available.

    python-telegram-bot only builds a JobQueue when the [job-queue] extra is
    installed; without it ``application.job_queue`` is None and scheduling
    would raise at startup rather than at the first fire.
    """
    job_queue = application.job_queue
    if job_queue is None:
        logger.warning("No JobQueue — SIP reminders disabled (install the job-queue extra)")
        return False
    job_queue.run_monthly(
        _sip_monthly_job,
        when=time(hour=SIP_REMINDER_HOUR_IST, minute=0, tzinfo=_IST),
        day=SIP_REMINDER_DAY,
    )
    job_queue.run_monthly(
        _portfolio_prescan_monthly_job,
        when=time(hour=PORTFOLIO_PRESCAN_HOUR_IST, minute=0, tzinfo=_IST),
        day=SIP_REMINDER_DAY,
    )
    job_queue.run_monthly(
        _portfolio_sip_plan_monthly_job,
        when=time(hour=SIP_REMINDER_HOUR_IST, minute=0, tzinfo=_IST),
        day=SIP_REMINDER_DAY,
    )
    return True


async def _register_commands(application: Application) -> None:
    await application.bot.set_my_commands(BOT_COMMANDS)
    try:
        me = await application.bot.get_me()
        if me.username:
            logger.info(
                "Inline suggestions: BotFather /setinline then type @%s + letters",
                me.username,
            )
    except Exception:
        logger.exception("Failed to log bot username for inline hint")


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set — cannot start the bot")

    application = (
        Application.builder().token(settings.telegram_bot_token).post_init(_register_commands).build()
    )
    application.add_handler(CommandHandler("preflight", handle_preflight))
    application.add_handler(CommandHandler("prescan", handle_prescan))
    application.add_handler(CommandHandler("candidates", handle_candidates))
    application.add_handler(CommandHandler("analyze", handle_analyze))
    application.add_handler(CommandHandler("refresh", handle_refresh))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(CommandHandler("spend", handle_spend))
    application.add_handler(CommandHandler("health", handle_health))
    application.add_handler(CommandHandler("sip", handle_sip))
    application.add_handler(CommandHandler("capital", handle_capital))
    application.add_handler(CommandHandler("hold", handle_hold))
    application.add_handler(CommandHandler("track", handle_track))
    application.add_handler(InlineQueryHandler(handle_inline_query))
    application.add_handler(CallbackQueryHandler(handle_symbol_pick))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_text))
    schedule_sip_reminder(application)
    return application


def main() -> None:
    setup_logging()
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
