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
takes roughly 5-10 minutes end to end, and a Stage 2 validation retry adds
another 5-8 minutes on top of that — Sonnet's adaptive thinking plus a
full 16-section report is genuinely slow to generate, not stuck.
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
from datetime import UTC, datetime
from types import SimpleNamespace

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
from stockbot.expected_return import format_expected_return_telegram
from stockbot.fetch.tickers import resolve_ticker
from stockbot.models import AmbiguousMatch, Analysis, TickerInfo
from stockbot.monitor.health_audit import run_health_audit
from stockbot.pipeline import (
    ANALYSIS_RUNTIME_CAP_SECONDS,
    PipelineResult,
    run_full_analysis,
)
from stockbot.portfolio_screener.eligibility import (
    check_deep_analysis_eligibility,
    format_analyze_gate_block,
)
from stockbot.portfolio_screener.outcome_log import build_candidates_messages
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig
from stockbot.report_digest import (
    TELEGRAM_MAX_MISSING,
    TELEGRAM_MAX_REASON_CHARS,
    TELEGRAM_MAX_REASONS,
    TELEGRAM_MAX_WATCH_CHARS,
    _clip,
    _compact_context_flags_line,
    build_compact_attachment_md,
)
from stockbot.storage import backfill_cached_verdicts, invalidate_cached_analyses

logger = logging.getLogger(__name__)

HEALTH_AUDIT_DAYS = 14

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
ANALYSIS_RUNTIME_CAP_MINUTES = ANALYSIS_RUNTIME_CAP_SECONDS // 60
ANALYZING_TEMPLATE = "⏳ Analyzing {company}... (usually 5-15 min — please don't restart the bot)"
DISCLAIMER = (
    "<i>Educational research only — not investment advice. "
    "Do your own due diligence before any trade.</i>"
)

# Tickers with a run_full_analysis currently in flight, keyed by normalized
# query text. Prevents a second /analyze for the same query from starting
# an overlapping (and separately billed) analysis while the first is still
# working — found live: nothing previously stopped this, and a restart
# mid-analysis silently abandoned the in-flight thread with no recovery.
_IN_FLIGHT: set[str] = set()

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


def _format_buy_range_line(verdict_json: dict) -> str:
    buy_zone = _money_pair(verdict_json.get("buy_zone_abs"))
    buy_range_allowed = verdict_json.get("buy_range_allowed")
    anti_chase, cash_gap_blocks, wc_gap_norm = _capital_range_gate_context(verdict_json)
    buy_zone_ok = (
        buy_zone is not None
        and buy_range_allowed is True
        and not cash_gap_blocks
        and not anti_chase
    )
    if buy_zone_ok:
        return f"Buy range: ₹{buy_zone[0]}–₹{buy_zone[1]}"
    if anti_chase:
        return "Buy range: not issued (anti-chase: pause new capital)"
    if cash_gap_blocks and wc_gap_norm:
        return f"Buy range: not issued (WC: {esc(wc_gap_norm)})"
    return "Buy range: not issued"


def _format_sell_range_line(verdict_json: dict) -> str:
    base_fv = _resolve_scenario_fair_value(verdict_json, "base")
    if base_fv is not None:
        return f"Sell range: ₹{base_fv[0]}–₹{base_fv[1]}"
    return "Sell range: unavailable"


def _format_add_more_range_line(verdict_json: dict) -> str:
    block_reason = add_more_range_blocked_reason(verdict_json)
    if block_reason:
        if block_reason.startswith("anti-chase"):
            return "Add-more range: not issued (anti-chase: pause new capital)"
        if block_reason.startswith("WC:"):
            wc_label = block_reason.removeprefix("WC: ").strip()
            return f"Add-more range: not issued (WC: {esc(wc_label)})"
        if block_reason.startswith("five-year test:"):
            label = block_reason.removeprefix("five-year test: ").strip()
            return f"Add-more range: not issued (five-year: {esc(label)})"
        if block_reason.startswith("thesis:"):
            label = block_reason.removeprefix("thesis: ").strip()
            return f"Add-more range: not issued (thesis: {esc(label)})"
        return "Add-more range: not issued"

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
    profit_review_line = _format_profit_review_line(v)
    if profit_review_line:
        action_range_lines.append(profit_review_line)

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
        lines.append(
            "<i>Digest attached; full §1–§16 report stored internally (same LLM run).</i>"
        )

    text = "\n".join(lines)
    if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
        text = text[: TELEGRAM_MAX_MESSAGE_LENGTH - 20].rstrip() + "\n(truncated)"
    return text


def format_ambiguous_reply(candidates: AmbiguousMatch) -> str:
    lines = ["Multiple companies match — reply with the exact symbol:"]
    for i, candidate in enumerate(candidates.candidates, start=1):
        lines.append(f"{i}. <code>{esc(candidate.symbol)}</code> — {esc(candidate.company_name)}")
    return "\n".join(lines)


async def _deliver_result(update: Update, result: PipelineResult, status_message) -> None:
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
            "(usually 5–15 min), then try again."
        )
        return

    if result.status == "budget_exceeded":
        await status_message.edit_text(
            f"Monthly budget reached (₹{result.spent_inr:.0f} spent this month). "
            f"No new analyses until next month — the cap is real, not advisory."
        )
        return

    if result.status == "insufficient_data":
        failures = "\n".join(f"- {esc(f)}" for f in (result.validation_failures or []))
        await status_message.edit_text(
            "Insufficient data for a confident view after validation. "
            f"This is an honest answer, not a bug.\n\n{failures}"
        )
        return

    if result.status == "analysis_cost_exceeded":
        await status_message.edit_text(
            f"This analysis hit its per-run cost cap (₹{result.spent_inr:.2f} spent) before "
            f"producing a validated verdict, and was stopped rather than left to keep retrying. "
            f"Try again, or send /spend to check the monthly total."
        )
        return

    if result.status == "analysis_runtime_exceeded":
        await status_message.edit_text(
            f"Analysis stopped after {ANALYSIS_RUNTIME_CAP_MINUTES} minutes to avoid burning "
            f"more LLM budget (₹{result.spent_inr:.2f} spent so far — Stage 1 only if Stage 2 "
            f"never completed). Try /prescan first, then /analyze again when you can wait."
        )
        return

    if result.status == "render_failed":
        await status_message.edit_text(
            f"The analysis passed validation but couldn't be rendered "
            f"(₹{result.spent_inr:.2f} spent, not wasted — logged either way): "
            f"{esc(result.render_error)}"
        )
        return

    analysis = result.analysis
    await status_message.edit_text(
        format_verdict_reply(analysis, staleness_banner=result.staleness_banner),
        parse_mode=ParseMode.HTML,
    )
    await _send_report_attachment(update, analysis)


def _write_report_file(analysis: Analysis, *, compact: bool = True):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "digest" if compact else "full"
    report_path = REPORTS_DIR / f"{analysis.ticker}_{analysis.run_date.isoformat()}_{suffix}.md"
    body = (
        build_compact_attachment_md(analysis)
        if compact
        else analysis.report_md
    )
    report_path.write_text(body, encoding="utf-8")
    return report_path


async def _send_report_attachment(update: Update, analysis: Analysis) -> None:
    report_path = await asyncio.to_thread(_write_report_file, analysis)
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


async def _run_and_reply(update: Update, query: str, *, force: bool = False) -> None:
    if settings.require_prescan_for_analyze and not force:
        allowed = await _check_analyze_eligibility_gate(update, query)
        if not allowed:
            return

    key = query.strip().upper()
    if key in _IN_FLIGHT:
        await update.message.reply_text(
            f"Already analyzing {esc(query)} — this takes several minutes; please wait for "
            f"that to finish instead of sending it again."
        )
        return

    _IN_FLIGHT.add(key)
    status_message = await update.message.reply_text(ANALYZING_TEMPLATE.format(company=esc(query)))
    progress_task = asyncio.create_task(_send_progress_updates(status_message, query))
    try:
        try:
            result = await asyncio.to_thread(run_full_analysis, query)
        except Exception as exc:  # bot's resilience boundary — a crash here must still reply, not vanish
            logger.exception("run_full_analysis failed for %r", query)
            await status_message.edit_text(f"Something went wrong: {esc(exc)}")
            return
    finally:
        progress_task.cancel()
        _IN_FLIGHT.discard(key)

    await _deliver_result(update, result, status_message)


async def handle_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    query, force = _parse_analyze_command_args(context.args)
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
    await _run_and_reply(update, query, force=force)


def _parse_analyze_command_args(args: list[str] | None) -> tuple[str, bool]:
    parts = list(args or [])
    force = False
    if parts and parts[0].lower() == "force":
        force = True
        parts = parts[1:]
    return " ".join(parts).strip(), force


async def handle_prescan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
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


def _parse_force_analyze_plain_text(text: str) -> tuple[str, bool] | None:
    """Accept 'force BEL' as a plain-text override for the eligibility gate."""
    stripped = text.strip()
    if stripped.lower().startswith("force "):
        rest = stripped[6:].strip()
        if rest:
            return rest, True
    return None


async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    text = (update.message.text or "").strip()
    if not text:
        return
    prescan_query = _parse_prescan_plain_text(text)
    if prescan_query is not None:
        _clear_awaiting_symbol(context)
        await _run_prescan_and_reply(update, prescan_query)
        return
    force_query = _parse_force_analyze_plain_text(text)
    if force_query is not None:
        _clear_awaiting_symbol(context)
        query, force = force_query
        await _run_and_reply(update, query, force=force)
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
    _clear_awaiting_symbol(context)
    args = list(context.args or [])
    chunks, err = await asyncio.to_thread(build_candidates_messages, args)
    if err:
        await update.message.reply_text(err, parse_mode=ParseMode.HTML)
        return
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
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
        "/analyze <company> — full deep analysis (requires prescan eligibility)\n"
        "/analyze force <symbol> — bypass gate (not recommended)\n"
        "/refresh SYMBOL — clear cached analysis for symbol\n"
        "/refresh backfill — recompute gates + expected_return on cached rows\n"
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


async def handle_spend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
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
]


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
    application.add_handler(CommandHandler("prescan", handle_prescan))
    application.add_handler(CommandHandler("candidates", handle_candidates))
    application.add_handler(CommandHandler("analyze", handle_analyze))
    application.add_handler(CommandHandler("refresh", handle_refresh))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(CommandHandler("spend", handle_spend))
    application.add_handler(CommandHandler("health", handle_health))
    application.add_handler(InlineQueryHandler(handle_inline_query))
    application.add_handler(CallbackQueryHandler(handle_symbol_pick))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_text))
    return application


def main() -> None:
    setup_logging()
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
