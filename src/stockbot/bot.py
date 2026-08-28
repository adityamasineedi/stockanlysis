"""Module 12 (part 2) — Telegram bot. Polling mode only — no webhook, no
web server, no public IP. python-telegram-bot v22 (confirmed installed;
CommandHandler/MessageHandler/Application.run_polling signatures verified
against the installed package before writing this).

Reply formatting reads verdict_json only, never report_md — the full
report goes out as a .md file attachment instead. HTML parse mode (not
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
from datetime import datetime

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from stockbot.config import LOGS_DIR, REPORTS_DIR, settings, setup_logging
from stockbot.costs import month_to_date_spend
from stockbot.constitution_gates import should_anti_chase_from_dict
from stockbot.expected_return import format_expected_return_telegram
from stockbot.models import AmbiguousMatch, Analysis
from stockbot.pipeline import (
    ANALYSIS_RUNTIME_CAP_SECONDS,
    PipelineResult,
    run_full_analysis,
)
from stockbot.portfolio_screener.eligibility import (
    check_deep_analysis_eligibility,
    format_analyze_gate_block,
)
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig
from stockbot.monitor.health_audit import run_health_audit

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


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


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


def _resolve_base_fair_value(verdict_json: dict) -> tuple[str, str] | None:
    """Headline FV is always the BASE range — never bear-low–bull-high.

    Prefer the Python-merged fair_value_base_abs; if missing (older saved
    analyses), recompute from valuation_inputs so Telegram never falls back
    to a wrong span or ₹None.
    """
    formatted = _money_pair(verdict_json.get("fair_value_base_abs"))
    if formatted is not None:
        return formatted

    raw_inputs = verdict_json.get("valuation_inputs")
    if not isinstance(raw_inputs, dict):
        return None
    try:
        from stockbot.llm.verdict import ValuationInputs, compute_valuation

        valuation = compute_valuation(ValuationInputs.model_validate(raw_inputs))
        return _money_pair(valuation.fair_value_base_abs)
    except Exception:
        logger.exception("Could not recompute fair_value_base_abs for Telegram card")
        return None


_CASH_GAP_BLOCK_MARKERS = (
    "operating cash flow was sharply negative",
    "ocf/pat",
    "σcfo/σpat",
    "cfo/pat",
    "cash conversion",
    "cumulative Σcfo",
    "cumulative cfo",
    "cumulative σcfo",
)


def _verdict_text_blob(verdict: dict) -> str:
    parts: list[str] = []
    for key in ("reasons_avoid", "reasons_buy"):
        for item in verdict.get(key) or []:
            parts.append(str(item))
    parts.append(str(verdict.get("biggest_watch") or ""))
    five_year = verdict.get("five_year_business_test") or {}
    if isinstance(five_year, dict):
        for key in ("evidence_against", "evidence_for"):
            for item in five_year.get(key) or []:
                parts.append(str(item))
    return " ".join(parts).lower()


def _unresolved_cash_gap_blocks_buy_zone(verdict: dict) -> bool:
    """Hide buy zones when WC gap is unresolved (incl. legacy cached analyses)."""
    wc_gap = verdict.get("wc_gap_classification")
    if wc_gap is not None and str(wc_gap).strip() != "":
        return str(wc_gap).strip().upper() != "TEMPORARY_BILLING_CYCLE"

    blob = _verdict_text_blob(verdict)
    return any(marker in blob for marker in _CASH_GAP_BLOCK_MARKERS)


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
) -> str:
    v = analysis.verdict_json
    buy_zone = _money_pair(v.get("buy_zone_abs"))
    buy_range_allowed = v.get("buy_range_allowed")
    wc_gap = v.get("wc_gap_classification")
    wc_gap_norm = str(wc_gap).strip().upper() if wc_gap else None
    cash_gap_blocks = _unresolved_cash_gap_blocks_buy_zone(v)
    anti_chase = bool(v.get("anti_chase_flag")) or should_anti_chase_from_dict(v)[0]
    buy_zone_ok = (
        buy_zone is not None
        and buy_range_allowed is True
        and not cash_gap_blocks
        and not anti_chase
        and (
            wc_gap_norm is None
            or wc_gap_norm == "TEMPORARY_BILLING_CYCLE"
        )
    )
    if buy_zone_ok:
        buy_zone_line = f"Buy Zone: ₹{buy_zone[0]}–₹{buy_zone[1]}"
    elif anti_chase:
        buy_zone_line = "Buy Zone: not issued (anti-chase: pause new capital)"
    elif cash_gap_blocks:
        label = wc_gap_norm or "RECONCILIATION_REQUIRED"
        buy_zone_line = f"Buy Zone: not issued (WC: {esc(label)})"
    elif wc_gap_norm and wc_gap_norm != "TEMPORARY_BILLING_CYCLE":
        buy_zone_line = f"Buy Zone: not issued (WC: {esc(wc_gap_norm)})"
    else:
        buy_zone_line = "Buy Zone: not issued"
    # fair_value_base_abs is Python-computed (compute_valuation, from the
    # model's valuation_inputs) and merged into verdict_json by
    # pipeline.py — not a field the model states directly under v3.
    fair_value = _resolve_base_fair_value(v) or ("?", "?")

    lines: list[str] = []
    if staleness_banner:
        lines.append(esc(staleness_banner))
        lines.append("")

    five_year = v.get("five_year_business_test") or {}
    five_year_answer = five_year.get("answer") if isinstance(five_year, dict) else None

    lines.extend(
        [
            f"<b>{esc(v.get('verdict', '?'))}</b> — {esc(analysis.ticker)}",
            f"Price: ₹{v.get('current_price_abs', '?')} (as of {esc(v.get('price_date', '?'))})",
            buy_zone_line,
            f"Fair Value (base): ₹{fair_value[0]}–₹{fair_value[1]}",
            f"Risk: {esc(v.get('risk', '?'))} · Confidence: {v.get('confidence', '?')}/10",
            f"Holding Period: {esc(v.get('holding_period', '?'))}",
        ]
    )
    stage2_line = _format_stage2_mode_line(v)
    if stage2_line:
        lines.append(stage2_line)
    if five_year_answer:
        lines.append(f"5y business test: {esc(str(five_year_answer))}")
    if wc_gap_norm:
        lines.append(f"WC gap: {esc(wc_gap_norm)}")
    if anti_chase:
        lines.append("Anti-chase: pause new capital — valuation recheck")
    tension = v.get("external_valuation_tension")
    if tension and str(tension).upper() not in ("NONE", ""):
        lines.append(f"Valuation tension: {esc(str(tension))}")
    if v.get("thesis_status"):
        lines.append(f"Thesis: {esc(str(v.get('thesis_status')))}")
    for er_line in format_expected_return_telegram(v.get("expected_return") or {}):
        lines.append(esc(er_line))
    lines.extend(
        [
            "",
            "<b>Why buy</b>",
        ]
    )
    for reason in v.get("reasons_buy") or []:
        lines.append(f"• {esc(reason)}")
    lines.append("")
    lines.append("<b>Why avoid</b>")
    for reason in v.get("reasons_avoid") or []:
        lines.append(f"• {esc(reason)}")
    lines.append("")
    lines.append(f"<b>Biggest watch:</b> {esc(v.get('biggest_watch', '?'))}")

    for item in analysis.missing:
        lines.append(f"⚠️ {esc(item)}")

    lines.append("")
    lines.append(DISCLAIMER)

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


def _write_report_file(analysis: Analysis):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{analysis.ticker}_{analysis.run_date.isoformat()}.md"
    report_path.write_text(analysis.report_md, encoding="utf-8")
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
    query, force = _parse_analyze_command_args(context.args)
    if not query:
        await update.message.reply_text(
            "Usage: /analyze <company name or symbol>\n"
            "       /analyze force <symbol> — bypass eligibility gate (not recommended)"
        )
        return
    await _run_and_reply(update, query, force=force)


def _parse_analyze_command_args(args: list[str] | None) -> tuple[str, bool]:
    parts = list(args or [])
    force = False
    if parts and parts[0].lower() == "force":
        force = True
        parts = parts[1:]
    return " ".join(parts).strip(), force


async def handle_prescan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text(
            "Usage: /prescan <symbol or name>\n"
            "Example: /prescan BEL\n\n"
            "Cheap check: is this stock worth expensive deep /analyze?"
        )
        return
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
    text = (update.message.text or "").strip()
    if not text:
        return
    prescan_query = _parse_prescan_plain_text(text)
    if prescan_query is not None:
        await _run_prescan_and_reply(update, prescan_query)
        return
    force_query = _parse_force_analyze_plain_text(text)
    if force_query is not None:
        query, force = force_query
        await _run_and_reply(update, query, force=force)
        return
    await _run_and_reply(update, text)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Commands:\n"
        "/prescan <symbol> — cheap check: worth deep analysis?\n"
        "/analyze <company> — full deep analysis (requires prescan eligibility)\n"
        "/analyze force <symbol> — bypass gate (not recommended)\n"
        "/spend — month-to-date cost\n"
        "/health — cost/token/quality audit (no LLM spend)\n\n"
        "Or send: prescan BEL\n\n"
        "Educational research only — not investment advice."
    )


async def handle_spend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    spent = await asyncio.to_thread(month_to_date_spend)
    await update.message.reply_text(
        f"Month-to-date spend: ₹{spent:.2f} of ₹{settings.monthly_budget_inr:.0f}"
    )


def _write_health_audit_file(markdown: str):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().date().isoformat()
    path = LOGS_DIR / f"health_audit_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


async def handle_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    BotCommand("prescan", "Cheap check: worth deep analysis?"),
    BotCommand("analyze", "Full deep analysis by name or symbol"),
    BotCommand("help", "Usage instructions"),
    BotCommand("spend", "Month-to-date cost"),
    BotCommand("health", "Cost/token/quality audit"),
]


async def _register_commands(application: Application) -> None:
    # Gives Telegram's own "/" command menu (name + description autocomplete
    # when typing "/") — not per-keystroke company-name suggestions, which
    # would be a different feature (inline mode) with its own BotFather setup.
    await application.bot.set_my_commands(BOT_COMMANDS)


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set — cannot start the bot")

    application = (
        Application.builder().token(settings.telegram_bot_token).post_init(_register_commands).build()
    )
    application.add_handler(CommandHandler("prescan", handle_prescan))
    application.add_handler(CommandHandler("analyze", handle_analyze))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(CommandHandler("spend", handle_spend))
    application.add_handler(CommandHandler("health", handle_health))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_text))
    return application


def main() -> None:
    setup_logging()
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
