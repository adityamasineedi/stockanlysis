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
another 5-8 minutes on top of that — Opus 5's default adaptive thinking
plus a full 16-section report is genuinely slow to generate, not stuck.
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

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from stockbot.config import REPORTS_DIR, settings, setup_logging
from stockbot.costs import month_to_date_spend
from stockbot.models import AmbiguousMatch, Analysis
from stockbot.pipeline import PipelineResult, run_full_analysis

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
ANALYZING_TEMPLATE = "⏳ Analyzing {company}... (usually 5-15 min — please don't restart the bot)"

# Tickers with a run_full_analysis currently in flight, keyed by normalized
# query text. Prevents a second /analyze for the same query from starting
# an overlapping (and separately billed) analysis while the first is still
# working — found live: nothing previously stopped this, and a restart
# mid-analysis silently abandoned the in-flight thread with no recovery.
_IN_FLIGHT: set[str] = set()


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def format_verdict_reply(analysis: Analysis) -> str:
    v = analysis.verdict_json
    buy_zone = v.get("buy_zone_abs") or [None, None]
    # fair_value_base_abs is Python-computed (compute_valuation, from the
    # model's valuation_inputs) and merged into verdict_json by
    # pipeline.py — not a field the model states directly under v3.
    fair_value = v.get("fair_value_base_abs") or [None, None]

    lines = [
        f"<b>{esc(v.get('verdict', '?'))}</b> — {esc(analysis.ticker)}",
        f"Price: ₹{v.get('current_price_abs', '?')} (as of {esc(v.get('price_date', '?'))})",
        f"Buy Zone: ₹{buy_zone[0]}–₹{buy_zone[1]}",
        f"Fair Value: ₹{fair_value[0]}–₹{fair_value[1]}",
        f"Risk: {esc(v.get('risk', '?'))} · Confidence: {v.get('confidence', '?')}/10",
        f"Holding Period: {esc(v.get('holding_period', '?'))}",
        "",
        "<b>Why buy</b>",
    ]
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

    if result.status == "render_failed":
        await status_message.edit_text(
            f"The analysis passed validation but couldn't be rendered "
            f"(₹{result.spent_inr:.2f} spent, not wasted — logged either way): "
            f"{esc(result.render_error)}"
        )
        return

    analysis = result.analysis
    await status_message.edit_text(format_verdict_reply(analysis), parse_mode=ParseMode.HTML)
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
                await status_message.edit_text(
                    f"⏳ Still analyzing {esc(query)}... ({minutes} min elapsed — Opus 5 report "
                    f"generation typically takes 5-15 min, please don't restart the bot)"
                )
            except Exception:
                # Cosmetic progress ping — a transient Telegram/network hiccup here (we've seen
                # httpx.ReadError on this exact polling connection live) must not take down the
                # analysis itself, but it's still logged rather than silently swallowed.
                logger.warning("Progress update edit failed for %r", query, exc_info=True)
    except asyncio.CancelledError:
        pass


async def _run_and_reply(update: Update, query: str) -> None:
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
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /analyze <company name or symbol>")
        return
    await _run_and_reply(update, query)


async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return
    await _run_and_reply(update, text)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send a company name, or use /analyze <company>. I'll fetch, analyze, and reply "
        "with a verdict plus the full report as a file.\n/spend shows month-to-date cost."
    )


async def handle_spend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    spent = await asyncio.to_thread(month_to_date_spend)
    await update.message.reply_text(
        f"Month-to-date spend: ₹{spent:.2f} of ₹{settings.monthly_budget_inr:.0f}"
    )


BOT_COMMANDS = [
    BotCommand("analyze", "Analyze a stock by name or symbol"),
    BotCommand("help", "Usage instructions"),
    BotCommand("spend", "Month-to-date cost"),
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
    application.add_handler(CommandHandler("analyze", handle_analyze))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(CommandHandler("spend", handle_spend))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_text))
    return application


def main() -> None:
    setup_logging()
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
