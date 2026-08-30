"""Post-process full Stage 2 output for Telegram delivery — no LLM token savings.

The pipeline still generates and stores the complete ``report_md`` for
validation and SQLite. This module only shortens what users see in-chat
and in the attached ``.md`` file.
"""

from __future__ import annotations

import re

from stockbot.models import Analysis

_BEGINNER_NEEDLE = "SHOULD I BUY?"
_JSON_FENCE_RE = re.compile(r"```json\b", re.IGNORECASE)
_FOOTER_NEEDLE = "Research and education, not investment advice"

TELEGRAM_MAX_REASONS = 2
TELEGRAM_MAX_REASON_CHARS = 120
TELEGRAM_MAX_WATCH_CHARS = 100
TELEGRAM_MAX_MISSING = 3
ATTACHMENT_BEGINNER_MAX_CHARS = 2_400
ATTACHMENT_MAX_CHARS = 12_000


def _clip(text: object, max_chars: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 1].rstrip() + "…"


def extract_beginner_summary(report_md: str, *, max_chars: int = ATTACHMENT_BEGINNER_MAX_CHARS) -> str:
    """Pull the Beginner Summary block from a rendered report."""
    if not report_md or _BEGINNER_NEEDLE not in report_md:
        return ""
    start = report_md.find(_BEGINNER_NEEDLE)
    # The needle is bare text, but the report writes the heading as markup —
    # "**SHOULD I BUY?**" in the v3 prompt. Slicing at the needle itself cut
    # the opening "**" and kept the closing one, so the digest opened with a
    # bare "SHOULD I BUY?**". Back up to the start of the line to keep it.
    start = report_md.rfind("\n", 0, start) + 1
    tail = report_md[start:]
    json_match = _JSON_FENCE_RE.search(tail)
    if json_match:
        tail = tail[: json_match.start()]
    footer_idx = tail.find(_FOOTER_NEEDLE)
    if footer_idx >= 0:
        tail = tail[:footer_idx]
    return _clip(tail.strip(), max_chars)


def _fallback_summary_from_verdict(verdict_json: dict) -> str:
    lines = [
        "**SHOULD I BUY?**",
        f"- **Decision:** {verdict_json.get('verdict', '?')}",
        f"- **Risk:** {verdict_json.get('risk', '?')} · **Confidence:** {verdict_json.get('confidence', '?')}/10",
        f"- **Holding period:** {verdict_json.get('holding_period', '?')}",
        "",
        "**Why buy**",
    ]
    for reason in (verdict_json.get("reasons_buy") or [])[:3]:
        lines.append(f"- {reason}")
    lines.extend(["", "**Why avoid**"])
    for reason in (verdict_json.get("reasons_avoid") or [])[:3]:
        lines.append(f"- {reason}")
    watch = verdict_json.get("biggest_watch")
    if watch:
        lines.extend(["", f"**Biggest watch:** {watch}"])
    return "\n".join(lines)


def build_compact_attachment_md(analysis: Analysis) -> str:
    """Shorter ``.md`` for Telegram — full report remains in DB."""
    beginner = extract_beginner_summary(analysis.report_md)
    if not beginner:
        beginner = _fallback_summary_from_verdict(analysis.verdict_json)

    header = (
        f"# {analysis.ticker} — {analysis.run_date.isoformat()}\n\n"
        "_Reading digest only. The complete validated report from this run is "
        "stored internally; this file omits the long §1–§16 sections._\n"
    )
    body = f"\n{beginner}\n"
    if analysis.missing:
        shown = analysis.missing[:TELEGRAM_MAX_MISSING]
        body += "\n**Data gaps**\n" + "\n".join(f"- {item}" for item in shown)
        if len(analysis.missing) > TELEGRAM_MAX_MISSING:
            body += f"\n- … +{len(analysis.missing) - TELEGRAM_MAX_MISSING} more\n"

    text = header + body
    if len(text) > ATTACHMENT_MAX_CHARS:
        text = text[: ATTACHMENT_MAX_CHARS - 20].rstrip() + "\n\n(truncated)\n"
    return text


def _compact_context_flags_line(
    *,
    five_year_answer: str | None,
    wc_gap_norm: str | None,
    anti_chase: bool,
    tension: object,
    thesis_status: object,
) -> str | None:
    parts: list[str] = []
    if five_year_answer:
        parts.append(f"5y test: {five_year_answer}")
    if wc_gap_norm:
        parts.append(f"WC: {wc_gap_norm}")
    if anti_chase:
        parts.append("anti-chase")
    if tension and str(tension).upper() not in ("NONE", ""):
        parts.append(f"tension: {tension}")
    if thesis_status:
        parts.append(f"thesis: {thesis_status}")
    if not parts:
        return None
    return " · ".join(parts)
