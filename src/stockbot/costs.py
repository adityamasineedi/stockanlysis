"""Module 9 (part 1) — cost tracking. Built before any LLM call, so Stage
1 (and everything after) has somewhere to log to from its first run.

Multi-provider (17D — the DeepSeek A/B test needs its own cost tracking,
not a bolt-on afterward). Each provider's pricing MODEL is structurally
different, not just different numbers:
  Anthropic — base input rate; cache READS at 0.1x; cache WRITES (opt-in
    via cache_control) at 1.25x (5m TTL) or 2.0x (1h TTL). No time-of-day
    variation.
  DeepSeek  — separate cache-hit/cache-miss input rates (caching is
    automatic/disk-based, never explicitly written to); no write-premium
    concept at all; peak/off-peak hours DOUBLE every rate uniformly.
Rates re-verified 2026-08-27 against:
  https://platform.claude.com/docs/en/about-claude/pricing
  https://api-docs.deepseek.com/quick_start/pricing/

FX rate comes from config.settings.usd_inr_rate — never hardcoded here, so
it can be updated without touching this module. Refresh USD_INR_RATE in
.env when spot drifts (cost estimates are wrong by the same %).

Storage: `llm_calls` table in the same SQLite file as `analyses`
(storage.py). This module owns its own table creation/migrations.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from stockbot.config import DB_PATH, settings

# Last date these USD/MTok tables were checked against the vendor docs above.
PRICING_VERIFIED_ON = "2026-08-27"

ANTHROPIC_PRICING_USD_PER_MTOK = {
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    # v3 migration (Stage 1 extraction model / A-B / screener ranker).
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # Not used in production pipeline; kept so an accidental call still logs.
    "claude-fable-5": {"input": 10.0, "output": 50.0},
}
CACHED_INPUT_DISCOUNT = 0.1
# Found live: switching both stages to ttl="1h" (see llm/extract.py,
# llm/verdict.py) without touching this made cost tracking wrong again —
# a 1h cache write is billed at 2x input price, not 1.25x. The two TTLs
# have genuinely different premiums; there is no single "cache write" rate.
CACHE_WRITE_PREMIUM_5M = 1.25
CACHE_WRITE_PREMIUM_1H = 2.0

# Off-peak (standard) rates — _deepseek_rate_multiplier doubles these
# during peak hours. Confirm against DeepSeek's own pricing page before
# relying on this for a real budget decision; pricing on fast-moving model
# APIs can change without much notice.
DEEPSEEK_PRICING_USD_PER_MTOK = {
    "deepseek-v4-flash": {"cache_hit_input": 0.007, "cache_miss_input": 0.22, "output": 0.66},
    "deepseek-v4-pro": {"cache_hit_input": 0.022, "cache_miss_input": 0.66, "output": 1.98},
}


def _deepseek_rate_multiplier(called_at: datetime) -> float:
    # Peak: 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday, at exactly 2x
    # the off-peak rate. All other hours — including the entire weekend —
    # are off-peak. In IST (UTC+5:30) that's 06:30-09:30 and 11:30-15:30,
    # squarely inside Indian market hours — do not assume off-peak just
    # because it "feels like" a quiet time to be calling an API.
    called_at_utc = called_at.astimezone(UTC)
    if called_at_utc.weekday() >= 5:  # Saturday=5, Sunday=6
        return 1.0
    hour = called_at_utc.hour
    is_peak = (1 <= hour < 4) or (6 <= hour < 10)
    return 2.0 if is_peak else 1.0


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            called_at TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
            cache_creation_1h_tokens INTEGER NOT NULL DEFAULT 0,
            thinking_tokens INTEGER NOT NULL DEFAULT 0,
            provider TEXT NOT NULL DEFAULT 'anthropic',
            cost_inr REAL NOT NULL
        )
        """
    )
    # Migrations for DBs created before a column existed — real rows already
    # exist from live calls, so this can't just be a fresh CREATE TABLE.
    # Checked via PRAGMA rather than a blind ALTER + catch, per the project
    # rule against swallowing exceptions: an unexpected ALTER failure (e.g.
    # a genuinely locked DB) should still surface loudly instead of being
    # mistaken for "column already exists".
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_calls)")}
    if "cache_creation_tokens" not in existing_columns:
        conn.execute("ALTER TABLE llm_calls ADD COLUMN cache_creation_tokens INTEGER NOT NULL DEFAULT 0")
    if "cache_creation_1h_tokens" not in existing_columns:
        conn.execute("ALTER TABLE llm_calls ADD COLUMN cache_creation_1h_tokens INTEGER NOT NULL DEFAULT 0")
    if "thinking_tokens" not in existing_columns:
        # Exact figure from response.usage.output_tokens_details.thinking_tokens
        # — kept purely for observability (PROJECT.md's "80% of Stage 2 output
        # is invisible thinking" finding was estimated from char counts; this
        # makes it measurable precisely, and lets a future architecture change
        # be checked against real before/after numbers instead of an estimate).
        conn.execute("ALTER TABLE llm_calls ADD COLUMN thinking_tokens INTEGER NOT NULL DEFAULT 0")
    if "provider" not in existing_columns:
        conn.execute("ALTER TABLE llm_calls ADD COLUMN provider TEXT NOT NULL DEFAULT 'anthropic'")
    return conn


def compute_cost_inr(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_creation_1h_tokens: int = 0,
    provider: str = "anthropic",
    called_at: datetime | None = None,
) -> float:
    if provider == "anthropic":
        # response.usage.input_tokens is the UNCACHED REMAINDER ONLY — the
        # API reports cache_read_input_tokens and cache_creation_input_tokens
        # as separate, additive fields, not a subset of input_tokens. Do not
        # subtract cached_tokens from input_tokens here (an earlier version
        # of this function did, harmless only because cached_tokens was
        # always 0 before caching was ever turned on).
        if model not in ANTHROPIC_PRICING_USD_PER_MTOK:
            raise ValueError(f"Unknown Anthropic model for pricing: {model!r}")
        rates = ANTHROPIC_PRICING_USD_PER_MTOK[model]
        # cache_creation_tokens is the TOTAL write (both TTLs combined, per
        # response.usage.cache_creation_input_tokens); cache_creation_1h_tokens
        # is the subset of that total written at the 1h TTL (per
        # response.usage.cache_creation.ephemeral_1h_input_tokens). The
        # remainder is billed at the 5m premium.
        cache_creation_5m_tokens = cache_creation_tokens - cache_creation_1h_tokens
        if cache_creation_5m_tokens < 0:
            raise ValueError(
                f"cache_creation_1h_tokens ({cache_creation_1h_tokens}) exceeds "
                f"cache_creation_tokens ({cache_creation_tokens})"
            )
        cost_usd = (
            input_tokens / 1_000_000 * rates["input"]
            + cached_tokens / 1_000_000 * rates["input"] * CACHED_INPUT_DISCOUNT
            + cache_creation_5m_tokens / 1_000_000 * rates["input"] * CACHE_WRITE_PREMIUM_5M
            + cache_creation_1h_tokens / 1_000_000 * rates["input"] * CACHE_WRITE_PREMIUM_1H
            + output_tokens / 1_000_000 * rates["output"]
        )
    elif provider == "deepseek":
        if cache_creation_tokens or cache_creation_1h_tokens:
            raise ValueError("cache_creation_tokens is an Anthropic-only concept — not applicable to DeepSeek")
        if model not in DEEPSEEK_PRICING_USD_PER_MTOK:
            raise ValueError(f"Unknown DeepSeek model for pricing: {model!r}")
        rates = DEEPSEEK_PRICING_USD_PER_MTOK[model]
        multiplier = _deepseek_rate_multiplier(called_at or datetime.now(UTC))
        # cached_tokens means CACHE-HIT input tokens here (DeepSeek's cache
        # is automatic/disk-based, never explicitly written to); input_tokens
        # means the cache-MISS remainder — a different split than Anthropic's,
        # not a subset relationship either way.
        cost_usd = (
            input_tokens / 1_000_000 * rates["cache_miss_input"] * multiplier
            + cached_tokens / 1_000_000 * rates["cache_hit_input"] * multiplier
            + output_tokens / 1_000_000 * rates["output"] * multiplier
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")
    return cost_usd * settings.usd_inr_rate


def log_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_creation_1h_tokens: int = 0,
    thinking_tokens: int = 0,
    provider: str = "anthropic",
    called_at: datetime | None = None,
) -> float:
    called_at = called_at or datetime.now(UTC)
    cost_inr = compute_cost_inr(
        model,
        input_tokens,
        output_tokens,
        cached_tokens,
        cache_creation_tokens,
        cache_creation_1h_tokens,
        provider,
        called_at,
    )
    with _connect() as conn:
        conn.execute(
            "INSERT INTO llm_calls "
            "(called_at, model, input_tokens, output_tokens, cached_tokens, cache_creation_tokens, "
            "cache_creation_1h_tokens, thinking_tokens, provider, cost_inr) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                called_at.isoformat(),
                model,
                input_tokens,
                output_tokens,
                cached_tokens,
                cache_creation_tokens,
                cache_creation_1h_tokens,
                thinking_tokens,
                provider,
                cost_inr,
            ),
        )
    return cost_inr


def month_to_date_spend(as_of: datetime | None = None) -> float:
    as_of = as_of or datetime.now(UTC)
    month_prefix = as_of.strftime("%Y-%m")
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_inr), 0) FROM llm_calls WHERE substr(called_at, 1, 7) = ?",
            (month_prefix,),
        ).fetchone()
    return float(row[0])


def check_budget() -> tuple[bool, float]:
    """(ok, spent_so_far). ok=False means the monthly cap is hit and the
    caller must refuse new paid analyses — not advisory, an actual block."""
    spent = month_to_date_spend()
    return spent < settings.monthly_budget_inr, spent
