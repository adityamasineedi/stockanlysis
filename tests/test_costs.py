"""Module 9 (costs) unit tests against a temp SQLite file — no real DB
touched, no network."""

import sqlite3
from datetime import UTC, datetime

import pytest

from stockbot import costs as costs_module
from stockbot.costs import check_budget, compute_cost_inr, log_call, month_to_date_spend


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(costs_module, "DB_PATH", tmp_path / "test_costs.sqlite3")
    monkeypatch.setattr(costs_module.settings, "usd_inr_rate", 90.0)


def test_compute_cost_sonnet_no_cache():
    # 1M input tokens @ $2, 1M output tokens @ $10 -> $12 -> ₹1080 at rate 90
    cost = compute_cost_inr("claude-sonnet-5", 1_000_000, 1_000_000, cached_tokens=0)
    assert cost == pytest.approx(12.0 * 90.0)


def test_compute_cost_opus_with_cache_discount():
    # input_tokens is the UNCACHED remainder only, per the real API — a
    # fully-cached-prefix call reports input_tokens=0 and cached_tokens
    # separately, additive, never as a subset to subtract out.
    # 1M cached tokens: 1M/1M * $5 * 0.1 = $0.5 -> ₹45
    cost = compute_cost_inr("claude-opus-5", 0, 0, cached_tokens=1_000_000)
    assert cost == pytest.approx(0.5 * 90.0)


def test_compute_cost_with_cache_write_premium():
    # 1M tokens written to cache: 1M/1M * $5 * 1.25 = $6.25 -> ₹562.50
    cost = compute_cost_inr("claude-opus-5", 0, 0, cache_creation_tokens=1_000_000)
    assert cost == pytest.approx(6.25 * 90.0)


def test_compute_cost_1h_cache_write_billed_at_2x_not_5m_premium():
    # Regression: switching both stages to ttl="1h" without updating this
    # would have silently under-billed every cache write at 1.25x instead
    # of the real 2x rate for a 1h write.
    # 1M tokens written at 1h TTL: 1M/1M * $5 * 2.0 = $10.00 -> ₹900
    cost = compute_cost_inr(
        "claude-opus-5", 0, 0, cache_creation_tokens=1_000_000, cache_creation_1h_tokens=1_000_000
    )
    assert cost == pytest.approx(10.0 * 90.0)


def test_compute_cost_mixed_5m_and_1h_cache_writes_billed_separately():
    # 600k at 1h ($5 * 2.0), 400k at 5m ($5 * 1.25)
    cost = compute_cost_inr(
        "claude-opus-5", 0, 0, cache_creation_tokens=1_000_000, cache_creation_1h_tokens=600_000
    )
    expected_usd = 600_000 / 1_000_000 * 5.0 * 2.0 + 400_000 / 1_000_000 * 5.0 * 1.25
    assert cost == pytest.approx(expected_usd * 90.0)


def test_compute_cost_rejects_1h_tokens_exceeding_total():
    with pytest.raises(ValueError, match="exceeds"):
        compute_cost_inr(
            "claude-opus-5", 0, 0, cache_creation_tokens=100, cache_creation_1h_tokens=200
        )


def test_compute_cost_input_and_cache_are_additive_not_overlapping():
    # A real cached call: some fresh input, some read from cache, some
    # output — all three must add, not have cached subtracted from input.
    cost = compute_cost_inr(
        "claude-sonnet-5", input_tokens=1000, output_tokens=500, cached_tokens=4000
    )
    expected_usd = 1000 / 1_000_000 * 2.0 + 4000 / 1_000_000 * 2.0 * 0.1 + 500 / 1_000_000 * 10.0
    assert cost == pytest.approx(expected_usd * 90.0)


def test_compute_cost_unknown_model_raises():
    with pytest.raises(ValueError):
        compute_cost_inr("gpt-4", 1000, 1000)


def test_log_call_persists_and_returns_cost():
    cost = log_call("claude-sonnet-5", 100_000, 5_000, cached_tokens=0)
    assert cost > 0
    assert month_to_date_spend() == pytest.approx(cost)


def test_log_call_persists_cache_creation_1h_and_thinking_tokens():
    log_call(
        "claude-sonnet-5",
        6_000,
        20_000,
        cache_creation_tokens=9_000,
        cache_creation_1h_tokens=9_000,
        thinking_tokens=17_000,
    )
    conn = sqlite3.connect(costs_module.DB_PATH)
    row = conn.execute(
        "SELECT cache_creation_tokens, cache_creation_1h_tokens, thinking_tokens FROM llm_calls"
    ).fetchone()
    assert row == (9_000, 9_000, 17_000)


def test_month_to_date_spend_sums_multiple_calls():
    c1 = log_call("claude-sonnet-5", 50_000, 2_000)
    c2 = log_call("claude-opus-5", 20_000, 3_000)
    assert month_to_date_spend() == pytest.approx(c1 + c2)


def test_month_to_date_spend_excludes_other_months():
    log_call("claude-sonnet-5", 50_000, 2_000)
    next_month = datetime(2027, 1, 15, tzinfo=UTC)
    assert month_to_date_spend(as_of=next_month) == 0.0


def test_month_to_date_spend_with_no_calls_is_zero():
    assert month_to_date_spend() == 0.0


def test_check_budget_ok_when_under_cap(monkeypatch):
    monkeypatch.setattr(costs_module.settings, "monthly_budget_inr", 1400.0)
    log_call("claude-sonnet-5", 10_000, 1_000)
    ok, spent = check_budget()
    assert ok is True
    assert spent > 0


def test_check_budget_blocks_when_cap_reached(monkeypatch):
    monkeypatch.setattr(costs_module.settings, "monthly_budget_inr", 1.0)
    log_call("claude-opus-5", 100_000, 20_000)  # comfortably exceeds ₹1
    ok, spent = check_budget()
    assert ok is False
    assert spent >= 1.0


def test_every_model_actually_used_by_the_pipeline_has_pricing():
    # Regression test: the real live v3-migration verification run crashed
    # here — Stage 1 switched to claude-haiku-4-5-20251001 but its pricing
    # entry was never added, so a real, successful, billed Haiku call
    # raised ValueError trying to log its own cost, and Stage 2 never ran.
    # This checks the actual MODEL constants pipeline code will call with,
    # not just that the pricing table has some model in it.
    from stockbot.llm.extract import MODEL as stage1_model
    from stockbot.llm.verdict import MODEL as stage2_model

    assert stage1_model in costs_module.ANTHROPIC_PRICING_USD_PER_MTOK
    assert stage2_model in costs_module.ANTHROPIC_PRICING_USD_PER_MTOK


def test_log_call_persists_cache_creation_tokens():
    # Found live: the Console's reported token totals ran ahead of our own
    # log's sum because cache WRITE tokens were billed correctly (via
    # compute_cost_inr) but never actually stored anywhere — only cache
    # READS had a column. This closes that observability gap.
    log_call("claude-opus-5", 1000, 500, cached_tokens=200, cache_creation_tokens=4300)
    with costs_module._connect() as conn:
        row = conn.execute(
            "SELECT cached_tokens, cache_creation_tokens FROM llm_calls ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row == (200, 4300)


def test_migration_adds_cache_creation_tokens_to_pre_existing_db(tmp_path, monkeypatch):
    # Regression test for the migration path itself: a DB created before
    # this column existed (real case — 16 rows already existed live) must
    # gain the column via ALTER TABLE, not silently ignore old rows or
    # crash on the next _connect().
    db_path = tmp_path / "pre_existing.sqlite3"
    monkeypatch.setattr(costs_module, "DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            called_at TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            cost_inr REAL NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO llm_calls (called_at, model, input_tokens, output_tokens, cached_tokens, cost_inr) "
        "VALUES ('2026-08-01T00:00:00+00:00', 'claude-opus-5', 1000, 500, 0, 10.0)"
    )
    conn.commit()
    conn.close()

    cost = log_call("claude-sonnet-5", 2000, 1000, cache_creation_tokens=500)
    assert cost > 0

    with costs_module._connect() as conn:
        rows = conn.execute(
            "SELECT cache_creation_tokens FROM llm_calls ORDER BY id"
        ).fetchall()
    assert rows == [(0,), (500,)]  # old row backfilled to 0, new row correct


def test_migration_adds_provider_column_to_pre_existing_db(tmp_path, monkeypatch):
    # Same migration pattern, for the provider column added in 17D.
    db_path = tmp_path / "pre_provider.sqlite3"
    monkeypatch.setattr(costs_module, "DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            called_at TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
            cost_inr REAL NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO llm_calls (called_at, model, input_tokens, output_tokens, cached_tokens, cost_inr) "
        "VALUES ('2026-08-01T00:00:00+00:00', 'claude-opus-5', 1000, 500, 0, 10.0)"
    )
    conn.commit()
    conn.close()

    log_call("deepseek-v4-flash", 2000, 1000, provider="deepseek")

    with costs_module._connect() as conn:
        rows = conn.execute("SELECT provider FROM llm_calls ORDER BY id").fetchall()
    assert rows == [("anthropic",), ("deepseek",)]  # old row backfilled, new row correct


def test_deepseek_cost_off_peak():
    # Monday 12:00 UTC -- outside both peak windows (01:00-04:00, 06:00-10:00)
    called_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)  # a real Monday
    cost = compute_cost_inr(
        "deepseek-v4-flash", input_tokens=1_000_000, output_tokens=1_000_000,
        cached_tokens=0, provider="deepseek", called_at=called_at,
    )
    # cache-miss input $0.22 + output $0.66 = $0.88 -> Rs79.20 at rate 90
    assert cost == pytest.approx(0.88 * 90.0)


def test_deepseek_cost_doubles_at_peak_hour():
    # Monday 07:00 UTC -- inside the 06:00-10:00 peak window
    called_at = datetime(2026, 8, 24, 7, 0, tzinfo=UTC)
    cost = compute_cost_inr(
        "deepseek-v4-flash", input_tokens=1_000_000, output_tokens=1_000_000,
        cached_tokens=0, provider="deepseek", called_at=called_at,
    )
    assert cost == pytest.approx(0.88 * 2 * 90.0)


def test_deepseek_cost_weekend_is_always_off_peak():
    # Saturday 07:00 UTC -- would be peak hours on a weekday, but weekends
    # are off-peak regardless of the hour.
    called_at = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)  # a real Saturday
    cost = compute_cost_inr(
        "deepseek-v4-flash", input_tokens=1_000_000, output_tokens=1_000_000,
        cached_tokens=0, provider="deepseek", called_at=called_at,
    )
    assert cost == pytest.approx(0.88 * 90.0)


def test_deepseek_cache_hit_tokens_billed_at_cheaper_rate():
    called_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    cost = compute_cost_inr(
        "deepseek-v4-flash", input_tokens=0, output_tokens=0,
        cached_tokens=1_000_000, provider="deepseek", called_at=called_at,
    )
    # cache-hit input $0.007 -> Rs0.63 at rate 90
    assert cost == pytest.approx(0.007 * 90.0)


def test_deepseek_rejects_cache_creation_tokens():
    with pytest.raises(ValueError, match="not applicable to DeepSeek"):
        compute_cost_inr(
            "deepseek-v4-flash", input_tokens=100, output_tokens=100,
            cache_creation_tokens=50, provider="deepseek",
        )


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        compute_cost_inr("some-model", 100, 100, provider="not-a-provider")


def test_openai_gpt4o_mini_pricing():
    # 1M in @ $0.15 + 1M out @ $0.60 = $0.75 → ₹67.5 at rate 90
    cost = compute_cost_inr(
        "gpt-4o-mini", 1_000_000, 1_000_000, provider="openai"
    )
    assert cost == pytest.approx(0.75 * 90.0)


def test_openai_cached_input_pricing():
    cost = compute_cost_inr(
        "gpt-4o-mini",
        input_tokens=0,
        output_tokens=0,
        cached_tokens=1_000_000,
        provider="openai",
    )
    assert cost == pytest.approx(0.075 * 90.0)
