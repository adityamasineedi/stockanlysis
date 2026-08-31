"""/sip command parsing, scheduling, and the monthly job.

Coroutines are driven with asyncio.run so the suite stays sync — there is no
pytest-asyncio in this project.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from stockbot.bot import (
    SIP_REMINDER_DAY,
    _parse_sip_setup,
    _sip_monthly_job,
    schedule_sip_reminder,
)
from stockbot.storage import SipPlan


def _plan(chat_id: int = 1, ticker: str = "BEL") -> SipPlan:
    return SipPlan(
        chat_id=chat_id,
        ticker=ticker,
        monthly_amount=5000.0,
        risk_profile="moderate",
        step_up_pct=0.0,
        horizon_years=20,
        started_at=datetime.now(UTC),
        active=True,
    )


def test_parse_sip_setup_accepts_amount_formats():
    assert _parse_sip_setup(["BEL", "5000"]) == ("BEL", 5000.0, 0.0)
    # Users paste amounts with separators and currency signs.
    assert _parse_sip_setup(["BEL", "5,000"]) == ("BEL", 5000.0, 0.0)
    assert _parse_sip_setup(["BEL", "₹5000"]) == ("BEL", 5000.0, 0.0)
    assert _parse_sip_setup(["BEL", "5000", "stepup", "10"]) == ("BEL", 5000.0, 10.0)
    assert _parse_sip_setup(["BEL", "5000", "step-up", "10%"]) == ("BEL", 5000.0, 10.0)


def test_parse_sip_setup_rejects_bad_input_with_a_message():
    for args in (["BEL"], [], ["BEL", "lots"], ["BEL", "0"], ["BEL", "-5"]):
        result = _parse_sip_setup(args)
        assert isinstance(result, str) and result, f"expected an error for {args!r}"
    # A bad step-up is reported rather than silently treated as zero.
    assert isinstance(_parse_sip_setup(["BEL", "5000", "stepup", "abc"]), str)
    assert isinstance(_parse_sip_setup(["BEL", "5000", "stepup", "-1"]), str)


def test_schedule_sip_reminder_without_job_queue_is_not_fatal():
    """Without the job-queue extra PTB sets job_queue to None; the bot must
    still start, just without reminders."""
    app = SimpleNamespace(job_queue=None)
    assert schedule_sip_reminder(app) is False


def test_schedule_sip_reminder_arms_a_monthly_job():
    recorded = {}

    class FakeQueue:
        def run_monthly(self, callback, when, day, **kwargs):
            recorded.update(callback=callback, when=when, day=day)
            return object()

    assert schedule_sip_reminder(SimpleNamespace(job_queue=FakeQueue())) is True
    assert recorded["day"] == SIP_REMINDER_DAY
    assert recorded["callback"] is _sip_monthly_job
    # Fires on IST, not the container's UTC.
    assert recorded["when"].tzinfo is not None
    assert recorded["when"].tzinfo.utcoffset(None).total_seconds() == 5.5 * 3600


class _RecordingBot:
    def __init__(self, fail_for: set[int] | None = None):
        self.sent: list[tuple[int, str]] = []
        self._fail_for = fail_for or set()

    async def send_message(self, chat_id, text, **kwargs):
        if chat_id in self._fail_for:
            raise RuntimeError("telegram is down for this chat")
        self.sent.append((chat_id, text))


def test_monthly_job_messages_every_active_plan(monkeypatch):
    from stockbot import bot

    plans = [_plan(1, "BEL"), _plan(2, "CRISIL")]
    monkeypatch.setattr(bot, "list_active_sip_plans", lambda: plans)
    monkeypatch.setattr(bot, "_build_sip_reminder", lambda chat_id, ticker: f"due {ticker}")

    telegram_bot = _RecordingBot()
    asyncio.run(_sip_monthly_job(SimpleNamespace(bot=telegram_bot)))

    assert [chat for chat, _ in telegram_bot.sent] == [1, 2]
    assert "due BEL" in telegram_bot.sent[0][1]
    assert "not investment advice" in telegram_bot.sent[0][1]


def test_monthly_job_keeps_going_after_one_chat_fails(monkeypatch):
    """One unreachable chat must not silently cancel everyone else's reminder."""
    from stockbot import bot

    monkeypatch.setattr(bot, "list_active_sip_plans", lambda: [_plan(1), _plan(2), _plan(3)])
    monkeypatch.setattr(bot, "_build_sip_reminder", lambda chat_id, ticker: "due")

    telegram_bot = _RecordingBot(fail_for={2})
    asyncio.run(_sip_monthly_job(SimpleNamespace(bot=telegram_bot)))

    assert [chat for chat, _ in telegram_bot.sent] == [1, 3]


def test_monthly_job_survives_a_reminder_build_failure(monkeypatch):
    from stockbot import bot

    def explode(chat_id, ticker):
        if chat_id == 1:
            raise ValueError("bad stored plan")
        return "due"

    monkeypatch.setattr(bot, "list_active_sip_plans", lambda: [_plan(1), _plan(2)])
    monkeypatch.setattr(bot, "_build_sip_reminder", explode)

    telegram_bot = _RecordingBot()
    asyncio.run(_sip_monthly_job(SimpleNamespace(bot=telegram_bot)))
    assert [chat for chat, _ in telegram_bot.sent] == [2]


def test_price_helper_returns_none_pair_when_fetch_fails(monkeypatch):
    """A dead price feed must not block the instalment reminder."""
    from stockbot.bot import _sip_price_and_high
    from stockbot.fetch import prices

    def boom(symbol):
        raise RuntimeError("yfinance unavailable")

    monkeypatch.setattr(prices, "fetch_price_data", boom)
    assert _sip_price_and_high("BEL") == (None, None)


def test_scenario_rates_prefer_the_stocks_own_analysis():
    from stockbot.sip import DEFAULT_SCENARIO_RATES_PCT
    from stockbot.sip_messages import resolve_scenario_rates

    generic = resolve_scenario_rates(None)
    assert generic.rates_pct == DEFAULT_SCENARIO_RATES_PCT
    assert "generic" in generic.source

    own = resolve_scenario_rates(
        {
            "expected_return": {
                "horizon_years": 3,
                "bear_cagr_range_pct": [-16.9, -13.1],
                "base_cagr_range_pct": [0.4, 3.7],
                "bull_cagr_range_pct": [13.1, 19.1],
            }
        }
    )
    assert own.rates_pct == (-15.0, 2.1, 16.1)  # midpoints
    assert "own" in own.source
    # Stretching a 3-year scenario over 20 years is itself an assumption.
    assert "3-year" in own.caveat

    # Malformed stored data falls back rather than raising.
    assert resolve_scenario_rates({"expected_return": {"base_cagr_range_pct": "nope"}}).rates_pct == (
        DEFAULT_SCENARIO_RATES_PCT
    )


def test_dip_block_always_carries_the_risk_note():
    from stockbot.sip_messages import TOPUP_RISK_NOTE, format_dip_block

    plan = _plan()
    text, dip = format_dip_block(plan, current_price=360, high_3m=420)
    assert dip == "DEEP"
    assert TOPUP_RISK_NOTE in text
    assert "₹5,000–₹10,000" in text

    calm, no_dip = format_dip_block(plan, current_price=419, high_3m=420)
    assert no_dip is None
    assert TOPUP_RISK_NOTE not in calm

    unknown, none_dip = format_dip_block(plan, current_price=400, high_3m=None)
    assert none_dip is None
    assert "history" in unknown


def test_projection_block_labels_generic_rates_as_assumptions():
    from stockbot.sip_messages import format_projection_block, resolve_scenario_rates

    text = format_projection_block(_plan(), resolve_scenario_rates(None))
    assert "not a forecast" in text
    assert "20 years" in text
    for label in ("Conservative", "Base", "Optimistic"):
        assert label in text


@pytest.mark.parametrize("step_up", [0.0, 10.0])
def test_projection_block_mentions_step_up_only_when_set(step_up):
    from stockbot.sip_messages import format_projection_block, resolve_scenario_rates

    plan = SipPlan(1, "BEL", 5000.0, "moderate", step_up, 20, datetime.now(UTC), True)
    text = format_projection_block(plan, resolve_scenario_rates(None))
    assert ("Step-up" in text) is bool(step_up)


@pytest.mark.parametrize(
    ("price", "expected"),
    [(360.0, "flat"), (400.0, "+"), (300.0, "−")],
)
def test_status_never_prints_minus_zero_for_break_even(price, expected):
    """Units are stored rounded, so a flat position can land a hair negative."""
    from stockbot.sip_messages import format_status, resolve_scenario_rates
    from stockbot.storage import SipLedgerSummary

    ledger = SipLedgerSummary(contributions=2, total_invested=7500.0, units_estimate=20.8333)
    text = format_status(_plan(), ledger, resolve_scenario_rates(None), current_price=price)
    assert expected in text
    assert "−₹0)" not in text and "−₹0 " not in text


def test_scenario_rates_read_stored_verdict_without_fetching_a_price(monkeypatch):
    """get_cached fetches the live price and refuses on a >10% move, so over a
    SIP's life it would reject every past analysis and quietly fall back to
    generic rates — while paying for a second price fetch to do it."""
    from datetime import timedelta

    from stockbot import bot

    fetches = []
    monkeypatch.setattr(
        bot, "get_latest_verdict_json",
        lambda t: (
            {
                "expected_return": {
                    "horizon_years": 3,
                    "bear_cagr_range_pct": [-16.9, -13.1],
                    "base_cagr_range_pct": [0.4, 3.7],
                    "bull_cagr_range_pct": [13.1, 19.1],
                }
            },
            datetime.now(UTC) - timedelta(days=800),
        ),
    )
    monkeypatch.setattr(bot, "_sip_price_and_high", lambda t: fetches.append(t) or (1.0, 2.0))

    rates = bot._sip_scenario_rates("BEL")
    assert rates.rates_pct == (-15.0, 2.1, 16.1)  # the stock's own scenarios
    assert fetches == []  # no price fetch to read stored numbers
    # A two-year-old scenario must not be presented as current.
    assert "year(s) ago" in rates.caveat
    assert "/analyze again" in rates.caveat


def test_scenario_rates_fall_back_when_nothing_is_stored(monkeypatch):
    from stockbot import bot
    from stockbot.sip import DEFAULT_SCENARIO_RATES_PCT

    monkeypatch.setattr(bot, "get_latest_verdict_json", lambda t: None)
    assert bot._sip_scenario_rates("BEL").rates_pct == DEFAULT_SCENARIO_RATES_PCT

    def boom(t):
        raise RuntimeError("db unreadable")

    monkeypatch.setattr(bot, "get_latest_verdict_json", boom)
    assert bot._sip_scenario_rates("BEL").rates_pct == DEFAULT_SCENARIO_RATES_PCT


def test_recent_analysis_carries_no_staleness_note(monkeypatch):
    from stockbot.sip_messages import resolve_scenario_rates

    fresh = resolve_scenario_rates(
        {"expected_return": {
            "horizon_years": 3,
            "bear_cagr_range_pct": [1.0, 2.0],
            "base_cagr_range_pct": [3.0, 4.0],
            "bull_cagr_range_pct": [5.0, 6.0],
        }},
        computed_at=datetime.now(UTC),
    )
    assert "ago" not in fresh.caveat
