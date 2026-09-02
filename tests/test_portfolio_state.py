"""Position sizing against declared capital — pure functions, no I/O."""

from __future__ import annotations

import pytest

from stockbot.portfolio_state import (
    DEFAULT_MAX_POSITION_PCT,
    concentration_breaches,
    headroom_inr,
    intended_position_inr,
    position_pct,
    position_value,
    size_position,
    tranche_amounts,
)


def test_size_position_against_declared_capital():
    # 25 BEL at ₹412.50 against ₹5,00,000 of capital, 10% cap.
    sizing = size_position(25, 412.50, 500_000)
    assert sizing.value_inr == 10_312.50
    assert sizing.pct_of_capital == pytest.approx(2.06)
    assert sizing.headroom_inr == 39_687.50  # 10% of capital, less what is held
    assert sizing.over_cap is False


def test_headroom_is_clamped_at_zero_when_over_cap():
    """A position past its cap has no room. A negative headroom would read as
    "you must sell", which is a different judgement this does not make."""
    sizing = size_position(200, 412.50, 500_000)
    assert sizing.pct_of_capital == pytest.approx(16.5)
    assert sizing.over_cap is True
    assert sizing.headroom_inr == 0.0


def test_headroom_of_an_empty_position_is_the_whole_allowance():
    assert headroom_inr(None, 500_000, 10.0) == 50_000.0
    assert headroom_inr(0, 500_000, 10.0) == 50_000.0


def test_tranche_amounts_sum_exactly_to_the_intended_position():
    """Four slices of a number not divisible by four lose paise; a plan whose
    parts don't add up invites doubt about the rest of it."""
    assert tranche_amounts(intended_position_inr(500_000)) == [12_500.0] * 4

    odd = tranche_amounts(10_000.01)
    assert len(odd) == 4
    assert sum(odd) == pytest.approx(10_000.01)
    assert odd[-1] != odd[0]  # remainder lands in the last tranche

    assert tranche_amounts(1000, count=1) == [1000.0]
    assert tranche_amounts(1000, count=0) is None


def test_custom_cap_overrides_the_default():
    assert DEFAULT_MAX_POSITION_PCT == 10.0
    tight = size_position(25, 412.50, 500_000, max_position_pct=2.0)
    assert tight.over_cap is True  # 2.06% breaches a 2% cap
    assert tight.headroom_inr == 0.0
    assert intended_position_inr(500_000, 8.0) == 40_000.0


def test_nothing_is_computed_without_a_usable_denominator():
    """A percentage against an unknown denominator looks authoritative and is
    arbitrary — withhold it, as summarize_sip_contributions withholds units."""
    assert position_pct(10_312.5, None) is None
    assert position_pct(10_312.5, 0) is None
    assert position_pct(10_312.5, -5) is None
    assert size_position(25, 412.50, None) is None
    assert intended_position_inr(None) is None
    assert headroom_inr(1000, None) is None


def test_non_finite_and_wrong_typed_inputs_are_refused():
    """float() parses "nan"/"inf" and `nan <= 0` is False — the same hole that
    once let a NaN SIP plan save and render "₹nan" everywhere."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert position_value(25, bad) is None
        assert position_value(bad, 412.50) is None
        assert position_pct(10_000, bad) is None
    assert position_value("25", 412.50) is None
    assert position_value(None, 412.50) is None
    # bool is an int subclass; True shares are not 1 share.
    assert position_value(True, 100) is None


def test_concentration_breaches_ranked_worst_first():
    breaches = concentration_breaches(
        {"BEL": 82_500.0, "CRISIL": 10_000.0, "XYZ": 60_000.0}, 500_000
    )
    assert breaches == [("BEL", 16.5), ("XYZ", 12.0)]


def test_uncomputable_positions_are_not_reported_as_compliant():
    """Absence of evidence is not a passing check — but it is also not a
    breach, so such rows are simply absent from the result."""
    breaches = concentration_breaches({"BEL": 82_500.0, "UNKNOWN": None}, 500_000)
    assert [t for t, _ in breaches] == ["BEL"]
    assert concentration_breaches({"BEL": 82_500.0}, None) == []
    assert concentration_breaches({}, 500_000) == []


def test_capital_and_hold_arg_parsing():
    from stockbot.bot import _parse_capital_args, _parse_hold_args

    assert _parse_capital_args(["500000"]) == (500_000.0, None, None)
    assert _parse_capital_args(["5,00,000"]) == (500_000.0, None, None)
    assert _parse_capital_args(["500000", "max", "8"]) == (500_000.0, 8.0, None)
    assert _parse_capital_args(["500000", "max", "10", "sector", "25"]) == (
        500_000.0,
        10.0,
        25.0,
    )
    assert _parse_capital_args(["500000", "sector", "20"]) == (500_000.0, None, 20.0)
    assert _parse_hold_args(["BEL", "25", "412.50"]) == ("BEL", 25.0, 412.50)
    assert _parse_hold_args(["BEL", "25", "₹412.50"]) == ("BEL", 25.0, 412.50)

    # Non-finite and out-of-range values are refused, not stored as ₹nan.
    for bad in (["nan"], ["inf"], ["0"], ["-5"], ["lots"]):
        assert isinstance(_parse_capital_args(bad), str), bad
    assert isinstance(_parse_capital_args(["500000", "max", "0"]), str)
    assert isinstance(_parse_capital_args(["500000", "max", "150"]), str)
    assert isinstance(_parse_capital_args(["500000", "max"]), str)   # value dropped
    assert isinstance(_parse_capital_args(["500000", "8"]), str)     # missing keyword
    assert isinstance(_parse_capital_args(["500000", "sector"]), str)
    for bad in (["BEL", "25"], ["BEL", "nan", "412.50"], ["BEL", "25", "0"]):
        assert isinstance(_parse_hold_args(bad), str), bad


def test_sector_concentration_breaches():
    from stockbot.portfolio_state import (
        DEFAULT_MAX_SECTOR_PCT,
        sector_concentration_breaches,
        sector_totals,
    )

    assert DEFAULT_MAX_SECTOR_PCT == 25.0
    positions = {"TCS": 80_000.0, "INFY": 70_000.0, "RELIANCE": 50_000.0}
    sectors = {"TCS": "Technology", "INFY": "Technology", "RELIANCE": "Energy"}
    totals = sector_totals(positions, sectors)
    assert totals["Technology"] == 150_000.0
    breaches = sector_concentration_breaches(positions, sectors, 500_000.0, 25.0)
    assert breaches == [("Technology", 30.0)]
    assert sector_concentration_breaches(positions, sectors, 500_000.0, 40.0) == []


def test_position_line_is_absent_without_capital_or_holding(monkeypatch, tmp_path):
    """No declared capital means no denominator; the line must not appear
    rather than quote a percentage of a guess."""
    from stockbot import bot, storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "pos.db")
    monkeypatch.setattr(bot, "_sip_price_and_high", lambda t: (412.50, 430.0))

    # Neither policy nor holding.
    assert bot._build_position_line(42, "BEL") is None

    # Holding but no policy.
    storage.save_holding(42, "BEL", 25, 400.0)
    assert bot._build_position_line(42, "BEL") is None

    # Policy but no holding in that name.
    storage.save_risk_policy(42, 500_000)
    assert bot._build_position_line(42, "CRISIL") is None

    # Both present, but no live price.
    monkeypatch.setattr(bot, "_sip_price_and_high", lambda t: (None, None))
    assert bot._build_position_line(42, "BEL") is None


def test_position_line_reports_value_share_and_headroom(monkeypatch, tmp_path):
    from stockbot import bot, storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "pos.db")
    monkeypatch.setattr(bot, "_sip_price_and_high", lambda t: (412.50, 430.0))
    storage.save_risk_policy(42, 500_000)
    storage.save_holding(42, "BEL", 25, 400.0)

    line = bot._build_position_line(42, "BEL")
    assert "₹10,312" in line          # 25 × 412.50
    assert "2.1% of capital" in line  # of ₹5,00,000
    assert "₹39,688 more" in line     # 10% cap less the position
    assert "cap 10%" in line


def test_position_line_flags_a_position_at_its_cap(monkeypatch, tmp_path):
    """Headroom is clamped at zero, so the line must say so rather than
    print "room for ₹0 more"."""
    from stockbot import bot, storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "pos.db")
    monkeypatch.setattr(bot, "_sip_price_and_high", lambda t: (412.50, 430.0))
    storage.save_risk_policy(42, 500_000)
    storage.save_holding(42, "BEL", 200, 400.0)  # 16.5% of capital

    line = bot._build_position_line(42, "BEL")
    assert "16.5% of capital" in line
    assert "at or over your cap" in line
    assert "room for" not in line
