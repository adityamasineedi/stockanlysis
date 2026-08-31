"""Unit tests for on-dip add-more range calculation."""

from stockbot.action_ranges import (
    add_more_range_blocked_reason,
    compute_add_more_zone_abs,
    resolve_add_more_zone_abs,
)


def test_add_more_on_dip_slices_bear_fv_below_buy_floor():
    zone = compute_add_more_zone_abs(
        fair_value_bear_abs=(300.0, 340.0),
        buy_zone_abs=(330.0, 355.0),
    )
    assert zone == (300.0, 330.0)


def test_add_more_without_buy_zone_uses_full_bear_band():
    zone = compute_add_more_zone_abs(
        fair_value_bear_abs=(300.0, 340.0),
        buy_zone_abs=None,
    )
    assert zone == (300.0, 340.0)


def test_add_more_invalid_when_buy_floor_at_bear_low():
    zone = compute_add_more_zone_abs(
        fair_value_bear_abs=(300.0, 340.0),
        buy_zone_abs=(300.0, 355.0),
    )
    assert zone is None


def test_resolve_add_more_from_verdict_json():
    verdict = {
        "risk": "MEDIUM",
        "add_range_allowed": True,
        "buy_range_allowed": True,
        "buy_zone_abs": [330.0, 355.0],
        "fair_value_bear_abs": [300.0, 340.0],
        "five_year_business_test": {"answer": "YES"},
        "anti_chase_flag": False,
        "wc_gap_classification": None,
    }
    assert resolve_add_more_zone_abs(verdict) == (300.0, 330.0)


def test_add_more_blocked_when_add_range_not_allowed():
    verdict = {
        "add_range_allowed": False,
        "anti_chase_flag": False,
        "wc_gap_classification": None,
    }
    assert add_more_range_blocked_reason(verdict) == "add range not allowed"


def test_add_more_blocked_on_thesis_broken():
    verdict = {
        "add_range_allowed": True,
        "thesis_status": "THESIS_BROKEN",
        "anti_chase_flag": False,
        "wc_gap_classification": None,
    }
    assert add_more_range_blocked_reason(verdict) == "thesis: THESIS_BROKEN"


def test_capital_range_reason_covers_shared_gates_only():
    """The shared gate set must stop at the three that block any new capital.
    Thesis and add_range_allowed are add-more-specific and must not leak into
    the buy line."""
    from stockbot.action_ranges import capital_range_blocked_reason

    assert (
        capital_range_blocked_reason({"five_year_business_test": {"answer": "UNCERTAIN"}})
        == "five-year test: UNCERTAIN"
    )
    assert (
        capital_range_blocked_reason({"wc_gap_classification": "WORKING_CAPITAL_STRESS"})
        == "WC: WORKING_CAPITAL_STRESS"
    )
    assert capital_range_blocked_reason({"anti_chase_flag": True}) == "anti-chase: pause new capital"
    # Add-more-only gates are invisible here, but still block add-more.
    thesis_only = {"add_range_allowed": True, "thesis_status": "THESIS_BROKEN"}
    assert capital_range_blocked_reason(thesis_only) is None
    assert add_more_range_blocked_reason(thesis_only) == "thesis: THESIS_BROKEN"


def test_buy_zone_price_ceiling_uses_shallowest_band_discount():
    """The ceiling is the highest price that could still carry a zone, so it
    comes from the band's *minimum* discount below the fair-value midpoint."""
    from stockbot.action_ranges import buy_zone_price_ceiling

    # JYOTHYLAB, live: FV 210-231 (mid 220.50), MEDIUM risk -> 20% floor.
    assert buy_zone_price_ceiling(
        {"risk": "MEDIUM", "fair_value_abs": [210.0, 231.0]}
    ) == (176.40, "MEDIUM")
    # LOW risk needs only a 10% margin, so its ceiling sits higher.
    assert buy_zone_price_ceiling({"risk": "LOW", "fair_value_abs": [100.0, 100.0]}) == (
        90.0,
        "LOW",
    )


def test_buy_zone_price_ceiling_none_without_inputs():
    from stockbot.action_ranges import buy_zone_price_ceiling

    assert buy_zone_price_ceiling({"risk": "MEDIUM"}) is None
    assert buy_zone_price_ceiling({"fair_value_abs": [210.0, 231.0]}) is None
    assert buy_zone_price_ceiling({"risk": "NONSENSE", "fair_value_abs": [1.0, 2.0]}) is None
    assert buy_zone_price_ceiling({"risk": "MEDIUM", "fair_value_abs": [0.0, 0.0]}) is None


def test_uncertain_with_evidence_not_blocked_in_trade_friendly_mode(monkeypatch):
    from stockbot.action_ranges import capital_range_blocked_reason

    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", True)
    verdict = {
        "five_year_business_test": {
            "answer": "UNCERTAIN",
            "confidence": "HIGH",
            "evidence_for": ["EPS up 3y", "debt down 3y"],
            "evidence_against": ["margin flat"],
        }
    }
    assert capital_range_blocked_reason(verdict) is None
