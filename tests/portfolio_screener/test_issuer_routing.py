"""Issuer classification, cash conversion, and eligibility routing."""

from __future__ import annotations

from stockbot.portfolio_screener.issuer_routing import (
    assess_cash_conversion,
    classify_issuer,
    decide_eligibility_route,
    fundamentals_fetch_failed,
    quality_override_applies,
)
from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig


def _base_nonfin(**kwargs: object) -> StockMetrics:
    defaults: dict[str, object] = {
        "ticker": "X",
        "sector": "Industrials",
        "industry": "Machinery",
        "market_cap_cr": 40000.0,
        "current_price_abs": 200.0,
        "years_available": 5,
        "revenue": 1200.0,
        "revenue_series": [800.0, 900.0, 1000.0, 1100.0, 1200.0],
        "net_income": 180.0,
        "net_income_series": [100.0, 120.0, 140.0, 160.0, 180.0],
        "eps": 18.0,
        "eps_series": [10.0, 12.0, 14.0, 16.0, 18.0],
        "operating_cash_flow": 70.0,
        "ocf_series": [110.0, 130.0, 150.0, 170.0, 70.0],
        "ocf_to_pat": 0.39,
        "roe": 22.0,
        "roce": 24.0,
        "debt_equity": 0.2,
        "interest_coverage": 12.0,
        "net_debt_ebitda": 0.4,
        "pe": 28.0,
        "pb": 4.0,
        "promoter_holding_pct": 51.0,
        "pledged_promoter_holding_pct": 0.0,
        "revenue_cagr_3y": 12.0,
        "eps_cagr_3y": 14.0,
    }
    defaults.update(kwargs)
    return StockMetrics(**defaults)  # type: ignore[arg-type]


def test_quality_override_uses_rounded_component_scores():
    """Display rounds 74.95 → 75; override must match so Mazdock-like names qualify."""
    from stockbot.portfolio_screener.models import (
        ComponentScores,
        DataValidationResult,
        HardFilterResult,
        QuantScreenResult,
    )

    quant = QuantScreenResult(
        ticker="MAZDOCK",
        base_score=58.0,
        red_flag_penalty=-0.2,
        final_quant_score=57.8,
        components=ComponentScores(
            business_quality=74.95,
            financial_strength=81.0,
            growth=83.0,
            cash_flow_quality=40.0,
            capital_efficiency=70.0,
            valuation=17.0,
            balance_sheet=70.0,
            earnings_quality=60.0,
            risk=69.0,
        ),
        red_flags=[],
        data_validation=DataValidationResult(
            ticker="MAZDOCK",
            data_completeness_score=0.96,
            data_quality_score=0.9,
            data_confidence="HIGH",
            missing_metrics={},
            contradictions=[],
            critical_ok=True,
        ),
        hard_filter=HardFilterResult(ticker="MAZDOCK", status="PASS", reasons=[]),
        sector="Industrials",
        industry="Aerospace & Defense",
    )
    assert quality_override_applies(quant) is True


def test_classify_bel_defence():
    m = _base_nonfin(
        ticker="BEL",
        sector="Industrials",
        industry="Aerospace & Defense",
    )
    assert classify_issuer(m) == "DEFENCE_EPC_PROJECT"


def test_classify_icici_bank():
    m = _base_nonfin(
        ticker="ICICIBANK",
        sector="Financial Services",
        industry="Private Sector Bank",
    )
    assert classify_issuer(m) == "BANK"


def test_classify_tata_power_utility():
    m = _base_nonfin(
        ticker="TATAPOWER",
        sector="Utilities",
        industry="Electric Utilities",
    )
    assert classify_issuer(m) == "UTILITY"


def test_classify_reliance_conglomerate():
    m = _base_nonfin(
        ticker="RELIANCE",
        sector="Energy",
        industry="Oil & Gas Refining & Marketing",
    )
    assert classify_issuer(m) == "CONGLOMERATE"


def test_cash_conversion_wc_watch_not_critical_when_3y_strong():
    m = _base_nonfin(
        ticker="BEL",
        sector="Industrials",
        industry="Aerospace & Defense",
        ocf_to_pat=0.37,
        # 3y cumulative OCF/PAT ≥ 0.80
        ocf_series=[150.0, 160.0, 70.0],
        net_income_series=[140.0, 150.0, 180.0],
    )
    cash = assess_cash_conversion(m, "DEFENCE_EPC_PROJECT")
    assert cash.status == "WATCH"
    assert cash.ocf_pat_3y is not None
    assert cash.ocf_pat_3y >= 0.80


def test_defence_escalated_watch_when_3y_cumulative_extreme():
    """Mazdock-style: 3y cumulative ~0.02 → ESCALATED_WATCH + cheap WC first."""
    m = _base_nonfin(
        ticker="MAZDOCK",
        sector="Industrials",
        industry="Aerospace & Defense",
        ocf_to_pat=-0.93,
        operating_cash_flow=-100.0,
        # ΣOCF≈10 / ΣPAT≈500 → ~0.02
        ocf_series=[50.0, 60.0, -100.0],
        net_income_series=[160.0, 170.0, 180.0],
        net_income=180.0,
    )
    cash = assess_cash_conversion(m, "DEFENCE_EPC_PROJECT")
    assert cash.status == "ESCALATED_WATCH"
    assert cash.ocf_pat_3y is not None
    assert cash.ocf_pat_3y < 0.25
    assert cash.years_used >= 2
    assert "cumulative" in cash.reason.lower() or "Σ" in cash.reason or "3y" in cash.reason
    assert "reported cash conversion" in cash.interpretation.lower()
    assert "may reflect" in cash.interpretation.lower()

    quant = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    # Force defence + override-like components via routing on real scores if possible
    decision = decide_eligibility_route(m, quant)
    assert decision.cash_conversion.status == "ESCALATED_WATCH"
    if decision.eligibility == "SECTOR_SPECIFIC_REVIEW":
        assert decision.next_action == "CHEAP_WC_RECONCILIATION_FIRST"
        assert "quality/growth/strength" in decision.key_reason.lower()
        assert "working-capital" in decision.key_risk.lower()


def test_loss_making_ocf_pat_not_pass():
    """Swiggy-style: negative PAT must not yield Cash conversion PASS via OCF/PAT."""
    m = _base_nonfin(
        ticker="SWIGGY",
        sector="Consumer Cyclical",
        industry="Internet Retail",
        ocf_to_pat=0.77,
        operating_cash_flow=-100.0,
        net_income=-130.0,
        ocf_series=[-80.0, -90.0, -100.0],
        net_income_series=[-110.0, -120.0, -130.0],
        roe=-20.0,
    )
    assert classify_issuer(m) == "LOSS_MAKING_GROWTH"
    cash = assess_cash_conversion(m, "LOSS_MAKING_GROWTH")
    assert cash.status == "NOT_APPLICABLE_WHILE_LOSS_MAKING"
    assert cash.status != "PASS"
    assert "not meaningful" in cash.interpretation.lower() or "negative" in cash.interpretation.lower()


def test_fmt_ratio_rounds_for_display():
    from stockbot.portfolio_screener.issuer_routing import fmt_ratio

    assert fmt_ratio(0.2506506180871828) == "0.25"
    assert fmt_ratio(None) == "null"

    m = _base_nonfin(
        ticker="BEL",
        sector="Industrials",
        industry="Aerospace & Defense",
        ocf_to_pat=0.25,
        ocf_series=[40.0, 50.0, 45.0],
        net_income_series=[100.0, 120.0, 180.0],
    )
    # 135/400 = 0.3375 → WATCH not escalated (<0.5 but >=0.25)
    cash = assess_cash_conversion(m, "DEFENCE_EPC_PROJECT")
    assert cash.status == "WATCH"
    assert cash.ocf_pat_3y is not None
    assert cash.ocf_pat_3y >= 0.25
    assert "cumulative" in (cash.interpretation or "").lower() or "threshold" in (
        cash.interpretation or ""
    ).lower()


def test_cash_conversion_critical_when_current_and_3y_weak():
    m = _base_nonfin(
        ocf_to_pat=0.30,
        ocf_series=[30.0, 35.0, 40.0],
        net_income_series=[100.0, 110.0, 120.0],
    )
    cash = assess_cash_conversion(m, "NON_FINANCIAL")
    assert cash.status == "CRITICAL"


def test_bel_quality_override_review_exception():
    m = _base_nonfin(
        ticker="BEL",
        sector="Industrials",
        industry="Aerospace & Defense",
        ocf_to_pat=0.25,
        operating_cash_flow=45.0,
        ocf_series=[40.0, 50.0, 45.0],
        net_income_series=[120.0, 130.0, 180.0],
        free_cash_flow=50.0,
        ebitda=250.0,
        operating_margin=18.0,
        ebitda_margin=20.0,
        current_ratio=1.6,
        pe=55.0,
        pb=10.0,
    )
    quant = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    decision = decide_eligibility_route(m, quant)
    assert decision.issuer_class == "DEFENCE_EPC_PROJECT"
    assert decision.cash_conversion.status == "WATCH"
    assert decision.eligibility == "SECTOR_SPECIFIC_REVIEW"
    assert decision.suitable_for_deep_analysis is True
    assert decision.quality_override is True
    assert decision.route == "DEFENCE_WC_REVIEW"


def test_icici_model_not_applicable():
    m = _base_nonfin(
        ticker="ICICIBANK",
        sector="Financial Services",
        industry="Private Sector Bank",
        debt_equity=8.0,  # bank leverage must not force hard reject
        ocf_to_pat=0.2,
        interest_coverage=1.0,
        pb=2.5,
        roe=16.0,
    )
    quant = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    decision = decide_eligibility_route(m, quant)
    assert decision.eligibility == "SECTOR_SPECIFIC_REVIEW"
    assert decision.route == "BANK_SCORECARD"
    assert decision.suitable_for_deep_analysis is True


def test_tata_power_utility_marginal():
    m = _base_nonfin(
        ticker="TATAPOWER",
        sector="Utilities",
        industry="Electric Utilities",
        debt_equity=1.8,
        interest_coverage=2.5,
        ocf_to_pat=0.95,
        ocf_series=[160.0, 170.0, 180.0],
        net_income_series=[150.0, 160.0, 180.0],
        roe=12.0,
        roce=10.0,
    )
    quant = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    decision = decide_eligibility_route(m, quant)
    assert decision.issuer_class == "UTILITY"
    assert decision.eligibility == "SECTOR_SPECIFIC_REVIEW"
    assert decision.suitable_for_deep_analysis is True


def test_reliance_conglomerate_sotp_review():
    m = _base_nonfin(
        ticker="RELIANCE",
        sector="Energy",
        industry="Oil & Gas Refining & Marketing",
        debt_equity=0.4,
        ocf_to_pat=1.1,
        ocf_series=[200.0, 220.0, 240.0],
        net_income_series=[180.0, 200.0, 220.0],
        operating_cash_flow=240.0,
        revenue_cagr_3y=4.0,  # soft aggregate growth must not auto-reject
    )
    quant = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    decision = decide_eligibility_route(m, quant)
    assert decision.issuer_class == "CONGLOMERATE"
    assert decision.eligibility == "SECTOR_SPECIFIC_REVIEW"
    assert decision.route in {
        "CONGLOMERATE_SOTP_REVIEW",
        "AUTO_DEEP",
        "SECTOR_SPECIFIC_REVIEW",
        "EXCEPTION_DEEP_REVIEW",
    }


def test_e2e_fetch_fail_data_unavailable():
    m = StockMetrics(
        ticker="E2E",
        years_available=0,
        missing={
            "revenue": "fundamentals fetch failed: timeout",
            "net_income": "fundamentals fetch failed: timeout",
            "operating_cash_flow": "fundamentals fetch failed: timeout",
            "eps": "fundamentals fetch failed: timeout",
        },
    )
    assert fundamentals_fetch_failed(m) is True
    quant = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    decision = decide_eligibility_route(m, quant)
    assert decision.eligibility == "DATA_UNAVAILABLE_RETRY"
    assert decision.suitable_for_deep_analysis is False
    assert "weak" not in decision.key_reason.lower() or "not" in decision.key_reason.lower()


def test_quant_below_60_not_always_not_suitable_with_override():
    m = _base_nonfin(
        ticker="BEL",
        sector="Industrials",
        industry="Aerospace & Defense",
        # Weak cash year but strong multi-year; rich valuation to drag composite
        ocf_to_pat=0.40,
        ocf_series=[140.0, 150.0, 80.0],
        net_income_series=[130.0, 140.0, 180.0],
        pe=80.0,
        pb=15.0,
        ev_ebitda=40.0,
    )
    quant = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    decision = decide_eligibility_route(m, quant)
    if quant.final_quant_score < 60:
        assert decision.eligibility not in {
            "NOT_SUITABLE_FOR_3Y_RESEARCH",
            "NOT_SUITABLE",
        } or decision.cash_conversion.status == "CRITICAL"
