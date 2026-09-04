"""Score calibration — avoid triple-counting OCF/PAT weakness."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.prescan_display import synthesize_why
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.red_flags import collect_red_flags
from stockbot.portfolio_screener.risk_scorer import score_earnings_quality
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig


def _hfcl_like() -> StockMetrics:
    """Mixed name: decent growth/leverage, weak ROE, negative OCF/PAT, rich P/E."""
    return StockMetrics(
        ticker="HFCL",
        sector="Technology",
        industry="Communication Equipment",
        market_cap_cr=35000.0,
        current_price_abs=230.0,
        revenue=6000.0,
        revenue_series=[4000.0, 4500.0, 5000.0, 5500.0, 6000.0],
        net_income=600.0,
        net_income_series=[300.0, 350.0, 400.0, 500.0, 600.0],
        eps=4.0,
        eps_series=[2.0, 2.5, 3.0, 3.5, 4.0],
        operating_profit=1100.0,
        operating_margin=0.19,
        operating_margin_series=[0.15, 0.16, 0.17, 0.18, 0.19],
        ebitda=1200.0,
        ebit=1100.0,
        operating_cash_flow=-375.0,
        ocf_series=[50.0, -20.0, 10.0, -100.0, -375.0],
        free_cash_flow=-500.0,
        fcf_series=[-50.0, -80.0, -100.0, -200.0, -500.0],
        roe=6.7,
        roce=11.0,
        debt_equity=0.36,
        net_debt_ebitda=0.97,
        interest_coverage=4.6,
        ocf_to_pat=-0.63,
        fcf_to_pat=-0.83,
        pe=60.0,
        revenue_cagr_3y=0.10,
        eps_cagr_3y=0.18,
        years_available=5,
        pledged_promoter_holding_pct=0.0,
    )


def test_ocf_pat_flag_does_not_stack_score_penalty() -> None:
    m = _hfcl_like()
    flags = collect_red_flags(m)
    ocf_flags = [f for f in flags if f.code.startswith("OCF_PAT")]
    assert ocf_flags, "cash-gap flag should still exist for routing"
    assert all(f.penalty == 0.0 for f in ocf_flags)


def test_earnings_quality_does_not_re_score_ocf_to_pat() -> None:
    m = _hfcl_like()
    # With OCF removed from earnings quality, a stable EPS path should not
    # collapse to ~20 solely because OCF/PAT is negative.
    eq = score_earnings_quality(m)
    assert eq >= 35.0


def test_hfcl_like_score_not_crushed_into_low_30s() -> None:
    m = _hfcl_like()
    q = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    # Pre-fix HFCL landed ~33 via cash score + EQ OCF + −10 flag.
    # After de-duplication, mixed names should sit mid-40s+, not low 30s.
    assert q.red_flag_penalty == 0.0 or q.red_flag_penalty > -10.0
    assert q.final_quant_score >= 40.0
    assert q.components.cash_flow_quality < 40.0  # cash weakness still counted once


def test_synthesize_why_names_cash_and_roe_not_vague_growth() -> None:
    why = synthesize_why(
        key_reason="quant_score 32.7 below 3y research floor; cash CRITICAL",
        quality=47.9,
        growth=73.6,
        strength=51.9,
        final_score=42.0,
        roe=6.7,
        ocf_pat=-0.63,
        pe=60.0,
    )
    assert "Growth look fine — see gate above" not in why
    assert "OCF/PAT" in why
    assert "ROE" in why
    assert "P/E" in why
