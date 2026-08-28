"""Tests for Stage 2 lite vs full routing."""

from dataclasses import dataclass
from unittest.mock import patch

from stockbot.analysis_routing import compute_stage2_routing, resolve_stage2_mode
from stockbot.llm.extract import ExtractionResult
from stockbot.models import RedFlag, TickerInfo

TICKER = TickerInfo(symbol="CLEAN", exchange="NSE", company_name="Clean Co", isin=None)


@dataclass
class _FakeQuant:
    red_flags: list
    data_validation: object


@dataclass
class _FakeValidation:
    data_confidence: str


@dataclass
class _FakeRoute:
    eligibility: str
    issuer_class: str


def _patch_prescan(*, issuer: str = "NON_FINANCIAL", flags: list | None = None, confidence: str = "HIGH"):
    quant = _FakeQuant(flags or [], _FakeValidation(confidence))
    route = _FakeRoute("AUTO_DEEP_ANALYSIS", issuer)
    return quant, route


@patch("stockbot.analysis_routing.fetch_universe_metrics")
@patch("stockbot.analysis_routing.compute_quant_score")
@patch("stockbot.analysis_routing.decide_eligibility_route")
def test_clean_non_financial_high_confidence_routes_lite(mock_route, mock_quant, mock_fetch):
    mock_fetch.return_value = [object()]
    quant, route = _patch_prescan()
    mock_quant.return_value = quant
    mock_route.return_value = route

    routing = compute_stage2_routing(TICKER)
    assert routing.stage2_mode == "LITE"
    assert routing.quant_red_flags_count == 0


@patch("stockbot.analysis_routing.fetch_universe_metrics")
@patch("stockbot.analysis_routing.compute_quant_score")
@patch("stockbot.analysis_routing.decide_eligibility_route")
def test_defence_issuer_forces_full(mock_route, mock_quant, mock_fetch):
    mock_fetch.return_value = [object()]
    quant, route = _patch_prescan(issuer="DEFENCE_EPC_PROJECT")
    mock_quant.return_value = quant
    mock_route.return_value = route

    routing = compute_stage2_routing(TICKER)
    assert routing.stage2_mode == "FULL"


def test_extraction_red_flags_upgrade_to_full():
    from stockbot.analysis_routing import AnalysisRouting

    lite_prescan = AnalysisRouting(
        stage2_mode="LITE",
        eligibility_verdict="AUTO_DEEP_ANALYSIS",
        issuer_class="NON_FINANCIAL",
        data_confidence="HIGH",
        quant_red_flags_count=0,
        reasons=("clean",),
    )
    extraction = ExtractionResult(
        red_flags_found=[
            RedFlag(
                headline="fraud probe",
                url="https://example.com",
                published_date=__import__("datetime").date(2026, 1, 1),
                found_by_query="fraud",
            )
        ]
    )
    assert resolve_stage2_mode(TICKER, extraction, prescan=lite_prescan) == "FULL"


@patch("stockbot.analysis_routing.settings")
def test_force_stage2_full_config_overrides_lite(mock_settings):
    mock_settings.force_stage2_full = True
    from stockbot.analysis_routing import AnalysisRouting

    lite_prescan = AnalysisRouting(
        stage2_mode="LITE",
        eligibility_verdict="AUTO_DEEP_ANALYSIS",
        issuer_class="NON_FINANCIAL",
        data_confidence="HIGH",
        quant_red_flags_count=0,
        reasons=("clean",),
    )
    assert resolve_stage2_mode(TICKER, ExtractionResult(), prescan=lite_prescan) == "FULL"
