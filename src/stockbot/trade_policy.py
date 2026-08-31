"""Trade-friendly policy — relaxes constitution gates without dropping safety rails.

When enabled (default), the bot issues buy ranges more often so SIP-style
investors are not blocked on UNCERTAIN names with real evidence, soft WC
labels, or prescan friction. NO answers, WORKING_CAPITAL_STRESS, and lazy
UNCERTAIN (no evidence) still block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stockbot.config import settings

if TYPE_CHECKING:
    from stockbot.llm.verdict import FiveYearBusinessTest

_WC_SOFT_CLASSIFICATIONS = frozenset({"INCONCLUSIVE", "DATA_OR_SCOPE_ERROR"})
_MIN_UNCERTAIN_EVIDENCE = 2


def trade_friendly_mode() -> bool:
    return settings.trade_friendly_mode


def prescan_required_for_analyze() -> bool:
    if trade_friendly_mode():
        return False
    return settings.require_prescan_for_analyze


def five_year_test_from_dict(raw: object) -> dict | None:
    if isinstance(raw, dict):
        return raw
    return None


def five_year_allows_buy_zone(test: FiveYearBusinessTest | dict | None) -> bool:
    """True when constitution may carry a buy/add range for this five-year answer."""
    if test is None:
        return True
    if isinstance(test, dict):
        answer = str(test.get("answer") or "").strip().upper()
        confidence = str(test.get("confidence") or "").strip().upper()
        evidence_for = test.get("evidence_for") or []
        evidence_against = test.get("evidence_against") or []
    else:
        answer = (test.answer or "").strip().upper()
        confidence = (test.confidence or "").strip().upper()
        evidence_for = list(test.evidence_for or ())
        evidence_against = list(test.evidence_against or ())

    if answer == "YES":
        return True
    if answer == "NO":
        return False
    if answer != "UNCERTAIN":
        return True

    if not trade_friendly_mode():
        return False

    if confidence == "LOW":
        return False
    if len(evidence_for) >= _MIN_UNCERTAIN_EVIDENCE:
        return True
    if confidence == "HIGH" and len(evidence_for) >= 1:
        return True
    return len(evidence_for) >= 1 and len(evidence_against) >= 1


def five_year_blocks_capital_range(verdict_json: dict) -> str | None:
    test = five_year_test_from_dict(verdict_json.get("five_year_business_test"))
    if test is None:
        return None
    if five_year_allows_buy_zone(test):
        return None
    answer = str(test.get("answer") or "").strip().upper()
    if answer:
        return f"five-year test: {answer}"
    return "five-year test: blocked"


def wc_gap_blocks_buy_zone(wc_gap_classification: object) -> bool:
    """Same semantics as constitution_gates.wc_gap_blocks_buy_zone — shared here."""
    if wc_gap_classification is None:
        return False
    text = str(wc_gap_classification).strip().upper()
    if not text:
        return False
    if trade_friendly_mode() and text in _WC_SOFT_CLASSIFICATIONS:
        return False
    return text != "TEMPORARY_BILLING_CYCLE"


def prescan_auto_deep_min_score() -> float:
    if trade_friendly_mode():
        return settings.prescan_auto_deep_min_score
    return 70.0


def effective_risk_discount_bands() -> dict[str, tuple[float, float]]:
    """Buy-zone margin-of-safety bands — slightly wider in trade-friendly mode."""
    if trade_friendly_mode():
        return {
            "LOW": (0.08, 0.15),
            "MEDIUM": (0.15, 0.25),
            "HIGH": (0.30, 1.00),
        }
    return {
        "LOW": (0.10, 0.15),
        "MEDIUM": (0.20, 0.25),
        "HIGH": (0.35, 1.00),
    }


def anti_chase_pe_threshold() -> float:
    return settings.anti_chase_pe_threshold


def trade_friendly_base_fv_buffer_pct() -> float:
    if trade_friendly_mode():
        return settings.trade_friendly_base_fv_buffer_pct
    return 0.005


def business_context_blocks_preflight(*, financial_years: int | None) -> bool:
    if financial_years is None:
        return True
    if financial_years >= 5:
        return False
    return not trade_friendly_mode()
