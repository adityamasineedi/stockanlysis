"""Portfolio Pre-Screener / Candidate Selection Engine.

Reduces a 40+ stock universe to ~10–18 candidates for the expensive
individual deep-analysis pipeline. Does not emit BUY/WATCH/SKIP.
"""

from stockbot.portfolio_screener.eligibility import (
    EligibilityResult,
    check_deep_analysis_eligibility,
)
from stockbot.portfolio_screener.pipeline import (
    handoff_to_deep_analysis,
    run_prescreen,
    run_prescreen_then_analyze,
)
from stockbot.portfolio_screener.scoring_config import (
    DEFAULT_RUN_CONFIG,
    SCREENING_WEIGHTS,
    ScreenerRunConfig,
)

__all__ = [
    "DEFAULT_RUN_CONFIG",
    "SCREENING_WEIGHTS",
    "EligibilityResult",
    "ScreenerRunConfig",
    "check_deep_analysis_eligibility",
    "handoff_to_deep_analysis",
    "run_prescreen",
    "run_prescreen_then_analyze",
]
