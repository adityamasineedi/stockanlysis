"""Screener-specific cost / savings tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

from stockbot.portfolio_screener.models import CostSummary

# Typical full deep analysis ceiling used for savings estimates (pipeline cap).
DEFAULT_DEEP_ANALYSIS_COST_INR = 80.0


@dataclass
class ScreenerCostTracker:
    stocks_processed: int = 0
    ai_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_inr: float = 0.0
    universe_size: int = 0
    deep_analysis_unit_cost_inr: float = DEFAULT_DEEP_ANALYSIS_COST_INR
    _events: list[dict[str, float | int | str]] = field(default_factory=list)

    def record_stock_processed(self) -> None:
        self.stocks_processed += 1

    def record_ai_call(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_inr: float,
    ) -> None:
        self.ai_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.estimated_cost_inr += cost_inr
        self._events.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_inr": cost_inr,
            }
        )

    def summary(self, *, final_candidates: int) -> CostSummary:
        # Savings vs analysing the full universe with the expensive pipeline.
        full_universe_cost = self.universe_size * self.deep_analysis_unit_cost_inr
        shortlist_cost = final_candidates * self.deep_analysis_unit_cost_inr
        saved = max(0.0, full_universe_cost - shortlist_cost - self.estimated_cost_inr)
        return CostSummary(
            stocks_processed=self.stocks_processed,
            ai_calls=self.ai_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            estimated_cost_inr=round(self.estimated_cost_inr, 2),
            estimated_deep_analysis_cost_saved_inr=round(saved, 2),
        )
