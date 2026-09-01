"""Unit tests for Stage 2 A/B benchmark (no live LLM)."""

from __future__ import annotations

import pytest

from stockbot.stage2_benchmark import (
    Stage2BenchmarkCell,
    Stage2BenchmarkReport,
    evaluate_gates,
    format_markdown,
)


def _cell(
    ticker: str,
    label: str,
    *,
    passed: bool = False,
    cost: float = 10.0,
    truncated: bool = False,
    parse_error: str | None = None,
) -> Stage2BenchmarkCell:
    return Stage2BenchmarkCell(
        ticker=ticker,
        model_label=label,
        provider="anthropic" if label != "deepseek-full" else "deepseek",
        model="test-model",
        cost_inr=cost,
        validation_passed=passed,
        validation_failures=[] if passed else ["missing section"],
        truncated=truncated,
        parse_error=parse_error,
        report_chars=5000 if passed else 1000,
    )


def test_evaluate_gates_haiku_passes_when_sonnet_baseline_and_cost_ok():
    cells = [
        _cell("GESHIP", "sonnet-full", passed=True, cost=40.0),
        _cell("ADVENZYMES", "sonnet-full", passed=True, cost=35.0),
        _cell("WAAREEENER", "sonnet-full", passed=False, cost=45.0),
        _cell("GESHIP", "haiku-full", passed=True, cost=18.0),
        _cell("ADVENZYMES", "haiku-full", passed=True, cost=20.0),
        _cell("WAAREEENER", "haiku-full", passed=True, cost=22.0),
    ]
    report = Stage2BenchmarkReport(
        generated_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        tickers=("GESHIP", "ADVENZYMES", "WAAREEENER"),
        cells=cells,
        stage1_costs_inr={"GESHIP": 5.0, "ADVENZYMES": 5.0, "WAAREEENER": 5.0},
    )
    gates, recommendation = evaluate_gates(report)
    assert gates["haiku-full"] is True
    assert "STAGE2_FULL_MODEL" in recommendation


def test_evaluate_gates_fails_when_no_passing_reports():
    cells = [
        _cell("GESHIP", "sonnet-full", passed=False, cost=40.0),
        _cell("ADVENZYMES", "sonnet-full", passed=False, cost=35.0),
        _cell("GESHIP", "deepseek-full", passed=False, cost=8.0),
        _cell("ADVENZYMES", "deepseek-full", passed=False, cost=9.0),
    ]
    report = Stage2BenchmarkReport(
        generated_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        tickers=("GESHIP", "ADVENZYMES"),
        cells=cells,
    )
    gates, recommendation = evaluate_gates(report)
    assert gates.get("deepseek-full") is False
    assert "Keep Sonnet FULL" in recommendation


def test_evaluate_gates_fails_on_truncation():
    cells = [
        _cell("GESHIP", "sonnet-full", passed=True, cost=40.0),
        _cell("GESHIP", "deepseek-full", passed=False, cost=8.0, truncated=True),
        _cell("ADVENZYMES", "deepseek-full", passed=True, cost=9.0, truncated=True),
    ]
    report = Stage2BenchmarkReport(
        generated_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        tickers=("GESHIP", "ADVENZYMES"),
        cells=cells,
    )
    gates, recommendation = evaluate_gates(report)
    assert gates.get("deepseek-full") is False
    assert "Keep Sonnet FULL" in recommendation


def test_format_markdown_includes_table():
    report = Stage2BenchmarkReport(
        generated_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        tickers=("GESHIP",),
        cells=[_cell("GESHIP", "sonnet-full", passed=True, cost=30.0)],
        recommendation="test",
        gates_passed={"haiku-full": False},
    )
    md = format_markdown(report)
    assert "Stage 2 A/B benchmark" in md
    assert "GESHIP" in md
    assert "sonnet-full" in md
