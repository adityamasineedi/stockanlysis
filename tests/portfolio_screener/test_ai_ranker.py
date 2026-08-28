"""Unit tests for pre-screener AI provider resolution."""

from __future__ import annotations

from stockbot.portfolio_screener.ai_ranker import _parse_ai_json, resolve_ai_ranker
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig


def test_auto_prefers_openai_when_key_present(monkeypatch):
    import stockbot.portfolio_screener.ai_ranker as mod

    monkeypatch.setattr(mod.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(mod.settings, "deepseek_api_key", "sk-ds")
    monkeypatch.setattr(mod.settings, "anthropic_api_key", "sk-ant")
    provider, model = resolve_ai_ranker(ScreenerRunConfig(ai_provider="auto"))
    assert provider == "openai"
    assert model == "gpt-4o-mini"


def test_auto_falls_back_to_deepseek(monkeypatch):
    import stockbot.portfolio_screener.ai_ranker as mod

    monkeypatch.setattr(mod.settings, "openai_api_key", "")
    monkeypatch.setattr(mod.settings, "deepseek_api_key", "sk-ds")
    monkeypatch.setattr(mod.settings, "anthropic_api_key", "sk-ant")
    provider, model = resolve_ai_ranker(ScreenerRunConfig(ai_provider="auto"))
    assert provider == "deepseek"
    assert model == "deepseek-v4-flash"


def test_parse_accepts_stocks_wrapper():
    raw = '{"stocks": [{"ticker": "TCS", "rank": 1, "ai_score": 80}]}'
    parsed = _parse_ai_json(raw)
    assert parsed[0]["ticker"] == "TCS"


def test_parse_accepts_bare_array():
    raw = '[{"ticker": "INFY", "rank": 1}]'
    parsed = _parse_ai_json(raw)
    assert parsed[0]["ticker"] == "INFY"
