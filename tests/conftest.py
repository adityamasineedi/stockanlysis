"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _strict_trade_policy_by_default(monkeypatch):
    """Keep legacy strict-gate expectations unless a test opts into trade-friendly."""
    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", False)
