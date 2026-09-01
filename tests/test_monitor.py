"""Tests for health audit monitor."""

import sys
from datetime import UTC, datetime

import pytest

from stockbot.costs import log_call
from stockbot.monitor.health_audit import run_health_audit


def test_run_health_audit_on_empty_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setattr("stockbot.storage.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.costs.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.LOGS_DIR", tmp_path / "logs")

    report = run_health_audit(days=7)
    assert report.days == 7
    assert isinstance(report.findings, list)


def test_detects_large_stage1_input(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setattr("stockbot.storage.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.costs.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.LOGS_DIR", tmp_path / "logs")

    log_call(
        "claude-sonnet-5",
        input_tokens=85_000,
        output_tokens=3_000,
        stage="stage1",
        ticker="MAZDOCK",
        called_at=datetime.now(UTC),
    )

    report = run_health_audit(days=1)
    titles = [f.title for f in report.findings]
    assert any("Stage 1 input" in t for t in titles)


def test_cache_write_warning_only_on_second_stage2_call(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setattr("stockbot.storage.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.costs.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.LOGS_DIR", tmp_path / "logs")

    now = datetime.now(UTC)
    log_call(
        "claude-sonnet-5",
        input_tokens=10_000,
        output_tokens=8_000,
        cache_creation_tokens=19_000,
        cached_tokens=0,
        stage="stage2",
        ticker="TORNTPOWER",
        called_at=now,
    )
    report_first = run_health_audit(days=1)
    first_titles = [f.title for f in report_first.findings]
    assert not any("cache" in t.lower() for t in first_titles)

    log_call(
        "claude-sonnet-5",
        input_tokens=10_000,
        output_tokens=8_000,
        cache_creation_tokens=19_000,
        cached_tokens=0,
        stage="stage2",
        ticker="TORNTPOWER",
        called_at=now,
    )
    report_second = run_health_audit(days=1)
    second_titles = [f.title for f in report_second.findings]
    assert any("cache" in t.lower() for t in second_titles)


def test_cache_write_warning_skips_when_prior_outside_1h(tmp_path, monkeypatch):
    from datetime import timedelta

    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setattr("stockbot.storage.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.costs.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.LOGS_DIR", tmp_path / "logs")

    first = datetime.now(UTC) - timedelta(hours=2)
    second = datetime.now(UTC)
    log_call(
        "claude-sonnet-5",
        input_tokens=10_000,
        output_tokens=8_000,
        cache_creation_tokens=19_000,
        cached_tokens=0,
        stage="stage2",
        ticker="HEROMOTOCO",
        called_at=first,
    )
    log_call(
        "claude-sonnet-5",
        input_tokens=10_000,
        output_tokens=8_000,
        cache_creation_tokens=19_000,
        cached_tokens=0,
        stage="stage2",
        ticker="HEROMOTOCO",
        called_at=second,
    )
    report = run_health_audit(days=1)
    titles = [f.title for f in report.findings]
    assert not any("cache" in t.lower() for t in titles)


def test_lite_max_tokens_raised_above_truncation_floor():
    from stockbot.llm import verdict as verdict_mod

    assert verdict_mod.LITE_MAX_TOKENS >= 16_384
    assert verdict_mod.MAX_TOKENS >= 48_000
    assert verdict_mod.MAX_TOKENS_CAP > verdict_mod.MAX_TOKENS
    assert verdict_mod.stage2_max_tokens("LITE", 0) == verdict_mod.LITE_MAX_TOKENS
    assert verdict_mod.stage2_max_tokens("LITE", 1) > verdict_mod.stage2_max_tokens("LITE", 0)
    assert verdict_mod.stage2_max_tokens("LITE", 2) == verdict_mod.LITE_MAX_TOKENS_CAP
    assert verdict_mod.stage2_max_tokens("FULL", 0) == verdict_mod.MAX_TOKENS
    assert verdict_mod.stage2_max_tokens("FULL", 1) == verdict_mod.MAX_TOKENS_CAP


def test_to_telegram_html_escapes_and_summarizes():
    from stockbot.monitor.health_audit import Finding, HealthAuditReport

    report = HealthAuditReport(
        generated_at=datetime.now(UTC),
        days=14,
        findings=[
            Finding(
                severity="critical",
                category="cost_leak",
                title="Near cap <script>",
                detail="MAZDOCK cost ₹77 & retries",
                evidence={"ticker": "MAZDOCK"},
            )
        ],
        summary={
            "mtd_spend_inr": 1047.77,
            "monthly_budget_inr": 1400.0,
            "llm_calls": 62,
            "analyses": 8,
        },
    )
    text = report.to_telegram_html()
    assert "<b>Health audit</b>" in text
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "₹1047.77" in text
    assert len(text) <= 4096


def test_to_telegram_html_shows_clean_bill_of_health():
    from stockbot.monitor.health_audit import HealthAuditReport

    report = HealthAuditReport(
        generated_at=datetime.now(UTC),
        days=7,
        findings=[],
        summary={"mtd_spend_inr": 10.0, "monthly_budget_inr": 1400.0},
    )
    assert "No critical or warning findings" in report.to_telegram_html()


def test_cli_exits_zero_when_no_critical(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setattr("stockbot.storage.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.costs.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.LOGS_DIR", tmp_path / "logs")

    from stockbot.monitor.cli import main

    monkeypatch.setattr(sys, "argv", ["stockbot-monitor", "--fail-on", "none"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
