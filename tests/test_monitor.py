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
