"""Tests for health audit monitor."""

import sys
from datetime import UTC, datetime, timedelta

import pytest

from stockbot.costs import log_call
from stockbot.monitor.health_audit import Finding, HealthAuditReport, run_health_audit
from stockbot.monitor.health_audit_state import (
    FindingDiff,
    HealthAuditState,
    StoredFinding,
    clear_health_audit_state,
    diff_findings,
    load_health_audit_state,
    log_cutoff,
    save_health_audit_state,
)


def _patch_audit_paths(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state = logs / "health_audit_state.json"
    monkeypatch.setattr("stockbot.storage.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.costs.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.DB_PATH", db_path)
    monkeypatch.setattr("stockbot.monitor.health_audit.LOGS_DIR", logs)
    monkeypatch.setattr("stockbot.monitor.health_audit_state.LOGS_DIR", logs)
    monkeypatch.setattr("stockbot.monitor.health_audit_state.STATE_PATH", state)
    return logs, state


def test_run_health_audit_on_empty_db(tmp_path, monkeypatch):
    _patch_audit_paths(tmp_path, monkeypatch)
    report = run_health_audit(days=7)
    assert report.days == 7
    assert isinstance(report.findings, list)


def test_detects_large_stage1_input(tmp_path, monkeypatch):
    _patch_audit_paths(tmp_path, monkeypatch)

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
    _patch_audit_paths(tmp_path, monkeypatch)

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
    report_first = run_health_audit(days=1, persist=False)
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
    report_second = run_health_audit(days=1, persist=False)
    second_titles = [f.title for f in report_second.findings]
    assert any("cache" in t.lower() for t in second_titles)


def test_cache_write_warning_skips_when_prior_outside_1h(tmp_path, monkeypatch):
    _patch_audit_paths(tmp_path, monkeypatch)

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
    report = run_health_audit(days=1, persist=False)
    titles = [f.title for f in report.findings]
    assert not any("cache" in t.lower() for t in titles)


def test_lite_max_tokens_raised_above_truncation_floor():
    from stockbot.llm import verdict as verdict_mod

    assert verdict_mod.LITE_MAX_TOKENS >= 32_768
    assert verdict_mod.MAX_TOKENS >= 48_000
    assert verdict_mod.MAX_TOKENS_CAP > verdict_mod.MAX_TOKENS
    assert verdict_mod.stage2_max_tokens("LITE", 0) == verdict_mod.LITE_MAX_TOKENS
    assert verdict_mod.stage2_max_tokens("LITE", 1) == verdict_mod.LITE_MAX_TOKENS_CAP
    assert verdict_mod.stage2_max_tokens("LITE", 2) == verdict_mod.LITE_MAX_TOKENS_CAP
    assert verdict_mod.stage2_max_tokens("FULL", 0) == verdict_mod.MAX_TOKENS
    assert verdict_mod.stage2_max_tokens("FULL", 1) == verdict_mod.MAX_TOKENS_CAP


def test_to_telegram_html_escapes_and_summarizes():
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


def test_event_date_and_dated_helpers():
    from stockbot.monitor.health_audit import _dated, _event_date

    assert _event_date("2026-09-01T14:30:00+00:00") == "2026-09-01"
    assert _event_date(datetime(2026, 9, 1, 14, 30, tzinfo=UTC)) == "2026-09-01"
    assert _dated("HEROMOTOCO", "2026-09-01T10:00:00+00:00", "cost ₹60.00.") == (
        "HEROMOTOCO (2026-09-01): cost ₹60.00."
    )


def test_expensive_analysis_detail_includes_created_date(tmp_path, monkeypatch):
    """Telegram /health only shows detail — dates must not hide in evidence JSON."""
    _patch_audit_paths(tmp_path, monkeypatch)
    from stockbot.storage import save_analysis
    import stockbot.storage as storage_module

    monkeypatch.setattr(storage_module, "DB_PATH", tmp_path / "test.sqlite3")
    row_id = save_analysis(
        ticker="HEROMOTOCO",
        verdict_json={"verdict": "WATCH", "expected_return": {"base": 0.1}},
        report_md="# report",
        brief_text="brief",
        stage1_tokens=100,
        stage2_tokens=200,
        cost_inr=60.0,
        validation_passed=True,
    )
    stamped = "2026-09-01T08:15:00+00:00"
    with storage_module._connect() as conn:
        conn.execute("UPDATE analyses SET created_at = ? WHERE id = ?", (stamped, row_id))

    report = run_health_audit(days=14, persist=False)
    expensive = [f for f in report.findings if f.title == "Expensive analysis run"]
    assert expensive
    assert "2026-09-01" in expensive[0].detail
    assert expensive[0].evidence.get("created_at") == stamped
    assert "2026-09-01" in report.to_telegram_html()


def test_to_telegram_html_shows_clean_bill_of_health():
    report = HealthAuditReport(
        generated_at=datetime.now(UTC),
        days=7,
        findings=[],
        summary={"mtd_spend_inr": 10.0, "monthly_budget_inr": 1400.0},
    )
    assert "No critical or warning findings" in report.to_telegram_html()


def test_diff_marks_resolved_and_new():
    previous = HealthAuditState(
        findings=[
            StoredFinding("warning", "quality", "Log pattern: validation_failed", "old"),
            StoredFinding("warning", "token_waste", "Stage 1 input high", "still"),
        ]
    )
    current = {"token_waste|Stage 1 input high", "cost_leak|New orphan"}
    result = diff_findings(previous, current)
    assert any(r.title == "Log pattern: validation_failed" for r in result.resolved)
    assert "cost_leak|New orphan" in result.new_keys
    assert "token_waste|Stage 1 input high" in result.open_keys


def test_log_cutoff_prefers_ignore_baseline():
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    state = HealthAuditState(
        ignore_log_before=datetime(2026, 9, 1, 0, 0, tzinfo=UTC),
        last_green_at=datetime(2026, 8, 20, 0, 0, tzinfo=UTC),
    )
    cutoff = log_cutoff(14, state, now=now)
    assert cutoff == state.ignore_log_before


def test_persist_resolves_when_finding_gone(tmp_path, monkeypatch):
    _logs, state_path = _patch_audit_paths(tmp_path, monkeypatch)
    save_health_audit_state(
        HealthAuditState(
            updated_at=datetime.now(UTC),
            findings=[
                StoredFinding(
                    "warning",
                    "quality",
                    "Log pattern: validation_failed",
                    "3 matching lines",
                )
            ],
        ),
        path=state_path,
    )
    report = run_health_audit(days=7, persist=True)
    assert report.diff is not None
    assert any("validation_failed" in r.title for r in report.diff.resolved)
    saved = load_health_audit_state(state_path)
    assert saved.findings == []


def test_clear_health_audit_prunes_reports(tmp_path, monkeypatch):
    logs, state_path = _patch_audit_paths(tmp_path, monkeypatch)
    stale = logs / "health_audit_2026-08-01.md"
    stale.write_text("# old\n", encoding="utf-8")
    result = clear_health_audit_state(path=state_path, prune_reports=True)
    assert result["pruned_reports"] == 1
    assert not stale.exists()
    state = load_health_audit_state(state_path)
    assert state.ignore_log_before is not None
    assert state.findings == []


def test_log_pattern_ignored_before_baseline(tmp_path, monkeypatch):
    logs, state_path = _patch_audit_paths(tmp_path, monkeypatch)
    now = datetime.now(UTC)
    save_health_audit_state(
        HealthAuditState(ignore_log_before=now, last_green_at=now),
        path=state_path,
    )
    old_ts = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    (logs / "stockbot.log").write_text(
        f"{old_ts} WARNING stockbot.pipeline: Stage 2 validation failed (attempt 1)\n",
        encoding="utf-8",
    )
    report = run_health_audit(days=14, persist=False)
    titles = [f.title for f in report.findings]
    assert not any("validation_failed" in t for t in titles)


def test_cli_exits_zero_when_no_critical(tmp_path, monkeypatch, capsys):
    _patch_audit_paths(tmp_path, monkeypatch)

    from stockbot.monitor.cli import main

    monkeypatch.setattr(sys, "argv", ["stockbot-monitor", "--fail-on", "none", "--no-persist"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_telegram_shows_resolved_section():
    report = HealthAuditReport(
        generated_at=datetime.now(UTC),
        days=7,
        findings=[],
        summary={},
        diff=FindingDiff(
            new_keys=frozenset(),
            open_keys=frozenset(),
            resolved=(
                StoredFinding("warning", "quality", "Log pattern: truncated", "gone"),
            ),
        ),
    )
    text = report.to_telegram_html()
    assert "Resolved since last audit" in text
    assert "prior issues cleared" in text


def test_verify_and_clear_refuses_when_warnings(tmp_path, monkeypatch):
    from stockbot.monitor.health_audit import verify_and_clear_health_audit

    logs, state_path = _patch_audit_paths(tmp_path, monkeypatch)
    now = datetime.now(UTC)
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    (logs / "stockbot.log").write_text(
        f"{ts} WARNING stockbot.pipeline: Stage 2 validation failed (attempt 1)\n",
        encoding="utf-8",
    )
    outcome = verify_and_clear_health_audit(days=14, fail_on="warning")
    assert outcome.cleared is False
    assert "NOT cleared" in outcome.reason
    state = load_health_audit_state(state_path)
    # Persist still recorded the open finding; ignore_log_before not force-cleared via clear()
    assert any("validation_failed" in f.title for f in (state.findings or []))


def test_verify_and_clear_clears_when_clean(tmp_path, monkeypatch):
    from stockbot.monitor.health_audit import verify_and_clear_health_audit

    logs, state_path = _patch_audit_paths(tmp_path, monkeypatch)
    stale = logs / "health_audit_2026-08-01.md"
    stale.write_text("# old\n", encoding="utf-8")
    outcome = verify_and_clear_health_audit(days=7, fail_on="warning")
    assert outcome.cleared is True
    assert not stale.exists()
    state = load_health_audit_state(state_path)
    assert state.findings == []
    assert state.ignore_log_before is not None


def test_cli_clear_without_force_exits_2(tmp_path, monkeypatch, capsys):
    _patch_audit_paths(tmp_path, monkeypatch)
    from stockbot.monitor.cli import main

    monkeypatch.setattr(sys, "argv", ["stockbot-monitor", "--clear"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "without verification" in err
