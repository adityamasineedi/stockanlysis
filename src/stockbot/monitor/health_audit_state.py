"""Persist last /health snapshot so resolved findings can be reported.

State lives under LOGS_DIR (same volume as stockbot.log / audit markdown).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stockbot.config import LOGS_DIR

logger = logging.getLogger(__name__)

STATE_PATH = LOGS_DIR / "health_audit_state.json"


@dataclass(frozen=True)
class StoredFinding:
    severity: str
    category: str
    title: str
    detail: str


@dataclass
class HealthAuditState:
    updated_at: datetime | None = None
    days: int = 14
    findings: list[StoredFinding] | None = None
    last_green_at: datetime | None = None
    ignore_log_before: datetime | None = None

    def finding_keys(self) -> set[str]:
        return {finding_key(f.category, f.title) for f in (self.findings or [])}


def finding_key(category: str, title: str) -> str:
    return f"{category.strip()}|{title.strip()}"


def _parse_dt(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def load_health_audit_state(path: Path | None = None) -> HealthAuditState:
    target = path or STATE_PATH
    if not target.exists():
        return HealthAuditState()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read health audit state from %s", target)
        return HealthAuditState()

    stored: list[StoredFinding] = []
    for item in payload.get("findings") or []:
        if not isinstance(item, dict):
            continue
        stored.append(
            StoredFinding(
                severity=str(item.get("severity") or "warning"),
                category=str(item.get("category") or ""),
                title=str(item.get("title") or ""),
                detail=str(item.get("detail") or ""),
            )
        )
    return HealthAuditState(
        updated_at=_parse_dt(payload.get("updated_at")),
        days=int(payload.get("days") or 14),
        findings=stored,
        last_green_at=_parse_dt(payload.get("last_green_at")),
        ignore_log_before=_parse_dt(payload.get("ignore_log_before")),
    )


def save_health_audit_state(state: HealthAuditState, path: Path | None = None) -> Path:
    target = path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "updated_at": (state.updated_at or datetime.now(UTC)).isoformat(),
        "days": state.days,
        "last_green_at": state.last_green_at.isoformat() if state.last_green_at else None,
        "ignore_log_before": (
            state.ignore_log_before.isoformat() if state.ignore_log_before else None
        ),
        "findings": [
            {
                "severity": f.severity,
                "category": f.category,
                "title": f.title,
                "detail": f.detail,
            }
            for f in (state.findings or [])
        ],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def clear_health_audit_state(
    *,
    path: Path | None = None,
    prune_reports: bool = True,
) -> dict[str, object]:
    """Baseline reset: stop counting old log lines; wipe open-finding ledger."""
    now = datetime.now(UTC)
    state = HealthAuditState(
        updated_at=now,
        findings=[],
        last_green_at=now,
        ignore_log_before=now,
    )
    save_health_audit_state(state, path=path)
    pruned = 0
    if prune_reports:
        logs = (path.parent if path is not None else LOGS_DIR)
        for report in logs.glob("health_audit_*.md"):
            try:
                report.unlink()
                pruned += 1
            except OSError:
                logger.warning("Could not delete %s", report)
    logger.info(
        "health_audit cleared ignore_log_before=%s pruned_reports=%d",
        now.isoformat(),
        pruned,
    )
    return {
        "ignore_log_before": now.isoformat(),
        "pruned_reports": pruned,
        "state_path": str(path or STATE_PATH),
    }


@dataclass(frozen=True)
class FindingDiff:
    new_keys: frozenset[str]
    open_keys: frozenset[str]
    resolved: tuple[StoredFinding, ...]


def diff_findings(
    previous: HealthAuditState,
    current_keys: set[str],
    *,
    current_by_key: dict[str, StoredFinding] | None = None,
) -> FindingDiff:
    prev_keys = previous.finding_keys()
    new_keys = frozenset(current_keys - prev_keys)
    open_keys = frozenset(current_keys & prev_keys)
    resolved_keys = prev_keys - current_keys
    resolved: list[StoredFinding] = []
    for item in previous.findings or []:
        key = finding_key(item.category, item.title)
        if key in resolved_keys:
            resolved.append(item)
    return FindingDiff(
        new_keys=new_keys,
        open_keys=open_keys,
        resolved=tuple(resolved),
    )


def log_cutoff(
    days: int,
    state: HealthAuditState,
    *,
    now: datetime | None = None,
) -> datetime:
    """Earliest timestamp to count log-pattern findings."""
    now = now or datetime.now(UTC)
    candidates = [now - timedelta(days=days)]
    if state.ignore_log_before is not None:
        candidates.append(state.ignore_log_before)
    if state.last_green_at is not None:
        candidates.append(state.last_green_at)
    return max(candidates)
