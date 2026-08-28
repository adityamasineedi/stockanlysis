"""CLI for stockbot quality / cost health audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stockbot.config import setup_logging
from stockbot.monitor.health_audit import run_health_audit


def main() -> None:
    # Windows consoles default to cp1252, which can't encode ₹
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Audit stockbot for cost leaks, token waste, and analysis quality issues."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Look back this many days (default: 14)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of markdown",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write report to this file (default: stdout only)",
    )
    parser.add_argument(
        "--fail-on",
        choices=("critical", "warning", "none"),
        default="critical",
        help="Exit code 1 if findings at/above this severity exist (default: critical)",
    )
    args = parser.parse_args()
    setup_logging()

    report = run_health_audit(days=max(1, args.days))

    if args.json:
        import json

        payload = {
            "generated_at": report.generated_at.isoformat(),
            "days": report.days,
            "summary": report.summary,
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "detail": f.detail,
                    "evidence": f.evidence,
                }
                for f in report.findings
            ],
        }
        text = json.dumps(payload, indent=2, default=str)
    else:
        text = report.to_markdown()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)

    if args.fail_on == "none":
        raise SystemExit(0)
    if args.fail_on == "critical" and report.critical_count:
        raise SystemExit(1)
    if args.fail_on == "warning" and (report.critical_count or report.warning_count):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
