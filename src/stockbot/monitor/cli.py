"""CLI for stockbot quality / cost health audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stockbot.config import setup_logging
from stockbot.monitor.health_audit import (
    clear_health_audit_state,
    run_health_audit,
    verify_and_clear_health_audit,
)


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
    parser.add_argument(
        "--verify-and-clear",
        action="store_true",
        help=(
            "Run audit first; clear log baseline only if verification is clean. "
            "Never clears when critical/warning findings remain."
        ),
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Unsafe: reset baseline without verifying (requires --force)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required with --clear to skip verification (not for deploy)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not update the finding ledger (read-only scan)",
    )
    args = parser.parse_args()
    setup_logging()

    if args.verify_and_clear and args.clear:
        print("Use either --verify-and-clear or --clear --force, not both.", file=sys.stderr)
        raise SystemExit(2)

    if args.clear:
        if not args.force:
            print(
                "Refusing --clear without verification.\n"
                "Use: stockbot-monitor --verify-and-clear\n"
                "Or emergency only: stockbot-monitor --clear --force",
                file=sys.stderr,
            )
            raise SystemExit(2)
        result = clear_health_audit_state()
        print(
            f"FORCED clear (no verification) at {result['ignore_log_before']}; "
            f"pruned {result['pruned_reports']} report(s)."
        )
        raise SystemExit(0)

    if args.verify_and_clear:
        # Deploy path: verify first; clear only when clean.
        # Default: warnings block clear. --fail-on critical allows warnings.
        fail_level: str = "warning"
        if args.fail_on == "critical":
            fail_level = "critical"
        outcome = verify_and_clear_health_audit(
            days=max(1, args.days),
            fail_on=fail_level,  # type: ignore[arg-type]
        )
        text = outcome.report.to_markdown()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(text)
        print(outcome.reason)
        if outcome.cleared and outcome.clear_meta:
            print(
                f"Cleared baseline at {outcome.clear_meta['ignore_log_before']}; "
                f"pruned {outcome.clear_meta['pruned_reports']} report(s)."
            )
        raise SystemExit(0 if outcome.cleared else 1)

    report = run_health_audit(days=max(1, args.days), persist=not args.no_persist)

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
