"""Stage 2 cost vs quality A/B benchmark — Sonnet vs Haiku vs DeepSeek.

Runs the same Sonnet Stage 1 extraction, then each candidate model on the
master-prompt FULL path. Validates outputs and scores against success gates
before any production model switch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from stockbot.ab_test import DEEPSEEK_MODEL
from stockbot.data_readiness import assemble_brief_for_analysis
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.llm.extract import run_stage1
from stockbot.llm.fixtures import FIXTURES_DIR, load_response_fixture
from stockbot.llm.stage2_deepseek import run_stage2_deepseek
from stockbot.llm.verdict import (
    LITE_MODEL,
    MODEL,
    TruncatedResponseError,
    VerdictParseError,
    run_stage2,
)
from stockbot.models import AmbiguousMatch, Brief, TickerInfo
from stockbot.validate import validate_report

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_TICKERS = ("GESHIP", "ADVENZYMES", "WAAREEENER")

Provider = Literal["anthropic", "deepseek"]

# Success gates from the A/B plan.
HAIKU_COST_GATE_INR = 25.0
DEEPSEEK_COST_GATE_INR = 20.0
SONNET_PASS_RATE_FLOOR_DELTA = 0.10
MIN_CHALLENGER_PASS_RATE = 2 / 3  # at least 2 of 3 tickers must validate


@dataclass(frozen=True)
class Stage2ModelSpec:
    label: str
    provider: Provider
    model: str
    enable_thinking: bool | None  # None = default adaptive on Sonnet FULL


BENCHMARK_MODELS: tuple[Stage2ModelSpec, ...] = (
    Stage2ModelSpec("sonnet-full", "anthropic", MODEL, None),
    Stage2ModelSpec("haiku-full", "anthropic", LITE_MODEL, False),
    Stage2ModelSpec("deepseek-full", "deepseek", DEEPSEEK_MODEL, False),
)


@dataclass
class Stage2BenchmarkCell:
    ticker: str
    model_label: str
    provider: str
    model: str
    cost_inr: float = 0.0
    validation_passed: bool = False
    validation_failures: list[str] = field(default_factory=list)
    truncated: bool = False
    parse_error: str | None = None
    report_chars: int = 0
    thinking_ratio: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class Stage2BenchmarkReport:
    generated_at: datetime
    tickers: tuple[str, ...]
    cells: list[Stage2BenchmarkCell]
    stage1_costs_inr: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    gates_passed: dict[str, bool] = field(default_factory=dict)

    @property
    def total_cost_inr(self) -> float:
        stage1 = sum(self.stage1_costs_inr.values())
        stage2 = sum(c.cost_inr for c in self.cells)
        return stage1 + stage2


def _resolve_ticker(query: str) -> TickerInfo:
    resolved = resolve_ticker(query, load_symbol_table())
    if resolved is None:
        raise ValueError(f"Ticker not found: {query!r}")
    if isinstance(resolved, AmbiguousMatch):
        raise ValueError(f"Ambiguous ticker {query!r}: {resolved.candidates}")  # noqa: TRY004
    return resolved


def _prepare_brief_and_extraction(ticker: TickerInfo) -> tuple[Brief, object, float]:
    brief, readiness = assemble_brief_for_analysis(ticker)
    if not readiness.ready_for_llm:
        raise RuntimeError(
            f"{ticker.symbol} not ready for LLM: {'; '.join(readiness.blockers)}"
        )
    extraction, stage1_usage = run_stage1(brief)
    return brief, extraction, float(stage1_usage["cost_inr"])


def _thinking_ratio(output_tokens: int, thinking_tokens: int) -> float | None:
    if output_tokens <= 0:
        return None
    return thinking_tokens / output_tokens


def run_stage2_cell(
    brief: Brief,
    extraction: object,
    spec: Stage2ModelSpec,
) -> Stage2BenchmarkCell:
    cell = Stage2BenchmarkCell(
        ticker=brief.ticker.symbol,
        model_label=spec.label,
        provider=spec.provider,
        model=spec.model,
    )
    try:
        if spec.provider == "deepseek":
            report_text, _verdict, usage = run_stage2_deepseek(
                brief, extraction, model=spec.model
            )
        else:
            report_text, _verdict, usage = run_stage2(
                brief,
                extraction,
                model=spec.model,
                mode="FULL",
                enable_thinking=spec.enable_thinking,
            )
        cell.cost_inr = float(usage["cost_inr"])
        cell.input_tokens = int(usage.get("input_tokens", 0))
        cell.output_tokens = int(usage.get("output_tokens", 0))
        cell.report_chars = len(report_text)
        think = int(usage.get("thinking_tokens", 0) or 0)
        cell.thinking_ratio = _thinking_ratio(cell.output_tokens, think)

        validation = validate_report(report_text, brief, stage2_mode="FULL")
        cell.validation_passed = validation.passed
        cell.validation_failures = list(validation.failures)
    except TruncatedResponseError as exc:
        cell.truncated = True
        cell.cost_inr = float(exc.cost_inr)
        cell.error = str(exc)
    except VerdictParseError as exc:
        cell.parse_error = str(exc)
        cell.error = str(exc)
    except Exception as exc:
        cell.error = str(exc)
        logger.exception(
            "Stage 2 benchmark failed for %s / %s", brief.ticker.symbol, spec.label
        )
    return cell


def _model_spec_for_fixture(stage: str, model: str) -> Stage2ModelSpec | None:
    if stage == "stage2_deepseek":
        return next((m for m in BENCHMARK_MODELS if m.label == "deepseek-full"), None)
    if stage == "stage2" and "haiku" in model:
        return next((m for m in BENCHMARK_MODELS if m.label == "haiku-full"), None)
    if stage == "stage2":
        return next((m for m in BENCHMARK_MODELS if m.label == "sonnet-full"), None)
    return None


def _validate_fixture_cell(
    brief: Brief,
    fixture_path: Path,
    spec: Stage2ModelSpec,
) -> Stage2BenchmarkCell:
    data = load_response_fixture(fixture_path)
    report_text = str(data.get("report_text") or "")
    usage = data.get("usage") or {}
    stop_reason = str(data.get("stop_reason") or "")
    cell = Stage2BenchmarkCell(
        ticker=brief.ticker.symbol,
        model_label=spec.label,
        provider=spec.provider,
        model=spec.model,
        report_chars=len(report_text),
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
    )
    if stop_reason in {"max_tokens", "length"}:
        cell.truncated = True
        cell.error = f"truncated at {stop_reason}"
        return cell
    try:
        validation = validate_report(report_text, brief, stage2_mode="FULL")
        cell.validation_passed = validation.passed
        cell.validation_failures = list(validation.failures)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        cell.parse_error = str(exc)
        cell.error = str(exc)
    return cell


def _load_stage1_costs_from_db(
    tickers: tuple[str, ...],
    *,
    date_prefix: str,
) -> dict[str, float]:
    import sqlite3

    from stockbot.config import DB_PATH

    costs: dict[str, float] = {}
    since = f"{date_prefix[:4]}-{date_prefix[4:6]}-{date_prefix[6:8]}"
    with sqlite3.connect(DB_PATH) as conn:
        for ticker in tickers:
            row = conn.execute(
                """
                SELECT cost_inr FROM llm_calls
                WHERE ticker = ? AND stage = 'stage1'
                  AND called_at >= ?
                ORDER BY called_at DESC LIMIT 1
                """,
                (ticker, since),
            ).fetchone()
            if row:
                costs[ticker] = float(row[0])
    return costs


def _load_cell_cost_from_db(ticker: str, stage: str, *, date_prefix: str) -> float:
    import sqlite3

    from stockbot.config import DB_PATH

    since = f"{date_prefix[:4]}-{date_prefix[4:6]}-{date_prefix[6:8]}"
    stage_filter = "stage2_deepseek" if stage == "stage2_deepseek" else "stage2"
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT cost_inr, model, output_tokens, thinking_tokens
            FROM llm_calls
            WHERE ticker = ? AND stage = ? AND called_at >= ?
            ORDER BY called_at DESC
            """,
            (ticker, stage_filter, since),
        ).fetchall()
    return rows


def collect_cells_from_fixtures(
    tickers: tuple[str, ...],
    *,
    fixture_dir: Path = FIXTURES_DIR,
    date_prefix: str | None = None,
) -> tuple[list[Stage2BenchmarkCell], dict[str, float]]:
    """Replay validation on saved Stage 2 fixtures (zero new LLM spend)."""
    date_prefix = date_prefix or datetime.now(UTC).strftime("%Y%m%d")
    cells: list[Stage2BenchmarkCell] = []
    stage1_costs = _load_stage1_costs_from_db(tickers, date_prefix=date_prefix)

    for ticker_query in tickers:
        ticker = _resolve_ticker(ticker_query)
        brief, readiness = assemble_brief_for_analysis(ticker)
        if not readiness.ready_for_llm:
            logger.warning("%s not ready — skipping fixture replay", ticker.symbol)
            continue

        pattern = f"*_{ticker.symbol}_{date_prefix}*"
        paths = sorted(fixture_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

        deepseek_paths = [p for p in paths if "stage2_deepseek" in p.name]
        anthropic_paths = [
            p for p in paths if p.name.startswith("stage2_") and "deepseek" not in p.name
        ]

        if len(anthropic_paths) >= 2:
            sized = sorted(
                anthropic_paths,
                key=lambda p: len(load_response_fixture(p).get("report_text") or ""),
                reverse=True,
            )
            assignments = [
                (sized[0], next(m for m in BENCHMARK_MODELS if m.label == "sonnet-full")),
                (sized[1], next(m for m in BENCHMARK_MODELS if m.label == "haiku-full")),
            ]
        else:
            assignments = [
                (p, next(m for m in BENCHMARK_MODELS if m.label == "sonnet-full"))
                for p in anthropic_paths[:1]
            ]

        for path, spec in assignments:
            cell = _validate_fixture_cell(brief, path, spec)
            db_rows = _load_cell_cost_from_db(
                ticker.symbol, "stage2", date_prefix=date_prefix
            )
            for cost, model, out, think in db_rows:
                if spec.model in str(model):
                    cell.cost_inr = float(cost)
                    cell.output_tokens = int(out or 0)
                    cell.thinking_ratio = _thinking_ratio(int(out or 0), int(think or 0))
                    break
            cells.append(cell)

        for path in deepseek_paths[-1:]:
            spec = next(m for m in BENCHMARK_MODELS if m.label == "deepseek-full")
            cell = _validate_fixture_cell(brief, path, spec)
            db_rows = _load_cell_cost_from_db(
                ticker.symbol, "stage2_deepseek", date_prefix=date_prefix
            )
            if db_rows:
                cell.cost_inr = float(db_rows[0][0])
                cell.output_tokens = int(db_rows[0][2] or 0)
            cells.append(cell)

    return cells, stage1_costs


def load_report_from_json(path: Path) -> Stage2BenchmarkReport:
    """Load a prior benchmark run for merging with new cells."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        Stage2BenchmarkCell(
            ticker=str(c["ticker"]),
            model_label=str(c["model_label"]),
            provider=str(c["provider"]),
            model=str(c["model"]),
            cost_inr=float(c.get("cost_inr", 0)),
            validation_passed=bool(c.get("validation_passed")),
            validation_failures=list(c.get("validation_failures") or []),
            truncated=bool(c.get("truncated")),
            parse_error=c.get("parse_error"),
            report_chars=int(c.get("report_chars", 0)),
            thinking_ratio=c.get("thinking_ratio"),
            input_tokens=int(c.get("input_tokens", 0)),
            output_tokens=int(c.get("output_tokens", 0)),
            error=c.get("error"),
        )
        for c in data.get("cells") or []
    ]
    report = Stage2BenchmarkReport(
        generated_at=datetime.fromisoformat(str(data["generated_at"])),
        tickers=tuple(data.get("tickers") or []),
        cells=cells,
        stage1_costs_inr={
            str(k): float(v) for k, v in (data.get("stage1_costs_inr") or {}).items()
        },
    )
    report.gates_passed, report.recommendation = evaluate_gates(report)
    return report


def merge_benchmark_reports(*reports: Stage2BenchmarkReport) -> Stage2BenchmarkReport:
    cells_by_key: dict[tuple[str, str], Stage2BenchmarkCell] = {}
    stage1: dict[str, float] = {}
    tickers: list[str] = []
    for report in reports:
        tickers.extend(report.tickers)
        stage1.update(report.stage1_costs_inr)
        for cell in report.cells:
            key = (cell.ticker, cell.model_label)
            existing = cells_by_key.get(key)
            if existing is None or (existing.error and not cell.error):
                cells_by_key[key] = cell
    cells = list(cells_by_key.values())
    unique_tickers = tuple(dict.fromkeys(tickers))
    merged = Stage2BenchmarkReport(
        generated_at=datetime.now(UTC),
        tickers=unique_tickers,
        cells=cells,
        stage1_costs_inr=stage1,
    )
    merged.gates_passed, merged.recommendation = evaluate_gates(merged)
    return merged


def run_benchmark_matrix(
    tickers: tuple[str, ...] = DEFAULT_BENCHMARK_TICKERS,
    models: tuple[Stage2ModelSpec, ...] = BENCHMARK_MODELS,
    *,
    skip_deepseek_without_key: bool = True,
    model_labels: tuple[str, ...] | None = None,
) -> Stage2BenchmarkReport:
    from stockbot.config import settings

    active_models = list(models)
    if model_labels:
        allowed = set(model_labels)
        active_models = [m for m in active_models if m.label in allowed]
    if skip_deepseek_without_key and not settings.deepseek_api_key:
        active_models = [m for m in active_models if m.provider != "deepseek"]
        logger.warning("DEEPSEEK_API_KEY unset — skipping deepseek-full cells")

    cells: list[Stage2BenchmarkCell] = []
    stage1_costs: dict[str, float] = {}

    for ticker_query in tickers:
        ticker = _resolve_ticker(ticker_query)
        logger.info("Benchmark Stage 1 (Sonnet) for %s…", ticker.symbol)
        try:
            brief, extraction, stage1_cost = _prepare_brief_and_extraction(ticker)
        except Exception as exc:
            logger.exception("Stage 1 failed for %s — skipping ticker", ticker.symbol)
            for spec in active_models:
                cells.append(
                    Stage2BenchmarkCell(
                        ticker=ticker.symbol,
                        model_label=spec.label,
                        provider=spec.provider,
                        model=spec.model,
                        error=f"Stage 1 failed: {exc}",
                    )
                )
            continue
        stage1_costs[ticker.symbol] = stage1_cost

        for spec in active_models:
            logger.info("Benchmark Stage 2 %s for %s…", spec.label, ticker.symbol)
            cells.append(run_stage2_cell(brief, extraction, spec))

    report = Stage2BenchmarkReport(
        generated_at=datetime.now(UTC),
        tickers=tickers,
        cells=cells,
        stage1_costs_inr=stage1_costs,
    )
    report.gates_passed, report.recommendation = evaluate_gates(report)
    return report


def evaluate_gates(report: Stage2BenchmarkReport) -> tuple[dict[str, bool], str]:
    """Score each challenger model against Sonnet baseline."""
    by_label: dict[str, list[Stage2BenchmarkCell]] = {}
    for cell in report.cells:
        by_label.setdefault(cell.model_label, []).append(cell)

    sonnet_cells = by_label.get("sonnet-full", [])
    sonnet_pass_rate = _pass_rate(sonnet_cells)
    gates: dict[str, bool] = {}

    for label in ("haiku-full", "deepseek-full"):
        cells = by_label.get(label, [])
        if not cells:
            gates[label] = False
            continue
        pass_rate = _pass_rate(cells)
        passing_costs = [c.cost_inr for c in cells if c.validation_passed and c.cost_inr > 0]
        median_cost = _median(passing_costs) if passing_costs else float("inf")
        truncations = sum(1 for c in cells if c.truncated)
        parse_failures = sum(1 for c in cells if c.parse_error)
        cost_gate = HAIKU_COST_GATE_INR if label == "haiku-full" else DEEPSEEK_COST_GATE_INR

        required_pass_rate = max(
            0.0,
            sonnet_pass_rate - SONNET_PASS_RATE_FLOOR_DELTA,
            MIN_CHALLENGER_PASS_RATE if len(cells) >= 3 else 0.0,
        )

        gates[label] = (
            pass_rate >= required_pass_rate
            and parse_failures == 0
            and truncations <= 1
            and bool(passing_costs)
            and median_cost <= cost_gate
        )

    recommendation = _build_recommendation(report, gates, sonnet_pass_rate)
    return gates, recommendation


def _pass_rate(cells: list[Stage2BenchmarkCell]) -> float:
    if not cells:
        return 0.0
    return sum(1 for c in cells if c.validation_passed) / len(cells)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _build_recommendation(
    report: Stage2BenchmarkReport,
    gates: dict[str, bool],
    sonnet_pass_rate: float,
) -> str:
    lines = [
        (
            f"Sonnet baseline validation pass rate: {sonnet_pass_rate:.0%} "
            f"({sum(1 for c in report.cells if c.model_label == 'sonnet-full' and c.validation_passed)}"
            f"/{sum(1 for c in report.cells if c.model_label == 'sonnet-full')})."
        ),
        f"Total benchmark spend: ₹{report.total_cost_inr:.2f}.",
        "",
    ]
    any_pass = False
    for label, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- **{label}**: gates {status}")
        if passed:
            any_pass = True
            spec = next(m for m in BENCHMARK_MODELS if m.label == label)
            label_cells = [c for c in report.cells if c.model_label == label]
            rate = _pass_rate(label_cells)
            lines.append(
                f"  → Set `STAGE2_FULL_MODEL={spec.model}` "
                f"and `STAGE2_FULL_THINKING=false` after manual review "
                f"({rate:.0%} validation pass on benchmark set)."
            )
    if not any_pass:
        lines.extend(
            [
                "",
                (
                    "No challenger passed all gates. **Keep Sonnet FULL** for production; "
                    "consider expanding LITE routing for clean AUTO_DEEP names instead."
                ),
            ]
        )
    return "\n".join(lines)


def format_markdown(report: Stage2BenchmarkReport) -> str:
    lines = [
        f"# Stage 2 A/B benchmark ({report.generated_at.date().isoformat()})",
        "",
        f"Tickers: {', '.join(report.tickers)}",
        (
            f"Total spend: **₹{report.total_cost_inr:.2f}** "
            f"(Stage 1 Sonnet: ₹{sum(report.stage1_costs_inr.values()):.2f}, "
            f"Stage 2: ₹{sum(c.cost_inr for c in report.cells):.2f})"
        ),
        "",
        "## Results",
        "",
        "| Ticker | Model | Cost ₹ | Valid | Trunc | Chars | Think% | Error |",
        "|--------|-------|--------|-------|-------|-------|--------|-------|",
    ]
    for cell in report.cells:
        think = f"{cell.thinking_ratio:.0%}" if cell.thinking_ratio is not None else "—"
        err = (cell.error or cell.parse_error or "")[:60]
        lines.append(
            f"| {cell.ticker} | {cell.model_label} | {cell.cost_inr:.2f} | "
            f"{'✓' if cell.validation_passed else '✗'} | "
            f"{'Y' if cell.truncated else '—'} | {cell.report_chars} | {think} | {err} |"
        )

    if report.cells:
        lines.extend(["", "## Validation failures (detail)", ""])
        for cell in report.cells:
            if not cell.validation_failures:
                continue
            lines.append(f"### {cell.ticker} / {cell.model_label}")
            for failure in cell.validation_failures[:8]:
                lines.append(f"- {failure}")
            lines.append("")

    lines.extend(["## Gate evaluation", ""])
    for label, passed in report.gates_passed.items():
        lines.append(f"- **{label}**: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Recommendation", "", report.recommendation, ""])
    return "\n".join(lines)


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    from stockbot.config import setup_logging

    sys.stdout.reconfigure(encoding="utf-8")
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Stage 2 cost/quality A/B — Sonnet vs Haiku vs DeepSeek on FULL master prompt."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_BENCHMARK_TICKERS),
        help="Comma-separated NSE symbols (default: GESHIP,ADVENZYMES,WAAREEENER)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/stage2_benchmark_latest.md"),
        help="Write markdown report here",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON dump of all cells",
    )
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Replay validation from saved fixtures only (no new LLM calls)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model labels: sonnet-full,haiku-full,deepseek-full",
    )
    parser.add_argument(
        "--date-prefix",
        default=None,
        help="Fixture filename date prefix YYYYMMDD (default: today UTC)",
    )
    parser.add_argument(
        "--merge-json",
        type=Path,
        action="append",
        default=None,
        help="Merge cells from prior JSON report(s) with this run (dedupe by ticker+model)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge --merge-json files and write report (no LLM, no brief fetch)",
    )
    args = parser.parse_args()

    tickers = tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip())
    model_labels = (
        tuple(m.strip() for m in args.models.split(",") if m.strip())
        if args.models
        else None
    )
    date_prefix = args.date_prefix or datetime.now(UTC).strftime("%Y%m%d")

    reports: list[Stage2BenchmarkReport] = []
    if args.merge_json:
        for path in args.merge_json:
            reports.append(load_report_from_json(path))

    if args.merge_only:
        if not reports:
            parser.error("--merge-only requires at least one --merge-json file")
    elif args.fixtures_only:
        cells, stage1 = collect_cells_from_fixtures(tickers, date_prefix=date_prefix)
        reports.append(
            Stage2BenchmarkReport(
                generated_at=datetime.now(UTC),
                tickers=tickers,
                cells=cells,
                stage1_costs_inr=stage1,
            )
        )
    else:
        reports.append(
            run_benchmark_matrix(
                tickers=tickers,
                model_labels=model_labels,
            )
        )

    report = merge_benchmark_reports(*reports)
    md = format_markdown(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.out}")

    if args.json:
        payload = {
            "generated_at": report.generated_at.isoformat(),
            "tickers": list(report.tickers),
            "total_cost_inr": report.total_cost_inr,
            "stage1_costs_inr": report.stage1_costs_inr,
            "gates_passed": report.gates_passed,
            "recommendation": report.recommendation,
            "cells": [
                {
                    "ticker": c.ticker,
                    "model_label": c.model_label,
                    "provider": c.provider,
                    "model": c.model,
                    "cost_inr": c.cost_inr,
                    "validation_passed": c.validation_passed,
                    "validation_failures": c.validation_failures,
                    "truncated": c.truncated,
                    "parse_error": c.parse_error,
                    "report_chars": c.report_chars,
                    "thinking_ratio": c.thinking_ratio,
                    "error": c.error,
                }
                for c in report.cells
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
