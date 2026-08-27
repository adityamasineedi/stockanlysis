"""Stage 1 extraction recall benchmark — reusable for every future model
decision on this stage, not just tonight's Sonnet/Haiku/DeepSeek question.

Ground truth is hand-verified directly against the raw fetched annual
report text (not against any model's output) for VMM, BEL, and JYOTHYLAB —
see GROUND_TRUTH below for the exact keyword confirmation. IRCTC is
excluded from the scored benchmark: its annual report has never
successfully fetched (PDF download succeeds, pdfplumber parsing times out
past 120s), so there is no ground truth to score recall against — it's
still worth including as a "does the model correctly report MISSING
rather than hallucinate" check, handled separately.

Recall only, not precision: this measures whether each model's extraction
mentions the governance findings that are actually present in the source
text. It does not penalize a model for reporting an *additional* finding
that isn't in GROUND_TRUTH (that's not a false positive in the sense that
matters here — an extra real finding is a bonus, not a hallucination,
unless it's fabricated, which recall alone can't detect).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stockbot.ab_test import DEEPSEEK_MODEL, _call_anthropic, _call_deepseek
from stockbot.brief import assemble_brief
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.llm.extract import ExtractionResult, build_user_message

# Keyword sets per finding, matched case-insensitively against the
# extraction's combined text (auditor_opinion_type, auditor_concerns,
# key_audit_matters, related_party_summary, extraction_gaps — a finding
# reported honestly as a gap still counts, since the point is whether the
# model surfaced awareness of it, not which field it landed in).
#
# "caro" was originally its own bare-keyword finding — wrong, not just
# imprecise. CARO commentary is a MANDATORY section of every Indian audit
# report, clean or not: BEL's actual CARO text reads "there are no
# qualifications or adverse remarks in these CARO reports" — a clean bill
# of health, not a red flag. Matching the literal word "CARO" conflates
# "the boilerplate section exists" with "there's an adverse finding in
# it", which invalidated the BEL ground truth entirely (see PROJECT.md's
# recall-benchmark note). Replaced with content-specific keywords for the
# one ticker (VMM) that has a genuine adverse CARO finding — dropped from
# BEL's ground truth below rather than kept as a mismatched category.
FINDING_KEYWORDS = {
    "rule_11g_audit_trail": ["11(g)", "audit trail", "audit-trail"],
    "caro_bank_stock_variance": ["stock statement", "book debt statement", "bank stock", "quarterly return"],
    "board_composition_reg17": ["regulation 17", "independent director"],
    "whistleblower": ["whistle"],
    "contingent_liabilities": ["contingent liabilit"],
}

# Ground truth: which findings are actually present in each ticker's real,
# fetched annual report text. Verified 2026-08-26 by grepping the raw
# section text directly (see PROJECT.md's recall-benchmark note) — not
# inferred from any model's extraction. BEL has no "caro_bank_stock_variance"
# entry: its CARO section was read directly and confirmed clean (no
# qualifications), so there is nothing there to recall.
GROUND_TRUTH: dict[str, set[str]] = {
    "VMM": {"rule_11g_audit_trail", "caro_bank_stock_variance", "whistleblower"},
    "BEL": {"rule_11g_audit_trail", "board_composition_reg17", "contingent_liabilities"},
    "JYOTHYLAB": {"rule_11g_audit_trail", "board_composition_reg17", "whistleblower", "contingent_liabilities"},
}

# Real Haiku Stage 1 fixtures already captured live tonight — reused here
# instead of paying for fresh Haiku calls just to re-confirm what's
# already been paid for once. No pre-existing Haiku fixture for VMM
# specifically — the earlier ab_test.py run that tested Haiku on VMM
# predated the fix that makes _call_anthropic save fixtures at all, so
# that real, billed call's raw response was never persisted. VMM's Haiku
# entry gets a fresh call below instead (now correctly logged/saved for
# next time).
HAIKU_FIXTURES = {
    "BEL": r"E:\stockanlysis\data\llm_fixtures\stage1_BEL_20260826T060419322134.json",
    "JYOTHYLAB": r"E:\stockanlysis\data\llm_fixtures\stage1_JYOTHYLAB_20260826T115623012667.json",
}

# Real Sonnet 5 Stage 1 fixture captured before the v3 migration switched
# Stage 1 to Haiku — reused for VMM instead of paying for a fresh Sonnet
# call to re-confirm what's already been paid for once.
SONNET_VMM_FIXTURE = r"E:\stockanlysis\data\llm_fixtures\stage1_VMM_20260826T050428664220.json"


@dataclass
class ModelResult:
    ticker: str
    provider_model: str
    result: ExtractionResult | None
    error: str | None = None


def _load_fixture_result(path: str) -> ExtractionResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExtractionResult.model_validate(json.loads(data["report_text"]))


def score_recall(result: ExtractionResult, ground_truth: set[str]) -> dict[str, bool]:
    haystack = " ".join(
        [
            result.auditor_opinion_type or "",
            *result.auditor_concerns,
            *result.key_audit_matters,
            result.related_party_summary or "",
            *result.extraction_gaps,
        ]
    ).lower()
    return {
        finding: any(kw in haystack for kw in FINDING_KEYWORDS[finding])
        for finding in ground_truth
    }


def recall_rate(scores: dict[str, bool]) -> float:
    if not scores:
        return 1.0
    return sum(scores.values()) / len(scores)


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    table = load_symbol_table()

    all_results: dict[str, dict[str, ModelResult]] = {t: {} for t in GROUND_TRUTH}

    for ticker in GROUND_TRUTH:
        print(f"\n{'=' * 20} {ticker} {'=' * 20}")
        resolved = resolve_ticker(ticker, table)
        brief = assemble_brief(resolved)
        user_message = build_user_message(brief)

        # Sonnet 5 — reuse the real pre-migration fixture for VMM; fresh
        # real calls for BEL/JYOTHYLAB (no pre-migration Sonnet fixture
        # exists for these — fixture-saving didn't exist yet when they
        # were first analyzed).
        if ticker == "VMM":
            sonnet_result = _load_fixture_result(SONNET_VMM_FIXTURE)
            all_results[ticker]["sonnet-5"] = ModelResult(ticker, "sonnet-5", sonnet_result)
        else:
            attempt = _call_anthropic(user_message, "claude-sonnet-5", ticker=ticker)
            all_results[ticker]["sonnet-5"] = ModelResult(ticker, "sonnet-5", attempt.result, attempt.error)

        # Haiku 4.5 — reuse real fixtures already captured tonight where
        # available; VMM has no saved Haiku fixture (see HAIKU_FIXTURES'
        # comment), so it gets a fresh real call instead.
        if ticker in HAIKU_FIXTURES:
            haiku_result = _load_fixture_result(HAIKU_FIXTURES[ticker])
            all_results[ticker]["haiku-4.5"] = ModelResult(ticker, "haiku-4.5", haiku_result)
        else:
            attempt = _call_anthropic(user_message, "claude-haiku-4-5-20251001", ticker=ticker)
            all_results[ticker]["haiku-4.5"] = ModelResult(ticker, "haiku-4.5", attempt.result, attempt.error)

        # DeepSeek V4 Flash — free tokens, call fresh for a clean run.
        ds_attempt = _call_deepseek(user_message, DEEPSEEK_MODEL, ticker=ticker)
        all_results[ticker]["deepseek-v4-flash"] = ModelResult(
            ticker, "deepseek-v4-flash", ds_attempt.result, ds_attempt.error
        )

    print(f"\n\n{'=' * 20} RECALL SUMMARY {'=' * 20}")
    per_model_totals: dict[str, list[float]] = {"sonnet-5": [], "haiku-4.5": [], "deepseek-v4-flash": []}
    for ticker, ground_truth in GROUND_TRUTH.items():
        print(f"\n{ticker} (ground truth: {sorted(ground_truth)})")
        for model_name, model_result in all_results[ticker].items():
            if model_result.result is None:
                print(f"  {model_name}: FAILED — {model_result.error}")
                per_model_totals[model_name].append(0.0)
                continue
            scores = score_recall(model_result.result, ground_truth)
            rate = recall_rate(scores)
            per_model_totals[model_name].append(rate)
            missed = [f for f, hit in scores.items() if not hit]
            print(f"  {model_name}: {rate:.0%} ({sum(scores.values())}/{len(scores)})" + (f" — missed: {missed}" if missed else ""))

    print(f"\n{'=' * 20} OVERALL (mean recall across {len(GROUND_TRUTH)} tickers) {'=' * 20}")
    for model_name, rates in per_model_totals.items():
        mean = sum(rates) / len(rates) if rates else 0.0
        print(f"  {model_name}: {mean:.0%}")


if __name__ == "__main__":
    main()
