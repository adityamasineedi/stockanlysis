"""Module 6 — annual report ingestion. The hardest module; verified each
step against real filings (RELIANCE, JYOTHYLAB) before relying on it.

Discovery: BSE's corporate-announcements UI is a client-rendered SPA with
no discoverable static endpoint (same dead end as Modules 1 and 4's BSE
sources). NSE has a confirmed, working annual-reports API instead — same
cookie-priming pattern as Module 4 (prime via the annual-reports page,
not the homepage):
    https://www.nseindia.com/api/annual-reports?index=equities&symbol=<SYMBOL>
Returns one record per filed year with a direct download link. Several
real wrinkles found while testing, in the order they'd bite:
  - Older filings are sometimes a .zip (one PDF inside) instead of a
    direct .pdf — both are handled.
  - pdfplumber correctly extracts curly apostrophes (U+2019) from the PDF
    text, e.g. "Independent Auditor's Report" with a curly '. A naive
    straight-quote substring search silently finds 0 hits for every
    apostrophe-bearing heading — apostrophes are normalized before
    matching, or every "Independent Auditor's Report" style heading
    would be missed as a hard failure mode, not a loud one.
  - "Independent Auditor's Report" also titles unrelated documents in a
    real filing (a corporate-governance compliance certificate). Both the
    real opinion and the false positive turned out to be addressed "to
    the members" — that salutation is not discriminating on its own. What
    actually distinguishes them is standard SA-format phrasing that only
    appears in a genuine opinion ("basis for opinion", "report on the
    audit", "we have audited") — see AUDITOR_OPINION_CONFIRM_PHRASES.
  - pdfplumber's text order can scramble a multi-column page layout: one
    real filing extracted "To," and "the Members" with an unrelated
    heading interleaved between them by the page's column structure. The
    confirming-phrase check therefore scans a wide window (300 chars)
    rather than expecting one fixed phrase immediately after the heading.
"""

from __future__ import annotations

import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pdfplumber

from stockbot.config import ANNUAL_REPORT_CACHE_DIR
from stockbot.models import ReportText

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
NSE_PRIMING_URL = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
NSE_ANNUAL_REPORTS_URL = "https://www.nseindia.com/api/annual-reports"

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50MB cap
SWING_WINDOW_PAGES = 3
TOKEN_CAP = 50_000
CHARS_PER_TOKEN_ESTIMATE = 4  # rough, dependency-free — no tokenizer in the stack
MIN_AVG_CHARS_PER_PAGE = 20  # below this, treat the PDF as scanned/image-only

# Priority order: filled in this order, lower priorities dropped first
# when TOKEN_CAP binds. See module docstring for why apostrophes below
# are written straight — matching normalizes both forms.
HEADING_PRIORITY: list[str] = [
    "Qualified Opinion",
    "Adverse Opinion",
    "Disclaimer of Opinion",
    "Independent Auditor's Report",
    "Emphasis of Matter",
    "Key Audit Matters",
    "Contingent Liabilit",
    "Related Party",
]


class AnnualReportError(Exception):
    pass


def find_latest_annual_report(symbol: str) -> dict | None:
    with httpx.Client(
        timeout=30.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        client.get(NSE_PRIMING_URL)  # sets the cookies the API call below requires
        response = client.get(
            NSE_ANNUAL_REPORTS_URL,
            params={"index": "equities", "symbol": symbol},
            headers={"Accept": "application/json", "Referer": NSE_PRIMING_URL},
        )
    if response.status_code == 404:
        return None
    response.raise_for_status()

    records = response.json().get("data", [])
    if not records:
        return None

    def _to_year(record: dict) -> int:
        try:
            return int(record.get("toYr", 0))
        except (TypeError, ValueError):
            return 0

    records.sort(key=_to_year, reverse=True)
    return records[0]


def _cache_path(symbol: str, record: dict, suffix: str) -> Path:
    from_yr = record.get("fromYr", "unknown")
    to_yr = record.get("toYr", "unknown")
    return ANNUAL_REPORT_CACHE_DIR / f"{symbol}_{from_yr}_{to_yr}{suffix}"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return

    ANNUAL_REPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        timeout=60.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client, client.stream("GET", url) as response:
        response.raise_for_status()
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes():
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_BYTES:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise AnnualReportError(
                        f"Annual report at {url} exceeds the {MAX_DOWNLOAD_BYTES} byte cap"
                    )
                f.write(chunk)


def _resolve_pdf_path(symbol: str, record: dict) -> Path:
    file_url = record["fileName"]
    if file_url.lower().endswith(".zip"):
        zip_path = _cache_path(symbol, record, ".zip")
        _download(file_url, zip_path)

        pdf_path = zip_path.with_suffix(".pdf")
        if not pdf_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                pdf_members = [n for n in archive.namelist() if n.lower().endswith(".pdf")]
                if not pdf_members:
                    raise AnnualReportError(f"No PDF found inside {zip_path.name}")
                largest = max(
                    pdf_members, key=lambda n: archive.getinfo(n).file_size
                )
                pdf_path.write_bytes(archive.read(largest))
        return pdf_path

    pdf_path = _cache_path(symbol, record, ".pdf")
    _download(file_url, pdf_path)
    return pdf_path


def _extract_pages(pdf_path: Path) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _is_scanned(pages_text: list[str]) -> bool:
    if not pages_text:
        return True
    total_chars = sum(len(t.strip()) for t in pages_text)
    return (total_chars / len(pages_text)) < MIN_AVG_CHARS_PER_PAGE


def _normalize_quotes(text: str) -> str:
    return text.replace("‘", "'").replace("’", "'")


def _heading_pattern(heading: str) -> re.Pattern:
    escaped = re.escape(_normalize_quotes(heading))
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(escaped, re.IGNORECASE)


AUDITOR_OPINION_HEADING = "Independent Auditor's Report"
AUDITOR_OPINION_CONFIRM_WINDOW = 300
# "members of" alone is NOT discriminating — verified live it also appears
# in an unrelated compliance certificate sharing the same heading (both
# are addressed "To the Members of <Company>"). "basis for opinion" and
# "report on the audit" are standard SA-format phrases that appeared only
# in the genuine financial-statements opinion across every real filing
# checked (RELIANCE, JYOTHYLAB) — including one where pdfplumber's column
# extraction scrambled "To," away from "the Members" entirely, which is
# why this checks a wide window for any of several phrases rather than
# one fixed salutation immediately after the heading.
AUDITOR_OPINION_CONFIRM_PHRASES = ("basis for opinion", "report on the audit", "we have audited")


def _find_heading_pages(pages_text: list[str], heading: str) -> list[int]:
    pattern = _heading_pattern(heading)
    hits: list[int] = []
    for i, text in enumerate(pages_text):
        normalized = _normalize_quotes(text)
        for match in pattern.finditer(normalized):
            if heading == AUDITOR_OPINION_HEADING:
                nearby = normalized[
                    match.end() : match.end() + AUDITOR_OPINION_CONFIRM_WINDOW
                ].lower()
                if not any(phrase in nearby for phrase in AUDITOR_OPINION_CONFIRM_PHRASES):
                    continue
            hits.append(i)
            break
    return hits


def _merge_ranges(hit_pages: list[int], window: int, total_pages: int) -> list[tuple[int, int]]:
    if not hit_pages:
        return []
    raw = sorted((max(0, p - window), min(total_pages - 1, p + window)) for p in hit_pages)
    merged = [raw[0]]
    for start, end in raw[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# A 250-char window turned out to still be too wide: on a real BEL filing
# it let an UNRELATED numeric table (a CSR-provision reconciliation, a
# date column header) further down the SAME page dilute/outscore the
# actual disclosure ("Contingent Liabilities 147 162", where the real
# figures sit within about a dozen characters of the heading match, not
# 250). A real note reads "<heading> <number> <number>" almost
# immediately; policy prose reads "<heading> are possible obligations..."
# or "<heading>:\na) The carrying amount..." — words, not digits, right
# at the start. Narrowed to the first 60 characters after the match,
# which is enough room for a currency symbol or short label before the
# first figure but not enough to reach an unrelated table further away.
_FIGURE_WINDOW_CHARS = 60


def _figure_density(text: str) -> float:
    """Fraction of characters that are digits."""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def _range_priority(pages_text: list[str], hits: list[int], pattern: re.Pattern, range_: tuple[int, int]) -> float:
    # A whole-block average (over the full ±SWING_WINDOW_PAGES padded
    # range) turned out too diluted to trust: found live on real KPITTECH
    # and BEL filings, a big merged range containing the ACTUAL numbered
    # note (e.g. "Contingent Liabilities 147 162") still scored LOWER than
    # a different range that had no real disclosure at all, just because
    # the real range also spanned several pages of surrounding prose that
    # diluted its average. What actually distinguishes a real disclosure
    # from accounting-policy prose using the same heading words is
    # narrower: the text immediately following the heading match itself —
    # a real note reads "Contingent Liabilities 147 162 ..." right after
    # the words; a policy paragraph reads "...as possible obligations
    # arising from past events..." instead. Score each range by the best
    # (max) such window across its own hit pages, not the padded average.
    start, end = range_
    best = 0.0
    for hit in hits:
        if not (start <= hit <= end):
            continue
        text = _normalize_quotes(pages_text[hit])
        for match in pattern.finditer(text):
            window = text[match.end() : match.end() + _FIGURE_WINDOW_CHARS]
            best = max(best, _figure_density(window))
    return best


def _extract_ranges_text(
    pages_text: list[str], hits: list[int], heading: str, ranges: list[tuple[int, int]]
) -> str:
    # A heading like "Contingent Liabilit" or "Related Party" can hit many
    # scattered pages — accounting-policy definitions early in the notes,
    # the actual quantified disclosure somewhere later. The old behaviour
    # concatenated ranges in page (document) order and truncation always
    # kept the front and cut the back, which silently dropped the real,
    # numbered note in favour of boilerplate policy prose on a real
    # KPITTECH report (confirmed: the disclosed figures never appeared in
    # what was sent to Stage 1 at all) and very nearly did the same on a
    # real BEL report. Reordering ranges by _range_priority — the range
    # whose actual hit page(s) read most like a numbered table, not the
    # padded block average — means truncation drops the least informative
    # prose first instead of always cutting from the back, regardless of
    # which page it happened to fall on.
    pattern = _heading_pattern(heading)
    ranges = sorted(
        ranges, key=lambda r: _range_priority(pages_text, hits, pattern, r), reverse=True
    )
    blocks = ["\n".join(pages_text[start : end + 1]) for start, end in ranges]
    return "\n\n---\n\n".join(blocks)


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def _truncate_to_budget(text: str, budget_tokens: int) -> str:
    """Keep the earliest page-ranges (first in document order) up to the
    remaining budget, rather than dropping the whole heading — a partial
    Independent Auditor's Report is far more useful than none at all,
    which is the entire point of this module."""
    max_chars = budget_tokens * CHARS_PER_TOKEN_ESTIMATE
    if max_chars <= 0:
        return ""
    truncated = text[:max_chars]
    last_break = truncated.rfind("\n\n")
    if last_break > max_chars * 0.5:
        truncated = truncated[:last_break]
    return truncated + "\n\n[TRUNCATED — exceeds the annual report token budget]"


def _build_sections(pages_text: list[str]) -> tuple[dict[str, str], bool, list[str]]:
    total_pages = len(pages_text)
    candidate_text: dict[str, str] = {}
    for heading in HEADING_PRIORITY:
        hits = _find_heading_pages(pages_text, heading)
        if not hits:
            continue
        ranges = _merge_ranges(hits, SWING_WINDOW_PAGES, total_pages)
        candidate_text[heading] = _extract_ranges_text(pages_text, hits, heading, ranges)

    sections: dict[str, str] = {}
    dropped: list[str] = []
    budget = TOKEN_CAP
    for heading in HEADING_PRIORITY:
        if heading not in candidate_text:
            continue
        text = candidate_text[heading]
        cost = _estimate_tokens(text)

        if cost <= budget:
            sections[heading] = text
            budget -= cost
        elif budget > 0:
            sections[heading] = _truncate_to_budget(text, budget)
            dropped.append(heading)
            budget = 0
        else:
            dropped.append(heading)

    return sections, bool(dropped), dropped


def fetch_annual_report(symbol: str) -> ReportText:
    record = find_latest_annual_report(symbol)
    if record is None:
        return ReportText(
            sections={},
            report_year=None,
            source_url=None,
            truncated=False,
            dropped_sections=[],
            source="nse_annual_reports",
            fetched_at=datetime.now(UTC),
        )

    pdf_path = _resolve_pdf_path(symbol, record)
    pages_text = _extract_pages(pdf_path)

    if _is_scanned(pages_text):
        return ReportText(
            sections={},
            report_year=int(record["toYr"]) if str(record.get("toYr", "")).isdigit() else None,
            source_url=record["fileName"],
            truncated=False,
            dropped_sections=[],
            source="nse_annual_reports",
            fetched_at=datetime.now(UTC),
        )

    sections, truncated, dropped = _build_sections(pages_text)

    return ReportText(
        sections=sections,
        report_year=int(record["toYr"]) if str(record.get("toYr", "")).isdigit() else None,
        source_url=record["fileName"],
        truncated=truncated,
        dropped_sections=dropped,
        source="nse_annual_reports",
        fetched_at=datetime.now(UTC),
    )
