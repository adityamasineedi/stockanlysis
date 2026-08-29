"""Module 6 (annual report) unit tests against synthetic page text — no
network, no real PDFs. Live behaviour (real NSE discovery, a real 187-page
RELIANCE filing, the curly-apostrophe extraction quirk, and the
compliance-certificate false positive for "Independent Auditor's Report")
was verified by hand during development — see the module docstring."""

from stockbot.fetch.annual_report import (
    HEADING_PRIORITY,
    TOKEN_CAP,
    _build_sections,
    _estimate_tokens,
    _figure_density,
    _find_heading_pages,
    _is_scanned,
    _merge_ranges,
    _normalize_quotes,
    _truncate_to_budget,
    parse_ar_business_summary,
)


def test_normalize_quotes_converts_curly_to_straight():
    assert _normalize_quotes("Auditor’s Report") == "Auditor's Report"
    assert _normalize_quotes("Auditor‘s Report") == "Auditor's Report"


def test_find_heading_pages_matches_curly_apostrophe_in_source_text():
    pages = [
        "some text",
        "Independent Auditor’s Report\nTo The Members of ABC Ltd\nBasis for Opinion...",
    ]
    hits = _find_heading_pages(pages, "Independent Auditor's Report")
    assert hits == [1]


def test_find_heading_pages_rejects_non_opinion_document_sharing_the_heading():
    # a compliance certificate using the same heading AND the same "to the
    # members" salutation, but none of the audit-specific confirming
    # phrases — this is the real false positive found in a live RELIANCE
    # filing (both documents are addressed "to the members", so that
    # salutation alone can't be the distinguishing signal)
    pages = [
        (
            "Independent Auditor's Report on compliance with SEBI regulations...\n"
            "To the Members of ABC Limited, we report on compliance matters."
        )
    ]
    hits = _find_heading_pages(pages, "Independent Auditor's Report")
    assert hits == []


def test_find_heading_pages_accepts_genuine_opinion_letter():
    pages = ["Independent Auditor's Report\nTo The Members of XYZ Limited\nBasis for Opinion..."]
    hits = _find_heading_pages(pages, "Independent Auditor's Report")
    assert hits == [0]


def test_find_heading_pages_accepts_opinion_letter_even_with_scrambled_salutation():
    # verified live: pdfplumber's column-order extraction can interleave
    # an unrelated heading between "To," and "the Members" — the wider,
    # multi-phrase window must still catch this via "report on the audit"
    pages = ["Independent Auditor's Report\nTo, SOME OTHER HEADING\nthe Members of Report on the Audit"]
    hits = _find_heading_pages(pages, "Independent Auditor's Report")
    assert hits == [0]


def test_find_heading_pages_case_insensitive_and_whitespace_flexible():
    pages = ["RELATED   PARTY\ntransactions disclosed below"]
    hits = _find_heading_pages(pages, "Related Party")
    assert hits == [0]


def test_find_heading_pages_matches_partial_stem():
    pages = ["Contingent Liabilities as at year end"]
    hits = _find_heading_pages(pages, "Contingent Liabilit")
    assert hits == [0]


def test_merge_ranges_combines_overlapping_windows():
    ranges = _merge_ranges([10, 11, 12], window=3, total_pages=100)
    assert ranges == [(7, 15)]


def test_merge_ranges_keeps_disjoint_windows_separate():
    ranges = _merge_ranges([5, 50], window=2, total_pages=100)
    assert ranges == [(3, 7), (48, 52)]


def test_merge_ranges_clamps_to_document_bounds():
    ranges = _merge_ranges([0, 99], window=3, total_pages=100)
    assert ranges == [(0, 3), (96, 99)]


def test_merge_ranges_empty_input():
    assert _merge_ranges([], window=3, total_pages=100) == []


def test_estimate_tokens_uses_char_heuristic():
    assert _estimate_tokens("a" * 400) == 100


def test_truncate_to_budget_respects_limit_and_marks_truncation():
    text = "word " * 5000  # 25,000 chars
    truncated = _truncate_to_budget(text, budget_tokens=100)  # 400 char budget
    assert len(truncated) <= 500  # 400 chars + marker text
    assert "TRUNCATED" in truncated


def test_truncate_to_budget_zero_budget_returns_empty():
    assert _truncate_to_budget("some text", budget_tokens=0) == ""


def test_is_scanned_detects_near_empty_pages():
    assert _is_scanned(["", "", ""]) is True
    assert _is_scanned([]) is True


def test_is_scanned_false_for_normal_text_density():
    assert _is_scanned(["word " * 500] * 10) is False


def test_build_sections_drops_lowest_priority_first_when_all_fit_individually_but_not_together():
    # each heading's own text is small, but far more headings exist than
    # fit together under a tiny cap — lower priority ones should drop first
    pages = []
    for i, heading in enumerate(HEADING_PRIORITY):
        pages.append(f"{heading}\nTo The Members of Test Co\n" + ("x" * 40))

    sections, truncated, dropped = _build_sections(pages)

    # Qualified Opinion is priority #1 and tiny — must survive
    assert "Qualified Opinion" in sections
    assert isinstance(truncated, bool)
    assert isinstance(dropped, list)


def test_build_sections_truncates_rather_than_fully_dropping_when_budget_partially_remains():
    # one huge heading that alone exceeds TOKEN_CAP
    huge_text = "x" * (TOKEN_CAP * 4 * 2)  # ~2x the token cap in characters
    pages = [f"Qualified Opinion\n{huge_text}"]

    sections, truncated, dropped = _build_sections(pages)

    assert truncated is True
    assert "Qualified Opinion" in dropped
    assert "Qualified Opinion" in sections  # partial content retained, not excluded
    assert len(sections["Qualified Opinion"]) > 0


def test_figure_density_ranks_numeric_tables_above_prose():
    prose = "the company measures contingent liabilities as indication for impairment testing purposes"
    table = "Contingent Liabilities 147 162 Reconciliation 31 205 2026 2025"
    assert _figure_density(table) > _figure_density(prose)


def test_figure_density_empty_text_is_zero():
    assert _figure_density("") == 0.0


def test_build_sections_prioritizes_figure_dense_page_over_earlier_prose_page_when_truncating():
    # Regression for a real bug found on live KPITTECH/BEL reports: a
    # heading with multiple hit pages used to concatenate them in document
    # order, and truncation always kept the FRONT of that concatenation —
    # so an earlier boilerplate accounting-policy page (repeating the
    # heading's own words in prose) survived while the real, numbered
    # disclosure many pages later got silently cut once the combined text
    # exceeded TOKEN_CAP. Made the prose page alone exceed the cap so the
    # only way the figures survive is if the figure-dense block is sorted
    # to the front before truncation, not left in document order.
    # Hit pages far enough apart (> 2*SWING_WINDOW_PAGES+1) that
    # _merge_ranges keeps their windows as two separate, non-overlapping
    # ranges rather than collapsing them into one contiguous block.
    prose_page = "Contingent Liabilit ies are possible obligations arising from past events. " * 4000
    figures_page = "Contingent Liabilit ies 147 162 2026 2025 Note 31 Reconciliation to carrying amounts"
    pages = [prose_page] + ["filler page, nothing relevant"] * 10 + [figures_page]

    sections, truncated, _dropped = _build_sections(pages)

    assert truncated is True
    assert "Contingent Liabilit" in sections
    assert "147" in sections["Contingent Liabilit"]


def test_build_sections_no_hits_returns_empty_without_error():
    pages = ["nothing relevant here"] * 5
    sections, truncated, dropped = _build_sections(pages)
    assert sections == {}
    assert truncated is False
    assert dropped == []


def test_build_sections_extracts_business_headings():
    pages = [
        "Management Discussion and Analysis\n"
        "Order book stands at Rs. 20,535 crore with execution over 2 years.\n"
        "Segment - Shipbuilding: revenue growth remained strong.",
        "Segment Information\nShipbuilding, Submarines, Refits",
    ]
    sections, _, _ = _build_sections(pages)
    assert any("Management Discussion" in k for k in sections)
    assert any("Order Book" in k or "Management Discussion" in k for k in sections)


def test_parse_ar_business_summary_extracts_order_book_and_segments():
    sections = {
        "Order Book": "The consolidated order book as on 31 Mar 2026 was Rs. 20,535 crore.",
        "Segment Information": "Segment - Shipbuilding: warships and submarines",
    }
    summary = parse_ar_business_summary(sections)
    assert summary is not None
    assert summary.order_book_cr == 20535.0
    assert summary.segments
