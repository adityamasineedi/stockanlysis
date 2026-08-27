"""Module 5 (news) unit tests against synthetic RSS XML — no network, no
HTTP-mocking dependency. Live behaviour (real Google News RSS structure,
real dedup across queries, a genuinely empty adversarial query) was
verified by hand against Jyothy Labs during development."""

from datetime import date

from stockbot.fetch.news import _build_url, _dedup_by_title, _parse_rss_items
from stockbot.models import RedFlag


def _rss(items: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f"<item><title>{title}</title><link>{link}</link>"
        f"<pubDate>{pub_date}</pubDate></item>"
        for title, link, pub_date in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{body}</channel></rss>'


def test_build_url_encodes_query():
    url = _build_url("Jyothy Labs SEBI")
    assert url.startswith("https://news.google.com/rss/search?q=")
    assert "Jyothy+Labs+SEBI" in url
    assert url.endswith("&hl=en-IN&gl=IN&ceid=IN:en")


def test_parse_rss_items():
    xml = _rss(
        [
            (
                "Headline One",
                "https://news.google.com/rss/articles/AAA",
                "Mon, 25 Aug 2026 07:00:00 GMT",
            ),
            (
                "Headline Two",
                "https://news.google.com/rss/articles/BBB",
                "Tue, 26 Aug 2026 08:00:00 GMT",
            ),
        ]
    )

    items = _parse_rss_items(xml, "Test Query")

    assert len(items) == 2
    assert items[0].headline == "Headline One"
    assert items[0].published_date == date(2026, 8, 25)
    assert items[0].found_by_query == "Test Query"


def test_parse_rss_items_skips_items_missing_a_date():
    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>No date item</title>"
        "<link>https://news.google.com/rss/articles/CCC</link></item>"
        "</channel></rss>"
    )
    assert _parse_rss_items(xml, "Test Query") == []


def test_parse_rss_items_returns_empty_list_for_no_items():
    xml = '<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
    assert _parse_rss_items(xml, "Test Query") == []


def test_dedup_merges_matching_titles_and_keeps_all_query_labels():
    items = [
        RedFlag("Company faces SEBI probe over disclosure", "url-a", date(2026, 1, 1), "Q1"),
        RedFlag(
            "Company faces SEBI probe over disclosure", "url-a-redirect", date(2026, 1, 1), "Q2"
        ),
        RedFlag("Totally unrelated headline", "url-b", date(2026, 1, 2), "Q1"),
    ]
    deduped = _dedup_by_title(items)

    assert len(deduped) == 2
    merged = next(i for i in deduped if "SEBI probe" in i.headline)
    assert merged.found_by_query == "Q1, Q2"


def test_dedup_does_not_duplicate_same_query_label():
    items = [
        RedFlag("Same headline", "url-a", date(2026, 1, 1), "Q1"),
        RedFlag("Same headline", "url-a-redirect", date(2026, 1, 1), "Q1"),
    ]
    deduped = _dedup_by_title(items)
    assert len(deduped) == 1
    assert deduped[0].found_by_query == "Q1"
