"""Module 5 — news and red flags. Google News RSS, no API key.

Verified live: news.google.com/rss/search?q=<query>&hl=en-IN&gl=IN&ceid=IN:en
returns a standard RSS 2.0 feed (~100 items per query), no cookie priming
needed.

Two things worth knowing before touching this module:
  - <link> is a Google News redirect URL that differs per query even for
    the same underlying article — dedup happens on normalized title
    similarity (rapidfuzz), never on URL.
  - An unquoted, multi-word query like "<company> auditor resignation" is
    a broad/fuzzy Google match, not an exact phrase — it can return AGM
    notices, earnings-call writeups, or a real-but-differently-worded hit
    (a live test surfaced a genuine CFO resignation under an "auditor
    resignation" query for one company). That's expected: this module
    only fetches and tags candidates by which query surfaced them: it
    does not judge relevance. Relevance is Stage 1's job — "code does
    retrieval, the model does judgment". queries_empty means the RSS feed
    had literally zero items for that query, not "zero relevant items".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential

from stockbot.config import HTTP_USER_AGENT
from stockbot.models import NewsItems, RedFlag

USER_AGENT = HTTP_USER_AGENT
RSS_BASE_URL = "https://news.google.com/rss/search"

GENERAL_NEWS_MONTHS = 12
GENERAL_NEWS_MAX_ITEMS = 15
TITLE_DEDUP_THRESHOLD = 90.0

RED_FLAG_QUERY_TEMPLATES = [
    "{company} SEBI",
    "{company} auditor resignation",
    "{company} promoter pledge",
    "{company} fraud investigation",
    "{company} rating downgrade",
]


def _build_url(query: str) -> str:
    return f"{RSS_BASE_URL}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_rss(query: str) -> list[RedFlag]:
    url = _build_url(query)
    with httpx.Client(
        timeout=20.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        response = client.get(url)
    response.raise_for_status()
    return _parse_rss_items(response.text, query)


def _parse_rss_items(xml: str, query: str) -> list[RedFlag]:
    soup = BeautifulSoup(xml, "lxml-xml")
    items: list[RedFlag] = []
    for item in soup.find_all("item"):
        title_tag = item.find("title")
        link_tag = item.find("link")
        pub_date_tag = item.find("pubDate")
        if title_tag is None or link_tag is None or pub_date_tag is None:
            continue
        try:
            published = parsedate_to_datetime(pub_date_tag.get_text(strip=True))
        except (TypeError, ValueError):
            continue
        items.append(
            RedFlag(
                headline=title_tag.get_text(strip=True),
                url=link_tag.get_text(strip=True),
                published_date=published.date(),
                found_by_query=query,
            )
        )
    return items


def _dedup_by_title(items: list[RedFlag]) -> list[RedFlag]:
    survivors: list[dict] = []
    for item in items:
        match = next(
            (
                s
                for s in survivors
                if fuzz.ratio(item.headline.lower(), s["headline"].lower())
                >= TITLE_DEDUP_THRESHOLD
            ),
            None,
        )
        if match is None:
            survivors.append(
                {
                    "headline": item.headline,
                    "url": item.url,
                    "published_date": item.published_date,
                    "queries": [item.found_by_query],
                }
            )
        elif item.found_by_query not in match["queries"]:
            match["queries"].append(item.found_by_query)

    return [
        RedFlag(
            headline=s["headline"],
            url=s["url"],
            published_date=s["published_date"],
            found_by_query=", ".join(s["queries"]),
        )
        for s in survivors
    ]


def general_news(company: str) -> list[RedFlag]:
    items = _fetch_rss(company)

    cutoff = (datetime.now(UTC) - timedelta(days=30 * GENERAL_NEWS_MONTHS)).date()
    recent = [i for i in items if i.published_date >= cutoff]
    recent.sort(key=lambda i: i.published_date, reverse=True)

    return _dedup_by_title(recent)[:GENERAL_NEWS_MAX_ITEMS]


def red_flag_news(company: str) -> tuple[list[RedFlag], list[str], list[str]]:
    queries = [template.format(company=company) for template in RED_FLAG_QUERY_TEMPLATES]

    all_items: list[RedFlag] = []
    queries_empty: list[str] = []
    for query in queries:
        results = _fetch_rss(query)
        if not results:
            queries_empty.append(query)
        all_items.extend(results)

    return _dedup_by_title(all_items), queries, queries_empty


def fetch_news(company: str) -> NewsItems | None:
    """Fetch general + red-flag news. Returns None only when every RSS call fails."""
    try:
        general = general_news(company)
        red_flags, queries_run, queries_empty = red_flag_news(company)
    except httpx.HTTPError:
        return None

    return NewsItems(
        general=general,
        red_flags=red_flags,
        queries_run=queries_run,
        queries_empty=queries_empty,
        source="google_news_rss",
        fetched_at=datetime.now(UTC),
    )
