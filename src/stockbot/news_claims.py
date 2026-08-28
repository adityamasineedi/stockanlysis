"""Extract unverified external claims from fetched news headlines."""

from __future__ import annotations

from stockbot.models import NewsItems
from stockbot.order_book_signals import extract_order_book_news_claims as extract_order_book_news_claims