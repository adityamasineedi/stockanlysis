"""Telegram access configuration tests."""

from stockbot.config import parse_telegram_allowed_chat_ids


def test_parse_allowed_chat_ids_empty_means_unrestricted():
    assert parse_telegram_allowed_chat_ids("") == frozenset()
    assert parse_telegram_allowed_chat_ids("  ") == frozenset()


def test_parse_allowed_chat_ids_comma_separated():
    assert parse_telegram_allowed_chat_ids("123, 456 ,789") == frozenset({123, 456, 789})
