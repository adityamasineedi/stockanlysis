from __future__ import annotations

from stockbot.bot_suggestions import parse_pick_callback
from stockbot.models import TickerInfo


def test_parse_pick_callback():
    assert parse_pick_callback("pick:prescan:BEL") == ("prescan", "BEL")
    assert parse_pick_callback("pick:analyze:HEROMOTOCO") == ("analyze", "HEROMOTOCO")
    assert parse_pick_callback("nope") is None


def test_build_symbol_pick_keyboard_truncates_long_labels():
    from stockbot.bot_suggestions import build_symbol_pick_keyboard

    tickers = [
        TickerInfo(
            symbol="REALLYLONGSYMBOL",
            exchange="NSE",
            company_name="A" * 40,
            isin=None,
        )
    ]
    keyboard = build_symbol_pick_keyboard(tickers, action="prescan")
    assert keyboard is not None
    assert keyboard.inline_keyboard[0][0].callback_data.startswith("pick:prescan:")
