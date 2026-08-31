"""Portfolio SIP bot parsing."""

from __future__ import annotations

from stockbot.bot import _parse_portfolio_paid_args, _parse_sip_amount_token


def test_parse_sip_amount_token():
    assert _parse_sip_amount_token("5000") == 5000.0
    assert _parse_sip_amount_token("₹3,213") == 3213.0
    assert _parse_sip_amount_token("BEL") is None


def test_parse_portfolio_paid_args():
    sym, amount, topup = _parse_portfolio_paid_args(["paid", "KAYNES", "3685"])
    assert sym == "KAYNES"
    assert amount == 3685.0
    assert topup is False

    sym, amount, topup = _parse_portfolio_paid_args(["paid", "BEL", "2500", "topup"])
    assert sym == "BEL"
    assert amount == 2500.0
    assert topup is True

    err = _parse_portfolio_paid_args(["paid", "BEL"])
    assert isinstance(err, str)
