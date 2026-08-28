"""Tests for NSE XBRL pledge parsing."""

from __future__ import annotations

from stockbot.fetch.shareholding import parse_pledge_pct_from_nse_xbrl

_FALSE_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">false</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
"""

_TRUE_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">true</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
"""


def test_pledge_zero_when_xbrl_flag_false():
    assert parse_pledge_pct_from_nse_xbrl(_FALSE_XML) == 0.0


def test_pledge_none_when_xbrl_flag_true_but_pct_not_parsed():
    assert parse_pledge_pct_from_nse_xbrl(_TRUE_XML) is None


def test_pledge_none_when_tag_missing():
    assert parse_pledge_pct_from_nse_xbrl("<root></root>") is None
