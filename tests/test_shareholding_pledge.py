"""Tests for NSE XBRL pledge parsing."""

from __future__ import annotations

from stockbot.fetch.shareholding import parse_pledge_pct_from_nse_xbrl

_FALSE_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">false</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
"""

_TRUE_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">true</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
"""

_RATIO_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">true</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
<in-bse-shp:NumberOfSharesEncumberedUnderPledged contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">7700000</in-bse-shp:NumberOfSharesEncumberedUnderPledged>
<in-bse-shp:NumberOfShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">974234554</in-bse-shp:NumberOfShares>
"""

_HIGH_PLEDGE_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">true</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
<in-bse-shp:NumberOfSharesEncumberedUnderPledged contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">349213558</in-bse-shp:NumberOfSharesEncumberedUnderPledged>
<in-bse-shp:NumberOfShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">423652000</in-bse-shp:NumberOfShares>
"""

_DIRECT_PCT_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">true</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
<in-bse-shp:EncumberedShareUnderPledgedAsPercentageOfNumberOfSharesHeld contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">12.5</in-bse-shp:EncumberedShareUnderPledgedAsPercentageOfNumberOfSharesHeld>
"""

_PROSE_XML = """
<in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged contextRef="MainI">true</in-bse-shp:WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged>
<note>Shares pledged representing 18.25% of their holding as on quarter end.</note>
"""


def test_pledge_zero_when_xbrl_flag_false():
    assert parse_pledge_pct_from_nse_xbrl(_FALSE_XML) == 0.0


def test_pledge_none_when_xbrl_flag_true_but_pct_not_parsed():
    assert parse_pledge_pct_from_nse_xbrl(_TRUE_XML) is None


def test_pledge_none_when_tag_missing():
    assert parse_pledge_pct_from_nse_xbrl("<root></root>") is None


def test_pledge_pct_from_promoter_group_share_ratio():
    assert parse_pledge_pct_from_nse_xbrl(_RATIO_XML) == 0.7904


def test_pledge_pct_from_high_promoter_group_share_ratio():
    assert parse_pledge_pct_from_nse_xbrl(_HIGH_PLEDGE_XML) == 82.4293


def test_pledge_pct_from_direct_percentage_tag():
    assert parse_pledge_pct_from_nse_xbrl(_DIRECT_PCT_XML) == 12.5


def test_pledge_pct_from_prose_fallback():
    assert parse_pledge_pct_from_nse_xbrl(_PROSE_XML) == 18.25
