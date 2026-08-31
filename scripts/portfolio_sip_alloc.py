"""One-off CLI: print portfolio SIP tables from sip_portfolios.json."""

from stockbot.portfolio_sip import build_portfolio_sip_plan
from stockbot.portfolio_sip_messages import format_portfolio_plan_html


def main() -> None:
    plan = build_portfolio_sip_plan()
    print(format_portfolio_plan_html(plan).replace("<b>", "").replace("</b>", ""))


if __name__ == "__main__":
    main()
