# Stage 2 Lite — short verdict path

You receive the same `<context>` blocks as the full master analysis (price,
financials, shareholding, Stage 1 extraction). You do **not** have web search.

Write a **compact** report (~1,500–2,500 words of prose) with these sections only:

1. **QUICK VERDICT** — verdict, price tokens, buy zone (or "not issued"), fair
   value base range tokens, risk, confidence X/10, holding period.
2. **WHY BUY** — 3–5 bullets with [FINANCIALS] / [EXTRACTION] citations.
3. **WHY AVOID** — 3–5 bullets with citations.
4. **VALUATION SUMMARY** — bear/base/bull EPS × multiple reasoning; use
   `{{fair_value_base_low}}`–`{{fair_value_base_high}}` for headline fair value.
5. **BEGINNER SUMMARY** — include **SHOULD I BUY?** with Decision, Price, Buy Zone,
   Fair Value, Risk, Confidence.
6. **EXPECTED RETURN (scenarios)** — state bear/base/bull **CAGR ranges** over 2–5
   years (not fixed "year 1 / year 2 / year 3" return ladders). Tie assumptions to
   supplied EPS/multiple/order-book/cash-flow evidence; mark external broker forecasts
   [UNVERIFIED] unless in context.

Apply the Quality-First constitution from the system prompt: complete
`five_year_business_test` before any buy/add zone; withhold buy/add ranges when
WC gap is unresolved; anti-chase when applicable. Weight last-3y FINANCIALS over
an early boom: `DECELERATING` / `NEGATIVE` → UNCERTAIN, not YES, and no full
Ideal Buy (research still allowed). Short history (~1 year) and missing latest
figures are `DATA REVIEW` / research — never invent numbers, never auto-reject
the business solely for incomplete data.

Use placeholder tokens (`{{current_price}}`, `{{rsi14}}`, etc.) for all numbers
Python will substitute — never hard-code price, technical, or fair-value figures.

At the very end, after the Beginner Summary, output the same fenced ` ```json `
block specified in `<pipeline_constraints>` (identical schema to the full path).

Footer (after JSON): *Research and education, not investment advice.*
