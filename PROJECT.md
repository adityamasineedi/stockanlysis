# stockbot — data contracts

This file is the source of truth for every type that crosses a module
boundary in this project. `src/stockbot/models.py` implements exactly
what's documented here — if the two ever disagree, this file is wrong
and should be fixed, not silently overridden in code.

## Units discipline

All money is **₹ crore as `float`**, unless the field name ends in:
- `_abs` — absolute rupees (e.g. a per-share price)
- `_pct` — a percentage, expressed as a plain number (e.g. `42.5` for 42.5%)

Never mix. Never store a formatted string like `"₹1,234cr"` in a data
field — formatting only happens at the Telegram-reply boundary
(`bot.py`), never earlier.

## Missing data

Every fetch-result type below carries `source` and a `fetched_at`
timestamp. A value that could not be determined is **always `None`** —
never `0`, never `"N/A"`, never `""`. `0` and `None` mean different
things (e.g. pledge: `0.0` = confirmed no pledge, `None` = pledge
status unconfirmed) and the LLM stages depend on being able to tell
them apart.

## DataFrame-holding dataclasses

Any dataclass holding a `pandas.DataFrame` field is declared
`@dataclass(frozen=True, eq=False)` — the dataclass-generated `__eq__`
calls `==` on every field including the DataFrame, and `DataFrame.__eq__`
raises `ValueError` on truth-value ambiguity rather than returning a bool.
All other dataclasses are `@dataclass(frozen=True)`.

## Types

### `TickerInfo`
A single resolved company/exchange listing.
| field | type | notes |
|---|---|---|
| `symbol` | `str` | e.g. `"TCS"` |
| `exchange` | `Literal["NSE","BSE"]` | |
| `company_name` | `str` | as listed |
| `isin` | `str \| None` | used for dual-listing dedup in Module 1 |

### `AmbiguousMatch`
Returned by `resolve_ticker` when two or more **genuinely different**
companies score close on a fuzzy name match. Never returned for the
same company cross-listed on NSE/BSE — that case resolves silently to
NSE (see `fetch/tickers.py`).
| field | type |
|---|---|
| `candidates` | `list[TickerInfo]` |
| `scores` | `list[float]` |

### `PriceData`
| field | type | notes |
|---|---|---|
| `current_price_abs` | `float` | from `ohlcv_unadjusted` |
| `price_date` | `date` | |
| `ohlcv_adjusted` | `pd.DataFrame` | `auto_adjust=True` — feeds SMA/RSI/S-R |
| `ohlcv_unadjusted` | `pd.DataFrame` | `auto_adjust=False` — feeds current price, 52wk H/L |
| `week52_high_abs` | `float` | from the actual 52-week window, not the 2yr series |
| `week52_low_abs` | `float` | |
| `source` | `str` | e.g. `"yfinance"` |
| `fetched_at` | `datetime` | |

Adjusted and unadjusted series diverge materially in India due to
frequent bonuses/splits — **each is used only for its stated purpose**,
never interchanged.

### `Technicals`
Computed by pure functions over `PriceData.ohlcv_adjusted` — no
network calls in this module.
| field | type | notes |
|---|---|---|
| `sma50` | `float \| None` | |
| `sma200` | `float \| None` | |
| `rsi14` | `float \| None` | Wilder's smoothing, not a simple average |
| `support_abs` | `list[float]` | swing lows, last 6 months |
| `resistance_abs` | `list[float]` | swing highs, last 6 months |
| `as_of_date` | `date` | |
| `source` | `str` | `"computed"` |
| `fetched_at` | `datetime` | |

### `Financials`
| field | type | notes |
|---|---|---|
| `pnl` | `pd.DataFrame` | |
| `balance_sheet` | `pd.DataFrame` | |
| `cash_flow` | `pd.DataFrame` | |
| `ratios` | `pd.DataFrame` | |
| `quarterly` | `pd.DataFrame` | last ~12 quarters, same row shape as `pnl` |
| `basis` | `Literal["consolidated","standalone"]` | **required, never inferred** |
| `years_available` | `int` | |
| `source` | `str` | e.g. `"screener"` |
| `fetched_at` | `datetime` | |

`basis` must be carried through to the brief and stated prominently —
a standalone P&L compared against a consolidated peer produces a
nonsense valuation and nothing downstream can catch the mismatch if
this field is dropped.

### `Shareholding`
| field | type | notes |
|---|---|---|
| `promoter_pct` | `float \| None` | |
| `pledge_pct_of_promoter_holding` | `float \| None` | **named for its denominator** — see below |
| `fii_pct` | `float \| None` | |
| `dii_pct` | `float \| None` | |
| `quarter` | `str \| None` | e.g. `"Q1FY26"` |
| `source` | `str` | `"NSE" \| "BSE" \| "Screener"` |
| `fetched_at` | `datetime` | |

**Pledge denominator.** Exchanges report pledge as a percentage of
*promoter holding*. Some third-party sources report it as a percentage
of *total equity* instead — these differ by roughly 2–3x, and the
downstream red-flag threshold (40% of promoter holding) only makes
sense against the correct denominator. If a source's denominator is
unclear, this field is `None`. Never convert between the two unless
both `promoter_pct` and the source's denominator are known with
certainty. `None` here means "unconfirmed," never "no pledge" — those
are different findings and must stay distinguishable downstream.

### `RedFlag`
A single news item, general or adversarial.
| field | type |
|---|---|
| `headline` | `str` |
| `url` | `str` |
| `published_date` | `date` |
| `found_by_query` | `str` |

### `NewsItems`
| field | type | notes |
|---|---|---|
| `general` | `list[RedFlag]` | last 12 months, max 15 |
| `red_flags` | `list[RedFlag]` | results from the 5 hardcoded adversarial queries |
| `queries_run` | `list[str]` | all 5, always |
| `queries_empty` | `list[str]` | subset of the above that returned nothing — code-known, never asked of the LLM |
| `source` | `str` | `"google_news_rss"` |
| `fetched_at` | `datetime` | |

### `ReportText`
| field | type | notes |
|---|---|---|
| `sections` | `dict[str,str]` | heading → extracted text, in priority order |
| `report_year` | `int \| None` | |
| `source_url` | `str \| None` | |
| `truncated` | `bool` | `True` if the 50K-token cap dropped anything |
| `dropped_sections` | `list[str]` | which headings were dropped, if truncated |
| `source` | `str` | `"BSE_annual_report"` |
| `fetched_at` | `datetime` | |

Section priority order when trimming to the token cap (lowest dropped
first): Qualified/Adverse/Disclaimer Opinion → Independent Auditor's
Report → Emphasis of Matter → Key Audit Matters → Contingent
Liabilities → Related Party.

### `Brief`
The single structured document everything downstream reads. Assembled
by `brief.py` from modules 1–6 in parallel.
| field | type |
|---|---|
| `ticker` | `TickerInfo` |
| `price` | `PriceData` | fatal if unavailable — assemble_brief does not degrade around a missing price |
| `technicals` | `Technicals` | computed from `price`, fatal alongside it |
| `financials` | `Financials \| None` | `None` = module failed entirely (not just a partial gap — see `Financials.basis` etc. for partial gaps within a successful fetch) |
| `shareholding` | `Shareholding \| None` | `None` = module failed entirely |
| `news` | `NewsItems \| None` | `None` = module failed entirely |
| `annual_report` | `ReportText` | never `None` — the fetcher itself returns an empty `ReportText` (sections={}) when no report is found, so "not found" and "fetch failed" are both representable without Optional here |
| `missing` | `list[str]` — every `"MISSING: <what> — <why>"` entry, never a silent omission |
| `token_count` | `int` |
| `confidence_ceiling` | `int` — degrade-loudly cap; 10 unless a fetch failure lowers it |
| `generated_at` | `datetime` |

### `ValidationResult`
| field | type |
|---|---|
| `passed` | `bool` |
| `failures` | `list[str]` |

### `Analysis`
The stored, final record of one run.
| field | type |
|---|---|
| `ticker` | `str` |
| `run_date` | `date` |
| `verdict_json` | `dict` — the fenced JSON block from Stage 2, never re-derived from prose |
| `report_md` | `str` |
| `costs` | `float` — ₹, this run's total spend |
| `validation` | `ValidationResult` |
| `missing` | `list[str]` — carried over from `Brief.missing` so cache hits can still show it, not just fresh runs |

## Confidence-ceiling rules (enforced in `brief.py`)

| condition | ceiling |
|---|---|
| default | 10 |
| `Financials` fetch failed entirely (no consolidated or standalone page usable) — `Brief.financials is None` | 4 |
| Annual report not found | 5 |
| (Stage 2's own pipeline-wide cap, applied later, not in the brief) | 7 |

No yfinance-basics fallback exists for financials (an earlier draft of this
doc referenced one from the original v1 architecture sketch; Prompt 3 was
never asked to build it and didn't). A total financials failure means
`Brief.financials` is `None`, not a degraded-but-present alternative.

## LLM API note: no `temperature` parameter

Sonnet 5 and Opus 5 reject the `temperature` (and `top_p`/`top_k`) parameter
outright — HTTP 400, not just ignored. Prompt 9's "Temperature 0" instruction
for Stage 1 predates this; `llm/extract.py` and `llm/verdict.py` never pass
`temperature`. Determinism for the extraction task comes from a narrow,
factual prompt, not a sampling knob — there's no other way to force it on
these models.

These stack — the lowest applicable ceiling wins. The `Brief.confidence_ceiling`
value is what Stage 2 is told never to exceed, on top of the pipeline-wide
cap of 7 injected directly into the Stage 2 prompt.

## LLM API note: prompt caching on both system prompts

Both `llm/extract.py`'s `SYSTEM_PROMPT` and `llm/verdict.py`'s master prompt
are byte-identical on every call regardless of ticker, so both carry
`cache_control: {"type": "ephemeral", "ttl": "1h"}` on the `system` block —
passed as `system=[{"type": "text", "text": ..., "cache_control": {...}}]`,
not a plain string. The 1h TTL (not the 5-minute default) covers Stage 2's
validation-retry path and back-to-back analyses that would miss a 5m window.
Cache *reads* bill at 0.1x base input. Cache *writes* bill at 1.25x (5m TTL)
or 2.0x (1h TTL) — production uses 1h, so writes are 2x.

Below the model's cacheable minimum (1024 tokens for Sonnet 5, 512 for
Opus 5) it's a silent no-op (`cache_creation_input_tokens: 0`), never an
error — so there's no downside to leaving the marker on Stage 1's smaller
system prompt too.

`costs.compute_cost_inr` bills `cached_tokens` (reads) at 0.1x and splits
`cache_creation_tokens` into 5m (1.25x) vs 1h (2.0x) write premiums, both
additive to `input_tokens` — the API's `input_tokens` is the *uncached
remainder only*, never a superset that cached/created tokens are subtracted
from. An earlier version of this function subtracted `cached_tokens` from
`input_tokens`, which was harmless only because caching had never been
turned on and `cached_tokens` was always 0; fixed before enabling caching
for real.

## v3 migration: bear-case calibration and valuation_inputs

Master prompt v3 replaced v2's direct `fair_value_abs`/`bear_fair_value_abs`
JSON fields with `valuation_inputs` (an EPS estimate + P/E multiple range
per bear/base/bull) — the model supplies inputs, `llm/verdict.compute_valuation`
does the multiplication. Found live on a real BEL report: the model's own
arithmetic (`30x × ₹7.6-8.4 → ₹240-₹280`) was off by ~10% (actual ₹228-252),
understating downside by 5 points. This is prevention, not detection — there
is no model-stated price left to check for drift against, because the model
is never asked to state one.

`VerdictJSON.bear_growth_justification` (optional, required only when
`eps_bear` exceeds trailing EPS) backs `validate._check_bear_eps_sanity`.
Found live on a real VMM report: the model's "bear" case assumed EPS ₹2.00
against a real trailing EPS of ₹1.91 — +11% growth mislabeled as the bear
case, on a 58x stock. `validate._check_bear_adequacy_for_high_multiple`
requires ≥30% downside for any stock above 40x trailing P/E, for the same
reason. Trailing EPS is read from `Financials.pnl.loc["EPS in Rs", "TTM"]`
— a reliable Screener.in row/column label, not a prose-regex extraction.

## Multi-provider cost tracking (17D)

`costs.py` now tracks `provider` per call (`"anthropic"` default, or
`"deepseek"`). DeepSeek's pricing model is structurally different, not just
different numbers: separate cache-hit/cache-miss input rates (its caching
is automatic/disk-based, never explicitly written to, so there's no
Anthropic-style write premium), and peak-hours (01:00-04:00 and 06:00-10:00
UTC, Mon-Fri) that double every rate uniformly — computed from the call's
own timestamp via `_deepseek_rate_multiplier`, never assumed off-peak.
Verified against `https://api-docs.deepseek.com/quick_start/pricing/`
and Anthropic's pricing page on 2026-08-27 (see `costs.PRICING_VERIFIED_ON`).
`ab_test.py` uses this to compare Stage 1
extraction quality (Claude Haiku 4.5 vs DeepSeek V4 Flash) before deciding
whether to move that stage to the cheaper provider.

## Single choke point for Anthropic calls (`llm/client.py`)

`extract.py::run_stage1` and `verdict.py::run_stage2` each used to
duplicate their own fixture-save + cost-log logic after the API call; when
`ab_test.py::_call_anthropic` was added later, it copied neither — a real,
billed Haiku call went completely untracked and was only caught by
inspection, not by any test or check. Logging placed *after* the call at
each call site is a gap waiting to be forgotten again at the next call
site; logging placed *inside* the one function every call site must route
through is not optional to skip.

`call_anthropic_and_log()` is that function: every non-DeepSeek call in
this codebase — Stage 1, Stage 2, and the A/B/recall-benchmark tooling —
goes through it, and it makes the call, saves the raw response as a
fixture, and logs the cost as one atomic step before returning. There is
no path to the Anthropic API elsewhere in the codebase. Tests that used to
monkeypatch `log_call` at `stockbot.llm.extract`/`stockbot.llm.verdict`
now patch it at `stockbot.llm.client` instead, since that's the only place
it's called from.

## Recall-benchmark ground truth: CARO is boilerplate, not a finding

`llm/recall_benchmark.py`'s ground truth originally scored "CARO" as a
finding by bare keyword match. That's wrong, not just imprecise: CARO
(Companies Auditor's Report Order) commentary is a *mandatory* section of
every Indian audit report, clean or not. BEL's actual CARO text reads
"there are no qualifications or adverse remarks in these CARO reports" —
a clean bill of health, not a red flag — so matching the literal word
"CARO" conflated "the boilerplate section exists" with "there's an
adverse finding in it," which invalidated BEL's entire ground truth (every
model's BEL score was inflated by one guaranteed-present, meaningless
match). Fixed by dropping the bare "caro" keyword, replacing it with
content-specific keywords (`caro_bank_stock_variance`, matched only for
VMM, which has a genuine adverse CARO finding about bank stock-statement
variance) — never score a mandatory-boilerplate section's mere presence as
a signal.

Corrected full-benchmark result (mean recall across VMM/BEL/JYOTHYLAB):
Sonnet 5 83%, Haiku 4.5 42%, DeepSeek V4 Flash 39%. Sonnet 5 stayed Stage
1's model on this evidence.
