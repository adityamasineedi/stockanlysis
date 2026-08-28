"""The single choke point for every Anthropic messages call in this
project. Found live, twice in two days: extract.py's run_stage1 and
verdict.py's run_stage2 each duplicated their own fixture-save +
cost-log logic, and ab_test.py's _call_anthropic was added later without
either — a real, billed call went completely untracked. Logging placed
*after* the call at each call site is a gap waiting to be forgotten again
at the next call site. Logging placed *inside* the one function every
call site must route through is not optional to skip.

call_anthropic_and_log is that function: it makes the actual API call,
saves the raw response as a fixture, and logs the cost, as one atomic
step. Every caller gets the response and cost back — there is no
Anthropic call anywhere else in this codebase that reaches the network
without going through here first.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from stockbot.costs import log_call
from stockbot.llm.fixtures import save_response_fixture

logger = logging.getLogger(__name__)

MAX_ANTHROPIC_ATTEMPTS = 3


def _is_retryable_anthropic_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _invoke_anthropic(
    client: Anthropic,
    *,
    stream: bool,
    kwargs: dict[str, Any],
) -> Any:
    if stream:
        with client.messages.stream(**kwargs) as message_stream:
            return message_stream.get_final_message()
    return client.messages.parse(**kwargs)


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_ANTHROPIC_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(_is_retryable_anthropic_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _invoke_anthropic_with_retries(
    client: Anthropic,
    *,
    stream: bool,
    kwargs: dict[str, Any],
) -> Any:
    return _invoke_anthropic(client, stream=stream, kwargs=kwargs)


def call_anthropic_and_log(
    client: Anthropic,
    *,
    stage: str,
    ticker: str,
    model: str,
    max_tokens: int,
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    output_format: type | None = None,
    thinking: dict[str, Any] | None = None,
    stream: bool = False,
) -> tuple[Any, float]:
    """Returns (response, cost_inr). `response` has the same shape
    regardless of `stream` — .content, .stop_reason, .usage — since
    stream=True internally calls .get_final_message() before returning.

    Transient network / 5xx / rate-limit errors are retried up to
    MAX_ANTHROPIC_ATTEMPTS times. Fixture save and cost logging run only
    after a successful response so failed attempts are not billed in our
    ledger (Anthropic may still bill partial work on their side).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if output_format is not None:
        kwargs["output_format"] = output_format
    if thinking is not None:
        kwargs["thinking"] = thinking

    response = _invoke_anthropic_with_retries(client, stream=stream, kwargs=kwargs)

    report_text = "".join(block.text for block in response.content if block.type == "text")
    save_response_fixture(
        stage=stage,
        ticker=ticker,
        report_text=report_text,
        stop_reason=response.stop_reason,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        },
    )
    # cache_creation is an Optional nested breakdown (added alongside 1h TTL
    # support) of the flat cache_creation_input_tokens total by which TTL
    # wrote it — needed because a 1h write is billed at 2x, a 5m write at
    # 1.25x, and there's no way to tell them apart from the flat total alone.
    cache_creation = getattr(response.usage, "cache_creation", None)
    cache_creation_1h_tokens = getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0
    # thinking_tokens: exact count of output_tokens spent on internal
    # reasoning, purely for cost observability (see PROJECT.md's "80% of
    # Stage 2 output is invisible thinking" finding).
    output_tokens_details = getattr(response.usage, "output_tokens_details", None)
    thinking_tokens = getattr(output_tokens_details, "thinking_tokens", 0) or 0
    cost_inr = log_call(
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        cache_creation_1h_tokens=cache_creation_1h_tokens,
        thinking_tokens=thinking_tokens,
        stage=stage,
        ticker=ticker,
    )

    return response, cost_inr
