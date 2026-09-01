"""Stage 2 FULL report generation via DeepSeek (OpenAI-compatible API).

Used by the Stage 2 cost/quality benchmark — not wired into the production
/analyze pipeline until A/B gates pass.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from stockbot.ab_test import DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from stockbot.config import settings
from stockbot.costs import log_call
from stockbot.llm.extract import ExtractionResult
from stockbot.llm.fixtures import save_response_fixture
from stockbot.llm.verdict import (
    MAX_TOKENS,
    TruncatedResponseError,
    VerdictJSON,
    build_user_message,
    extract_verdict_json,
    load_stage2_system_prompt,
)
from stockbot.models import Brief

logger = logging.getLogger(__name__)

STAGE2_DEEPSEEK_TIMEOUT_SECONDS = 600.0


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError)),
)
def _post_deepseek_chat(*, api_key: str, payload: dict) -> httpx.Response:
    response = httpx.post(
        f"{DEEPSEEK_API_BASE.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=STAGE2_DEEPSEEK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


def run_stage2_deepseek(
    brief: Brief,
    extraction: ExtractionResult,
    *,
    extra_instruction: str | None = None,
    max_tokens: int | None = None,
    model: str = DEEPSEEK_MODEL,
) -> tuple[str, VerdictJSON, dict]:
    """Run master-prompt Stage 2 via DeepSeek; returns (report_text, verdict, usage)."""
    api_key = settings.deepseek_api_key
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — cannot run Stage 2 DeepSeek benchmark")

    call_max_tokens = max_tokens or MAX_TOKENS
    system_prompt = load_stage2_system_prompt("FULL")
    user_message = build_user_message(brief, extraction, extra_instruction)

    response = _post_deepseek_chat(api_key=api_key, payload={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": call_max_tokens,
        })
    data = response.json()
    choice = data["choices"][0]
    report_text = choice["message"]["content"] or ""
    stop_reason = choice.get("finish_reason", "unknown")
    usage_raw = data.get("usage", {}) or {}
    input_tokens = int(usage_raw.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage_raw.get("completion_tokens", 0) or 0)
    cached_tokens = int(usage_raw.get("prompt_cache_hit_tokens", 0) or 0)
    miss_tokens = max(0, input_tokens - cached_tokens) if cached_tokens else input_tokens

    save_response_fixture(
        stage="stage2_deepseek",
        ticker=brief.ticker.symbol,
        report_text=report_text,
        stop_reason=stop_reason,
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cached_tokens,
        },
    )

    called_at = datetime.now(UTC)
    cost_inr = log_call(
        model=model,
        input_tokens=miss_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        provider="deepseek",
        called_at=called_at,
        stage="stage2_deepseek",
        ticker=brief.ticker.symbol,
    )

    if stop_reason == "length":
        raise TruncatedResponseError(cost_inr, len(report_text), call_max_tokens)

    verdict = extract_verdict_json(report_text)
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_inr": cost_inr,
        "stage2_mode": "FULL",
        "provider": "deepseek",
        "model": model,
        "thinking_tokens": 0,
    }
    logger.info(
        "%s Stage 2 DeepSeek (%s) cost ₹%.2f, %d chars",
        brief.ticker.symbol,
        model,
        cost_inr,
        len(report_text),
    )
    return report_text, verdict, usage
