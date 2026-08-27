"""Raw LLM response capture and replay.

Rule: nothing that can be tested without an API call should cost one.
Every real Stage 1/Stage 2 response is saved here (text, stop_reason,
usage — before any parsing is attempted), so a response already paid for
can be replayed through the exact same downstream code (parsing,
validation, retry-feedback assembly, Telegram formatting) at zero cost,
as many times as needed. A truncated or otherwise broken response is the
most valuable fixture of all — it's the only one that reproduces that
failure path.

ReplayClient is a drop-in for the `client=` parameter already accepted by
run_stage1/run_stage2 — no changes to their call sites are needed to use
it; it makes calling code not know or care that no network request is
being made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from stockbot.config import DATA_DIR

FIXTURES_DIR = DATA_DIR / "llm_fixtures"


def save_response_fixture(
    stage: str, ticker: str, report_text: str, stop_reason: str, usage: dict
) -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = FIXTURES_DIR / f"{stage}_{ticker}_{timestamp}.json"
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "ticker": ticker,
                "report_text": report_text,
                "stop_reason": stop_reason,
                "usage": usage,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def load_response_fixture(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"
    parsed_output: Any = None


@dataclass
class _FakeResponse:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: _FakeUsage | None = None
    parsed_output: Any = None


class _ReplayStream:
    """Mimics the context-manager returned by client.messages.stream() —
    Stage 2 switched to streaming (see verdict.py) because the Anthropic
    SDK refuses non-streaming calls once max_tokens is high enough that
    generation could plausibly exceed 10 minutes."""

    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def get_final_message(self) -> _FakeResponse:
        return self._response


class ReplayClient:
    """Drop-in for the Anthropic client, backed by a saved fixture instead
    of a live API call. Usage: run_stage2(brief, extraction,
    client=ReplayClient("data/llm_fixtures/stage2_JYOTHYLAB_....json"))."""

    def __init__(self, fixture_path: str | Path, output_format: type | None = None):
        data = load_response_fixture(fixture_path)
        self._report_text: str = data["report_text"]
        self._stop_reason: str = data["stop_reason"]
        self._usage: dict = data["usage"]
        self._output_format = output_format
        self.messages = self  # client.messages.create/parse/stream -> self.create/self.parse/self.stream

    def _build_response(self) -> _FakeResponse:
        parsed_output = None
        if self._output_format is not None:
            from pydantic import TypeAdapter

            try:
                parsed_output = TypeAdapter(self._output_format).validate_json(self._report_text)
            except Exception:  # noqa: BLE001 - mirrors the real SDK's tolerant parsed_output=None
                parsed_output = None

        block = _FakeTextBlock(text=self._report_text, parsed_output=parsed_output)
        usage = _FakeUsage(
            input_tokens=self._usage.get("input_tokens", 0),
            output_tokens=self._usage.get("output_tokens", 0),
            cache_read_input_tokens=(
                self._usage.get("cache_read_input_tokens") or self._usage.get("cached_tokens") or 0
            ),
        )
        response = _FakeResponse(
            content=[block], stop_reason=self._stop_reason, usage=usage, parsed_output=parsed_output
        )
        return response

    def create(self, **kwargs) -> _FakeResponse:
        return self._build_response()

    def parse(self, **kwargs) -> _FakeResponse:
        self._output_format = kwargs.get("output_format", self._output_format)
        return self._build_response()

    def stream(self, **kwargs) -> _ReplayStream:
        self._output_format = kwargs.get("output_format", self._output_format)
        return _ReplayStream(self._build_response())
