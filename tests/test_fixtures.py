"""llm/fixtures.py unit tests — save/load roundtrip and ReplayClient
behaviour, entirely local, no network, no API cost. This is the
infrastructure that makes the rest of Stage 1/Stage 2 debugging free."""

import json

from stockbot.llm import fixtures as fixtures_module
from stockbot.llm.extract import ExtractionResult
from stockbot.llm.fixtures import (
    ReplayClient,
    load_response_fixture,
    save_response_fixture,
)


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)

    path = save_response_fixture(
        stage="stage2",
        ticker="TEST",
        report_text="some report text",
        stop_reason="end_turn",
        usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
    )

    assert path.exists()
    data = load_response_fixture(path)
    assert data["stage"] == "stage2"
    assert data["ticker"] == "TEST"
    assert data["report_text"] == "some report text"
    assert data["stop_reason"] == "end_turn"
    assert data["usage"]["output_tokens"] == 50


def test_save_response_fixture_handles_unicode(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)
    path = save_response_fixture(
        stage="stage1",
        ticker="TEST",
        report_text="₹1,234.56 crore figures here",
        stop_reason="end_turn",
        usage={"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0},
    )
    data = load_response_fixture(path)
    assert "₹1,234.56" in data["report_text"]


def test_replay_client_create_returns_report_text(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)
    path = save_response_fixture(
        stage="stage2",
        ticker="TEST",
        report_text="```json\n{}\n```",
        stop_reason="max_tokens",
        usage={"input_tokens": 5000, "output_tokens": 16000, "cache_read_input_tokens": 0},
    )

    client = ReplayClient(path)
    response = client.messages.create(model="claude-opus-5", max_tokens=16000, messages=[])

    assert response.content[0].text == "```json\n{}\n```"
    assert response.stop_reason == "max_tokens"
    assert response.usage.input_tokens == 5000
    assert response.usage.output_tokens == 16000


def test_replay_client_stream_mimics_the_streaming_context_manager(tmp_path, monkeypatch):
    # Stage 2 switched to client.messages.stream() (see verdict.py) since
    # the SDK refuses non-streaming calls once max_tokens is high enough
    # that generation could exceed 10 minutes — ReplayClient must support
    # the same "with ... as stream: stream.get_final_message()" shape.
    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)
    path = save_response_fixture(
        stage="stage2",
        ticker="TEST",
        report_text="```json\n{}\n```",
        stop_reason="end_turn",
        usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
    )

    client = ReplayClient(path)
    with client.messages.stream(model="claude-sonnet-5", max_tokens=32000, messages=[]) as stream:
        response = stream.get_final_message()

    assert response.content[0].text == "```json\n{}\n```"
    assert response.stop_reason == "end_turn"


def test_replay_client_reproduces_a_real_truncation_fixture(tmp_path, monkeypatch):
    # This is exactly the scenario the rule was written for: a truncated
    # response, saved once, replayed as many times as needed for free.
    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)
    path = save_response_fixture(
        stage="stage2",
        ticker="JYOTHYLAB",
        report_text="partial report with no closing json fence",
        stop_reason="max_tokens",
        usage={"input_tokens": 30000, "output_tokens": 16000, "cache_read_input_tokens": 0},
    )

    client = ReplayClient(path)
    response = client.messages.create()
    assert response.stop_reason == "max_tokens"
    assert "no closing json fence" in response.content[0].text


def test_replay_client_parse_sets_parsed_output_on_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)
    valid_json = json.dumps({"auditor_opinion_type": "clean"})
    path = save_response_fixture(
        stage="stage1",
        ticker="TEST",
        report_text=valid_json,
        stop_reason="end_turn",
        usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
    )

    client = ReplayClient(path, output_format=ExtractionResult)
    response = client.messages.parse(output_format=ExtractionResult)

    assert response.parsed_output is not None
    assert response.parsed_output.auditor_opinion_type == "clean"


def test_replay_client_parse_returns_none_on_unparseable_text(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)
    path = save_response_fixture(
        stage="stage1",
        ticker="TEST",
        report_text="not valid json at all",
        stop_reason="max_tokens",
        usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
    )

    client = ReplayClient(path, output_format=ExtractionResult)
    response = client.messages.parse(output_format=ExtractionResult)
    assert response.parsed_output is None
