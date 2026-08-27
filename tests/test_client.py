"""Unit tests for Anthropic client retry helpers — no network."""

from anthropic import APIConnectionError, APIStatusError, APITimeoutError

from stockbot.llm.client import _is_retryable_anthropic_error


def test_connection_errors_are_retryable():
    assert _is_retryable_anthropic_error(APIConnectionError(request=object())) is True


def test_timeout_errors_are_retryable():
    assert _is_retryable_anthropic_error(APITimeoutError(request=object())) is True


def test_server_status_errors_are_retryable():
    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = 503
    assert _is_retryable_anthropic_error(exc) is True


def test_client_status_errors_are_not_retryable():
    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = 400
    assert _is_retryable_anthropic_error(exc) is False


def test_generic_exception_is_not_retryable():
    assert _is_retryable_anthropic_error(ValueError("nope")) is False
