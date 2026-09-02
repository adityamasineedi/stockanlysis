"""Cooperative cancel for long-running bot operations (/stop).

Uses a threading.Event so sync pipeline code in worker threads can observe
cancellation without asyncio coupling.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cancel = threading.Event()
_operation: str | None = None


class OperationCancelled(Exception):
    """Raised when the user sends /stop during a cancellable operation."""


def begin_operation(label: str) -> None:
    global _operation
    with _lock:
        _cancel.clear()
        _operation = label
    logger.info("operation started: %s", label)


def end_operation() -> None:
    with _lock:
        global _operation
        _operation = None
        _cancel.clear()


def active_operation() -> str | None:
    with _lock:
        return _operation


def request_cancel() -> str | None:
    """Signal stop. Returns the operation label when something was running."""
    with _lock:
        if _operation is None:
            return None
        _cancel.set()
        label = _operation
    logger.info("cancel requested for operation: %s", label)
    return label


def cancel_requested() -> bool:
    return _cancel.is_set()


def raise_if_cancelled() -> None:
    if _cancel.is_set():
        raise OperationCancelled("cancelled by user")
