"""Regression tests for session attribution inside the streaming worker thread.

``interruptible_streaming_api_call`` runs the provider call on a worker thread.
The ``[session]`` tag that every log line carries comes from a
``threading.local`` in :mod:`hermes_logging`, so that thread starts unbound and
everything it logs is formatted WITHOUT a session tag.

In a process serving concurrent sessions this is the difference between a
diagnosable failure and an anonymous one: the worker is where a stream failure
is logged, so the one line carrying the cause is the one line that names no
session.

These tests assert the binding's contract: it attributes the worker's records
to the agent's session, it never leaks into the caller's thread, and it stays
best-effort (a session-less agent or a broken logging module must not raise
inside the worker).
"""

import contextlib
import importlib
import logging
import threading

from agent.chat_completion_helpers import bind_worker_session_context

SESSION_ID = "20260101_000000_abcdef01"


def _live_hermes_logging():
    """Resolve ``hermes_logging`` the way the production helper does.

    Other tests in this suite reload modules, so the object bound at import
    time is not necessarily the one ``bind_worker_session_context`` writes
    to.  Going through ``sys.modules`` on every call keeps these tests
    order-independent.
    """
    return importlib.import_module("hermes_logging")


class _Agent:
    """Minimal agent-like object — only ``session_id`` matters here."""

    def __init__(self, session_id=None):
        if session_id is not None:
            self.session_id = session_id


class _SessionLessAgent:
    """An agent object that doesn't expose ``session_id`` at all."""


@contextlib.contextmanager
def _live_record_factory():
    """Reinstall the session record factory from the live module.

    The factory is a process-global installed at import time, so a stale
    wrapper left behind by a module reload would read a different
    thread-local than the one the binding writes.  The previous factory is
    restored on exit.
    """
    previous_factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    _live_hermes_logging()._install_session_record_factory()
    try:
        yield
    finally:
        logging.setLogRecordFactory(previous_factory)


def _session_tag_of_a_new_record():
    """``session_tag`` of a record created on the CURRENT thread."""
    record = logging.getLogger("stream-worker-test").makeRecord(
        "stream-worker-test", logging.INFO, __file__, 1, "msg", None, None
    )
    return getattr(record, "session_tag", None)


def _session_tag_in_worker(agent):
    """Bind + emit one LogRecord on a fresh thread; return its ``session_tag``."""
    captured = {}

    def _worker():
        bind_worker_session_context(agent)
        captured["tag"] = _session_tag_of_a_new_record()

    with _live_record_factory():
        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join()
    return captured["tag"]


def test_worker_records_carry_the_agent_session_tag():
    # Regression: this record used to be emitted unbound, so a stream failure
    # in one of several concurrent sessions could not be attributed to any.
    assert _session_tag_in_worker(_Agent(SESSION_ID)) == f" [{SESSION_ID}]"


def test_binding_does_not_leak_into_the_calling_thread():
    _live_hermes_logging().clear_session_context()
    _session_tag_in_worker(_Agent(SESSION_ID))
    with _live_record_factory():
        assert _session_tag_of_a_new_record() == ""


def test_session_less_agent_leaves_the_worker_unbound():
    assert _session_tag_in_worker(_SessionLessAgent()) == ""
    assert _session_tag_in_worker(_Agent(None)) == ""


def test_binding_swallows_logging_failures(monkeypatch):
    def _boom(_session_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(_live_hermes_logging(), "set_session_context", _boom)
    # Best-effort by contract: the worker must run the API call even when the
    # logging module is unusable.
    assert _session_tag_in_worker(_Agent(SESSION_ID)) == ""
