"""Shutdown watchdog + loop heartbeat coverage for #66892.

The drain path is asyncio-based; a frozen loop makes every asyncio timeout
structurally unable to fire. These tests pin the out-of-loop backstop
(thread watchdog) and the loop-liveness heartbeat file contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import gateway.shutdown_watchdog as shutdown_watchdog_module
import pytest

from gateway.shutdown_watchdog import (
    DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S,
    _sweep_stale_tick_sockets,
    arm_shutdown_watchdog,
    get_loop_heartbeat_path,
    get_loop_tick_socket_path,
    get_shutdown_watchdog_dump_path,
    loop_heartbeat_forever,
    resolve_shutdown_watchdog_delay,
    write_loop_heartbeat,
)

def test_resolve_shutdown_watchdog_delay_adds_grace():
    assert resolve_shutdown_watchdog_delay(180) == 180 + DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay(0) == DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay("bad") == DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay(10, grace_s=5) == 15.0


def test_arm_shutdown_watchdog_fires_with_dump_and_exit(tmp_path):
    done = threading.Event()
    fired = threading.Event()
    dump = tmp_path / "logs" / "watchdog.log"
    snapshot_calls = []
    exit_codes = []

    def snapshot():
        snapshot_calls.append(1)
        return {"active_agents": 1, "draining": True}

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    with patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit):
        arm_shutdown_watchdog(
            0.15,
            done_event=done,
            snapshot_fn=snapshot,
            dump_path=dump,
            exit_code=9,
        )
        assert fired.wait(timeout=5.0), "watchdog did not fire"

    assert exit_codes == [9]
    assert snapshot_calls == [1]
    assert dump.is_file()
    text = dump.read_text(encoding="utf-8")
    assert "shutdown_watchdog_fired" in text
    assert "faulthandler dump" in text
    assert get_shutdown_watchdog_dump_path(tmp_path).name == "gateway-shutdown-watchdog.log"




async def _run_heartbeat_until_payload(tmp_path, timeout_s=10.0):
    """Run loop_heartbeat_forever as a task until a heartbeat payload exists.

    Returns (task, payload). Cancels the task and awaits it (suppressing
    CancelledError) before returning so the tick server is closed cleanly.
    """
    task = asyncio.ensure_future(
        loop_heartbeat_forever(interval_s=1.0, home=tmp_path)
    )
    heartbeat_path = get_loop_heartbeat_path(tmp_path)
    deadline = time.monotonic() + timeout_s
    payload = None
    while time.monotonic() < deadline:
        if heartbeat_path.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
                if payload:
                    break
                payload = None
        await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    if payload is None:
        pytest.fail(
            f"heartbeat payload did not appear at {heartbeat_path} within "
            f"{timeout_s}s"
        )
    return payload


@pytest.mark.platforms("windows")
@pytest.mark.asyncio
async def test_loop_tick_witness_arms_over_tcp_on_windows(tmp_path, caplog):
    """Non-POSIX never touches AF_UNIX; the witness arms over TCP loopback.

    Runs on native Windows (``os.name == "nt"`` for real) rather than faking
    the platform from Linux — see "Don't fake the host OS" in AGENTS.md.
    """
    start_unix_server_calls = []

    def _forbid_start_unix_server(*args, **kwargs):
        start_unix_server_calls.append((args, kwargs))
        raise AssertionError("start_unix_server must not be called on non-POSIX")

    with patch.object(
        shutdown_watchdog_module.asyncio,
        "start_unix_server",
        side_effect=_forbid_start_unix_server,
        # ``asyncio.start_unix_server`` does not exist on native Windows, so
        # without create=True patch.object itself raises AttributeError.
        # create=True arms the forbidden-call tripwire anyway and mock
        # deletes the created attribute on exit.
        create=True,
    ), caplog.at_level(logging.DEBUG, logger="gateway.shutdown_watchdog"):
        payload = await _run_heartbeat_until_payload(tmp_path)

    # (a) the AF_UNIX server was never attempted
    assert start_unix_server_calls == []
    # (b) no warning about an unavailable tick socket
    assert not [
        r
        for r in caplog.records
        if r.levelname == "WARNING"
        and "Loop tick socket unavailable" in r.getMessage()
    ]
    # (c) the witness is armed over TCP and the port is published
    assert payload["loop_tick_socket"] is True
    assert 0 < int(payload["loop_tick_tcp_port"]) <= 65535
    # (d) the POSIX socket node was never created
    assert not list(tmp_path.glob("**/gateway.loop-tick.*.sock"))


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_loop_tick_witness_arms_on_posix(tmp_path):
    payload = await _run_heartbeat_until_payload(tmp_path)
    assert payload["loop_tick_socket"] is True


# --- long HERMES_HOME: the witness socket must not depend on the home's length -------------------

_SUN_PATH_BUDGET = 100  # bytes; sockaddr_un.sun_path is 104 on macOS / 108 on Linux


def _long_home(base: Path, tag: str = "h") -> Path:
    home = base / (tag * max(1, _SUN_PATH_BUDGET + 1 - len(str(base)) - 1))
    home.mkdir()
    assert len(str(home).encode()) > _SUN_PATH_BUDGET
    return home


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _bind_node(path: Path) -> None:
    """Leave a real AF_UNIX node at ``path`` (what a SIGKILLed gateway leaves behind)."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(str(path))
    finally:
        srv.close()
    assert path.is_socket()


@pytest.mark.platforms("posix")
def test_tick_socket_path_stays_in_home_when_it_fits(tmp_path):
    with tempfile.TemporaryDirectory(prefix="hwd-") as short:
        path = get_loop_tick_socket_path(Path(short), 4242)
    assert path == Path(short) / "state" / "gateway.loop-tick.4242.sock"


@pytest.mark.platforms("posix")
def test_tick_socket_path_falls_back_to_short_tmp_path_for_long_home(tmp_path):
    home = _long_home(tmp_path)
    path = get_loop_tick_socket_path(home, 4242)
    assert len(str(path).encode()) <= _SUN_PATH_BUDGET
    assert path.parent in {Path(tempfile.gettempdir()), Path("/tmp")}
    assert path.name.startswith("hermes-lt-") and path.name.endswith(".4242.sock")
    # per-home and per-pid: another home or pid never collides
    assert get_loop_tick_socket_path(home, 4243) != path
    assert get_loop_tick_socket_path(_long_home(tmp_path, "g"), 4242) != path


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_loop_tick_witness_arms_and_answers_with_long_home(tmp_path, caplog):
    from hermes_cli.gateway import _probe_loop_tick_socket

    home = _long_home(tmp_path)
    caplog.set_level(logging.DEBUG, logger="gateway.shutdown_watchdog")
    task = asyncio.ensure_future(loop_heartbeat_forever(interval_s=1.0, home=home))
    try:
        heartbeat_path = get_loop_heartbeat_path(home)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not heartbeat_path.is_file():
            await asyncio.sleep(0.05)
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8-sig"))
        node = get_loop_tick_socket_path(home)
        assert payload["loop_tick_socket"] is True
        assert not [r for r in caplog.records
                    if r.levelname == "WARNING" and "Loop tick socket unavailable" in r.getMessage()]
        assert node.is_socket()
        assert (node.stat().st_mode & 0o077) == 0, "fallback node must not be group/world accessible"
        assert await asyncio.to_thread(_probe_loop_tick_socket, os.getpid(), home) is True
        # a node owned by another uid is not evidence: the fallback lives in a shared temp dir
        real_uid = os.getuid()  # windows-footgun: ok — posix-gated test
        with patch.object(os, "getuid", return_value=real_uid + 1):
            assert await asyncio.to_thread(_probe_loop_tick_socket, os.getpid(), home) is None
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert not node.exists(), "own node must be unlinked on shutdown"


@pytest.mark.platforms("posix")
def test_sweep_removes_dead_pid_fallback_socket_and_keeps_live_one(tmp_path):
    home = _long_home(tmp_path)
    own = get_loop_tick_socket_path(home, os.getpid())
    dead = get_loop_tick_socket_path(home, _dead_pid())
    foreign = get_loop_tick_socket_path(_long_home(tmp_path, "g"), _dead_pid())
    live_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    live = get_loop_tick_socket_path(home, live_proc.pid)
    nodes = [own, dead, live, foreign]
    try:
        for node in nodes:
            _bind_node(node)
        _sweep_stale_tick_sockets(own)
        assert not dead.exists()
        assert live.exists()
        assert own.exists()
        assert foreign.exists(), "another home's node (different hash) is not ours to sweep"
    finally:
        live_proc.kill()
        live_proc.wait()
        for node in nodes:
            node.unlink(missing_ok=True)
