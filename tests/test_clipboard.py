import subprocess
from unittest.mock import Mock

import pytest

from open_trading_agents.ui import clipboard


@pytest.fixture
def desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate display discovery from the machine running the tests.

    Args:
        monkeypatch: Temporary environment and platform replacements.
    """
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(clipboard.sys, "platform", "linux")


def test_wayland_failure_falls_back_to_x11_clipboard(desktop: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass Unicode as stdin and use CLIPBOARD rather than the primary selection.

    Args:
        desktop: Isolated desktop environment.
        monkeypatch: Isolated process runner.
    """
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: f"/usr/bin/{name}")
    run = Mock(side_effect=[subprocess.TimeoutExpired("wl-copy", 3), subprocess.CompletedProcess([], 0)])
    monkeypatch.setattr(clipboard.subprocess, "run", run)

    text = "Prices 한국어\n$(no shell execution)"
    assert clipboard.copy_desktop_text(text)
    assert run.call_args.args[0] == ("/usr/bin/xclip", "-selection", "clipboard", "-in")
    assert run.call_args.kwargs["input"] == text.encode()
    assert run.call_count == 2


@pytest.mark.parametrize("remote", ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"])
def test_remote_session_does_not_write_server_clipboard(desktop: None, monkeypatch: pytest.MonkeyPatch, remote: str) -> None:
    """Keep remote copies on the terminal path even when the server has a display.

    Args:
        desktop: Isolated desktop environment.
        monkeypatch: Isolated environment and process runner.
        remote: SSH environment marker.
    """
    monkeypatch.setenv(remote, "set")
    monkeypatch.setenv("DISPLAY", ":0")
    run = Mock()
    monkeypatch.setattr(clipboard.subprocess, "run", run)
    assert not clipboard.copy_desktop_text("text")
    run.assert_not_called()


def test_missing_helpers_leave_terminal_clipboard_available(desktop: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing local utilities are a normal fallback, not a TUI error.

    Args:
        desktop: Isolated desktop environment.
        monkeypatch: Simulated missing executables.
    """
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    assert not clipboard.copy_desktop_text("text")
