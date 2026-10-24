import asyncio
import re
from typing import Any

import pytest
from textual.app import App
from textual.drivers.headless_driver import HeadlessDriver

from open_trading_agents.analysis.models import AgentEvent, EventSink
from open_trading_agents.ui.app import TradingApp

SAVE_TITLE = "\x1b[22;0t"
RESTORE_TITLE = "\x1b[23;0t"


class RecordingDriver(HeadlessDriver):
    def __init__(
        self,
        app: App[Any],
        *,
        debug: bool = False,
        mouse: bool = True,
        size: tuple[int, int] | None = None,
    ) -> None:
        """Capture terminal-title controls without touching the user's terminal.

        Args:
            app: Application using this isolated driver.
            debug: Framework debug setting.
            mouse: Framework mouse setting.
            size: Fixed test terminal dimensions.
        """
        super().__init__(app, debug=debug, mouse=mouse, size=size)
        self.title_controls: list[str] = []

    @property
    def is_headless(self) -> bool:
        """Enable the terminal-output path while retaining the headless driver's isolation."""
        return False

    def write(self, data: str) -> None:
        """Retain title controls and discard ordinary screen rendering.

        Args:
            data: Raw terminal output produced by the application.
        """
        if "\x1b]0;" in data or SAVE_TITLE in data or RESTORE_TITLE in data:
            self.title_controls.append(data)

    @property
    def titles(self) -> list[str]:
        """Return the title payloads actually written through the terminal driver."""
        return re.findall(r"\x1b\]0;([^\x07]*)\x07", "".join(self.title_controls))


@pytest.mark.parametrize("terminal_event,title", [("run_end", "Open Trading Agents · NVDA"), ("error", "Failed · Open Trading Agents · NVDA")])
async def test_title_animates_during_silence_and_stops_at_terminal_state(terminal_event: str, title: str) -> None:
    """Animate actual terminal titles without new model events or transcript reindexing.

    Args:
        terminal_event: Workflow completion or failure event.
        title: Expected stable title after that event.
    """
    app = TradingApp(ticker="NVDA")
    app.driver_class = RecordingDriver

    async with app.run_test(headless=False, size=(120, 36)) as pilot:
        driver = app._driver
        assert isinstance(driver, RecordingDriver)
        await app.receive(AgentEvent("stage_start", "Workflow", "analysis"))
        await app.receive(AgentEvent("text", "Market Analyst", "analysis", "Evidence\n" * 100))
        app.flush_events()
        await pilot.pause()
        offsets = app.agent_views["Market Analyst"]._offsets
        first = len(driver.titles)

        await pilot.pause(0.35)

        active_titles = driver.titles[first:]
        assert len(set(active_titles)) >= 2
        assert all("Open Trading Agents" in title and "NVDA" in title for title in active_titles)
        assert all(title.split(" ", 1)[0] in app._running_spinner.frames for title in active_titles)
        assert app.agent_views["Market Analyst"]._offsets is offsets

        await app.receive(AgentEvent(terminal_event, "Workflow", "reports", "Provider failed" if terminal_event == "error" else ""))
        app.flush_events()
        assert driver.titles[-1] == title
        written = len(driver.titles)

        await pilot.pause(0.25)

        assert len(driver.titles) == written

    controls = "".join(driver.title_controls)
    assert controls.count(SAVE_TITLE) == controls.count(RESTORE_TITLE) == 1
    assert controls.endswith(RESTORE_TITLE)


async def test_quit_restores_the_title_after_cancelling_the_workflow() -> None:
    """Stop title animation and restore the prior title after owned work finishes cancelling."""
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def run(sink: EventSink) -> None:
        """Remain active until quit cancels this isolated workflow.

        Args:
            sink: Workflow event receiver, unused by this silent fixture.
        """
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    app = TradingApp(run, ticker="NVDA")
    app.driver_class = RecordingDriver

    async with app.run_test(headless=False) as pilot:
        driver = app._driver
        assert isinstance(driver, RecordingDriver)
        await started.wait()
        await pilot.pause(0.15)
        assert driver.titles[-1].split(" ", 1)[0] in app._running_spinner.frames

        await pilot.press("q")

    assert cleaned.is_set()
    assert any("Stopped" in title for title in driver.titles)
    controls = "".join(driver.title_controls)
    assert controls.count(SAVE_TITLE) == controls.count(RESTORE_TITLE) == 1
    assert controls.endswith(RESTORE_TITLE)


async def test_headless_runs_emit_no_terminal_title_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep nonterminal output free of title escape sequences.

    Args:
        monkeypatch: Capture writes made by the normal headless driver.
    """
    writes: list[str] = []

    def capture(driver: HeadlessDriver, data: str) -> None:
        """Observe raw output without forwarding it to stdout.

        Args:
            driver: Framework driver receiving the data.
            data: Raw terminal output.
        """
        writes.append(data)

    monkeypatch.setattr(HeadlessDriver, "write", capture)
    app = TradingApp(ticker="NVDA")

    async with app.run_test() as pilot:
        await app.receive(AgentEvent("stage_start", "Workflow", "analysis"))
        app.flush_events()
        await pilot.pause(0.15)
        await app.receive(AgentEvent("run_end", "Workflow", "reports"))
        app.flush_events()

    assert not any("\x1b]0;" in data or SAVE_TITLE in data or RESTORE_TITLE in data for data in writes)


async def test_title_payload_cannot_contain_terminal_control_characters() -> None:
    """Keep a malformed display ticker from terminating or injecting a title control."""
    app = TradingApp(ticker="NVDA")
    app.driver_class = RecordingDriver

    async with app.run_test(headless=False) as pilot:
        driver = app._driver
        assert isinstance(driver, RecordingDriver)
        await pilot.pause()
        app.ticker = "NVDA\x1b]0;injected\x07\n\x9c"

        app._refresh_terminal_title("⠋")
        title = driver.titles[-1]

        assert "NVDA" in title
        assert all(character.isprintable() for character in title)
        assert driver.title_controls[-1].count("\x1b]0;") == 1
        app.ticker = "NVDA"
