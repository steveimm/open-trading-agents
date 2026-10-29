import asyncio
from pathlib import Path

import pytest
from textual.widgets import Input, Static, Tabs

from open_trading_agents.analysis.chat import ReportChat
from open_trading_agents.analysis.models import AgentEvent, AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.dataflows.tools import TOOL_NAMES
from open_trading_agents.ui.app import TradingApp


class Session:
    def __init__(self, emit: EventSink) -> None:
        """Record native conversation turns and stream fixture responses."""
        self.session_id = "chat-session"
        self.emit = emit
        self.prompts: list[str] = []
        self.closed = False
        self.wait: asyncio.Event | None = None

    async def prompt(self, text: str) -> AgentResult:
        """Stream one answer while optionally holding the response open."""
        self.prompts.append(text)
        await self.emit(AgentEvent("prompt", "Report Assistant", "chat", text))
        await self.emit(AgentEvent("text", "Report Assistant", "chat", "Answer"))
        if self.wait is not None:
            await self.wait.wait()
        return AgentResult("Answer", self.session_id)

    async def close(self) -> None:
        """Record native resource cleanup."""
        self.closed = True


@pytest.mark.parametrize("allow_execution,web_research", [(True, True), (True, False), (False, True), (False, False)])
async def test_report_context_and_native_history(tmp_path: Path, allow_execution: bool, web_research: bool) -> None:
    """Supply the entire report once, preserve subsequent turns, and display only user questions."""
    reports = tmp_path / "reports"
    reports.mkdir()
    report = "# Full report\n" + "Evidence\n" * 2000 + "Final recommendation"
    (reports / "complete_report.md").write_text(report)
    opened = []
    events = []

    async def emit(event: AgentEvent) -> None:
        """Collect public chat events."""
        events.append(event)

    async def factory(config: HarnessConfig, spec: AgentSpec, sink: EventSink) -> Session:
        """Give chat all analyst tools through the existing run context."""
        expected_tools = set(TOOL_NAMES) if web_research else set(TOOL_NAMES) - {"search_web", "fetch_webpage"}
        assert spec.allow_execution == allow_execution and set(spec.tools) == expected_tools
        command = spec.tool_server_command
        assert command[command.index("--context") + 1] == str(tmp_path / "tool-context.json")
        assert command[command.index("--artifact-dir") + 1] == str(tmp_path / "chat" / "data")
        assert set(command[command.index("--tools") + 1].split(",")) == expected_tools
        assert spec.system_prompt == chat.system_prompt
        assert "The first user message contains the report followed by a question." in spec.system_prompt
        assert "Later messages continue the same conversation." in spec.system_prompt
        assert "Respond in Korean unless the user requests another language." in spec.system_prompt
        assert "limit requests to its analysis date" in spec.system_prompt
        assert "required JSON" not in spec.system_prompt
        assert "rating values in English" not in spec.system_prompt
        assert "Return your report or decision once" not in spec.system_prompt
        assert "exports the reports" not in spec.system_prompt
        assert "Write the report in" not in spec.system_prompt
        for term in ("analyst", "workflow", "harness", "portfolio manager"):
            assert term not in spec.system_prompt.lower()
        assert "Financial data tools provide price history" in spec.system_prompt
        assert ("Code execution is unavailable" in spec.system_prompt) == (not allow_execution)
        assert ("Web search and page retrieval are unavailable" in spec.system_prompt) == (not web_research)
        if not allow_execution:
            assert "You can execute code" not in spec.system_prompt
        session = Session(sink)
        opened.append(session)
        return session

    chat = ReportChat(tmp_path, HarnessConfig(), "Korean", factory, allow_execution=allow_execution, web_research=web_research)
    await chat.ask("What are the risks?", emit)
    await chat.ask("Explain the second risk.", emit)
    assert len(opened) == 1
    assert report in opened[0].prompts[0]
    assert opened[0].prompts[1] == "Explain the second risk."
    assert [event.text for event in events if event.kind == "prompt"] == ["What are the risks?", "Explain the second risk."]
    await chat.close()
    assert opened[0].closed


@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_report_actions_and_chat_typing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]) -> None:
    """Open the saved HTML, type ordinary shortcut characters, and prevent overlapping chat requests."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "complete_report.md").write_text("# Complete report\nEvidence")
    (reports / "complete_report.html").write_text("<h1>Complete report</h1>")
    opened = []
    sessions = []
    monkeypatch.setattr("open_trading_agents.ui.app.webbrowser.open_new_tab", lambda uri: opened.append(uri) or True)

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> Session:
        """Hold the response open to exercise duplicate submission and shutdown."""
        session = Session(emit)
        session.wait = asyncio.Event()
        sessions.append(session)
        return session

    chat = ReportChat(tmp_path, HarnessConfig(), "English", factory)
    app = TradingApp(run_dir=tmp_path, report_chat=chat)
    async with app.run_test(size=size) as pilot:
        assert not app.query("#open-report")
        assert not app.check_action("open_report", ())
        assert app.query_one("#chat-input", Input).disabled
        await app.receive(AgentEvent("run_end", "Workflow", "reports"))
        app.flush_events()
        await pilot.pause()
        app.query_one("#stage-tabs", Tabs).active = "tab-reports"
        await pilot.pause()
        assert "b open in browser" in str(app.query_one("#shortcuts", Static).render())
        assert "Total" not in str(app.query_one("#run-status", Static).render())
        assert "00:" not in str(app.query_one("#stage-status", Static).render())
        await pilot.press("b")
        assert opened == [(reports / "complete_report.html").as_uri()]
        app.query_one("#stage-tabs", Tabs).active = "tab-chat"
        await pilot.pause()
        field = app.query_one("#chat-input", Input)
        assert not app.check_action("open_report", ())
        assert "open in browser" not in str(app.query_one("#shortcuts", Static).render())
        assert not field.has_focus
        assert "q quit" in str(app.query_one("#shortcuts", Static).render())
        assert "Ctrl+C" not in str(app.query_one("#shortcuts", Static).render())
        assert "Total" not in str(app.query_one("#run-status", Static).render())
        assert "00:" not in str(app.query_one("#stage-status", Static).render())
        assert chat.system_prompt in app.stage_views["chat"].visible_plain()
        await pilot.press("enter")
        assert field.has_focus
        assert field.region.bottom <= size[1]
        assert not app.query("Button")
        assert field.region.right <= size[0]
        assert field.region.y >= app.query_one("#chat-conversation").region.bottom
        assert app.stage_views["chat"].size.height > 0
        await pilot.press("b", "q", "f", "m", "space", "left_square_bracket", "right_square_bracket", "left", "right")
        assert field.value == "bqfm []"
        await pilot.press("enter")
        await pilot.pause()
        assert len(sessions) == 1 and len(sessions[0].prompts) == 1
        assert "Answer" in app.stage_views["chat"].plain()
        assert field.has_focus and field.value == ""
        await pilot.press(*"second question")
        assert field.value == "second question"
        await pilot.press("enter")
        assert len(sessions[0].prompts) == 1
        sessions[0].wait.set()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert sessions[0].prompts[1] == "second question"
        assert field.has_focus and field.value == ""
        assert app.error is None
    assert sessions[0].closed


@pytest.mark.parametrize("quit_keys", [("escape", "q"), ("ctrl+c",)])
async def test_chat_failure_and_quit_leave_report_intact(tmp_path: Path, quit_keys: tuple[str, ...]) -> None:
    """Keep failed chat independent of the research result and close an interrupted response."""
    reports = tmp_path / "reports"
    reports.mkdir()
    report = "# Saved report"
    (reports / "complete_report.md").write_text(report)
    session = None
    attempts = 0

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> Session:
        """Fail startup once, then keep a response in flight for cancellation."""
        nonlocal attempts, session
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Model unavailable")
        session = Session(emit)
        session.wait = asyncio.Event()
        return session

    chat = ReportChat(tmp_path, HarnessConfig(), "English", factory)
    app = TradingApp(run_dir=tmp_path, report_chat=chat)
    async with app.run_test() as pilot:
        await app.receive(AgentEvent("run_end", "Workflow", "reports"))
        app.flush_events()
        await pilot.pause()
        app.query_one("#stage-tabs", Tabs).active = "tab-chat"
        await pilot.pause()
        field = app.query_one("#chat-input", Input)
        await pilot.press("enter")
        field.value = "Question"
        await pilot.press("enter")
        await pilot.pause()
        assert "Model unavailable" in app.stage_views["chat"].plain()
        assert app.error is None and app._failure is None
        assert field.has_focus
        assert not app.check_action("compose_chat", ())
        field.value = "Retry"
        await pilot.press("enter")
        await pilot.pause()
        assert session is not None and not session.closed
        assert field.has_focus
        assert "ctrl+c" not in str(app.query_one("#shortcuts", Static).render()).lower()
        await pilot.press(*quit_keys)
        assert session.closed
    assert (reports / "complete_report.md").read_text() == report


@pytest.mark.parametrize("raises", [False, True])
async def test_browser_failure_keeps_report_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raises: bool) -> None:
    """Report browser failures in the UI without failing the completed analysis."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "complete_report.md").write_text("Report")
    (reports / "complete_report.html").write_text("Report")
    notices = []

    def open_page(uri: str) -> bool:
        """Simulate a missing browser or failed browser command."""
        if raises:
            raise OSError("No browser")
        return False

    monkeypatch.setattr("open_trading_agents.ui.app.webbrowser.open_new_tab", open_page)
    app = TradingApp(run_dir=tmp_path)
    monkeypatch.setattr(app, "notify", lambda message, **kwargs: notices.append(message))
    async with app.run_test() as pilot:
        await app.receive(AgentEvent("run_end", "Workflow", "reports"))
        app.flush_events()
        await pilot.pause()
        app.query_one("#stage-tabs", Tabs).active = "tab-reports"
        await pilot.pause()
        assert "b open in browser" in str(app.query_one("#shortcuts", Static).render())
        await pilot.press("b")
        assert notices and "Could not open a browser" in notices[0]
        assert app.error is None and app.report_document.plain() == "Report"


async def test_chat_respects_execution_web_and_budget_settings(tmp_path: Path) -> None:
    """Apply run-level tool restrictions to the chat native session."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "complete_report.md").write_text("Report")

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> Session:
        """Check the native capabilities after run-level restrictions are applied."""
        assert not spec.allow_execution
        assert set(spec.tools) == set(TOOL_NAMES) - {"search_web", "fetch_webpage"}
        assert spec.max_tool_rounds == 7
        return Session(emit)

    async def emit(event: AgentEvent) -> None:
        """Accept fixture events without a terminal."""

    chat = ReportChat(tmp_path, HarnessConfig(), "English", factory, allow_execution=False, web_research=False, max_tool_rounds=7)
    try:
        await chat.ask("Explain", emit)
    finally:
        await chat.close()
