import asyncio
import io
import tomllib
from pathlib import Path
from typing import Any

import pytest
import typer
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.plain_text import PlainTextOutput
from typer.testing import CliRunner

from open_trading_agents.ui import configuration as configuration_ui

application = typer.Typer()
runner = CliRunner()


@application.command()
def configure(path: Path) -> None:
    """Exercise the data editor through Typer's real input and masking handling.

    Args:
        path: Isolated configuration destination.
    """
    configuration_ui.configure_data_interactively(path)


@pytest.fixture
def configuration_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate application storage and reject agent setup during data configuration.

    Args:
        tmp_path: Temporary user home and configuration destination.
        monkeypatch: Scoped home and discovery replacements.

    Returns:
        An initially absent configuration path.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def unexpected(*args: Any, **kwargs: Any) -> None:
        """Reject profile lookup or model discovery from the data-only editor."""
        raise AssertionError("Data configuration tried to configure an agent")

    monkeypatch.setattr(configuration_ui, "list_presets", unexpected)
    monkeypatch.setattr(configuration_ui, "discover_models", unexpected)
    return tmp_path / ".open-trading-agents" / "config.toml"


def select_menus(monkeypatch: pytest.MonkeyPatch, path: Path, answers: list[int]) -> list[tuple[str, list[str]]]:
    """Select menu rows while checking that no edits reach disk before Save.

    Args:
        monkeypatch: Scoped menu replacement.
        path: Configuration file that must remain unchanged during editing.
        answers: Selected indexes, with exhaustion simulating cancellation.

    Returns:
        Presented menu labels and rows for privacy checks.
    """
    original = path.read_bytes() if path.exists() else None
    remaining = iter(answers)
    shown: list[tuple[str, list[str]]] = []

    def choose(label: str, choices: list[str], default: int = 0) -> int:
        """Retain visible choices and select a row without changing the source file."""
        assert (path.read_bytes() if path.exists() else None) == original
        shown.append((label, choices))
        if label == "Data sources":
            assert choices == ["Credentials", "Source selection", "Save"]
        if label == "Credentials":
            assert choices[-1] == "Back"
            assert all(choice.rsplit(" · ", 1)[1] == "Not set" or choice.rsplit(" · ", 1)[1].startswith("*") for choice in choices[:2])
            assert choices[2].endswith((" · Set", " · Not set"))
        try:
            return next(remaining)
        except StopIteration as exc:
            raise typer.Abort() from exc

    monkeypatch.setattr(configuration_ui, "_choose", choose)
    return shown


def record_prompts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Observe real prompt options without replacing their input or hiding behavior.

    Args:
        monkeypatch: Scoped Typer prompt wrapper.

    Returns:
        Prompt labels and options supplied by the editor.
    """
    original = configuration_ui.typer.prompt
    received: list[tuple[str, dict[str, Any]]] = []

    def prompt(label: str, **options: Any) -> str:
        """Record prompt options before delegating to Typer's real input handling."""
        received.append((label, options))
        return original(label, **options)

    monkeypatch.setattr(configuration_ui.typer, "prompt", prompt)
    return received


def test_data_editor_masks_keys_and_saves_all_fields_without_agent_setup(
    configuration_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save optional data access while keeping API keys out of prompts and menu summaries.

    Args:
        configuration_path: Isolated destination with no saved presets.
        monkeypatch: Scoped input adapters.
    """
    shown = select_menus(monkeypatch, configuration_path, [0, 0, 1, 2, 3, 2])
    prompts = record_prompts(monkeypatch)
    keys = iter(["fred-private-key", "alpha-private-key"])
    monkeypatch.setattr(configuration_ui, "_api_key_prompt", lambda label: next(keys))
    result = runner.invoke(application, [str(configuration_path)], input="Research Team contact@example.com\n")

    assert result.exit_code == 0, result.output
    assert tomllib.loads(configuration_path.read_text())["data"] == {
        "fred_api_key": "fred-private-key",
        "alpha_vantage_api_key": "alpha-private-key",
        "sec_edgar_user_agent": "Research Team contact@example.com",
    }
    assert [label for label, _ in prompts] == ["SEC EDGAR contact identity"]
    assert all(not options.get("hide_input", False) for _, options in prompts)
    assert all(options["default"] == "" and options["show_default"] is False for _, options in prompts)
    assert "fred-private-key" not in result.output and "alpha-private-key" not in result.output
    menu_text = str(shown)
    assert not any(value in menu_text for value in ("fred-private-key", "alpha-private-key", "contact@example.com"))
    assert [choices for label, choices in shown if label == "Credentials"][-1][:2] == ["FRED (macro data) · ****e-key", "Alpha Vantage (alternative market/news data) · ****e-key"]
    assert "macroeconomic series" in result.output and "alternative source" in result.output and "contact email" in result.output
    assert "Saved data sources:" in result.output
    assert not (configuration_path.parent / "presets").exists()


def test_data_editor_retains_blank_updates_removes_entries_and_preserves_other_settings(
    configuration_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edit existing credentials atomically without echoing them or replacing unrelated preferences.

    Args:
        configuration_path: Isolated configuration with existing settings and credentials.
        monkeypatch: Scoped input adapters.
    """
    configuration_path.parent.mkdir()
    configuration_path.write_text('''# retained preferences
output_language = "Korean"
[harness]
kind = "pi"
model = "unchanged-model"
[data]
fred_api_key = "existing-fred-key"
alpha_vantage_api_key = "existing-alpha-key"
sec_edgar_user_agent = "Existing Contact old@example.com"
web_search_backend = "yahoo"
[data.data_vendors]
news_data = "yfinance"
''')
    shown = select_menus(monkeypatch, configuration_path, [0, 0, 0, 1, 0, 2, 2, 2, 1, 3, 2])
    keys = iter(["", "replacement-alpha-key"])
    monkeypatch.setattr(configuration_ui, "_api_key_prompt", lambda label: next(keys))
    result = runner.invoke(application, [str(configuration_path)])

    assert result.exit_code == 0, result.output
    saved = tomllib.loads(configuration_path.read_text())
    assert saved["data"]["fred_api_key"] == "existing-fred-key"
    assert saved["data"]["alpha_vantage_api_key"] == "replacement-alpha-key"
    assert "sec_edgar_user_agent" not in saved["data"]
    assert saved["output_language"] == "Korean" and saved["harness"] == {"kind": "pi", "model": "unchanged-model"}
    assert saved["data"]["web_search_backend"] == "yahoo"
    assert saved["data"]["data_vendors"] == {"news_data": "yfinance"}
    assert "# retained preferences" in configuration_path.read_text()
    visible = result.output + str(shown)
    assert not any(value in visible for value in ("existing-fred-key", "existing-alpha-key", "replacement-alpha-key", "old@example.com"))
    assert [choices for label, choices in shown if label == "Credentials"][-1][2] == "SEC EDGAR (contact identity) · Not set"
    assert all(choices == ["Update", "Remove", "Back"] for label, choices in shown if label not in {"Data sources", "Credentials"})


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("cancel_at", ["menu", "prompt"])
def test_cancel_after_editing_leaves_configuration_unchanged(
    configuration_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool, cancel_at: str,
) -> None:
    """Discard pending credentials when setup is cancelled at either kind of input.

    Args:
        configuration_path: Isolated configuration destination.
        monkeypatch: Scoped input adapters.
        existing: Whether a configuration file already exists.
        cancel_at: Menu selection or the next text prompt.
    """
    original = b'output_language = "Korean"\n' if existing else None
    if original is not None:
        configuration_path.parent.mkdir()
        configuration_path.write_bytes(original)
    answers = [0, 0] if cancel_at == "menu" else [0, 0, 1]
    select_menus(monkeypatch, configuration_path, answers)
    def cancel_prompt(label: str) -> str:
        """Interrupt the second prompt after one credential has been staged."""
        if label == "Alpha Vantage API key":
            raise typer.Abort()
        return "discarded-private-key"

    monkeypatch.setattr(configuration_ui, "_api_key_prompt", cancel_prompt)
    result = runner.invoke(application, [str(configuration_path)])

    assert result.exit_code != 0
    assert (configuration_path.read_bytes() if configuration_path.exists() else None) == original
    assert "discarded-private-key" not in result.output
    assert "Saved data sources:" not in result.output
    if not existing:
        assert not configuration_path.parent.exists()


@pytest.mark.parametrize("actions", [
    [("a", "a"), ("bcde", "abcde"), ("f", "abcdef")],
    [("\x1b[200~prefix-secret-ABCDE\x1b[201~", "prefix-secret-ABCDE"),
     ("\x7f", "prefix-secret-ABCD"), ("\x1b[D\x1b[Dx", "prefix-secret-ABxCD"),
     ("\x01z", "zprefix-secret-ABxCD"), ("\x05\x7fQ", "zprefix-secret-ABxCQ")],
])
async def test_api_key_input_masks_live_typing_paste_and_editing(
    monkeypatch: pytest.MonkeyPatch, actions: list[tuple[str, str]],
) -> None:
    """Render masked live input while keeping pasted and edited credentials unchanged.

    Args:
        monkeypatch: Scoped observation of real Questionary prompts.
        actions: Keyboard chunks and the original buffer value expected after each edit.
    """
    output = io.StringIO()
    loop = asyncio.get_running_loop()
    started, updated = asyncio.Event(), asyncio.Event()
    frames: list[tuple[str, str]] = []
    questions = []
    original = configuration_ui.questionary.text

    def question(label: str, **options: Any) -> Any:
        """Observe rendered frames from the actual masked input widget."""
        assert options["default"] == "" and options["erase_when_done"] is True and options["is_password"] is True
        result = original(label, **options)
        assert result.application.current_buffer.text == ""

        def rendered(app: Any) -> None:
            """Record the visible screen after rendering each buffer value."""
            screen = app.renderer.last_rendered_screen
            if screen is not None:
                visible = "\n".join("".join(character.char for _, character in sorted(row.items()))
                                    for _, row in sorted(screen.data_buffer.items()))
                frames.append((app.current_buffer.text, visible))
                loop.call_soon_threadsafe(updated.set)

        result.application.after_render += rendered
        questions.append(result)
        loop.call_soon_threadsafe(started.set)
        return result

    monkeypatch.setattr(configuration_ui.questionary, "text", question)
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=PlainTextOutput(output)):
        pending = asyncio.create_task(asyncio.to_thread(configuration_ui._api_key_prompt, "API key"))
        await asyncio.wait_for(started.wait(), 3)
        try:
            for keys, expected in actions:
                keyboard.send_text(keys)
                async with asyncio.timeout(3):
                    while True:
                        updated.clear()
                        if any(value == expected for value, _ in frames):
                            break
                        await updated.wait()
                visible = next(screen for value, screen in reversed(frames) if value == expected)
                preview = "*" * len(expected) if len(expected) <= 5 else f"****{expected[-5:]}"
                assert preview in visible
                assert expected not in visible
            keyboard.send_text("\r")
            assert await asyncio.wait_for(pending, 3) == actions[-1][1]
        finally:
            if not pending.done():
                keyboard.send_text("\x03")
                with pytest.raises(typer.Abort):
                    await asyncio.wait_for(pending, 3)

    assert questions[0].application.erase_when_done
    assert actions[-1][1] not in output.getvalue()
    assert "prefix-secret" not in output.getvalue()


def test_api_key_input_blank_and_cancel_do_not_reveal_or_prefill_existing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fresh secret entry blank and preserve cancellation through the real keyboard input.

    Args:
        monkeypatch: Scoped prompt observation.
    """
    original = configuration_ui.questionary.text

    def question(label: str, **options: Any) -> Any:
        """Check empty initial input and erased completion on each actual prompt."""
        assert options["default"] == "" and options["erase_when_done"] is True
        return original(label, **options)

    monkeypatch.setattr(configuration_ui.questionary, "text", question)
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=PlainTextOutput(io.StringIO())):
        keyboard.send_text("\r")
        assert configuration_ui._api_key_prompt("API key") == ""
        keyboard.send_text("\x03")
        with pytest.raises(typer.Abort):
            configuration_ui._api_key_prompt("API key")


@pytest.mark.parametrize("save", [True, False])
def test_source_selection_applies_fallbacks_and_replaces_only_related_overrides(
    configuration_path: Path, monkeypatch: pytest.MonkeyPatch, save: bool,
) -> None:
    """Save a usable Alpha Vantage chain with its key, or discard the whole edit on cancellation.

    Args:
        configuration_path: Isolated settings file with per-tool overrides.
        monkeypatch: Menu, key-entry, and vendor replacements.
        save: Whether to save or interrupt after staging source selections.
    """
    from open_trading_agents.config import load_settings
    from open_trading_agents.dataflows import router
    from open_trading_agents.dataflows.config import run_config
    from open_trading_agents.dataflows.errors import VendorUnavailableError
    from open_trading_agents.diagnostics import _selected_vendors

    configuration_path.parent.mkdir()
    configuration_path.write_text('''# Keep this preference
output_language = "Korean"
[data]
web_search_backend = "yahoo"
[data.tool_vendors]
get_news = "yfinance"
get_indicators = "yfinance"
''')
    original = configuration_path.read_bytes()
    answers = [0, 1, 3, 1, 3, 4] + ([2] if save else [])
    shown = select_menus(monkeypatch, configuration_path, answers)
    monkeypatch.setattr(configuration_ui, "_api_key_prompt", lambda label: "private-alpha-value")
    result = runner.invoke(application, [str(configuration_path)])

    assert "private-alpha-value" not in result.output + str(shown)
    if not save:
        assert result.exit_code != 0
        assert configuration_path.read_bytes() == original
        return

    assert result.exit_code == 0, result.output
    saved = tomllib.loads(configuration_path.read_text())
    assert saved["data"]["alpha_vantage_api_key"] == "private-alpha-value"
    assert saved["data"]["data_vendors"] == {"news_data": "yfinance,alpha_vantage"}
    assert saved["data"]["tool_vendors"] == {"get_indicators": "yfinance"}
    assert saved["data"]["web_search_backend"] == "yahoo"
    assert saved["output_language"] == "Korean"
    assert "# Keep this preference" in configuration_path.read_text()
    assert "individual tool overrides" in result.output
    assert any("News and insider data · Enabled" in choices for _, choices in shown)
    settings = load_settings(configuration_path)
    assert "alpha_vantage" in _selected_vendors(settings)

    calls = []

    def unavailable(*args: Any) -> str:
        """Simulate an unavailable first choice to exercise the saved fallback order."""
        calls.append("yfinance")
        raise VendorUnavailableError("fixture unavailable")

    def fallback(*args: Any) -> str:
        """Return evidence from the explicitly selected second source."""
        calls.append("alpha_vantage")
        return "fixture news"

    monkeypatch.setitem(router.VENDOR_METHODS, "get_news", {"yfinance": unavailable, "alpha_vantage": fallback})
    with run_config(settings.data):
        assert router.route_to_vendor("get_news", "NVDA", "2024-01-01", "2024-01-02") == "fixture news"
    assert calls == ["yfinance", "alpha_vantage"]


def test_source_toggles_disable_alpha_and_keep_sec_before_yahoo_when_enabling_fundamentals(
    configuration_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disable an old Alpha-first chain and enable fundamentals without dropping filing evidence.

    Args:
        configuration_path: Isolated saved source preferences.
        monkeypatch: Menu input replacement.
    """
    configuration_path.parent.mkdir()
    configuration_path.write_text('[data.data_vendors]\nnews_data = "alpha_vantage,yfinance"\n')
    shown = select_menus(monkeypatch, configuration_path, [1, 3, 2, 4, 2])
    result = runner.invoke(application, [str(configuration_path)])

    assert result.exit_code == 0, result.output
    vendors = tomllib.loads(configuration_path.read_text())["data"]["data_vendors"]
    assert vendors == {"news_data": "yfinance", "fundamental_data": "sec_edgar,yfinance,alpha_vantage"}
    menus = [choices for label, choices in shown if label == "Alpha Vantage fallback"]
    assert menus[0][3] == "News and insider data · Enabled"
    assert menus[-1][3] == "News and insider data · Disabled"
    assert menus[-1][2] == "Fundamentals and financial statements · Enabled"


def test_saving_normalizes_existing_alpha_first_categories_and_tool_overrides(
    configuration_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply Yahoo-first ordering to old saved selections without requiring toggle cycles.

    Args:
        configuration_path: Isolated legacy source configuration.
        monkeypatch: Save-only menu input.
    """
    configuration_path.parent.mkdir()
    configuration_path.write_text('''[data.data_vendors]
news_data = "alpha_vantage,yfinance"
fundamental_data = "sec_edgar,alpha_vantage"
[data.tool_vendors]
get_stock_data = "alpha_vantage"
''')
    select_menus(monkeypatch, configuration_path, [2])
    result = runner.invoke(application, [str(configuration_path)])

    assert result.exit_code == 0, result.output
    data = tomllib.loads(configuration_path.read_text())["data"]
    assert data["data_vendors"] == {"news_data": "yfinance,alpha_vantage", "fundamental_data": "sec_edgar,yfinance,alpha_vantage"}
    assert data["tool_vendors"] == {"get_stock_data": "yfinance,alpha_vantage"}
