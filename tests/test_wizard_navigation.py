from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from open_trading_agents.config import HarnessSettings, Settings
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice
from open_trading_agents.ui import configuration as ui


@pytest.mark.parametrize("key", ["\r", "\x1b[C"])
def test_right_and_enter_confirm_same_selection(key: str) -> None:
    """Exercise the real menu's existing confirmation path with both supported keys.

    Args:
        key: Enter or Right arrow escape sequence.
    """
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\x1b[B" + key)
        assert ui._choose("Model", ["first", "second"]) == 1


def test_left_returns_without_becoming_an_abort() -> None:
    """Left must signal parent navigation without selecting or cancelling the whole wizard."""
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\x1b[D")
        with pytest.raises(ui.BackNavigation):
            ui._choose("Model", ["first", "second"])


def test_runtime_backtracks_and_skips_unavailable_provider_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain prior model choices and return directly to Harness for Claude Code.

    Args:
        monkeypatch: Isolated menus and native model catalogs.
    """
    steps = iter([
        ("Harness", 1), ("Provider", 1), ("Model", 2), ("Reasoning effort", None),
        ("Model", None), ("Provider", None), ("Harness", 2), ("Model", None),
        ("Harness", 2), ("Model", 1), ("Reasoning effort", 2),
    ])
    seen = []

    def choose(label: str, choices: list[str], default: int = 0) -> int:
        """Check the screen order and simulate backward navigation."""
        expected, answer = next(steps)
        assert label == expected
        seen.append((label, default))
        if answer is None:
            raise ui.BackNavigation()
        return answer

    monkeypatch.setattr(ui, "_choose", choose)
    monkeypatch.setattr(ui, "discover_models", lambda selected: ModelCatalog((ModelChoice("opus", "Opus"),))
                        if selected.kind == "claude-code" else ModelCatalog((ModelChoice("small", "Small", "local"),
                                                                             ModelChoice("large", "Large", "local"))))
    monkeypatch.setattr(ui, "effort_choices", lambda current, catalog: ("low", "high"))
    selected = ui._runtime(HarnessSettings(), {})
    assert selected == HarnessSettings(kind="claude-code", model="opus", effort="high")
    assert seen[4:7] == [("Model", 2), ("Provider", 1), ("Harness", 1)]
    assert seen[8] == ("Harness", 2)


def test_agent_editor_back_preserves_assignments_and_returns_to_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back out of custom setup, profile selection, and both editor modes without losing edits.

    Args:
        monkeypatch: Isolated profile catalog and selections.
    """
    profiles = {"base": HarnessSettings(model="base"), "fast": HarnessSettings(model="fast")}
    monkeypatch.setattr(ui, "list_presets", lambda: list(profiles))
    monkeypatch.setattr(ui, "load_preset", profiles.__getitem__)
    steps = iter([
        ("Agent configuration", "Simple"), ("Simple agent configuration", "Working agents"),
        ("Working agents", "Configure for this run"), ("Harness", None), ("Working agents", None),
        ("Simple agent configuration", "Working agents"), ("Working agents", "fast"),
        ("Simple agent configuration", None), ("Agent configuration", "Advanced"),
        ("Advanced agent configuration", None), ("Agent configuration", None),
    ])

    def choose(label: str, choices: list[str], default: int = 0, **kwargs: object) -> int:
        """Walk the nested editors and verify inherited manager choices stay intact."""
        expected, answer = next(steps)
        assert label == expected
        if label == "Advanced agent configuration":
            assert "Market Analyst · fast" in choices and "Research Manager · fast" in choices
        if answer is None:
            raise ui.BackNavigation()
        return next(index for index, choice in enumerate(choices) if choice.startswith(answer))

    monkeypatch.setattr(ui, "_choose", choose)
    with pytest.raises(ui.BackNavigation) as back:
        ui.select_agent_settings(Settings(harness=profiles["base"]))
    assert back.value.settings is not None
    assert back.value.settings.model_settings(role="market").model == "fast"
    assert back.value.settings.model_settings(role="research_manager").model == "fast"


def test_run_wizard_back_retains_date_depth_and_agent_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revisit earlier inputs before constructing a workflow or creating run files.

    Args:
        tmp_path: Unwritten run location.
        monkeypatch: Isolated text inputs and selectors.
    """
    dates = []
    depths = []
    agent_calls = 0
    monkeypatch.setattr(ui.typer, "prompt", lambda label: "NVDA")
    monkeypatch.setattr(ui, "select_run_portfolio", lambda: None)
    monkeypatch.setattr(ui, "select_output_directory", lambda directory: None)

    def date(previous: str | None = None) -> str:
        """Offer the previous date after returning from Research depth."""
        dates.append(previous)
        return "2024-03-15" if previous is None else "2024-03-16"

    def depth(settings: Settings) -> Settings:
        """Return once, then update the existing depth on the next visit."""
        depths.append(settings.research_rounds)
        if len(depths) == 1:
            raise ui.BackNavigation()
        return settings.model_copy(update={"research_rounds": 3 if len(depths) == 2 else 5})

    def agents(settings: Settings) -> Settings:
        """Return to depth with edited assignments, then accept them unchanged."""
        nonlocal agent_calls
        agent_calls += 1
        if agent_calls == 1:
            raise ui.BackNavigation(settings.model_copy(update={"quick": HarnessSettings(model="chosen")}))
        assert settings.quick.model == "chosen"
        return settings

    monkeypatch.setattr(ui, "select_analysis_date", date)
    monkeypatch.setattr(ui, "select_research_depth", depth)
    monkeypatch.setattr(ui, "select_agent_settings", agents)
    selected = ui.select_run_inputs(Settings(state_dir=tmp_path), None, None, None, None, ask_portfolio=True)
    assert dates == [None, "2024-03-15"] and depths == [1, 1, 3]
    assert selected.ticker == "NVDA" and selected.analysis_date == "2024-03-16"
    assert selected.settings.research_rounds == 5 and selected.settings.quick.model == "chosen"
    assert not list(tmp_path.iterdir())


def test_configure_returns_from_editor_to_section_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A top-level editor's Left returns to configuration categories without writing files.

    Args:
        tmp_path: Isolated user configuration directory.
        monkeypatch: Simulated terminal and navigation events.
    """
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from open_trading_agents import cli

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))
    visits = []

    def section(previous: str | None = None) -> str:
        """Restore the category cursor on return."""
        visits.append(previous)
        return "data" if previous is None else "portfolio"

    def back(path: Path) -> None:
        """Leave the data editor before saving."""
        raise ui.BackNavigation()

    opened = []
    monkeypatch.setattr(ui, "select_configuration_section", section)
    monkeypatch.setattr(ui, "configure_data_interactively", back)
    monkeypatch.setattr(ui, "configure_portfolio_interactively", opened.append)
    result = CliRunner().invoke(cli.app, ["configure"])
    assert result.exit_code == 0, result.output
    assert visits == [None, "data"]
    assert opened == [tmp_path / ".open-trading-agents" / "portfolio.json"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("command,editor", [
    ("agents", "configure_interactively"), ("data", "configure_data_interactively"),
    ("portfolio", "configure_portfolio_interactively"),
])
def test_left_at_direct_configuration_root_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, editor: str,
) -> None:
    """A direct editor has no parent category menu and can return without an error.

    Args:
        tmp_path: Isolated user home.
        monkeypatch: Simulated terminal and root-level Left event.
        command: Direct configure subcommand.
        editor: Interactive editor used by that command.
    """
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from open_trading_agents import cli

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))

    def back(path: Path) -> None:
        """Return before writing any configuration."""
        raise ui.BackNavigation()

    monkeypatch.setattr(ui, editor, back)
    result = CliRunner().invoke(cli.app, ["configure", command])
    assert result.exit_code == 0, result.output
    assert "failed" not in result.output.lower()
    assert not list(tmp_path.iterdir())


def test_final_profile_confirmation_back_does_not_save_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return to the preset name and save only the revised final choice.

    Args:
        tmp_path: Isolated user home and profile files.
        monkeypatch: Controlled profile and final confirmation steps.
    """
    from open_trading_agents.config import list_presets, load_preset

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = tmp_path / ".open-trading-agents" / "config.toml"
    monkeypatch.setattr(ui, "_runtime", lambda current, catalogs: HarnessSettings(model="chosen"))
    names = []

    def name(previous: str | None = None) -> tuple[str, bool]:
        """Revisit the entered name while all changes are still pending."""
        assert not path.exists() and not list_presets()
        names.append(previous)
        return ("first" if previous is None else "revised"), False

    def confirm(label: str, default: bool = False) -> bool:
        """Go back once, then approve the revised name."""
        if len(names) == 1:
            raise ui.BackNavigation()
        return True

    monkeypatch.setattr(ui, "_preset", name)
    monkeypatch.setattr(ui, "_confirm", confirm)
    ui.configure_interactively(path)
    assert names == [None, "first"]
    assert list_presets() == ("revised",)
    assert load_preset("revised").model == "chosen"
