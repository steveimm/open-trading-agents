from typing import Any

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from questionary.prompts.common import InquirerControl
from questionary.question import Question

from open_trading_agents.ui import configuration as configuration_ui


@pytest.fixture
def menus(monkeypatch: pytest.MonkeyPatch) -> list[Question]:
    """Capture actual terminal menus without replacing their renderer or keyboard handling.

    Args:
        monkeypatch: Scoped replacement of the menu-construction function.

    Returns:
        Constructed Questionary menus for inspection after input completes.
    """
    captured: list[Question] = []
    select = configuration_ui.questionary.select

    def capture(*args: Any, **kwargs: Any) -> Question:
        """Retain each native menu while forwarding its original arguments."""
        question = select(*args, **kwargs)
        assert question.application.erase_when_done
        captured.append(question)
        return question

    monkeypatch.setattr(configuration_ui.questionary, "select", capture)
    return captured


def test_default_suffix_stays_on_initial_choice_when_cursor_moves(menus: list[Question]) -> None:
    """Keep the default marker distinct from the movable pointer and return the original option value."""
    choices = ["pi", "codex", "claude"]
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\x1b[B\r")
        assert configuration_ui._choose("Harness", choices, default=1) == 2
    control = next(control for control in menus[0].application.layout.find_all_controls() if isinstance(control, InquirerControl))
    assert [choice.title for choice in control.choices] == ["pi", "codex (default)", "claude"]
    assert [choice.value for choice in control.choices] == [0, 1, 2]
    assert control.get_pointed_at().value == 2
    assert choices == ["pi", "codex", "claude"]
    assert any(style == "class:pointer" and "›" in text for style, text in control._get_choice_tokens())


def test_default_and_cursor_options_have_no_reverse_or_background_highlight(menus: list[Question]) -> None:
    """Remove prompt_toolkit's inherited reverse-video selected style while keeping Enter defaults."""
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\r")
        assert configuration_ui._choose("Provider", ["Harness default", "local"], default=1) == 1
    style = menus[0].application._merged_style
    for name in ("selected", "highlighted"):
        attributes = style.get_attrs_for_style_str(f"class:{name}")
        assert not attributes.reverse
        assert not attributes.bgcolor


@pytest.mark.parametrize("keys, expected", [("cloud", 0), ("j", 0), ("k", 0), ("\x0e", 0), ("\x10", 0), ("cloud\x1b[B", 1)])
def test_only_arrow_keys_change_selection(menus: list[Question], keys: str, expected: int) -> None:
    """Ignore typed search text and alternate shortcuts while preserving arrow navigation.

    Args:
        menus: Captured terminal menus.
        keys: Typed characters and optional arrow keys before confirmation.
        expected: Selected index after the supported navigation keys.
    """
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text(f"{keys}\r")
        assert configuration_ui._choose("Model", ["local-small", "cloud-large", "local-large"], default=0) == expected
    control = next(control for control in menus[0].application.layout.find_all_controls() if isinstance(control, InquirerControl))
    assert control.choices[0].title == "local-small (default)"
    assert control.search_filter is None
    assert control.filtered_choices == control.choices
    assert control.get_pointed_at().value == expected
