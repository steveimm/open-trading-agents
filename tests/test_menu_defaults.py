from typing import Any

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.formatted_text import fragment_list_to_text
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


@pytest.mark.parametrize("choices", [["pi", "codex", "claude-code"], ["astra (selected)", "pi", "claude-code"]])
def test_choice_labels_stay_verbatim_when_cursor_moves(menus: list[Question], choices: list[str]) -> None:
    """Keep caller-provided preset labels independent of the initial choice and movable pointer.

    Args:
        menus: Captured terminal menus.
        choices: Ordinary options or a saved profile explicitly labelled as selected.
    """
    expected_labels = choices.copy()
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\x1b[B\r")
        assert configuration_ui._choose("Harness", choices, default=1) == 2
    control = next(control for control in menus[0].application.layout.find_all_controls() if isinstance(control, InquirerControl))
    assert [fragment_list_to_text(choice.title) for choice in control.choices] == expected_labels
    assert [choice.value for choice in control.choices] == [0, 1, 2]
    assert control.get_pointed_at().value == 2
    assert choices == expected_labels
    assert any(style == "class:pointer" and "›" in text for style, text in control._get_choice_tokens())


def test_cursor_highlight_uses_the_theme_without_reverse_video(menus: list[Question]) -> None:
    """Render the current row with the requested background and leave other rows unhighlighted.

    Args:
        menus: Captured terminal menus with renderer-level style defaults.
    """
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\r")
        assert configuration_ui._choose("Provider", ["Harness default", "local"], default=1) == 1
    application = menus[0].application
    control = next(control for control in application.layout.find_all_controls() if isinstance(control, InquirerControl))
    for index, choice in enumerate(control.choices):
        for token_style, _ in choice.title:
            attributes = application.renderer.style.get_attrs_for_style_str(token_style)
            assert not attributes.reverse
            assert attributes.bgcolor == ("303547" if index == 1 else "")


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
    assert fragment_list_to_text(control.choices[0].title) == "local-small"
    assert control.search_filter is None
    assert control.filtered_choices == control.choices
    assert control.get_pointed_at().value == expected
