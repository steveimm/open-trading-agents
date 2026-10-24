import socket
import subprocess
from pathlib import Path
from typing import Any, NoReturn

import pytest

from open_trading_agents import diagnostics
from open_trading_agents.config import HarnessSettings, default_config_path, preset_path, save_preset


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate configuration and prevent diagnostics from contacting services or starting a harness.

    Args:
        tmp_path: Temporary application home.
        monkeypatch: Scoped filesystem, executable, and external-work replacements.

    Returns:
        The absent application configuration path.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    installed = {"pi": "/fixture/bin/pi", "claude": "/fixture/bin/claude"}
    monkeypatch.setattr(diagnostics.shutil, "which", installed.get)

    def unexpected(*args: Any, **kwargs: Any) -> NoReturn:
        """Reject network or subprocess work from these local checks."""
        raise AssertionError("doctor attempted external work")

    monkeypatch.setattr(socket.socket, "connect", unexpected)
    monkeypatch.setattr(socket.socket, "connect_ex", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    return default_config_path()


def test_configured_data_and_default_profile_are_reported_without_values(config_path: Path) -> None:
    """Inspect selected profiles and data access without disclosing credentials or modifying files.

    Args:
        config_path: Isolated application configuration.
    """
    save_preset("daily", HarnessSettings(kind="pi", model="selected-model"))
    config_path.write_text('''default_preset = "daily"
[data]
fred_api_key = "private-fred-key"
alpha_vantage_api_key = "private-alpha-key"
sec_edgar_user_agent = "Private Contact person@example.com"
[data.tool_vendors]
get_news = "alpha_vantage"
''')
    paths = (config_path, preset_path("daily"))
    before = {path: path.read_bytes() for path in paths}

    checks = dict(diagnostics.doctor_checks(config_path))

    assert checks["Config"] == f"Readable and valid: {config_path}"
    assert checks["pi"] == "/fixture/bin/pi" and checks["codex"] == "Not installed"
    assert checks["claude"] == "/fixture/bin/claude"
    assert checks["Profiles"] == "1 valid saved profile(s)"
    assert checks["Default profile"] == "Configured: daily"
    assert checks["FRED"] == checks["Alpha Vantage"] == checks["SEC EDGAR"] == "Configured"
    visible = str(checks)
    for value in ("private-fred-key", "private-alpha-key", "Private Contact", "person@example.com"):
        assert value not in visible
    assert "verified" not in visible.lower()
    assert {path: path.read_bytes() for path in paths} == before


def test_missing_keys_distinguish_selected_sources_and_optional_vendors(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report actionable missing settings while treating unused Alpha Vantage access as optional.

    Args:
        config_path: Isolated configuration using the default data vendors.
        monkeypatch: Scoped obsolete environment credentials.
    """
    config_path.parent.mkdir()
    config_path.write_text('[harness]\nkind = "pi"\n')
    monkeypatch.setenv("FRED_API_KEY", "ignored-environment-secret")

    checks = dict(diagnostics.doctor_checks(config_path))

    assert "uv run open-trading-agents configure agents" in checks["Profiles"]
    assert checks["FRED"].startswith("Missing API key") and "uv run open-trading-agents configure data" in checks["FRED"]
    assert checks["Alpha Vantage"] == "Not configured (optional, not selected)"
    assert "package fallback" in checks["SEC EDGAR"] and "configure data" in checks["SEC EDGAR"]
    assert "ignored-environment-secret" not in str(checks)


@pytest.mark.parametrize("override_all", [False, True])
def test_data_checks_use_selected_file_and_effective_tool_overrides(config_path: Path, override_all: bool) -> None:
    """Respect a custom configuration path and tool overrides when deciding which keys are missing.

    Args:
        config_path: Global configuration that must not supply the selected file's credentials.
        override_all: Whether every news tool overrides the Alpha Vantage category selection.
    """
    config_path.parent.mkdir()
    config_path.write_text('[data]\nfred_api_key = "global-private-key"\nalpha_vantage_api_key = "global-alpha-key"\n')
    selected = config_path.parent.parent / "custom settings.toml"
    text = '[data.data_vendors]\nnews_data = "alpha_vantage"\n[data.tool_vendors]\nget_news = "yfinance"\n'
    if override_all:
        text += 'get_global_news = "yfinance"\nget_insider_transactions = "yfinance"\n'
    selected.write_text(text)

    checks = dict(diagnostics.doctor_checks(selected))

    assert str(selected) in checks["Config"]
    assert checks["FRED"].startswith("Missing API key")
    assert f"--config '{selected}'" in checks["FRED"]
    if override_all:
        assert checks["Alpha Vantage"] == "Not configured (optional, not selected)"
    else:
        assert checks["Alpha Vantage"].startswith("Missing API key")
    assert "global-private-key" not in str(checks) and "global-alpha-key" not in str(checks)


@pytest.mark.parametrize("invalid_file", [False, True])
def test_missing_or_invalid_default_profile_is_reported_alongside_data(config_path: Path, invalid_file: bool) -> None:
    """Keep a broken default profile distinct from valid application and data configuration.

    Args:
        config_path: Isolated application configuration selecting a broken default.
        invalid_file: Whether the selected default exists but contains invalid settings.
    """
    save_preset("valid", HarnessSettings(kind="pi"))
    config_path.write_text('default_preset = "broken"\n[data]\nfred_api_key = "private-fred-key"\n')
    if invalid_file:
        preset_path("broken").write_text('harness = "private-invalid-harness-value"\n')

    checks = dict(diagnostics.doctor_checks(config_path))

    assert checks["Config"].startswith("Readable and valid")
    assert "Missing or invalid: broken" in checks["Default profile"]
    assert "configure agents" in checks["Default profile"]
    assert checks["FRED"] == "Configured"
    assert ("invalid (broken)" in checks["Profiles"]) == invalid_file
    assert "private-invalid-harness-value" not in str(checks) and "private-fred-key" not in str(checks)


def test_effective_role_conflict_is_reported_after_applying_default_profile(config_path: Path) -> None:
    """Validate sparse role overrides against the selected default while retaining independent data checks.

    Args:
        config_path: Isolated configuration whose provider override conflicts with its default harness.
    """
    save_preset("daily", HarnessSettings(kind="claude", model="selected-model"))
    config_path.write_text('''default_preset = "daily"
[agents.bull]
provider = "private-provider-selection"
[data]
fred_api_key = "private-fred-key"
''')

    checks = dict(diagnostics.doctor_checks(config_path))

    assert checks["Default profile"] == "Configured: daily"
    assert checks["Config"].startswith("Readable with invalid agent selections:")
    assert checks["Agent selections"] == "Invalid: bull. Check [agents] in the configuration."
    assert checks["FRED"] == "Configured"
    assert "Alpha Vantage" in checks and "SEC EDGAR" in checks
    assert "private-provider-selection" not in str(checks) and "private-fred-key" not in str(checks)


@pytest.mark.parametrize("contents", [
    '[data]\nfred_api_key = "private-secret"\ninvalid = [',
    'research_rounds = "private-secret"\n',
    '[data]\nfred_api_key = ["private-secret"]\n',
])
def test_invalid_configuration_does_not_echo_values(config_path: Path, contents: str) -> None:
    """Surface malformed TOML, invalid settings, and credential types without exposing their contents.

    Args:
        config_path: Isolated malformed configuration.
        contents: Invalid TOML or settings carrying a value that must remain private.
    """
    config_path.parent.mkdir()
    config_path.write_text(contents)

    checks = dict(diagnostics.doctor_checks(config_path))

    assert checks["Config"].startswith("Invalid:")
    assert checks["Data sources"].startswith("Not checked")
    assert "private-secret" not in str(checks)
    assert config_path.read_text() == contents


def test_missing_configuration_does_not_create_application_state(config_path: Path) -> None:
    """Explain initial setup without creating a file, profile, run, or cache.

    Args:
        config_path: Absent isolated application configuration.
    """
    checks = dict(diagnostics.doctor_checks(config_path))

    assert checks["Config"].startswith("Missing:") and "configure agents" in checks["Config"]
    assert checks["Profiles"].startswith("Missing.")
    assert checks["Data sources"].startswith("Not checked")
    assert not config_path.parent.exists()


def test_unreadable_configuration_is_reported_without_aborting_other_local_checks(config_path: Path) -> None:
    """Report a non-file configuration path while retaining executable diagnostics.

    Args:
        config_path: Isolated directory supplied in place of a TOML file.
    """
    config_path.mkdir(parents=True)
    checks = dict(diagnostics.doctor_checks(config_path))

    assert checks["Config"].startswith("Unreadable:")
    assert checks["pi"] == "/fixture/bin/pi"
    assert checks["Data sources"].startswith("Not checked")
