import shlex
import shutil
from pathlib import Path

from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.config import Settings, default_config_path, list_presets, load_preset, load_settings
from open_trading_agents.dataflows.config import build_data_config, read_data_credentials
from open_trading_agents.dataflows.router import TOOLS_CATEGORIES, VENDOR_METHODS


def _selected_vendors(settings: Settings) -> set[str]:
    """Resolve credential-bearing vendors using the same tool-over-category precedence as requests.

    Args:
        settings: Parsed application settings.

    Returns:
        Vendors present in at least one effective configured fallback chain.

    Raises:
        ValueError: Vendor selection tables contain invalid values or unsupported chains.
    """
    data = build_data_config(settings.data)
    categories, tools = data["data_vendors"], data["tool_vendors"]
    if any(not isinstance(table, dict) or any(not isinstance(value, str) for value in table.values())
           for table in (categories, tools)):
        raise ValueError("Invalid vendor selection tables")

    selected: set[str] = set()
    for category, details in TOOLS_CATEGORIES.items():
        for method in details["tools"]:
            configured = tools.get(method, categories.get(category, "default"))
            explicit = {value.strip() for value in configured.split(",") if value.strip() not in {"", "default"}}
            available = set(VENDOR_METHODS[method])
            chain = explicit & available if explicit else available
            if not chain:
                raise ValueError("Unsupported vendor selection")
            selected.update(chain)

    return selected


def _data_checks(settings: Settings, path: Path, action: str) -> list[tuple[str, str]]:
    """Report local credential presence without printing or testing credential values.

    Args:
        settings: Effective application settings.
        path: Selected credential-bearing configuration file.
        action: Command for updating this configuration's data access.

    Returns:
        Data-source configuration diagnostics.
    """
    try:
        credentials = read_data_credentials(path)
    except (OSError, ValueError):
        return [("Data sources", f"Unreadable or invalid credential fields. Run {action}.")]
    try:
        selected = _selected_vendors(settings)
    except ValueError:
        return [("Data sources", "Invalid vendor selections. Check [data.data_vendors] and [data.tool_vendors].")]

    checks = []
    for label, vendor, key in (("FRED", "fred", "fred_api_key"), ("Alpha Vantage", "alpha_vantage", "alpha_vantage_api_key")):
        if credentials[key]:
            message = "Configured" if vendor in selected else "Configured (optional, not selected)"
        elif vendor in selected:
            message = f"Missing API key for selected data source. Run {action}."
        else:
            message = "Not configured (optional, not selected)"
        checks.append((label, message))

    if credentials["sec_edgar_user_agent"]:
        message = "Configured" if "sec_edgar" in selected else "Configured (not selected)"
    elif "sec_edgar" in selected:
        message = f"Contact identity not set (package fallback). Run {action}."
    else:
        message = "Contact identity not set (not selected)"
    checks.append(("SEC EDGAR", message))
    return checks


def doctor_checks(config_path: Path) -> list[tuple[str, str]]:
    """Inspect local application, profile, executable, and data-source configuration.

    Args:
        config_path: Application configuration file selected for this inspection.

    Returns:
        Label/message pairs containing local findings and setup commands, never secrets.
    """
    source = config_path.expanduser().resolve()
    suffix = "" if source == default_config_path().resolve() else f" --config {shlex.quote(str(source))}"
    agents_action = f"uv run open-trading-agents configure agents{suffix}"
    data_action = f"uv run open-trading-agents configure data{suffix}"
    settings: Settings | None = None

    try:
        settings = load_settings(source)
        config_message = f"Readable and valid: {source}"
    except FileNotFoundError:
        config_message = f"Missing: {source}. Run {agents_action}."
    except OSError:
        config_message = f"Unreadable: {source}. Check the file and its permissions."
    except ValueError:
        config_message = f"Invalid: {source}. Fix the TOML configuration before running an analysis."

    checks = [("Config", config_message)]
    checks.extend((name, shutil.which(executable) or "Not installed")
                  for name, executable in (("pi", "pi"), ("codex", "codex"), ("claude-code", "claude")))
    try:
        names = list_presets()
        invalid = []
        for name in names:
            try:
                load_preset(name)
            except (OSError, ValueError):
                invalid.append(name)
        if not names:
            profile_message = f"Missing. Run {agents_action} to create a saved profile."
        elif invalid:
            profile_message = f"{len(names) - len(invalid)} valid, {len(invalid)} invalid ({', '.join(invalid)}). Run {agents_action}."
        else:
            profile_message = f"{len(names)} valid saved profile(s)"
    except OSError:
        profile_message = f"Unreadable profile directory. Check its permissions or run {agents_action}."
    checks.append(("Profiles", profile_message))

    if settings is None:
        checks.append(("Data sources", "Not checked until the configuration file is available and valid."))
        return checks

    check_agents = True
    if settings.default_preset:
        try:
            settings = load_settings(source, use_default_preset=True)
            checks.append(("Default profile", f"Configured: {settings.default_preset}"))
        except (OSError, ValueError):
            check_agents = False
            checks.append(("Default profile", f"Missing or invalid: {settings.default_preset}. Run {agents_action}."))
    else:
        checks.append(("Default profile", "Not selected (using configured agent settings)"))

    invalid_roles = []
    if check_agents:
        for role in ROLES:
            try:
                settings.model_settings(role=role)
            except ValueError:
                invalid_roles.append(role)
    if invalid_roles:
        checks[0] = ("Config", f"Readable with invalid agent selections: {source}")
        checks.append(("Agent selections", f"Invalid: {', '.join(invalid_roles)}. Check [agents] in the configuration."))

    checks.extend(_data_checks(settings, source, data_action))
    return checks
