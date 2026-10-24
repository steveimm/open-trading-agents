import json
import stat
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_trading_agents.analysis.models import AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.analysis.workflow import Workflow
from open_trading_agents.config import Settings, load_settings, save_data_configuration, save_harness_selection
from open_trading_agents.dataflows.config import (
    DATA_CREDENTIAL_KEYS,
    build_data_config,
    get_config,
    get_data_credential,
    read_data_credentials,
    run_config,
)
from open_trading_agents.persistence.storage import RunStore


def test_data_credential_updates_preserve_comments_and_secure_the_file(tmp_path: Path) -> None:
    """Update and clear selected fields without replacing unrelated preferences or comments.

    Args:
        tmp_path: Isolated configuration location.
    """
    path = tmp_path / "config.toml"
    assert read_data_credentials(path) == dict.fromkeys(DATA_CREDENTIAL_KEYS, "")
    path.write_text('# Preferences\nresearch_rounds = 3\n[data]\nnews_article_limit = 12 # retained\n')
    path.chmod(0o640)
    values = {"fred_api_key": "fake-fred-secret", "alpha_vantage_api_key": "fake-alpha-secret",
              "sec_edgar_user_agent": "Fixture Research fixture@example.invalid"}
    assert save_data_configuration(path, values) == path.resolve()
    assert read_data_credentials(path) == values
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert '# Preferences' in path.read_text() and '# retained' in path.read_text()
    assert load_settings(path).data == {"news_article_limit": 12}

    save_harness_selection(path, model="updated-model")
    assert read_data_credentials(path) == values
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    save_data_configuration(path, {"fred_api_key": "  "})
    stored = tomllib.loads(path.read_text())
    assert "fred_api_key" not in stored["data"]
    assert stored["research_rounds"] == 3 and stored["data"]["news_article_limit"] == 12
    assert read_data_credentials(path) == {**values, "fred_api_key": ""}


def test_data_credentials_stay_out_of_runtime_configuration_and_refresh_after_rotation(tmp_path: Path) -> None:
    """Keep only the file locator in tool settings and reread keys for each request.

    Args:
        tmp_path: Isolated configuration location.
    """
    path = tmp_path / "selected.toml"
    save_data_configuration(path, {"fred_api_key": "fake-initial-secret"})
    raw = {"credentials_file": str(path), "fred_api_key": "fake-injected-secret", "news_article_limit": 7}
    assert Settings(data=raw).data == {"news_article_limit": 7}
    assert "fred_api_key" not in build_data_config(raw)
    settings = Settings()
    settings.data.update(raw)
    assert settings.model_dump()["data"] == {"news_article_limit": 7}

    with run_config(raw):
        assert get_config()["credentials_file"] == str(path)
        assert not set(DATA_CREDENTIAL_KEYS).intersection(get_config())
        assert get_data_credential("fred_api_key") == "fake-initial-secret"
        assert get_data_credential("alpha_vantage_api_key") == ""
        save_data_configuration(path, {"fred_api_key": "fake-rotated-secret"})
        assert get_data_credential("fred_api_key") == "fake-rotated-secret"
    assert raw["fred_api_key"] == "fake-injected-secret"


def test_data_credential_errors_do_not_echo_values_or_modify_files(tmp_path: Path) -> None:
    """Reject invalid configuration without disclosing supplied credential contents.

    Args:
        tmp_path: Isolated configuration location.
    """
    secret = "fake-sensitive-value-must-not-appear"
    with pytest.raises(ValidationError) as invalid_settings:
        Settings(data={"fred_api_key": secret}, analysts=())
    assert secret not in str(invalid_settings.value)
    with pytest.raises(ValidationError) as invalid_value:
        Settings(data={"fred_api_key": [secret]})
    assert secret not in str(invalid_value.value)

    path = tmp_path / "config.toml"
    with pytest.raises(ValueError) as unknown:
        save_data_configuration(path, {"unsupported": secret})
    assert secret not in str(unknown.value) and not path.exists()

    path.write_text(f"[data]\nfred_api_key = {secret}\n")
    original = path.read_bytes()
    for reader in (read_data_credentials, load_settings):
        with pytest.raises(ValueError) as malformed:
            reader(path)
        assert secret not in str(malformed.value)
    with pytest.raises(ValueError) as malformed_save:
        save_data_configuration(path, {"fred_api_key": "replacement"})
    assert secret not in str(malformed_save.value)
    assert path.read_bytes() == original


async def test_workflow_persists_only_credential_source_and_resumes_after_rotation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep source selection across resume without copying keys into any run artifact.

    Args:
        tmp_path: Isolated home, source configuration, and run storage.
        monkeypatch: Scoped home-directory replacement.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    source = tmp_path / "custom" / "settings.toml"
    values = {"fred_api_key": "fake-private-fred-value", "alpha_vantage_api_key": "fake-private-alpha-value",
              "sec_edgar_user_agent": "Fake Private Identity fixture@example.invalid"}
    save_data_configuration(source, values)
    settings = load_settings(source).model_copy(update={"state_dir": tmp_path / "state", "analysts": ("market",), "memory_enabled": False})

    class Session:
        session_id = "fixture-session"

        async def prompt(self, text: str) -> AgentResult:
            """Return schema-valid fixture output without external requests.

            Args:
                text: Workflow task supplied to the synthetic session.

            Returns:
                A completed synthetic decision.
            """
            result = {"recommendation": "Hold", "rationale": "Evidence", "strategic_actions": "Wait", "action": "Hold",
                      "reasoning": "Evidence", "rating": "Hold", "executive_summary": "Wait", "investment_thesis": "Evidence"}
            return AgentResult(json.dumps(result), self.session_id)

        async def close(self) -> None:
            """Release the fixture without external effects."""

        async def cancel(self) -> None:
            """Cancel the fixture without external effects."""

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> Session:
        """Construct an offline session for the real workflow.

        Args:
            config: Selected native runtime.
            spec: Role and workspace settings.
            emit: Workflow event recorder.

        Returns:
            A synthetic session that never contacts a model.
        """
        return Session()

    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory, config_path=source)
    result = await workflow.run()
    store = RunStore(result.directory)
    try:
        assert store.metadata("config_path") == str(source.resolve())
        assert store.metadata("input")["settings"]["data"] == {}
    finally:
        store.close()
    context = json.loads(workflow.context_file.read_text())
    assert context["config"]["credentials_file"] == str(source.resolve())
    for path in result.directory.rglob("*"):
        if path.is_file():
            assert all(value.encode() not in path.read_bytes() for value in values.values())

    save_data_configuration(source, {"fred_api_key": "fake-rotated-value"})
    resumed = Workflow.resume(result.directory, session_factory=factory)
    try:
        assert resumed.config_path == source.resolve()
        with run_config(resumed.data_config):
            assert get_data_credential("fred_api_key") == "fake-rotated-value"
    finally:
        resumed.store.close()


def test_legacy_tool_context_uses_global_credentials_without_a_locator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve old context files through the shared config when no source was recorded.

    Args:
        tmp_path: Isolated home and legacy run location.
        monkeypatch: Scoped home-directory replacement.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    source = tmp_path / ".open-trading-agents" / "config.toml"
    save_data_configuration(source, {"fred_api_key": "fake-global-key"})
    workflow = Workflow("NVDA", "2026-01-02", Settings(state_dir=tmp_path / "state"))
    with workflow.store.db:
        workflow.store.db.execute("DELETE FROM metadata WHERE key = 'config_path'")
    context = json.loads(workflow.context_file.read_text())
    context["config"].pop("credentials_file")
    workflow.context_file.write_text(json.dumps(context))
    workflow.store.close()

    resumed = Workflow.resume(workflow.directory)
    try:
        assert resumed.config_path == source.resolve()
        with run_config(context["config"]):
            assert get_data_credential("fred_api_key") == "fake-global-key"
    finally:
        resumed.store.close()


def test_existing_legacy_context_can_bind_an_explicit_config_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep old backtest subprocess tools aligned with a newly supplied credential source.

    Args:
        tmp_path: Isolated source and saved run location.
        monkeypatch: Scoped home-directory replacement.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    source = tmp_path / "custom.toml"
    save_data_configuration(source, {"fred_api_key": "fake-custom-key"})
    settings = Settings(state_dir=tmp_path / "state")
    workflow = Workflow("NVDA", "2026-01-02", settings, run_id="legacy")
    with workflow.store.db:
        workflow.store.db.execute("DELETE FROM metadata WHERE key = 'config_path'")
    context = json.loads(workflow.context_file.read_text())
    context["config"].pop("credentials_file")
    workflow.context_file.write_text(json.dumps(context))
    workflow.store.close()

    reopened = Workflow("NVDA", "2026-01-02", settings, run_id="legacy", config_path=source)
    try:
        saved = json.loads(reopened.context_file.read_text())
        assert saved["config"] == {**context["config"], "credentials_file": str(source.resolve())}
        with run_config(saved["config"]):
            assert get_data_credential("fred_api_key") == "fake-custom-key"
    finally:
        reopened.store.close()
