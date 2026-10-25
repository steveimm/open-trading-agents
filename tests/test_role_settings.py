import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_trading_agents.analysis.backtest import _manifest
from open_trading_agents.analysis.models import AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.analysis.workflow import Workflow
from open_trading_agents.config import HarnessSettings, Settings, load_settings, preset_path, save_preset
from open_trading_agents.reporting.reports import write_reports


def test_role_overrides_inherit_their_group_and_reset_on_harness_changes() -> None:
    """Keep partial selectors within their working/manager baseline and isolate other harnesses."""
    settings = Settings(
        harness=HarnessSettings(kind="pi", provider="local", model="shared", effort="low", command=("custom-pi",)),
        quick=HarnessSettings(model="worker", extra_args=("--custom",)),
        deep=HarnessSettings(kind="codex", model="manager", provider="manager-provider", effort="medium"),
        agents={
            "bull": HarnessSettings(effort="high"),
            "bear": HarnessSettings(kind="claude-code", model="bear-model"),
            "social": HarnessSettings(model=None),
            "portfolio_manager": HarnessSettings(kind="pi", model="portfolio-model"),
        },
    )
    bull = settings.model_settings(role="bull")
    assert (bull.kind, bull.provider, bull.model, bull.effort) == ("pi", "local", "worker", "high")
    assert bull.command == ("custom-pi",) and bull.extra_args == ("--custom",)
    assert settings.model_settings(role="social").model is None
    assert settings.model_settings(role="trader") == settings.model_settings()
    assert settings.model_settings(role="research_manager") == settings.model_settings(True)

    for role, harness in (("bear", "claude-code"), ("portfolio_manager", "pi")):
        selected = settings.model_settings(role=role)
        assert selected.kind == harness
        assert selected.provider is selected.effort is None
        assert selected.command == selected.extra_args == ()

    bull.model = "changed-bull"
    assert settings.model_settings().model == "worker"
    assert settings.harness.model == "shared"


def test_unknown_role_selectors_fail_before_run_creation() -> None:
    """Reject misspelled role keys in configuration and runtime lookup."""
    with pytest.raises(ValidationError, match="Unknown agent roles: bullish"):
        Settings(agents={"bullish": HarnessSettings(model="wrong-role")})
    with pytest.raises(ValueError, match="Unknown agent role: bullish"):
        Settings().model_settings(role="bullish")


def test_partial_role_fields_follow_changed_group_defaults() -> None:
    """Keep explicitly configured effort while inheriting a newly selected working profile."""
    settings = Settings(
        harness=HarnessSettings(kind="pi", model="old-worker", provider="old-provider"),
        agents={"bull": HarnessSettings(effort="high")},
    )
    assert settings.model_settings(role="bull").model == "old-worker"
    assert settings.model_dump()["agents"] == {"bull": {"effort": "high"}}

    settings.quick = HarnessSettings(kind="codex", model="new-worker", provider="new-provider", effort="low")
    restored = Settings.model_validate(settings.model_dump())
    selected = restored.model_settings(role="bull")
    assert (selected.kind, selected.model, selected.provider, selected.effort) == ("codex", "new-worker", "new-provider", "high")
    assert restored.model_dump()["agents"] == {"bull": {"effort": "high"}}


def test_default_preset_precedes_effective_role_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the default Pi profile before checking a provider-only override against a stale Claude group.

    Args:
        tmp_path: Isolated home and configuration directory.
        monkeypatch: Scoped home-directory replacement.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("local", HarnessSettings(kind="pi", model="selected-worker"))
    path = tmp_path / ".open-trading-agents" / "config.toml"
    path.write_text('default_preset = "local"\n[harness]\nkind = "claude-code"\n[agents.bull]\nprovider = "local-provider"\n')

    settings = load_settings(path, use_default_preset=True)
    selected = settings.model_settings(role="bull")
    assert (selected.kind, selected.model, selected.provider) == ("pi", "selected-worker", "local-provider")
    assert settings.model_dump()["agents"] == {"bull": {"provider": "local-provider"}}


def test_default_preset_labels_only_roles_with_matching_public_selectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the applied default profile without guessing names for overridden runtimes.

    Args:
        tmp_path: Isolated preset and configuration storage.
        monkeypatch: Scoped home-directory replacement.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    profile = HarnessSettings(kind="pi", provider="local", model="worker", effort="high")
    save_preset("daily", profile)
    save_preset("another-identical-profile", profile)
    path = tmp_path / ".open-trading-agents" / "config.toml"
    path.write_text('''default_preset = "daily"
[agents.market]
command = ["custom-pi"]
extra_args = ["--custom"]
[agents.bull]
model = "different-model"
[agents.bear]
effort = "low"
[agents.trader]
provider = "different-provider"
[agents.portfolio_manager]
kind = "codex"
''')

    settings = load_settings(path, use_default_preset=True)
    unmatched = {"bull", "bear", "trader", "portfolio_manager"}
    assert settings.agent_presets == {role: "daily" for role in ROLES if role not in unmatched}
    assert settings.model_settings(role="market").command == ("custom-pi",)
    assert settings.model_settings(role="market").profile_identity() == profile.profile_identity()
    assert settings.model_settings(role="bull").model == "different-model"
    assert load_settings(path).agent_presets == {}


def test_preset_labeling_does_not_validate_roles_before_runtime_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave an invalid inherited provider available for a later explicit harness override.

    Args:
        tmp_path: Isolated preset and configuration storage.
        monkeypatch: Scoped home-directory replacement.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("native", HarnessSettings(kind="claude-code"))
    path = tmp_path / ".open-trading-agents" / "config.toml"
    path.write_text('default_preset = "native"\n[agents.bull]\nprovider = "local"\n')

    settings = load_settings(path, use_default_preset=True)

    assert "bull" not in settings.agent_presets
    assert settings.agents["bull"].provider == "local"
    settings.harness = HarnessSettings(kind="pi", model="worker")
    assert settings.model_settings(role="bull").provider == "local"


@pytest.mark.parametrize("labels", [{"bullish": "daily"}, {"bull": "../outside"}, {"bull": ""}])
def test_agent_preset_labels_require_valid_roles_and_names(labels: dict[str, str]) -> None:
    """Reject invalid role metadata without requiring valid labels to reference existing files.

    Args:
        labels: An invalid role key or preset-name spelling.
    """
    with pytest.raises(ValidationError):
        Settings(agent_presets=labels)

    assert Settings(agent_presets={"bull": "missing-but-valid-profile"}).agent_presets == {"bull": "missing-but-valid-profile"}


@pytest.mark.parametrize("remove_profile", [False, True])
def test_exact_preset_labels_survive_resume_after_profiles_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove_profile: bool,
) -> None:
    """Restore selected names and pinned runtimes independently of mutable profile files.

    Args:
        tmp_path: Isolated presets and run storage.
        monkeypatch: Scoped home-directory replacement.
        remove_profile: Whether saved profiles are deleted or replaced after the run starts.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    profile = HarnessSettings(kind="codex", model="original-model", effort="high")
    for name in ("daily", "exact-alias"):
        save_preset(name, profile)
    settings = Settings(state_dir=tmp_path / "state", harness=profile, default_preset="daily",
                        agent_presets={"market": "daily", "bull": "exact-alias"})
    assert Settings.model_validate_json(settings.model_dump_json()).agent_presets == settings.agent_presets
    workflow = Workflow("NVDA", "2026-01-02", settings)
    workflow.store.save_session("Bull Researcher", "original-session")
    workflow.store.close()

    for name in ("daily", "exact-alias"):
        if remove_profile:
            preset_path(name).unlink()
        else:
            save_preset(name, HarnessSettings(kind="claude-code", model="replacement-model"), overwrite=True)

    resumed = Workflow.resume(workflow.directory)
    try:
        assert resumed.settings.agent_presets == {"market": "daily", "bull": "exact-alias"}
        assert resumed.settings.model_settings(role="bull") == profile
        assert resumed.store.session("Bull Researcher") == "original-session"
    finally:
        resumed.store.close()


def test_existing_run_preserves_original_labels_but_rejects_runtime_changes(tmp_path: Path) -> None:
    """Ignore replacement display labels without allowing changed native execution settings.

    Args:
        tmp_path: Isolated run storage.
    """
    settings = Settings(state_dir=tmp_path, harness=HarnessSettings(model="pinned-model"), agent_presets={"bull": "original"})
    workflow = Workflow("NVDA", "2026-01-02", settings, run_id="fixed")
    workflow.store.close()
    relabeled = settings.model_copy(update={"agent_presets": {"bull": "replacement", "market": "new-label"}})

    resumed = Workflow("NVDA", "2026-01-02", relabeled, run_id="fixed")
    try:
        assert resumed.settings.agent_presets == {"bull": "original"}
        assert resumed.store.metadata("input")["settings"]["agent_presets"] == {"bull": "original"}
        assert relabeled.agent_presets == {"bull": "replacement", "market": "new-label"}
    finally:
        resumed.store.close()

    changed = relabeled.model_copy(update={"harness": HarnessSettings(model="different-model")})
    with pytest.raises(ValueError, match="inputs differ"):
        Workflow("NVDA", "2026-01-02", changed, run_id="fixed")


@pytest.mark.parametrize("kind", ["claude-code", "codex"])
def test_sparse_role_configuration_preserves_inherited_kind_on_resume(tmp_path: Path, kind: str) -> None:
    """Serialize only explicit role fields while restoring their original group selection.

    Args:
        tmp_path: Isolated run storage.
        kind: Non-Pi working harness whose identity must remain inherited.
    """
    settings = Settings(
        state_dir=tmp_path,
        quick=HarnessSettings(kind=kind, model="group-model"),
        agents={"bull": HarnessSettings(effort="high")},
    )
    restored = Settings.model_validate_json(settings.model_dump_json())
    workflow = Workflow("NVDA", "2026-01-02", restored)
    assert workflow.store.metadata("input")["settings"]["agents"] == {"bull": {"effort": "high"}}
    workflow.store.close()

    resumed = Workflow.resume(workflow.directory)
    try:
        selected = resumed.settings.model_settings(role="bull")
        assert (selected.kind, selected.model, selected.effort) == (kind, "group-model", "high")
        assert resumed.settings.agents["bull"].model_fields_set == {"effort"}
    finally:
        resumed.store.close()


def test_invalid_effective_role_settings_do_not_create_run_storage(tmp_path: Path) -> None:
    """Reject a provider override inherited by Claude before opening any run files.

    Args:
        tmp_path: Isolated application storage parent.
    """
    settings = Settings(state_dir=tmp_path / "state", harness=HarnessSettings(kind="claude-code"),
                        agents={"bull": HarnessSettings(provider="unsupported")})
    with pytest.raises(ValidationError, match="Claude Code selects its provider"):
        Workflow("NVDA", "2026-01-02", settings)

    assert not settings.state_dir.exists()


async def test_every_role_keeps_its_runtime_selection_and_native_session_on_resume(tmp_path: Path) -> None:
    """Dispatch all analysis and supporting roles independently through persisted settings.

    Args:
        tmp_path: Isolated run storage.
    """
    opened: list[tuple[HarnessConfig, AgentSpec]] = []

    class Session:
        def __init__(self, spec: AgentSpec) -> None:
            """Retain the supplied native session reference.

            Args:
                spec: Role identity and optional previous session identifier.
            """
            self.session_id = spec.session_id or f"native:{spec.agent}"

        async def prompt(self, text: str) -> AgentResult:
            """Return offline evidence for a dispatched role.

            Args:
                text: Newly submitted task.

            Returns:
                A completed response with the preserved session reference.
            """
            return AgentResult("Synthetic evidence", self.session_id)

        async def close(self) -> None:
            """Release the fixture without external effects."""

        async def cancel(self) -> None:
            """Cancel the fixture without external effects."""

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> Session:
        """Record each effective role selection passed to the harness.

        Args:
            config: Effective native harness configuration.
            spec: Role identity and session reference.
            emit: Workflow event recorder.

        Returns:
            An offline native-session fixture.
        """
        opened.append((config, spec))
        return Session(spec)

    kinds = ("pi", "codex", "claude-code")
    selections = {key: HarnessSettings(kind=kinds[index % 3], model=f"{key}-model", effort="high")
                  for index, key in enumerate(ROLES)}
    settings = Settings(state_dir=tmp_path, memory_enabled=False, agents=selections)
    restored = Settings.model_validate_json(settings.model_dump_json())
    assert restored.model_dump() == settings.model_dump()
    workflow = Workflow("NVDA", "2026-01-02", restored, session_factory=factory)

    try:
        for key, role in ROLES.items():
            name = f"{role.name} example" if key == "reflection" else None
            await workflow._call(key, key, "Initial evidence", session_name=name)
    finally:
        for session in workflow.sessions.values():
            await session.close()
        workflow.store.close()

    resumed = Workflow.resume(workflow.directory, session_factory=factory)
    try:
        for key, role in ROLES.items():
            name = f"{role.name} example" if key == "reflection" else None
            await resumed._call(f"next:{key}", key, "New evidence", session_name=name)
    finally:
        for session in resumed.sessions.values():
            await session.close()
        resumed.store.close()

    count = len(ROLES)
    assert len(opened) == count * 2
    for index, key in enumerate(ROLES):
        original, restored = opened[index], opened[index + count]
        assert original[0] == restored[0] == selections[key].runtime()
        assert original[1].session_id is None
        assert restored[1].session_id == f"native:{original[1].agent}"


def test_saved_inputs_without_role_overrides_remain_resumable(tmp_path: Path) -> None:
    """Treat the absent new setting in older saved runs as an empty override map.

    Args:
        tmp_path: Isolated run storage.
    """
    settings = Settings(state_dir=tmp_path, memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings)
    inputs = workflow.store.metadata("input")
    inputs["settings"].pop("agents")
    inputs["settings"].pop("agent_presets")
    workflow.store.set_metadata("input", inputs)
    workflow.store.save_session("Bull Researcher", "existing-native-session")
    workflow.store.close()

    resumed = Workflow.resume(workflow.directory)
    try:
        assert resumed.settings.agents == {}
        assert resumed.settings.agent_presets == {}
        assert resumed.store.session("Bull Researcher") == "existing-native-session"
    finally:
        resumed.store.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_saved_screening_configuration_is_removed_without_changing_other_run_inputs(tmp_path: Path, enabled: bool) -> None:
    """Resume historical runs after removing the obsolete role and its optional filtering flag.

    Args:
        tmp_path: Isolated run storage.
        enabled: Historical screening setting recorded before the role was removed.
    """
    settings = Settings(state_dir=tmp_path, memory_enabled=False,
                        agents={"bull": HarnessSettings(effort="high"), "reflection": HarnessSettings(model="reviewer")},
                        agent_presets={"bull": "selected-bull"})
    workflow = Workflow("NVDA", "2026-01-02", settings)
    inputs = workflow.store.metadata("input")
    inputs["settings"]["social_screening"] = enabled
    inputs["settings"]["agents"]["screening"] = {"kind": "pi", "model": "obsolete-screening-model"}
    inputs["settings"]["agent_presets"]["screening"] = "obsolete-screening-preset"
    workflow.store.set_metadata("input", inputs)
    workflow.store.save_session("Bull Researcher", "existing-bull-session")
    workflow.store.begin_step("analyst:social", "Sentiment Analyst", "Original input")
    workflow.store.finish_step("analyst:social", "Completed sentiment report")
    original_context = workflow.context_file.read_bytes()
    workflow.store.close()

    resumed = Workflow.resume(workflow.directory)
    try:
        assert resumed.settings.model_dump() == settings.model_dump()
        assert resumed.store.metadata("input")["settings"] == settings.model_dump(mode="json")
        assert resumed.store.session("Bull Researcher") == "existing-bull-session"
        assert resumed.store.step("analyst:social")["output"] == "Completed sentiment report"
        assert resumed.context_file.read_bytes() == original_context
    finally:
        resumed.store.close()


@pytest.mark.parametrize("content", [
    "social_screening = false\n",
    '[agents.screening]\nmodel = "obsolete"\n',
    '[agent_presets]\nscreening = "obsolete"\n',
])
def test_user_configuration_rejects_removed_screening_settings(tmp_path: Path, content: str) -> None:
    """Limit obsolete-key cleanup to saved runs rather than accepting removed user options.

    Args:
        tmp_path: Isolated configuration location.
        content: One removed setting supplied in user TOML.
    """
    path = tmp_path / "config.toml"
    path.write_text(content)

    with pytest.raises(ValidationError):
        load_settings(path)


def test_backtest_fingerprint_preserves_old_defaults_and_pins_role_overrides(tmp_path: Path) -> None:
    """Keep existing sweep identities while detecting changes to individual agent profiles.

    Args:
        tmp_path: Isolated backtest storage.
    """
    settings = Settings(state_dir=tmp_path)
    old_inputs = {"settings": settings.model_dump(mode="json", exclude={"agents", "agent_presets"}), "portfolio": None}
    old_inputs["settings"]["social_screening"] = False
    old_digest = hashlib.sha256(json.dumps(old_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert _manifest(["NVDA"], ["2026-01-02"], settings, None)["configuration_fingerprint"] == old_digest
    settings.agent_presets = {"market": "display-label"}
    assert _manifest(["NVDA"], ["2026-01-02"], settings, None)["configuration_fingerprint"] == old_digest

    settings.agents["bear"] = HarnessSettings(kind="claude-code", model="bear-model")
    selected = _manifest(["NVDA"], ["2026-01-02"], settings, None)["configuration_fingerprint"]
    old_inputs["settings"]["agents"] = settings.model_dump(mode="json")["agents"]
    previous_role_digest = hashlib.sha256(json.dumps(old_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert selected == previous_role_digest
    assert selected != old_digest
    settings.agents["bear"].model = "different-bear-model"
    assert _manifest(["NVDA"], ["2026-01-02"], settings, None)["configuration_fingerprint"] != selected


def test_unknown_runtime_models_use_only_their_role_configuration(tmp_path: Path) -> None:
    """Keep observed models authoritative and label fallback selectors for the affected role.

    Args:
        tmp_path: Isolated report output location.
    """
    settings = {
        "quick_harness": "pi", "quick_model": "generic-worker", "deep_model": "generic-manager",
        "runtime_agents": [
            {"agent": "Bull Researcher", "harness": "pi", "model": "observed-bull"},
            {"agent": "Bear Researcher", "harness": "claude-code", "model": None},
            {"agent": "Reflection 42", "harness": "codex", "model": None},
            {"agent": "Unknown Agent", "harness": "pi", "model": None},
        ],
        "configured_agents": [
            {"role": "bull", "agent": "Bull Researcher", "harness": "pi", "model": "configured-bull"},
            {"role": "bear", "agent": "Bear Researcher", "harness": "claude-code", "model": "configured-bear"},
            {"role": "reflection", "agent": "Reflection", "harness": "codex", "model": "configured-reflection"},
        ],
    }
    target = write_reports(tmp_path, "NVDA", "2026-01-02", {}, "REVIEW", settings, [])
    for text in (target.read_text(), target.with_suffix(".html").read_text()):
        assert "observed-bull" in text and "configured-bull" not in text
        assert "configured-bear (Bear Researcher)" in text
        assert "configured-reflection (Reflection 42)" in text
        assert "generic-worker" not in text and "generic-manager" not in text


def test_legacy_claude_run_resumes_with_canonical_harness_names(tmp_path: Path) -> None:
    """Normalize saved group and role identities without invalidating resumable native sessions.

    Args:
        tmp_path: Isolated recorded run.
    """
    profile = HarnessSettings(kind="claude-code", model="sonnet")
    settings = Settings(state_dir=tmp_path, harness=profile, quick=profile, deep=profile, agents={"bear": profile})
    workflow = Workflow("NVDA", "2026-01-02", settings)
    recorded = workflow.store.metadata("input")
    for name in ("harness", "quick", "deep"):
        recorded["settings"][name]["kind"] = "claude"
    recorded["settings"]["agents"]["bear"]["kind"] = "claude"
    workflow.store.set_metadata("input", recorded)
    workflow.store.save_session("Bear Researcher", "legacy-native-session")
    workflow.store.close()

    resumed = Workflow.resume(workflow.directory)
    try:
        assert resumed.settings.model_settings().kind == "claude-code"
        assert resumed.settings.model_settings(True).kind == "claude-code"
        assert resumed.settings.model_settings(role="bear").kind == "claude-code"
        assert resumed.store.session("Bear Researcher") == "legacy-native-session"
    finally:
        resumed.store.close()
