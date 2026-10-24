import json
import os
import sys
import time
from pathlib import Path

import pytest

from open_trading_agents.config import HarnessSettings
from open_trading_agents.harnesses import catalog


def test_pi_catalog_uses_native_available_models_and_exact_selectors(tmp_path: Path) -> None:
    """Offer configured providers and exact native model IDs without starting generation."""
    script = tmp_path / "pi.py"
    script.write_text('''import sys
assert "--list-models" in sys.argv and "--offline" in sys.argv
assert "--model" not in sys.argv and "--print" not in sys.argv
for row in [
    ("provider", "model", "context", "max-out", "thinking", "images"),
    ("local", "team/model-27b", "256K", "32K", "yes", "no"),
    ("other-provider", "team/model-27b", "128K", "16K", "no", "yes"),
]:
    print(f"{row[0]:<20}  {row[1]:<30}  {row[2]:<8}  {row[3]:<8}  {row[4]:<8}  {row[5]}")
''')
    result = catalog.discover_models(HarnessSettings(command=(sys.executable, str(script))))
    assert result.notice is None
    assert [(model.provider, model.id) for model in result.models] == [
        ("local", "team/model-27b"), ("other-provider", "team/model-27b"),
    ]


def test_codex_catalog_uses_native_pagination_without_a_thread(tmp_path: Path) -> None:
    """Use native model selectors across pages and never create a model turn."""
    script, requests = tmp_path / "codex.py", tmp_path / "requests.jsonl"
    script.write_text('''import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    with path.open("a") as handle:
        handle.write(line)
    method = request["method"]
    if method == "initialize":
        result = {"userAgent": "fixture"}
    elif method == "initialized":
        continue
    elif method == "model/list":
        if request["params"]["cursor"] is None:
            result = {"data": [{"id": "picker-id", "model": "native-a", "displayName": "Native A"}], "nextCursor": "page2"}
        else:
            assert request["params"]["cursor"] == "page2"
            result = {"data": [{"id": "other-id", "model": "native-b", "displayName": "Native B"},
                               {"model": "hidden-model", "hidden": True}], "nextCursor": None}
    else:
        raise AssertionError(method)
    print(json.dumps({"method": "native/notice", "params": {}}), flush=True)
    print(json.dumps({"id": request["id"], "result": result}), flush=True)
''')
    result = catalog.discover_models(HarnessSettings(kind="codex", command=(sys.executable, str(script), str(requests)), provider="configured"))
    assert result.notice is None
    assert [(model.id, model.label, model.provider) for model in result.models] == [
        ("native-a", "Native A", "configured"), ("native-b", "Native B", "configured"),
    ]
    assert [json.loads(line)["method"] for line in requests.read_text().splitlines()] == [
        "initialize", "initialized", "model/list", "model/list",
    ]


def test_claude_choices_follow_installed_help_instead_of_fixed_model_versions(tmp_path: Path) -> None:
    """Native help defines the aliases even when new model families are introduced."""
    script = tmp_path / "claude.py"
    script.write_text('''import sys
assert sys.argv[-1] == "--help"
print("""Options:
  --model <model>  Provide an alias (e.g. 'family-a', 'family-b')
                   or a model's full name.
  -n, --name <name>  Set a name, for example 'unrelated'.
""")
''')
    result = catalog.discover_models(HarnessSettings(kind="claude", command=(sys.executable, str(script))))
    assert [model.id for model in result.models] == ["family-a", "family-b"]
    assert all(model.provider is None for model in result.models)
    assert result.notice and "aliases" in result.notice


def test_failed_discovery_does_not_expose_native_diagnostics(tmp_path: Path) -> None:
    """A native startup error permits manual selection without displaying sensitive stderr."""
    script = tmp_path / "broken.py"
    script.write_text('import sys\nprint("api-key-secret", file=sys.stderr)\nsys.exit(1)\n')
    result = catalog.discover_models(HarnessSettings(command=(sys.executable, str(script))))
    assert not result.models
    assert result.notice and "could not list models" in result.notice
    assert "api-key-secret" not in result.notice


def test_custom_cli_arguments_do_not_offer_a_different_native_profile(tmp_path: Path) -> None:
    """Custom startup arguments require manual selection rather than another profile's catalog."""
    script, started = tmp_path / "custom.py", tmp_path / "started"
    script.write_text(f'from pathlib import Path\nPath({str(started)!r}).touch()\n')
    result = catalog.discover_models(HarnessSettings(kind="codex", command=(sys.executable, str(script)), extra_args=("--profile", "work")))
    assert not started.exists()
    assert not result.models
    assert result.notice and "extra CLI arguments" in result.notice


@pytest.mark.skipif(os.name != "posix", reason="Process group cleanup requires POSIX")
def test_discovery_timeout_terminates_descendants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stalled catalog wrapper cannot leave a child running after the setup prompt returns."""
    script, started, leaked = tmp_path / "slow.py", tmp_path / "started", tmp_path / "leaked"
    script.write_text('''import subprocess
import sys
import time
from pathlib import Path
subprocess.Popen([sys.executable, "-c", "import time; from pathlib import Path; time.sleep(0.8); Path(__import__('sys').argv[1]).touch()", sys.argv[2]])
Path(sys.argv[1]).touch()
time.sleep(10)
''')
    monkeypatch.setattr(catalog, "_TIMEOUT_SECONDS", 0.25)
    before = time.monotonic()
    result = catalog.discover_models(HarnessSettings(command=(sys.executable, str(script), str(started), str(leaked))))
    assert time.monotonic() - before < 2
    assert result.notice and "timed out" in result.notice
    assert started.exists()
    time.sleep(0.9)
    assert not leaked.exists()


def test_catalog_output_limit_closes_a_noisy_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected unbounded command output falls back to manual model selection."""
    script = tmp_path / "noisy.py"
    script.write_text('import time\nprint("x" * 4096, flush=True)\ntime.sleep(10)\n')
    monkeypatch.setattr(catalog, "_OUTPUT_LIMIT", 128)
    before = time.monotonic()
    result = catalog.discover_models(HarnessSettings(command=(sys.executable, str(script))))
    assert time.monotonic() - before < 2
    assert not result.models
    assert result.notice and "could not list models" in result.notice
