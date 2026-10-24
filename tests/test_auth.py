"""Credential storage and the auth commands."""

from __future__ import annotations

import os
import stat

import pytest
from typer.testing import CliRunner

from aij.cli import app
from aij.credentials import CredentialStore, credentials_path, mask_secret
from aij.errors import ConfigError

runner = CliRunner()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AIJ_CREDENTIALS", str(tmp_path / "creds.toml"))
    return tmp_path / "creds.toml"


# -- store -------------------------------------------------------------------


def test_empty_store_has_no_profiles(store) -> None:
    creds = CredentialStore()
    assert creds.load() == {}
    assert creds.exists() is False


def test_add_and_read_back(store) -> None:
    creds = CredentialStore()
    creds.add("main", kind="openai", model="gpt-4o-mini", api_key="sk-abc")
    values = creds.get("main")
    assert values["kind"] == "openai"
    assert values["model"] == "gpt-4o-mini"
    assert values["api_key"] == "sk-abc"


def test_first_profile_becomes_default(store) -> None:
    creds = CredentialStore()
    creds.add("first", kind="openai", model="m", api_key="k")
    assert creds.default_name() == "first"


def test_second_profile_is_not_default(store) -> None:
    creds = CredentialStore()
    creds.add("first", kind="openai", model="m", api_key="k")
    creds.add("second", kind="openai", model="m2", api_key="k2")
    assert creds.default_name() == "first"


def test_make_default_moves_the_marker(store) -> None:
    creds = CredentialStore()
    creds.add("first", kind="openai", model="m", api_key="k")
    creds.add("second", kind="openai", model="m2", api_key="k2", make_default=True)
    assert creds.default_name() == "second"
    assert "default" not in creds.get("first")


def test_file_is_owner_only(store) -> None:
    """A key readable by other users on a shared box is a credential leak."""
    creds = CredentialStore()
    creds.add("main", kind="openai", model="m", api_key="sk-secret")
    mode = stat.S_IMODE(os.stat(store).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_file_roundtrips_special_characters(store) -> None:
    creds = CredentialStore()
    tricky = 'sk-"quoted"-and-\\backslash'
    creds.add("main", kind="openai", model="m", api_key=tricky)
    assert creds.get("main")["api_key"] == tricky


def test_remove_profile(store) -> None:
    creds = CredentialStore()
    creds.add("a", kind="openai", model="m", api_key="k")
    creds.add("b", kind="openai", model="m", api_key="k")
    assert creds.remove("a") is True
    assert creds.names() == ["b"]


def test_remove_missing_returns_false(store) -> None:
    assert CredentialStore().remove("nope") is False


def test_removing_default_promotes_another(store) -> None:
    creds = CredentialStore()
    creds.add("a", kind="openai", model="m", api_key="k", make_default=True)
    creds.add("b", kind="openai", model="m", api_key="k")
    creds.remove("a")
    assert creds.default_name() == "b"


def test_base_url_is_stored(store) -> None:
    creds = CredentialStore()
    creds.add(
        "corp",
        kind="openai-compatible",
        model="llama",
        api_key="gw",
        base_url="https://gateway.internal/llm",
    )
    assert creds.get("corp")["base_url"] == "https://gateway.internal/llm"


def test_resolve_uses_default(store) -> None:
    creds = CredentialStore()
    creds.add("a", kind="openai", model="m", api_key="k", make_default=True)
    creds.add("b", kind="openai", model="m", api_key="k")
    name, _ = creds.resolve()
    assert name == "a"


def test_resolve_single_profile_without_default(store) -> None:
    creds = CredentialStore()
    creds.add("only", kind="openai", model="m", api_key="k")
    # Clear the implicit default to exercise the single-profile path.
    profiles = creds.load()
    profiles["only"].pop("default", None)
    creds._write(profiles)
    name, _ = creds.resolve()
    assert name == "only"


def test_resolve_ambiguous_errors(store) -> None:
    creds = CredentialStore()
    creds.add("a", kind="openai", model="m", api_key="k")
    creds.add("b", kind="openai", model="m", api_key="k")
    profiles = creds.load()
    for values in profiles.values():
        values.pop("default", None)
    creds._write(profiles)
    with pytest.raises(ConfigError, match="pick one"):
        creds.resolve()


def test_resolve_named_missing_errors(store) -> None:
    creds = CredentialStore()
    creds.add("a", kind="openai", model="m", api_key="k")
    with pytest.raises(ConfigError, match="no profile named"):
        creds.resolve("nope")


def test_resolve_on_empty_store_returns_none(store) -> None:
    assert CredentialStore().resolve() is None


def test_invalid_profile_name_rejected(store) -> None:
    with pytest.raises(ConfigError, match="profile name"):
        CredentialStore().add("bad name", kind="openai", model="m", api_key="k")


def test_to_provider_config_requires_model(store) -> None:
    creds = CredentialStore()
    with pytest.raises(ConfigError, match="no model"):
        creds.to_provider_config({"kind": "openai", "api_key": "k"})


def test_to_provider_config_requires_key(store, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    creds = CredentialStore()
    with pytest.raises(ConfigError, match="no API key"):
        creds.to_provider_config({"kind": "openai", "model": "m"})


def test_to_provider_config_falls_back_to_env_key(store, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    creds = CredentialStore()
    config = creds.to_provider_config({"kind": "openai", "model": "m"})
    assert config.api_key == "sk-from-env"


def test_ollama_needs_no_key(store) -> None:
    creds = CredentialStore()
    config = creds.to_provider_config({"kind": "ollama", "model": "llama3"})
    assert config.kind == "ollama"
    assert config.api_key is None


def test_credentials_path_honors_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AIJ_CREDENTIALS", str(tmp_path / "x.toml"))
    assert credentials_path() == tmp_path / "x.toml"


def test_credentials_path_honors_xdg(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AIJ_CREDENTIALS", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert credentials_path() == tmp_path / "aij" / "credentials.toml"


def test_corrupt_file_raises(store) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("this is not toml [[[")
    with pytest.raises(ConfigError, match="invalid TOML"):
        CredentialStore().load()


def test_mask_secret() -> None:
    assert mask_secret(None) == "(none)"
    assert mask_secret("short") == "*****"
    masked = mask_secret("sk-1234567890abcdef")
    assert "1234" in masked or "cdef" in masked
    assert "567890" not in masked


# -- CLI ---------------------------------------------------------------------


def test_auth_add_writes_profile(store) -> None:
    result = runner.invoke(
        app,
        ["auth", "add", "main", "-p", "openai", "-m", "gpt-4o-mini", "-k", "sk-abc"],
    )
    assert result.exit_code == 0, result.output
    assert CredentialStore().get("main")["model"] == "gpt-4o-mini"


def test_auth_add_refuses_overwrite_without_force(store) -> None:
    CredentialStore().add("main", kind="openai", model="m", api_key="k")
    result = runner.invoke(app, ["auth", "add", "main", "-p", "openai", "-m", "other", "-k", "k2"])
    assert result.exit_code == 1
    assert CredentialStore().get("main")["model"] == "m"


def test_auth_add_force_overwrites(store) -> None:
    CredentialStore().add("main", kind="openai", model="m", api_key="k")
    result = runner.invoke(
        app,
        ["auth", "add", "main", "-p", "openai", "-m", "other", "-k", "k2", "--force"],
    )
    assert result.exit_code == 0
    assert CredentialStore().get("main")["model"] == "other"


def test_auth_add_rejects_unknown_provider(store) -> None:
    result = runner.invoke(app, ["auth", "add", "main", "-p", "wat", "-m", "m", "-k", "k"])
    assert result.exit_code == 2


def test_auth_add_from_env(store, monkeypatch) -> None:
    monkeypatch.setenv("MY_CORP_KEY", "sk-from-var")
    result = runner.invoke(
        app,
        [
            "auth",
            "add",
            "corp",
            "-p",
            "openai",
            "-m",
            "m",
            "--from-env",
            "MY_CORP_KEY",
        ],
    )
    assert result.exit_code == 0, result.output
    assert CredentialStore().get("corp")["api_key"] == "sk-from-var"


def test_auth_add_from_env_missing(store, monkeypatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    result = runner.invoke(
        app,
        ["auth", "add", "corp", "-p", "openai", "-m", "m", "--from-env", "NOPE_KEY"],
    )
    assert result.exit_code == 2


def test_auth_list_empty(store) -> None:
    result = runner.invoke(app, ["auth", "list"])
    assert result.exit_code == 0
    assert "no profiles saved" in result.output


def test_auth_list_shows_profiles(store) -> None:
    CredentialStore().add("main", kind="openai", model="gpt-4o-mini", api_key="sk-secret-key")
    result = runner.invoke(app, ["auth", "list"])
    assert result.exit_code == 0
    assert "main" in result.output
    assert "gpt-4o-mini" in result.output


def test_auth_list_never_leaks_full_key(store) -> None:
    CredentialStore().add("main", kind="openai", model="m", api_key="sk-verysecretvalue12345")
    result = runner.invoke(app, ["auth", "list", "--show-keys"])
    assert "verysecretvalue" not in result.output


def test_auth_remove(store) -> None:
    CredentialStore().add("main", kind="openai", model="m", api_key="k")
    result = runner.invoke(app, ["auth", "remove", "main", "--yes"])
    assert result.exit_code == 0
    assert CredentialStore().names() == []


def test_auth_remove_missing(store) -> None:
    result = runner.invoke(app, ["auth", "remove", "nope", "--yes"])
    assert result.exit_code == 1


def test_auth_default(store) -> None:
    creds = CredentialStore()
    creds.add("a", kind="openai", model="m", api_key="k")
    creds.add("b", kind="openai", model="m", api_key="k")
    result = runner.invoke(app, ["auth", "default", "b"])
    assert result.exit_code == 0
    assert CredentialStore().default_name() == "b"


def test_auth_path(store) -> None:
    result = runner.invoke(app, ["auth", "path"])
    assert result.exit_code == 0
    assert "creds.toml" in result.output


def test_scan_uses_default_profile_without_flags(store) -> None:
    """The whole point: 'aij scan' should just work after auth add."""
    CredentialStore().add(
        "main", kind="openai", model="gpt-4o-mini", api_key="sk-x", make_default=True
    )
    result = runner.invoke(
        app, ["scan", "--dry-run", "--objective", "cyber", "--technique", "direct"]
    )
    assert result.exit_code == 0, result.output
    assert "gpt-4o-mini" in result.output


def test_scan_profile_flag_selects_named(store) -> None:
    creds = CredentialStore()
    creds.add("a", kind="openai", model="model-a", api_key="k", make_default=True)
    creds.add("b", kind="openai", model="model-b", api_key="k")
    result = runner.invoke(
        app,
        ["scan", "--profile", "b", "--dry-run", "--objective", "cyber", "--technique", "direct"],
    )
    assert result.exit_code == 0
    assert "model-b" in result.output


def test_explicit_model_beats_profile(store) -> None:
    CredentialStore().add("a", kind="openai", model="from-profile", api_key="k")
    result = runner.invoke(
        app,
        [
            "scan",
            "--dry-run",
            "--model",
            "from-flag",
            "--objective",
            "cyber",
            "--technique",
            "direct",
        ],
    )
    assert result.exit_code == 0
    assert "from-flag" in result.output


# -- auth test against a live endpoint --------------------------------------


def test_auth_test_succeeds_against_working_endpoint(store, mock_target) -> None:
    from tests.conftest import MockTarget

    target: MockTarget = mock_target
    target.responder = lambda messages: "pong"
    creds = CredentialStore()
    creds.add(
        "live",
        kind="openai-compatible",
        model="mock",
        api_key="k",
        base_url=target.url,
        make_default=True,
    )
    result = runner.invoke(app, ["auth", "test", "live"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output
    assert "pong" in result.output


def test_auth_test_fails_on_401(store, mock_target) -> None:
    from tests.conftest import MockTarget

    target: MockTarget = mock_target
    target.fail_times = 10
    target.fail_status = 401
    creds = CredentialStore()
    creds.add("live", kind="openai-compatible", model="mock", api_key="bad", base_url=target.url)
    result = runner.invoke(app, ["auth", "test", "live"])
    assert result.exit_code == 1
    assert "rejected" in result.output or "not entitled" in result.output


def test_auth_test_fails_on_404(store, mock_target) -> None:
    from tests.conftest import MockTarget

    target: MockTarget = mock_target
    target.fail_times = 10
    target.fail_status = 404
    creds = CredentialStore()
    creds.add("live", kind="openai-compatible", model="ghost", api_key="k", base_url=target.url)
    result = runner.invoke(app, ["auth", "test", "live"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_auth_test_without_profiles(store) -> None:
    result = runner.invoke(app, ["auth", "test"])
    assert result.exit_code == 2
    assert "no profiles" in result.output


def test_auth_test_safety_block_still_counts_as_reachable(store, mock_target) -> None:
    """A filter response proves key, model, and URL all work."""
    from tests.conftest import MockTarget

    target: MockTarget = mock_target
    target.block_mode = "refusal_field"
    creds = CredentialStore()
    creds.add("live", kind="openai-compatible", model="mock", api_key="k", base_url=target.url)
    result = runner.invoke(app, ["auth", "test", "live"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output
