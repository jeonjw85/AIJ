"""Config resolution and precedence."""

from __future__ import annotations

import pytest

from aij.config import load_run_config, slugify_run_id
from aij.errors import ConfigError

TOML = """
[provider]
kind = "openai"
model = "gpt-4o-mini"
temperature = 0.3

[scan]
concurrency = 7
max_cost_usd = 1.5
retries = 5

[corpus]
objectives = ["cyber"]
techniques = ["direct"]
"""


@pytest.fixture
def env() -> dict[str, str]:
    return {"OPENAI_API_KEY": "sk-test"}


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "aij.toml"
    path.write_text(TOML)
    return path


def test_loads_toml(config_file, env) -> None:
    config = load_run_config(env=env)
    assert config.provider.kind == "openai"
    assert config.provider.model == "gpt-4o-mini"
    assert config.provider.temperature == 0.3
    assert config.concurrency == 7
    assert config.max_cost_usd == 1.5
    assert config.retries == 5
    assert config.objectives == ["cyber"]
    assert config.techniques == ["direct"]


def test_overrides_beat_toml(config_file, env) -> None:
    config = load_run_config(overrides={"concurrency": 2, "model": "gpt-4o"}, env=env)
    assert config.concurrency == 2
    assert config.provider.model == "gpt-4o"


def test_env_beats_toml_for_endpoint(config_file) -> None:
    env = {"OPENAI_API_KEY": "sk-test", "AIJ_MODEL": "from-env"}
    config = load_run_config(env=env)
    assert config.provider.model == "from-env"


def test_missing_model_is_error(env) -> None:
    with pytest.raises(ConfigError, match="no model configured"):
        load_run_config(env=env)


def test_missing_api_key_is_error(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="missing credentials"):
        load_run_config(overrides={"model": "gpt-4o"}, env={})


def test_ollama_needs_no_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_run_config(overrides={"model": "llama3", "kind": "ollama"}, env={"PATH": ""})
    assert config.provider.kind == "ollama"
    assert config.provider.api_key is None


def test_api_key_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "aij.toml"
    path.write_text(
        """
[provider]
kind = "openai"
model = "m"
api_key_env = "MY_KEY"
"""
    )
    config = load_run_config(env={"MY_KEY": "sk-custom"})
    assert config.provider.api_key == "sk-custom"
    assert config.provider.api_key_env == "MY_KEY"


def test_unknown_provider_kind(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="unknown provider kind"):
        load_run_config(overrides={"model": "m", "kind": "wat"}, env={})


def test_openai_compatible_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_run_config(
        overrides={"model": "m", "kind": "compatible", "base_url": "http://x/v1"},
        env={"OPENAI_API_KEY": "k"},
    )
    assert config.provider.kind == "openai-compatible"


def test_missing_config_path_is_error(tmp_path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_run_config(config_path=tmp_path / "nope.toml", env={"OPENAI_API_KEY": "k"})


def test_run_id_is_slugified(config_file, env) -> None:
    config = load_run_config(run_id="My Run / 2026!!", env=env)
    assert config.run_id == "My-Run-2026"


def test_auto_run_id_shape(config_file, env) -> None:
    config = load_run_config(env=env)
    assert config.run_id.startswith("run-")


def test_slugify_run_id_handles_empty() -> None:
    assert slugify_run_id("///") == "run"


def test_concurrency_must_be_positive(config_file, env) -> None:
    with pytest.raises(ConfigError, match="concurrency"):
        load_run_config(overrides={"concurrency": 0}, env=env)


def test_anthropic_base_url_env_is_honored(tmp_path, monkeypatch) -> None:
    """A shell already configured for the SDK should just work."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "aij.toml"
    path.write_text('[provider]\nkind = "anthropic"\nmodel = "claude-x"\n')
    config = load_run_config(
        env={
            "ANTHROPIC_API_KEY": "sk-ant",
            "ANTHROPIC_BASE_URL": "https://proxy.internal/anthropic",
        }
    )
    assert config.provider.base_url == "https://proxy.internal/anthropic"


def test_explicit_base_url_beats_env(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_run_config(
        overrides={
            "model": "m",
            "kind": "openai",
            "base_url": "https://explicit.example/v1",
        },
        env={"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://env.example/v1"},
    )
    assert config.provider.base_url == "https://explicit.example/v1"


def test_aij_base_url_beats_provider_env(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_run_config(
        overrides={"model": "m", "kind": "openai"},
        env={
            "OPENAI_API_KEY": "k",
            "OPENAI_BASE_URL": "https://openai.example/v1",
            "AIJ_BASE_URL": "https://override.example/v1",
        },
    )
    assert config.provider.base_url == "https://override.example/v1"


def test_build_provider_honors_env_base_url() -> None:
    """Without this, a shell pointed at a gateway silently hits the real vendor."""
    from aij.config import build_provider

    config = build_provider(
        {"kind": "anthropic", "model": "m"},
        {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "https://proxy.internal/anthropic"},
    )
    assert config.base_url == "https://proxy.internal/anthropic"


def test_explicit_base_url_beats_env_in_build_provider() -> None:
    from aij.config import build_provider

    config = build_provider(
        {"kind": "openai", "model": "m", "base_url": "https://explicit/v1"},
        {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://env/v1"},
    )
    assert config.base_url == "https://explicit/v1"


def test_no_env_falls_back_to_default() -> None:
    from aij.config import build_provider

    config = build_provider({"kind": "openai", "model": "m"}, {"OPENAI_API_KEY": "k"})
    assert config.base_url == "https://api.openai.com/v1"
