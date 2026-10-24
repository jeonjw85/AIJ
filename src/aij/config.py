"""Configuration loading: TOML files, environment overrides, CLI precedence."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from aij.credentials import CredentialStore
from aij.errors import ConfigError
from aij.models import ProviderConfig, RunConfig

DEFAULT_CONFIG_NAMES = ("aij.toml", ".aij.toml")
DEFAULT_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": "",
}
DEFAULT_BASE_URL_ENVS = {
    "openai": "OPENAI_BASE_URL",
    "openai-compatible": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "gemini": "GEMINI_BASE_URL",
    "ollama": "OLLAMA_HOST",
}
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434",
}
_RUN_ID_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify_run_id(value: str) -> str:
    cleaned = _RUN_ID_SAFE.sub("-", value.strip()).strip("-")
    return cleaned[:80] or "run"


def find_config(explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    env_path = os.environ.get("AIJ_CONFIG")
    if env_path:
        candidate = Path(env_path).expanduser()
        if not candidate.is_file():
            raise ConfigError(f"AIJ_CONFIG points at a missing file: {candidate}")
        return candidate
    for name in DEFAULT_CONFIG_NAMES:
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    return None


def _load_toml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - passthrough
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def _coerce_float(value: Any, key: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number, got {value!r}") from exc


def _coerce_int(value: Any, key: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer, got {value!r}") from exc


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _env_map(env: dict[str, str] | None) -> dict[str, str]:
    return dict(os.environ) if env is None else env


def build_provider(raw: dict[str, Any], env: dict[str, str] | None = None) -> ProviderConfig:
    env = _env_map(env)
    kind = str(raw.get("kind") or raw.get("type") or "openai").strip()
    aliases = {
        "openai_compatible": "openai-compatible",
        "compatible": "openai-compatible",
        "google": "gemini",
        "google-ai": "gemini",
    }
    kind = aliases.get(kind, kind)
    if kind not in DEFAULT_KEY_ENVS:
        raise ConfigError(
            f"unknown provider kind {kind!r}; expected one of {sorted(DEFAULT_KEY_ENVS)}"
        )

    model = str(raw.get("model") or "").strip()
    if not model:
        raise ConfigError("provider.model is required")

    api_key_env = raw.get("api_key_env") or DEFAULT_KEY_ENVS.get(kind) or None
    api_key = raw.get("api_key")
    if not api_key and api_key_env:
        api_key = env.get(str(api_key_env))
    if kind != "ollama" and not api_key:
        hint = f"set {api_key_env} or provider.api_key" if api_key_env else "set provider.api_key"
        raise ConfigError(f"missing credentials for {kind}: {hint}")

    # Explicit config wins; then the standard per-provider env var; then the
    # built-in default. Without the middle step, a shell already pointed at a
    # gateway via ANTHROPIC_BASE_URL would silently talk to the real vendor.
    base_url = raw.get("base_url")
    if not base_url:
        env_name = DEFAULT_BASE_URL_ENVS.get(kind)
        if env_name:
            base_url = env.get(env_name)
    if not base_url:
        base_url = DEFAULT_BASE_URLS.get(kind)
    extra_headers = raw.get("extra_headers") or {}
    extra_body = raw.get("extra_body") or {}
    if not isinstance(extra_headers, dict) or not isinstance(extra_body, dict):
        raise ConfigError("provider.extra_headers and provider.extra_body must be tables")

    return ProviderConfig(
        kind=kind,  # type: ignore[arg-type]
        model=model,
        base_url=str(base_url) if base_url else None,
        api_key=str(api_key) if api_key else None,
        api_key_env=str(api_key_env) if api_key_env else None,
        temperature=float(raw.get("temperature", 1.0)),
        top_p=(float(raw["top_p"]) if raw.get("top_p") is not None else None),
        max_tokens=int(raw.get("max_tokens", 1024)),
        timeout_s=float(raw.get("timeout_s", 60.0)),
        extra_headers={str(k): str(v) for k, v in extra_headers.items()},
        extra_body=extra_body,
    )


def load_run_config(
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
    env: dict[str, str] | None = None,
    profile: str | None = None,
    credentials: CredentialStore | None = None,
    use_profiles: bool = True,
) -> RunConfig:
    """Merge stored profile -> TOML -> env -> explicit overrides.

    Precedence is deliberately "most specific wins": a flag beats the
    environment, the environment beats the project file, and a saved profile is
    the fallback that makes ``aij scan`` work with no arguments at all.
    """
    env = _env_map(env)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    path = find_config(config_path)
    data = _load_toml(path)

    provider_raw = dict(data.get("provider") or {})
    scan_raw = dict(data.get("scan") or {})
    corpus_raw = dict(data.get("corpus") or {})

    # Environment wins over the file for endpoint selection, so CI can retarget
    # the same config without editing it.
    if env.get("AIJ_MODEL"):
        provider_raw["model"] = env["AIJ_MODEL"]
    if env.get("AIJ_PROVIDER"):
        provider_raw["kind"] = env["AIJ_PROVIDER"]
    if env.get("AIJ_BASE_URL"):
        provider_raw["base_url"] = env["AIJ_BASE_URL"]
    else:
        # Standard per-provider base URL variables, so a shell already set up for
        # the SDKs works without repeating the URL on the command line.
        kind = str(provider_raw.get("kind") or provider_raw.get("type") or "openai")
        for env_name, env_kind in (
            ("OPENAI_BASE_URL", "openai"),
            ("OPENAI_BASE_URL", "openai-compatible"),
            ("ANTHROPIC_BASE_URL", "anthropic"),
            ("GEMINI_BASE_URL", "gemini"),
            ("OLLAMA_HOST", "ollama"),
        ):
            if kind == env_kind and env.get(env_name):
                provider_raw["base_url"] = env[env_name]
                break

    provider_raw.update({k: v for k, v in overrides.items() if k in ProviderConfig.model_fields})

    # Fill anything still missing from a saved profile. Explicit config always
    # wins, so a project file with a fixed model is not overridden silently.
    if use_profiles:
        store = credentials or CredentialStore()
        resolved = store.resolve(profile)
        if resolved is not None:
            _, values = resolved
            for key in ("kind", "model", "base_url", "api_key"):
                if not provider_raw.get(key) and values.get(key):
                    provider_raw[key] = values[key]
            for key in ("temperature", "max_tokens", "timeout_s"):
                if key not in provider_raw and key in values:
                    provider_raw[key] = values[key]

    if not provider_raw.get("model"):
        raise ConfigError(
            "no model configured: pass --model, set provider.model, or run "
            "'aij auth add' to save a profile"
        )

    scan = RunConfig.model_fields
    merged: dict[str, Any] = {}
    for key in scan:
        if key in ("run_id", "provider", "created_at", "aij_version"):
            continue
        if key in overrides:
            merged[key] = overrides[key]
        elif key in scan_raw:
            merged[key] = scan_raw[key]

    for key in ("concurrency", "max_requests", "retries"):
        if key in merged:
            coerced = _coerce_int(merged[key], key)
            if coerced is None:
                merged.pop(key)
            else:
                merged[key] = coerced
    for key in ("requests_per_second", "max_cost_usd", "max_seconds"):
        if key in merged:
            coerced_f = _coerce_float(merged[key], key)
            if coerced_f is None:
                merged.pop(key)
            else:
                merged[key] = coerced_f
    if "stop_on_success" in merged:
        merged["stop_on_success"] = _coerce_bool(merged["stop_on_success"])
    if "seed" in merged:
        merged["seed"] = _coerce_int(merged["seed"], "seed")

    if not merged.get("objectives"):
        merged["objectives"] = list(
            corpus_raw.get("objectives") or overrides.get("objectives") or []
        )
    if not merged.get("techniques"):
        merged["techniques"] = list(
            corpus_raw.get("techniques") or overrides.get("techniques") or []
        )

    if merged.get("concurrency", 1) < 1:
        raise ConfigError("scan.concurrency must be >= 1")

    from aij import __version__

    resolved_run_id = slugify_run_id(run_id or str(overrides.get("run_id") or "") or _auto_run_id())
    return RunConfig(
        run_id=resolved_run_id,
        provider=build_provider(provider_raw, env),
        aij_version=__version__,
        **merged,
    )


def _auto_run_id() -> str:
    from aij.models import utcnow

    return utcnow().strftime("run-%Y%m%dT%H%M%SZ")
