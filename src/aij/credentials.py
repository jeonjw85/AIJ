"""Stored endpoint profiles: provider, model, base URL, and API key.

Keys live in a single file with 0600 permissions, never in the project tree and
never in a report. The file is plain TOML because an operator may need to audit
or hand-edit it; that is a deliberate trade against an OS keychain, which would
make the tool unusable over SSH.

Layout::

    [profiles.openai]
    kind = "openai"
    model = "gpt-4o-mini"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-..."

    [profiles.corp]
    kind = "openai-compatible"
    model = "llama-3.1-70b"
    base_url = "https://gateway.internal/llm/v1"
    api_key = "gw-..."
    default = true
"""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path
from typing import Any

from aij.errors import ConfigError
from aij.models import ProviderConfig

CREDENTIALS_VERSION = 1
KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}
BASE_URL_ENV_VARS = {
    "openai": "OPENAI_BASE_URL",
    "openai-compatible": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "gemini": "GEMINI_BASE_URL",
    "ollama": "OLLAMA_HOST",
}


def credentials_path() -> Path:
    """Resolve the profile file, honoring XDG and an explicit override."""
    override = os.environ.get("AIJ_CREDENTIALS")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return root / "aij" / "credentials.toml"


class CredentialStore:
    """Read and write named endpoint profiles."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else credentials_path()

    # -- reading ------------------------------------------------------------

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            raw = tomllib.loads(self.path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{self.path}: invalid TOML: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read {self.path}: {exc}") from exc

        profiles = raw.get("profiles", raw)
        if not isinstance(profiles, dict):
            raise ConfigError(f"{self.path}: expected a [profiles] table")
        return {
            str(name): dict(values) for name, values in profiles.items() if isinstance(values, dict)
        }

    def names(self) -> list[str]:
        return sorted(self.load())

    def get(self, name: str) -> dict[str, Any]:
        profiles = self.load()
        if name not in profiles:
            known = ", ".join(sorted(profiles)) or "none"
            raise ConfigError(f"no profile named {name!r}; saved profiles: {known}")
        return profiles[name]

    def default_name(self) -> str | None:
        for name, values in self.load().items():
            if values.get("default"):
                return name
        return None

    def resolve(self, name: str | None = None) -> tuple[str, dict[str, Any]] | None:
        """Return the profile to use, or None when the store is empty.

        An explicit name must exist. With no name, the profile marked default is
        used, and a single unmarked profile is accepted without ceremony.
        """
        profiles = self.load()
        if not profiles:
            if name:
                raise ConfigError(f"no profile named {name!r}; {self.path} does not exist")
            return None
        if name:
            return name, self.get(name)
        marked = self.default_name()
        if marked:
            return marked, profiles[marked]
        if len(profiles) == 1:
            only = next(iter(profiles))
            return only, profiles[only]
        known = ", ".join(sorted(profiles))
        raise ConfigError(
            f"several profiles are saved ({known}); pick one with --profile or "
            "mark a default with 'aij auth add NAME --default'"
        )

    def to_provider_config(
        self, values: dict[str, Any], *, model: str | None = None
    ) -> ProviderConfig:
        """Build a ProviderConfig, filling gaps from the environment."""
        kind = str(values.get("kind") or "openai")
        resolved_model = model or values.get("model")
        if not resolved_model:
            raise ConfigError("the profile has no model; pass --model or set one with auth add")

        api_key = values.get("api_key")
        if not api_key:
            env_name = KEY_ENV_VARS.get(kind)
            if env_name:
                api_key = os.environ.get(env_name)

        base_url = values.get("base_url")
        if not base_url:
            env_name = BASE_URL_ENV_VARS.get(kind)
            if env_name:
                base_url = os.environ.get(env_name)

        if kind != "ollama" and not api_key:
            raise ConfigError(
                f"the profile has no API key for {kind}; "
                f"set {KEY_ENV_VARS.get(kind, 'an API key')} or re-add the profile"
            )

        return ProviderConfig(
            kind=kind,  # type: ignore[arg-type]
            model=str(resolved_model),
            base_url=str(base_url) if base_url else None,
            api_key=str(api_key) if api_key else None,
            temperature=float(values.get("temperature", 1.0)),
            max_tokens=int(values.get("max_tokens", 1024)),
            timeout_s=float(values.get("timeout_s", 60.0)),
        )

    # -- writing ------------------------------------------------------------

    def add(
        self,
        name: str,
        *,
        kind: str,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        make_default: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        if not name or any(c in name for c in " .[]\n\t"):
            raise ConfigError(
                "profile name must be non-empty and cannot contain spaces, dots, or brackets"
            )
        profiles = self.load()
        if make_default:
            for values in profiles.values():
                values.pop("default", None)

        entry: dict[str, Any] = {"kind": kind, "model": model}
        if base_url:
            entry["base_url"] = base_url
        if api_key:
            entry["api_key"] = api_key
        if make_default or len(profiles) == 0:
            entry["default"] = True
        if extra:
            entry.update(extra)
        profiles[name] = entry
        self._write(profiles)
        return self.path

    def remove(self, name: str) -> bool:
        profiles = self.load()
        if name not in profiles:
            return False
        was_default = profiles[name].get("default")
        del profiles[name]
        if was_default and profiles:
            # Promote another profile so the next command still works.
            next(iter(sorted(profiles)))
            profiles[sorted(profiles)[0]]["default"] = True
        self._write(profiles)
        return True

    def set_default(self, name: str) -> None:
        profiles = self.load()
        if name not in profiles:
            raise ConfigError(f"no profile named {name!r}")
        for values in profiles.values():
            values.pop("default", None)
        profiles[name]["default"] = True
        self._write(profiles)

    def _write(self, profiles: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"version = {CREDENTIALS_VERSION}", ""]
        for name in sorted(profiles):
            values = profiles[name]
            lines.append(f"[profiles.{_quote(name)}]")
            for key in (
                "kind",
                "model",
                "base_url",
                "api_key",
                "temperature",
                "max_tokens",
                "timeout_s",
            ):
                if key in values and values[key] is not None:
                    lines.append(f"{key} = {_toml_value(values[key])}")
            if values.get("default"):
                lines.append("default = true")
            lines.append("")

        payload = "\n".join(lines)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        # Create with 0600 from the start so the key is never world-readable,
        # not even for the moment between write and chmod.
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        tmp.replace(self.path)
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)


def _quote(name: str) -> str:
    if name.replace("-", "").replace("_", "").isalnum():
        return name
    escaped = name.replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def mask_secret(value: str | None, *, keep: int = 4) -> str:
    """Render a key for display without revealing it."""
    if not value:
        return "(none)"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}…{value[-keep:]} ({len(value)} chars)"
