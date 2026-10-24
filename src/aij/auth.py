"""The ``aij auth`` command group: save, inspect, and test endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from aij.credentials import (
    BASE_URL_ENV_VARS,
    KEY_ENV_VARS,
    CredentialStore,
    mask_secret,
)
from aij.errors import AIJError, ConfigError
from aij.models import Message, Role
from aij.providers import build_provider

auth_app = typer.Typer(
    name="auth",
    help="Save API keys and endpoints so scans run without repeating flags.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)

PROVIDER_KINDS = ["openai", "anthropic", "gemini", "ollama", "openai-compatible"]


def _choose_provider() -> str:
    """Interactive provider picker.

    ``typer.prompt(type=...)`` changed shape across Typer versions, so the
    choices are validated here instead of relying on it.
    """
    while True:
        raw = typer.prompt(f"Provider ({'/'.join(PROVIDER_KINDS)})", default="openai").strip()
        if raw in PROVIDER_KINDS:
            return raw
        console.print(f"[red]unknown provider[/red] {raw!r}")


@auth_app.command("add")
def add_command(
    name: Annotated[str, typer.Argument(help="Profile name, e.g. 'openai' or 'corp-gateway'.")],
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="openai | anthropic | gemini | ollama | openai-compatible",
        ),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Default model.")] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", "-k", help="API key. Omit to be prompted (hidden input)."),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", "-u", help="Endpoint base URL or gateway root.")
    ] = None,
    api_key_env: Annotated[
        str | None,
        typer.Option("--from-env", help="Read the key from this env var instead of storing it."),
    ] = None,
    make_default: Annotated[
        bool, typer.Option("--default", help="Use this profile when none is named.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing profile.")] = False,
    store_path: Annotated[
        Path | None, typer.Option("--store", help="Override the credentials file location.")
    ] = None,
) -> None:
    """Save an endpoint profile.

    With no flags this prompts for everything, which is the intended first-run
    path. The key is read with hidden input so it never lands in shell history.
    """
    store = CredentialStore(store_path)

    if store.exists() and name in store.load() and not force:
        err_console.print(
            f"[yellow]profile {escape(name)!r} already exists[/yellow]; pass --force to overwrite"
        )
        raise typer.Exit(1)

    if not provider:
        provider = _choose_provider()
    if provider not in PROVIDER_KINDS:
        err_console.print(
            f"[red]unknown provider[/red] {escape(provider)!r}; "
            f"expected {', '.join(PROVIDER_KINDS)}"
        )
        raise typer.Exit(2)

    if not model:
        model = typer.prompt("Default model", default={"ollama": "llama3.1"}.get(provider, ""))
        if not model:
            err_console.print("[red]a model is required[/red]")
            raise typer.Exit(2)

    resolved_key: str | None = None
    if api_key_env:
        import os

        resolved_key = os.environ.get(api_key_env)
        if not resolved_key:
            err_console.print(f"[red]{escape(api_key_env)} is not set[/red] in this shell")
            raise typer.Exit(2)
    elif api_key:
        resolved_key = api_key
    elif provider != "ollama":
        env_name = KEY_ENV_VARS.get(provider, "")
        from_env = __import__("os").environ.get(env_name) if env_name else None
        if from_env:
            use_it = typer.confirm(f"Use {env_name} from the current environment?", default=True)
            if use_it:
                resolved_key = from_env
        if not resolved_key:
            resolved_key = typer.prompt(
                f"{provider} API key", hide_input=True, default="", show_default=False
            )
            if not resolved_key:
                err_console.print("[red]an API key is required[/red] (or use --from-env)")
                raise typer.Exit(2)

    if not base_url and provider in {"openai-compatible", "ollama"}:
        # These two have no useful public default, so ask.
        base_url = (
            typer.prompt(
                "Base URL",
                default="http://localhost:11434" if provider == "ollama" else "",
                show_default=provider == "ollama",
            )
            or None
        )
    elif not base_url and provider not in PROVIDER_KINDS:
        base_url = None

    try:
        path = store.add(
            name,
            kind=provider,
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            make_default=make_default,
        )
    except ConfigError as exc:
        err_console.print(f"[red]error[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    console.print(f"[green]saved[/green] profile [cyan]{escape(name)}[/cyan] → {escape(str(path))}")
    console.print(
        f"  {escape(provider)} / {escape(model)}"
        + (f" · {escape(base_url)}" if base_url else "")
        + f" · key {escape(mask_secret(resolved_key))}"
    )
    console.print(f"  next: [bold]aij scan --profile {escape(name)} --dry-run[/bold]")


@auth_app.command("list")
def list_command(
    store_path: Annotated[
        Path | None, typer.Option("--store", help="Override the credentials file location.")
    ] = None,
    show_keys: Annotated[
        bool, typer.Option("--show-keys", help="Print masked keys and base URLs.")
    ] = False,
) -> None:
    """List saved profiles."""
    store = CredentialStore(store_path)
    profiles = store.load()
    if not profiles:
        console.print(f"[dim]no profiles saved[/dim] ({escape(str(store.path))})")
        console.print("add one with [bold]aij auth add openai[/bold]")
        return

    table = Table(title=f"profiles ({len(profiles)})")
    table.add_column("name", style="cyan")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("default", justify="center")
    if show_keys:
        table.add_column("base URL")
        table.add_column("key")
    for name in sorted(profiles):
        values = profiles[name]
        row = [
            escape(name),
            escape(str(values.get("kind", "?"))),
            escape(str(values.get("model", "?"))),
            "●" if values.get("default") else "",
        ]
        if show_keys:
            row.append(escape(str(values.get("base_url") or "(default)")))
            row.append(escape(mask_secret(values.get("api_key"))))
        table.add_row(*row)
    console.print(table)
    console.print(f"[dim]{escape(str(store.path))}[/dim]")


@auth_app.command("remove")
def remove_command(
    name: Annotated[str, typer.Argument(help="Profile to delete.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    store_path: Annotated[
        Path | None, typer.Option("--store", help="Override the credentials file location.")
    ] = None,
) -> None:
    """Delete a saved profile."""
    store = CredentialStore(store_path)
    if name not in store.load():
        err_console.print(f"[red]no profile named[/red] {escape(name)!r}")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Delete profile {name!r}?"):
        raise typer.Exit(0)
    store.remove(name)
    console.print(f"[green]removed[/green] {escape(name)}")


@auth_app.command("default")
def default_command(
    name: Annotated[str, typer.Argument(help="Profile to use when none is named.")],
    store_path: Annotated[
        Path | None, typer.Option("--store", help="Override the credentials file location.")
    ] = None,
) -> None:
    """Mark a profile as the default."""
    store = CredentialStore(store_path)
    try:
        store.set_default(name)
    except ConfigError as exc:
        err_console.print(f"[red]error[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    console.print(f"[green]default[/green] → {escape(name)}")


@auth_app.command("test")
def test_command(
    name: Annotated[
        str | None, typer.Argument(help="Profile to test. Defaults to the default profile.")
    ] = None,
    store_path: Annotated[
        Path | None, typer.Option("--store", help="Override the credentials file location.")
    ] = None,
    timeout: Annotated[float, typer.Option("--timeout", help="Seconds to wait.")] = 30.0,
) -> None:
    """Send one real request to verify the profile works.

    This is the check that catches the failure modes a saved key alone cannot:
    wrong key, unpaid account, missing model entitlement, wrong base URL path.
    """
    store = CredentialStore(store_path)
    try:
        resolved = store.resolve(name)
    except ConfigError as exc:
        err_console.print(f"[red]error[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if resolved is None:
        err_console.print(
            f"[red]no profiles saved[/red] in {escape(str(store.path))}; run 'aij auth add' first"
        )
        raise typer.Exit(2)

    profile_name, values = resolved
    try:
        config = store.to_provider_config(values)
    except ConfigError as exc:
        err_console.print(f"[red]profile {escape(profile_name)}[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    async def check() -> None:
        import httpx

        provider = build_provider(config)
        console.print(
            f"testing [cyan]{escape(profile_name)}[/cyan] → "
            f"{escape(config.kind)}/{escape(config.model)}"
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await provider.complete(
                    [Message(role=Role.USER, content="Reply with the single word: pong")],
                    client=client,
                    max_retries=0,
                )
        except AIJError as exc:
            status = getattr(exc, "status", None)
            err_console.print(f"[red]failed[/red] {escape(str(exc))}")
            if status in (401, 403):
                err_console.print(
                    "  The key was rejected or the account is not entitled to this model. "
                    "Check the key, the model name, and your billing."
                )
            elif status == 404:
                err_console.print(
                    "  Model or URL path not found. Check --model and the base URL "
                    f"({escape(str(config.base_url or 'default'))})."
                )
            elif status is None:
                err_console.print(
                    f"  Could not reach {escape(str(config.base_url or 'the default endpoint'))}. "
                    "Check the base URL and your network."
                )
            raise typer.Exit(1) from exc

        if response.blocked:
            # A safety block still proves the key, model, and URL all work.
            console.print(
                f"[green]ok[/green] reached the endpoint (it returned a safety block: "
                f"{escape(response.block_reason or 'filtered')})"
            )
        else:
            console.print(f"[green]ok[/green] reply: {escape(repr(response.text.strip()[:80]))}")
        console.print(f"  {response.usage.total_tokens} tokens · {response.latency_ms:.0f}ms")

    asyncio.run(check())


@auth_app.command("path")
def path_command(
    store_path: Annotated[
        Path | None, typer.Option("--store", help="Override the credentials file location.")
    ] = None,
) -> None:
    """Print where credentials are stored."""
    store = CredentialStore(store_path)
    state = "exists" if store.exists() else "not created yet"
    console.print(f"{escape(str(store.path))} [dim]({state})[/dim]")


__all__ = ["BASE_URL_ENV_VARS", "KEY_ENV_VARS", "auth_app"]
