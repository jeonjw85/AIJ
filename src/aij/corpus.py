"""Objective corpus loading and validation."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from aij.errors import CorpusError
from aij.models import Objective

BUNDLED_PACKAGE = "aij.data.objectives"


def _parse_document(payload: object, source: str) -> list[Objective]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        entries = payload.get("objectives", payload.get("items"))
        if entries is None:
            raise CorpusError(f"{source}: mapping must contain an 'objectives' key")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise CorpusError(f"{source}: expected a list or mapping, got {type(payload).__name__}")

    if not isinstance(entries, list):
        raise CorpusError(f"{source}: 'objectives' must be a list")

    parsed: list[Objective] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CorpusError(f"{source}: objective #{index} is not a mapping")
        merged = dict(entry)
        merged.setdefault("category", Path(source).stem)
        try:
            parsed.append(Objective.model_validate(merged))
        except ValidationError as exc:
            raise CorpusError(f"{source}: objective #{index} invalid: {exc}") from exc
    return parsed


def load_objective_file(path: Path) -> list[Objective]:
    if not path.is_file():
        raise CorpusError(f"objective file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        documents = list(yaml.safe_load_all(text))
    except yaml.YAMLError as exc:
        raise CorpusError(f"{path}: invalid YAML: {exc}") from exc
    objectives: list[Objective] = []
    for document in documents:
        objectives.extend(_parse_document(document, str(path)))
    return objectives


def load_bundled_objectives() -> list[Objective]:
    objectives: list[Objective] = []
    try:
        root = resources.files(BUNDLED_PACKAGE)
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging error
        raise CorpusError("bundled corpus is missing from the installation") from exc
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith((".yaml", ".yml")):
            continue
        with entry.open("r", encoding="utf-8") as handle:
            documents = list(yaml.safe_load_all(handle))
        for document in documents:
            objectives.extend(_parse_document(document, entry.name))
    return objectives


def load_objectives(
    specs: Iterable[str] | None = None,
    *,
    corpus_dirs: Iterable[Path] | None = None,
    extra_files: Iterable[Path] | None = None,
    include_bundled: bool = True,
) -> list[Objective]:
    """Resolve objectives from bundled data plus user YAML.

    ``specs`` accepts objective ids, category names, glob patterns, or the
    literal ``all``. Ordering follows the order specs were given, and every
    objective appears at most once.
    """
    pool: dict[str, Objective] = {}
    if include_bundled:
        for objective in load_bundled_objectives():
            _add(pool, objective)

    paths: list[Path] = []
    for directory in corpus_dirs or []:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            raise CorpusError(f"corpus directory not found: {directory}")
        paths.extend(sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")))
    for extra in extra_files or []:
        paths.append(Path(extra).expanduser())
    for path in paths:
        for objective in load_objective_file(path):
            _add(pool, objective)

    requested = [s for s in (specs or []) if s and s != "all"]
    if not requested:
        return sorted((o for o in pool.values() if not o.opt_in), key=lambda o: (o.category, o.id))

    # A selector can match nothing because it is wrong, or because it matches
    # only opt-in objectives. Those need different messages.
    from fnmatch import fnmatch

    opt_in_hits = {
        o.id
        for o in pool.values()
        if o.opt_in and any(o.category == spec or fnmatch(o.id, spec) for spec in requested)
    }

    selected: list[Objective] = []
    seen: set[str] = set()
    for spec in requested:
        matches = _match(pool, spec)
        if not matches:
            if opt_in_hits:
                raise CorpusError(
                    f"{spec!r} matches only opt-in objective(s): "
                    f"{', '.join(sorted(opt_in_hits))}. Name the exact id to run one."
                )
            raise CorpusError(f"no objectives match {spec!r}; run 'aij list objectives' to see ids")
        for objective in matches:
            if objective.id not in seen:
                seen.add(objective.id)
                selected.append(objective)
    return selected


def _add(pool: dict[str, Objective], objective: Objective) -> None:
    existing = pool.get(objective.id)
    if existing is not None and existing != objective:
        raise CorpusError(f"duplicate objective id {objective.id!r} with conflicting definitions")
    pool[objective.id] = objective


def _match(pool: dict[str, Objective], spec: str) -> list[Objective]:
    from fnmatch import fnmatch

    # Naming the exact id is the only way to reach an opt-in objective.
    if spec in pool:
        return [pool[spec]]
    candidates = [o for o in pool.values() if not o.opt_in]
    matched = [o for o in candidates if o.category == spec or fnmatch(o.id, spec)]
    if matched:
        return sorted(matched, key=lambda o: o.id)
    matched = [o for o in candidates if spec in o.tags]
    return sorted(matched, key=lambda o: o.id)
