"""Technique registry and loading from user-supplied Python plugins."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path

from aij.errors import CorpusError
from aij.models import TechniqueSpec
from aij.techniques.base import Technique
from aij.techniques.multi import (
    CrescendoTechnique,
    PersonaDriftTechnique,
    RefusalRetryTechnique,
)
from aij.techniques.single import (
    AcademicTechnique,
    AuthorityTechnique,
    Base64Technique,
    ChainOfThoughtTechnique,
    CharacterSpacingTechnique,
    DirectTechnique,
    HypotheticalTechnique,
    ManyShotTechnique,
    PayloadSplittingTechnique,
    PersonaTechnique,
    PrefixInjectionTechnique,
    RefusalSuppressionTechnique,
    RoleplayInterviewTechnique,
    Rot13Technique,
    TranslationTechnique,
)

BUILTIN: tuple[type[Technique], ...] = (
    DirectTechnique,
    PersonaTechnique,
    AuthorityTechnique,
    RefusalSuppressionTechnique,
    HypotheticalTechnique,
    AcademicTechnique,
    RoleplayInterviewTechnique,
    ChainOfThoughtTechnique,
    PayloadSplittingTechnique,
    ManyShotTechnique,
    PrefixInjectionTechnique,
    Base64Technique,
    Rot13Technique,
    CharacterSpacingTechnique,
    TranslationTechnique,
    CrescendoTechnique,
    RefusalRetryTechnique,
    PersonaDriftTechnique,
)


class TechniqueRegistry:
    """Id-indexed collection of technique classes."""

    def __init__(self, techniques: Iterable[type[Technique]] = ()) -> None:
        self._by_id: dict[str, type[Technique]] = {}
        for technique in techniques:
            self.register(technique)

    def register(self, technique: type[Technique]) -> None:
        if not issubclass(technique, Technique):
            raise CorpusError(f"{technique!r} is not a Technique subclass")
        spec = getattr(technique, "spec", None)
        if not isinstance(spec, TechniqueSpec):
            raise CorpusError(f"{technique.__name__} must define a TechniqueSpec 'spec'")
        if spec.id in self._by_id and self._by_id[spec.id] is not technique:
            raise CorpusError(f"duplicate technique id {spec.id!r}")
        self._by_id[spec.id] = technique

    def get(self, technique_id: str) -> type[Technique]:
        try:
            return self._by_id[technique_id]
        except KeyError as exc:
            raise CorpusError(
                f"unknown technique {technique_id!r}; available: {', '.join(self.ids())}"
            ) from exc

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def specs(self) -> list[TechniqueSpec]:
        return sorted((t.spec for t in self._by_id.values()), key=lambda s: s.id)

    def resolve(self, specs: Iterable[str] | None) -> list[type[Technique]]:
        requested = [s for s in (specs or []) if s and s != "all"]
        if not requested:
            return [self._by_id[i] for i in self.ids()]

        selected: list[type[Technique]] = []
        seen: set[str] = set()
        for spec in requested:
            matches = (
                [spec]
                if spec in self._by_id
                else sorted(i for i in self._by_id if fnmatch(i, spec))
            )
            if not matches:
                raise CorpusError(f"no techniques match {spec!r}; run 'aij list techniques'")
            for technique_id in matches:
                if technique_id not in seen:
                    seen.add(technique_id)
                    selected.append(self._by_id[technique_id])
        return selected

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, technique_id: object) -> bool:
        return technique_id in self._by_id


def load_plugin_file(path: Path, registry: TechniqueRegistry) -> list[str]:
    """Import a Python file and register every Technique subclass it defines."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise CorpusError(f"technique plugin not found: {path}")
    module_name = f"aij_plugin_{path.stem}_{abs(hash(path)) % 10_000}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise CorpusError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CorpusError(f"{path}: plugin raised {type(exc).__name__}: {exc}") from exc

    added: list[str] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module_name:
            continue
        if not issubclass(obj, Technique) or obj is Technique:
            continue
        registry.register(obj)
        added.append(obj.spec.id)
    if not added:
        raise CorpusError(f"{path}: no Technique subclasses found")
    return added


def build_registry(plugin_paths: Iterable[Path] | None = None) -> TechniqueRegistry:
    registry = TechniqueRegistry(BUILTIN)
    for path in plugin_paths or []:
        load_plugin_file(Path(path), registry)
    return registry


DEFAULT_REGISTRY = TechniqueRegistry(BUILTIN)
