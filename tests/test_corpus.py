"""Corpus loading, selection, and plugin registration."""

from __future__ import annotations

import pytest

from aij.corpus import load_bundled_objectives, load_objective_file, load_objectives
from aij.errors import CorpusError
from aij.models import Severity
from aij.techniques import build_registry, load_plugin_file


def test_bundled_corpus_parses() -> None:
    objectives = load_bundled_objectives()
    assert len(objectives) >= 20
    ids = {o.id for o in objectives}
    assert "cyber.reverse-shell" in ids
    assert "content.minor-safety-canary" in ids


def test_bundled_ids_are_unique() -> None:
    objectives = load_bundled_objectives()
    ids = [o.id for o in objectives]
    assert len(ids) == len(set(ids))


def test_every_bundled_objective_has_markers() -> None:
    for objective in load_bundled_objectives():
        assert objective.success_markers, f"{objective.id} has no success markers"


def test_canary_is_redacted() -> None:
    objectives = {o.id: o for o in load_bundled_objectives()}
    assert objectives["content.minor-safety-canary"].redact_transcript is True


def test_select_by_category() -> None:
    selected = load_objectives(["cyber"])
    assert selected
    assert all(o.category == "cyber" for o in selected)


def test_select_by_id() -> None:
    selected = load_objectives(["cyber.reverse-shell"])
    assert [o.id for o in selected] == ["cyber.reverse-shell"]


def test_select_by_glob() -> None:
    selected = load_objectives(["harm.*"])
    assert len(selected) >= 4
    assert all(o.category == "harm" for o in selected)


def test_all_returns_every_non_opt_in_objective() -> None:
    everything = load_bundled_objectives()
    opt_in = [o for o in everything if o.opt_in]
    assert opt_in, "expected at least one opt-in objective in the corpus"
    assert len(load_objectives(["all"])) == len(everything) - len(opt_in)
    assert len(load_objectives(None)) == len(everything) - len(opt_in)


def test_opt_in_objective_runs_only_when_named() -> None:
    canary = next(o for o in load_bundled_objectives() if o.opt_in)
    # Category selection skips it.
    assert canary.id not in {o.id for o in load_objectives(["content"])}
    # A glob that hits only the opt-in objective explains itself.
    with pytest.raises(CorpusError, match="opt-in"):
        load_objectives(["*minor-safety*"])
    # Exact id reaches it.
    assert [o.id for o in load_objectives([canary.id])] == [canary.id]


def test_unknown_selector_raises() -> None:
    with pytest.raises(CorpusError, match="no objectives match"):
        load_objectives(["does.not.exist"])


def test_duplicate_ids_across_specs_are_deduped() -> None:
    selected = load_objectives(["cyber", "cyber.reverse-shell"])
    ids = [o.id for o in selected]
    assert len(ids) == len(set(ids))


def test_custom_file(tmp_path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        """
objectives:
  - id: custom.one
    category: custom
    severity: low
    prompt: test
    success_markers: ["x"]
"""
    )
    objectives = load_objective_file(path)
    assert len(objectives) == 1
    assert objectives[0].severity is Severity.LOW
    assert objectives[0].category == "custom"


def test_custom_dir_is_loaded(tmp_path) -> None:
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "mine.yaml").write_text(
        """
objectives:
  - id: custom.dir
    category: custom
    prompt: test
    success_markers: ["x"]
"""
    )
    objectives = load_objectives(["custom.dir"], corpus_dirs=[directory])
    assert len(objectives) == 1


def test_invalid_yaml_raises(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("objectives: [unclosed")
    with pytest.raises(CorpusError, match="invalid YAML"):
        load_objective_file(path)


def test_missing_markers_is_allowed(tmp_path) -> None:
    path = tmp_path / "sparse.yaml"
    path.write_text(
        """
objectives:
  - id: custom.sparse
    category: custom
    prompt: test
"""
    )
    objectives = load_objective_file(path)
    assert objectives[0].success_markers == []


def test_invalid_id_rejected(tmp_path) -> None:
    path = tmp_path / "badid.yaml"
    path.write_text(
        """
objectives:
  - id: "bad id with spaces"
    prompt: test
"""
    )
    with pytest.raises(CorpusError, match="invalid"):
        load_objective_file(path)


def test_duplicate_conflicting_definitions_raise(tmp_path) -> None:
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "a.yaml").write_text(
        """
objectives:
  - id: cyber.reverse-shell
    category: cyber
    prompt: conflicting
    success_markers: ["different"]
"""
    )
    with pytest.raises(CorpusError, match="conflicting"):
        load_objectives(["cyber.reverse-shell"], corpus_dirs=[directory])


# -- techniques --------------------------------------------------------------


def test_builtin_registry() -> None:
    registry = build_registry()
    assert len(registry) >= 15
    assert "direct" in registry
    assert "crescendo" in registry


def test_resolve_glob() -> None:
    registry = build_registry()
    selected = registry.resolve(["encoding-*"])
    assert len(selected) == 3


def test_resolve_unknown_raises() -> None:
    registry = build_registry()
    with pytest.raises(CorpusError, match="no techniques match"):
        registry.resolve(["nonexistent"])


def test_plugin_registration(tmp_path) -> None:
    plugin = tmp_path / "plug.py"
    plugin.write_text(
        """
from aij.models import Message, TechniqueSpec
from aij.techniques.base import Technique, TechniqueContext, user


class MyTechnique(Technique):
    spec = TechniqueSpec(id="mine", name="Mine", description="custom")

    def initial(self, objective, ctx: TechniqueContext):
        return [user(objective.prompt + " [wrapped]")]
"""
    )
    registry = build_registry([plugin])
    assert "mine" in registry
    technique = registry.get("mine")()
    messages = technique.initial(
        load_objectives(["cyber.reverse-shell"])[0],
        __import__("aij.techniques.base", fromlist=["TechniqueContext"]).TechniqueContext(
            rng=__import__("random").Random(0)
        ),
    )
    assert messages[0].content.endswith("[wrapped]")


def test_plugin_without_techniques_raises(tmp_path) -> None:
    plugin = tmp_path / "empty.py"
    plugin.write_text("x = 1\n")
    with pytest.raises(CorpusError, match="no Technique subclasses"):
        load_plugin_file(plugin, build_registry())


def test_plugin_syntax_error_raises(tmp_path) -> None:
    plugin = tmp_path / "broken.py"
    plugin.write_text("def oops(:\n")
    with pytest.raises(CorpusError, match="plugin raised"):
        load_plugin_file(plugin, build_registry())


def test_duplicate_plugin_id_raises(tmp_path) -> None:
    plugin = tmp_path / "dup.py"
    plugin.write_text(
        """
from aij.models import TechniqueSpec
from aij.techniques.base import Technique


class Dup(Technique):
    spec = TechniqueSpec(id="direct", name="Dup")

    def initial(self, objective, ctx):
        return []
"""
    )
    with pytest.raises(CorpusError, match="duplicate technique id"):
        load_plugin_file(plugin, build_registry())


def test_unreachable_min_marker_hits_rejected(tmp_path) -> None:
    """Duplicated markers must not satisfy a higher hit threshold."""
    tmp_path_corpus = tmp_path / "bad.yaml"
    tmp_path_corpus.write_text(
        """
objectives:
  - id: custom.unreachable
    category: custom
    prompt: test
    success_markers: ["one", "one"]
    min_marker_hits: 2
"""
    )
    with pytest.raises(CorpusError, match="unreachable"):
        load_objective_file(tmp_path_corpus)


def test_bundled_corpus_has_no_unreachable_objectives() -> None:
    for objective in load_bundled_objectives():
        distinct = len(set(objective.success_markers))
        assert objective.min_marker_hits <= max(distinct, 1), (
            f"{objective.id}: min_marker_hits={objective.min_marker_hits} "
            f"but only {distinct} distinct markers"
        )
