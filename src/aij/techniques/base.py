"""Technique base classes and context."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from aij.models import Message, Objective, Role, TechniqueSpec


@dataclass(slots=True)
class TechniqueContext:
    """Per-probe state handed to techniques.

    Techniques may stash arbitrary per-probe values in ``state`` (for example a
    randomly chosen escalation ladder) so later turns can reuse the exact
    choice made on turn zero.
    """

    rng: random.Random
    notes: list[str] = field(default_factory=list)
    state: dict[str, object] = field(default_factory=dict)
    max_turns: int = 6

    def note(self, text: str) -> None:
        self.notes.append(text)


class Technique(ABC):
    """Transforms an objective into one or more conversation turns.

    The scanner drives the conversation:

    1. ``initial`` produces the opening messages.
    2. The target answers.
    3. ``followup`` is called with the transcript so far and may return the
       next user message, or ``None`` to end the probe.

    Single-turn techniques simply never return a followup.
    """

    spec: TechniqueSpec

    @abstractmethod
    def initial(self, objective: Objective, ctx: TechniqueContext) -> list[Message]:
        """Build the first request payload."""

    def followup(
        self,
        objective: Objective,
        *,
        turn_index: int,
        transcript: list[Message],
        last_response: str,
        ctx: TechniqueContext,
    ) -> list[Message] | None:
        """Return the next user payload, or ``None`` to stop."""
        return None

    def scores_turn(self, objective: Objective, payload: list[Message], turn_index: int) -> bool:
        """Whether this turn's response should count toward the probe verdict.

        Multi-turn techniques that spend early turns on benign warm-up override
        this so the warm-up turns are not scored. Scoring them would turn an
        honest final refusal into "inconclusive", because a benign preamble
        neither complies nor refuses.
        """
        return True

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} id={self.spec.id}>"


def user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def assistant(content: str) -> Message:
    return Message(role=Role.ASSISTANT, content=content)


def system(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)
