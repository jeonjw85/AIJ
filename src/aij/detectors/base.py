"""Detector interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aij.models import Detection, Objective


class Detector(ABC):
    """Decides whether a response complied with an objective."""

    name: str = "detector"

    @abstractmethod
    async def detect(
        self,
        objective: Objective,
        response_text: str,
        *,
        turn_index: int = 0,
    ) -> Detection:
        """Return a verdict for one response."""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} name={self.name}>"
