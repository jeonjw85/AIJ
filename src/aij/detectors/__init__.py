"""Detector registry."""

from __future__ import annotations

from aij.detectors.base import Detector
from aij.detectors.heuristic import HeuristicDetector, score_refusal
from aij.detectors.judge import EscalatingDetector, JudgeDetector

__all__ = [
    "Detector",
    "EscalatingDetector",
    "HeuristicDetector",
    "JudgeDetector",
    "score_refusal",
]
