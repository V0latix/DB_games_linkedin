"""Core data structures shared across games and solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolveMetrics:
    """Standard metrics collected during solving."""

    nodes: int = 0
    backtracks: int = 0
    time_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveResult:
    """Container for a solve attempt."""

    solved: bool
    solution: Any | None
    metrics: SolveMetrics
    error: str | None = None
