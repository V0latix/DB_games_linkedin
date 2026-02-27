"""Helpers for timing and counting search metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class Timer:
    """Simple wall-clock timer."""

    _start: float | None = None

    def start(self) -> None:
        self._start = perf_counter()

    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        return (perf_counter() - self._start) * 1000.0
