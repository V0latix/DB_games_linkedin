"""Validation for Zip puzzle solutions."""

from __future__ import annotations

from dataclasses import dataclass

from .model import Cell, ZipPuzzle, ZipSolution


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None = None


def validate_solution(puzzle: ZipPuzzle, solution: ZipSolution) -> ValidationResult:
    n = puzzle.n
    path = solution.path

    if len(path) != n * n:
        return ValidationResult(False, "path length must be n*n")

    seen: set[Cell] = set()
    for idx, cell in enumerate(path):
        r, c = cell
        if not (0 <= r < n and 0 <= c < n):
            return ValidationResult(False, f"cell {cell} out of bounds")
        if cell in seen:
            return ValidationResult(False, f"cell {cell} visited more than once")
        seen.add(cell)
        if idx > 0:
            prev = path[idx - 1]
            if cell not in puzzle.neighbors[prev]:
                return ValidationResult(False, f"non-adjacent step {prev}->{cell}")

    if len(seen) != n * n:
        return ValidationResult(False, "path does not cover all cells")

    positions = {cell: i for i, cell in enumerate(path)}
    max_k = max(puzzle.numbers.keys())
    for k in range(1, max_k + 1):
        if k not in puzzle.numbers:
            return ValidationResult(False, f"missing number k={k}")

    last_index = -1
    for k in range(1, max_k + 1):
        cell = puzzle.numbers[k]
        if cell not in positions:
            return ValidationResult(False, f"number k={k} not visited")
        idx = positions[cell]
        if idx <= last_index:
            return ValidationResult(False, f"numbers not in order at k={k}")
        last_index = idx

    return ValidationResult(True, None)
