"""Core data structures for the Zip puzzle."""

from __future__ import annotations

from dataclasses import dataclass

Cell = tuple[int, int]


@dataclass
class ZipPuzzle:
    game: str
    n: int
    numbers: dict[int, Cell]
    walls: set[tuple[Cell, Cell]]
    neighbors: dict[Cell, list[Cell]]


@dataclass
class ZipSolution:
    path: list[Cell]

    def order_grid(self, n: int) -> list[list[int]]:
        grid = [[-1 for _ in range(n)] for _ in range(n)]
        for idx, (r, c) in enumerate(self.path):
            grid[r][c] = idx
        return grid
