"""Parser for the Zip puzzle JSON format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model import Cell, ZipPuzzle


@dataclass
class _NumberEntry:
    k: int
    r: int
    c: int


def _validate_bounds(n: int, r: int, c: int) -> None:
    if not (0 <= r < n and 0 <= c < n):
        msg = f"cell ({r},{c}) is out of bounds for n={n}"
        raise ValueError(msg)


def _normalize_wall(a: Cell, b: Cell) -> tuple[Cell, Cell]:
    return (a, b) if a <= b else (b, a)


def _neighbors(n: int, r: int, c: int) -> list[Cell]:
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [(rr, cc) for rr, cc in candidates if 0 <= rr < n and 0 <= cc < n]


def _build_neighbors(n: int, walls: set[tuple[Cell, Cell]]) -> dict[Cell, list[Cell]]:
    neighbors: dict[Cell, list[Cell]] = {}
    for r in range(n):
        for c in range(n):
            cell_neighbors: list[Cell] = []
            for nr, nc in _neighbors(n, r, c):
                edge = _normalize_wall((r, c), (nr, nc))
                if edge in walls:
                    continue
                cell_neighbors.append((nr, nc))
            neighbors[(r, c)] = cell_neighbors
    return neighbors


def parse_puzzle_dict(payload: dict) -> ZipPuzzle:
    game = str(payload.get("game", ""))
    if game != "zip":
        msg = f"expected game='zip', got {game!r}"
        raise ValueError(msg)

    n = int(payload["n"])
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)

    numbers_raw = payload.get("numbers", [])
    if not numbers_raw:
        msg = "numbers must contain at least k=1"
        raise ValueError(msg)

    numbers: dict[int, Cell] = {}
    positions: set[Cell] = set()

    for raw in numbers_raw:
        entry = _NumberEntry(k=int(raw["k"]), r=int(raw["r"]), c=int(raw["c"]))
        if entry.k <= 0:
            msg = f"number k must be >= 1, got {entry.k}"
            raise ValueError(msg)
        _validate_bounds(n, entry.r, entry.c)
        if entry.k in numbers:
            msg = f"duplicate number k={entry.k}"
            raise ValueError(msg)
        cell = (entry.r, entry.c)
        if cell in positions:
            msg = f"duplicate number position {cell}"
            raise ValueError(msg)
        numbers[entry.k] = cell
        positions.add(cell)

    max_k = max(numbers.keys())
    expected = set(range(1, max_k + 1))
    if set(numbers.keys()) != expected:
        msg = f"numbers must be contiguous 1..{max_k}"
        raise ValueError(msg)

    walls_raw = payload.get("walls", [])
    walls: set[tuple[Cell, Cell]] = set()
    for raw in walls_raw:
        r1, c1 = int(raw["r1"]), int(raw["c1"])
        r2, c2 = int(raw["r2"]), int(raw["c2"])
        _validate_bounds(n, r1, c1)
        _validate_bounds(n, r2, c2)
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            msg = f"wall endpoints must be orthogonally adjacent: ({r1},{c1})-({r2},{c2})"
            raise ValueError(msg)
        edge = _normalize_wall((r1, c1), (r2, c2))
        walls.add(edge)

    neighbors = _build_neighbors(n, walls)

    return ZipPuzzle(
        game=game,
        n=n,
        numbers=numbers,
        walls=walls,
        neighbors=neighbors,
    )


def parse_puzzle_file(path: str | Path) -> ZipPuzzle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_puzzle_dict(data)
