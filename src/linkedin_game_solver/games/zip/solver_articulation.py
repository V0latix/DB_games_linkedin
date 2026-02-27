"""Backtracking solver with articulation-point pruning for Zip."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.metrics import Timer
from ...core.types import SolveMetrics, SolveResult
from .model import Cell, ZipPuzzle, ZipSolution
from .validator import validate_solution


@dataclass
class _State:
    path: list[Cell]
    visited: set[Cell]
    next_k: int


def _free_neighbors(puzzle: ZipPuzzle, visited: set[Cell], cell: Cell) -> list[Cell]:
    return [nbr for nbr in puzzle.neighbors[cell] if nbr not in visited]


def _isolated_cells(puzzle: ZipPuzzle, visited: set[Cell], current: Cell) -> bool:
    unvisited = []
    for r in range(puzzle.n):
        for c in range(puzzle.n):
            cell = (r, c)
            if cell not in visited:
                unvisited.append(cell)
    if not unvisited:
        return False

    remaining = len(unvisited)
    for cell in unvisited:
        if _free_neighbors(puzzle, visited, cell):
            continue
        if remaining == 1 and current in puzzle.neighbors[cell]:
            continue
        return True
    return False


def _components_ok(puzzle: ZipPuzzle, visited: set[Cell]) -> bool:
    unvisited = {(r, c) for r in range(puzzle.n) for c in range(puzzle.n) if (r, c) not in visited}
    if not unvisited:
        return True
    stack = [next(iter(unvisited))]
    seen = {stack[0]}
    while stack:
        cell = stack.pop()
        for nxt in puzzle.neighbors[cell]:
            if nxt in unvisited and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == unvisited


def _is_checkpoint(puzzle: ZipPuzzle, cell: Cell) -> int | None:
    for k, pos in puzzle.numbers.items():
        if pos == cell:
            return k
    return None


def _degree_prune_ok(puzzle: ZipPuzzle, visited: set[Cell], current: Cell) -> bool:
    unvisited = {(r, c) for r in range(puzzle.n) for c in range(puzzle.n) if (r, c) not in visited}
    if not unvisited:
        return True

    degree_one = 0
    for cell in unvisited:
        degree = 0
        for nbr in puzzle.neighbors[cell]:
            if nbr in unvisited or nbr == current:
                degree += 1
        if degree == 0:
            return False
        if degree == 1:
            degree_one += 1

    return degree_one <= 2


def solve_articulation(puzzle: ZipPuzzle, time_limit_s: float | None = None) -> SolveResult:
    timer = Timer()
    timer.start()
    metrics = SolveMetrics()
    limit_ms = None if time_limit_s is None else max(0.0, time_limit_s * 1000.0)

    if 1 not in puzzle.numbers:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error="Invalid puzzle: number 1 is required.",
        )

    start = puzzle.numbers[1]
    max_k = max(puzzle.numbers.keys())
    initial_next = 2 if max_k >= 2 else max_k + 1

    state = _State(path=[start], visited={start}, next_k=initial_next)

    def timed_out() -> bool:
        return limit_ms is not None and timer.elapsed_ms() >= limit_ms

    def dfs(current: Cell) -> bool | None:
        if timed_out():
            return None
        if len(state.path) == puzzle.n * puzzle.n:
            return True
        if _isolated_cells(puzzle, state.visited, current):
            metrics.backtracks += 1
            return False
        if not _components_ok(puzzle, state.visited):
            metrics.backtracks += 1
            return False
        if not _degree_prune_ok(puzzle, state.visited, current):
            metrics.backtracks += 1
            return False

        neighbors = _free_neighbors(puzzle, state.visited, current)
        if not neighbors:
            metrics.backtracks += 1
            return False

        neighbors.sort(key=lambda cell: len(_free_neighbors(puzzle, state.visited, cell)))

        for nxt in neighbors:
            if timed_out():
                return None
            k = _is_checkpoint(puzzle, nxt)
            if k is not None and k != state.next_k:
                continue
            metrics.nodes += 1

            prev_next = state.next_k
            if k is not None:
                state.next_k += 1

            state.visited.add(nxt)
            state.path.append(nxt)
            result = dfs(nxt)
            if result is None:
                return None
            if result:
                return True

            state.path.pop()
            state.visited.remove(nxt)
            state.next_k = prev_next

        metrics.backtracks += 1
        return False

    solved = dfs(start)
    metrics.time_ms = timer.elapsed_ms()

    if solved is None:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error="Timeout: solver exceeded the time limit.",
        )
    if not solved:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error="No solution found under the given constraints.",
        )

    solution = ZipSolution(path=list(state.path))
    validation = validate_solution(puzzle, solution)
    if not validation.ok:
        return SolveResult(
            solved=False,
            solution=None,
            metrics=metrics,
            error=f"Solver produced an invalid solution: {validation.reason}",
        )

    return SolveResult(solved=True, solution=solution, metrics=metrics, error=None)
