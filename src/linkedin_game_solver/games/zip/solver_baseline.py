"""Baseline backtracking solver for Zip."""

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


def solve_baseline(puzzle: ZipPuzzle, time_limit_s: float | None = None) -> SolveResult:
    """DFS/backtracking to build a Hamiltonian path respecting numbered checkpoints."""

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

    def is_checkpoint(cell: Cell) -> int | None:
        for k, pos in puzzle.numbers.items():
            if pos == cell:
                return k
        return None

    def dfs(current: Cell) -> bool | None:
        if timed_out():
            return None
        if len(state.path) == puzzle.n * puzzle.n:
            return True

        neighbors = sorted(puzzle.neighbors[current])
        for nxt in neighbors:
            if timed_out():
                return None
            if nxt in state.visited:
                continue
            metrics.nodes += 1

            k = is_checkpoint(nxt)
            if k is not None and k != state.next_k:
                continue

            state.visited.add(nxt)
            state.path.append(nxt)
            prev_next = state.next_k
            if k is not None and k == state.next_k:
                state.next_k += 1

            result = dfs(nxt)
            if result is None:
                return None
            if result:
                return True

            state.next_k = prev_next
            state.path.pop()
            state.visited.remove(nxt)

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
