"""Heuristic backtracking solver for Zip."""

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


def _is_checkpoint(puzzle: ZipPuzzle, cell: Cell) -> int | None:
    for k, pos in puzzle.numbers.items():
        if pos == cell:
            return k
    return None


def _free_neighbors(puzzle: ZipPuzzle, state: _State, cell: Cell) -> list[Cell]:
    return [n for n in puzzle.neighbors[cell] if n not in state.visited]


def _isolated_cells(puzzle: ZipPuzzle, state: _State, current: Cell) -> bool:
    unvisited = []
    for r in range(puzzle.n):
        for c in range(puzzle.n):
            cell = (r, c)
            if cell in state.visited:
                continue
            unvisited.append(cell)
    if not unvisited:
        return False

    remaining = len(unvisited)
    for cell in unvisited:
        if _free_neighbors(puzzle, state, cell):
            continue
        if remaining == 1 and current in puzzle.neighbors[cell]:
            continue
        return True
    return False


def _components_ok(puzzle: ZipPuzzle, state: _State) -> bool:
    unvisited = {(r, c) for r in range(puzzle.n) for c in range(puzzle.n) if (r, c) not in state.visited}
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


def _lcv_score(puzzle: ZipPuzzle, state: _State, current: Cell, candidate: Cell) -> int:
    score = 0
    for neighbor in _free_neighbors(puzzle, state, candidate):
        score += len(_free_neighbors(puzzle, state, neighbor))
    return score


def _solve(
    puzzle: ZipPuzzle,
    time_limit_s: float | None,
    use_lcv: bool,
) -> SolveResult:
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
        if _isolated_cells(puzzle, state, current):
            metrics.backtracks += 1
            return False
        if not _components_ok(puzzle, state):
            metrics.backtracks += 1
            return False

        neighbors = _free_neighbors(puzzle, state, current)
        if not neighbors:
            metrics.backtracks += 1
            return False

        neighbors.sort(key=lambda cell: len(_free_neighbors(puzzle, state, cell)))
        if use_lcv:
            neighbors.sort(key=lambda cell: -_lcv_score(puzzle, state, current, cell))

        for nxt in neighbors:
            if timed_out():
                return None
            metrics.nodes += 1
            k = _is_checkpoint(puzzle, nxt)
            if k is not None and k != state.next_k:
                continue
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


def solve_heuristic(puzzle: ZipPuzzle, time_limit_s: float | None = None) -> SolveResult:
    return _solve(puzzle, time_limit_s=time_limit_s, use_lcv=True)


def solve_heuristic_nolcv(puzzle: ZipPuzzle, time_limit_s: float | None = None) -> SolveResult:
    return _solve(puzzle, time_limit_s=time_limit_s, use_lcv=False)
