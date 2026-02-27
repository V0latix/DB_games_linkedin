"""Central solver registry for the Zip puzzle."""

from __future__ import annotations

from collections.abc import Callable

from ...core.types import SolveResult
from .model import ZipPuzzle
from .solver_articulation import solve_articulation
from .solver_baseline import solve_baseline
from .solver_forced import solve_forced
from .solver_heuristic import solve_heuristic, solve_heuristic_nolcv

ZipSolver = Callable[[ZipPuzzle, float | None], SolveResult]

_SOLVERS: dict[str, ZipSolver] = {
    "baseline": solve_baseline,
    "forced": solve_forced,
    "articulation": solve_articulation,
    "heuristic": solve_heuristic,
    "heuristic_nolcv": solve_heuristic_nolcv,
}


def list_solvers() -> list[str]:
    return sorted(_SOLVERS)


def get_solver(name: str) -> ZipSolver:
    solver = _SOLVERS.get(name)
    if solver is None:
        known = ", ".join(list_solvers())
        msg = f"Unknown zip solver {name!r}. Known solvers: {known}."
        raise ValueError(msg)
    return solver
