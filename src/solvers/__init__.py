"""Solver layer: specialized solvers for different equation types."""

from .base import BaseSolver, SolverResult
from .general import GeneralSolver

__all__ = ["BaseSolver", "SolverResult", "GeneralSolver"]
