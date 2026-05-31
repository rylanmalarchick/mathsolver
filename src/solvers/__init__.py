"""Solver layer: specialized solvers for different equation types."""

from .base import BaseSolver, SolverRegistry, SolverResult
from .calculus_solver import CalculusSolver
from .general import GeneralSolver
from .ode_solver import ODESolver
from .physics_solver import PhysicsSolver

__all__ = [
    "BaseSolver",
    "SolverResult",
    "SolverRegistry",
    "GeneralSolver",
    "PhysicsSolver",
    "ODESolver",
    "CalculusSolver",
    "get_default_registry",
]


def get_default_registry() -> SolverRegistry:
    """
    Create and return a solver registry with all solvers at standard priorities.

    Priority order (lower = higher priority):
    - Physics: 10 (template-based physics formulas)
    - ODE: 20 (differential equations)
    - Calculus: 25 (derivatives, integrals, limits)
    - General: 100 (fallback for everything else)
    """
    registry = SolverRegistry()
    registry.register(PhysicsSolver(), priority=10)
    registry.register(ODESolver(), priority=20)
    registry.register(CalculusSolver(), priority=25)
    registry.register(GeneralSolver(), priority=100)
    return registry
