"""
Base solver interface and common result types.

All solvers inherit from BaseSolver and return SolverResult.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, TypeVar
from dataclasses import dataclass, field
import time
import signal
import sympy as sp

from ..models import Equation, Solution, SolutionStep, SolveRequest
from ..utils.errors import SolveTimeoutError

T = TypeVar("T")


@dataclass
class SolverResult:
    """
    Result from a solver operation.

    Wraps Solution with additional metadata about the solve attempt.
    """

    success: bool
    solution: Optional[Solution] = None
    error_message: Optional[str] = None
    solver_name: str = ""

    @classmethod
    def failure(cls, message: str, solver_name: str = "") -> "SolverResult":
        """Create a failed result."""
        return cls(success=False, error_message=message, solver_name=solver_name)

    @classmethod
    def from_solution(cls, solution: Solution, solver_name: str = "") -> "SolverResult":
        """Create a successful result from a Solution."""
        return cls(success=True, solution=solution, solver_name=solver_name)


class BaseSolver(ABC):
    """
    Abstract base class for equation solvers.

    Subclasses implement solve() for their specific equation types.
    """

    # Human-readable name for this solver
    name: str = "BaseSolver"

    # Description of what this solver handles
    description: str = "Base solver class"

    @abstractmethod
    def can_solve(self, equation: Equation) -> bool:
        """
        Check if this solver can handle the given equation.

        Args:
            equation: Equation to check

        Returns:
            True if this solver can attempt to solve it
        """
        pass

    @abstractmethod
    def solve(self, request: SolveRequest) -> SolverResult:
        """
        Attempt to solve the equation.

        Args:
            request: SolveRequest with equation and parameters

        Returns:
            SolverResult indicating success/failure and solution
        """
        pass

    def _create_step(
        self, step_num: int, operation: str, expr: sp.Basic
    ) -> SolutionStep:
        """Helper to create a solution step."""
        return SolutionStep(
            step_number=step_num,
            operation=operation,
            equation_state=expr,
            latex_repr=sp.latex(expr),
        )

    def _timed_solve(self, solve_func, *args, **kwargs):
        """
        Wrapper that times the solve operation.

        Returns (result, elapsed_ms)
        """
        start = time.perf_counter()
        result = solve_func(*args, **kwargs)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return result, elapsed_ms

    def _timed_solve_with_timeout(
        self,
        solve_func: Callable[..., T],
        timeout_seconds: float,
        *args,
        **kwargs,
    ) -> tuple:
        """
        Wrapper that times the solve operation with a timeout.

        Uses SIGALRM on Unix for true timeout (Windows falls back to no timeout).

        Args:
            solve_func: Function to call
            timeout_seconds: Maximum time allowed (seconds)
            *args, **kwargs: Passed to solve_func

        Returns:
            (result, elapsed_ms)

        Raises:
            SolveTimeoutError: If operation times out
        """

        def timeout_handler(signum, frame):
            raise SolveTimeoutError(timeout_seconds)

        start = time.perf_counter()

        # Try to use signal-based timeout (Unix only)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            try:
                result = solve_func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except AttributeError:
            # Windows doesn't have SIGALRM - just run without timeout
            result = solve_func(*args, **kwargs)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return result, elapsed_ms


class SolverRegistry:
    """
    Registry of available solvers.

    Maintains priority order for solver selection.
    """

    def __init__(self):
        self._solvers: List[BaseSolver] = []

    def register(self, solver: BaseSolver, priority: int = 100):
        """
        Register a solver with given priority (lower = higher priority).
        """
        self._solvers.append((priority, solver))
        self._solvers.sort(key=lambda x: x[0])

    def get_solver(self, equation: Equation) -> Optional[BaseSolver]:
        """
        Get the highest-priority solver that can handle this equation.
        """
        for _, solver in self._solvers:
            if solver.can_solve(equation):
                return solver
        return None

    def get_all_capable(self, equation: Equation) -> List[BaseSolver]:
        """Get all solvers that can handle this equation."""
        return [solver for _, solver in self._solvers if solver.can_solve(equation)]

    @property
    def solvers(self) -> List[BaseSolver]:
        """Get all registered solvers in priority order."""
        return [solver for _, solver in self._solvers]
