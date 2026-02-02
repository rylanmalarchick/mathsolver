"""
General-purpose solver using SymPy.

Handles algebraic equations, polynomial solving, and general symbolic math.
This is the fallback solver when no specialized solver matches.
"""

from typing import List, Optional, Union
import sympy as sp
from sympy import Eq, Symbol, solve, simplify, latex

from .base import BaseSolver, SolverResult
from ..models import Equation, Solution, SolutionStep, SolveRequest, EquationType


class GeneralSolver(BaseSolver):
    """
    General-purpose algebraic solver using SymPy.

    Handles:
    - Single equations (solve for variable)
    - Polynomial equations
    - Systems of equations
    - Simplification

    This is the fallback solver for anything not handled by specialized solvers.
    """

    name = "GeneralSolver"
    description = "General algebraic solver using SymPy"

    def can_solve(self, equation: Equation) -> bool:
        """
        General solver can attempt almost anything.

        Returns False only for things it definitely can't handle.
        """
        expr = equation.sympy_expr

        # Can't solve if no free symbols (nothing to solve for)
        if not expr.free_symbols:
            return False

        return True

    def solve(self, request: SolveRequest) -> SolverResult:
        """
        Solve the equation using SymPy's solve().
        """
        equation = request.equation
        target = request.target_variable
        expr = equation.sympy_expr

        # If no target specified, use first variable
        if target is None:
            variables = list(expr.free_symbols)
            if not variables:
                return SolverResult.failure(
                    "No variables to solve for", solver_name=self.name
                )
            target = sorted(variables, key=str)[0]

        steps = []
        step_num = 1

        # Step 1: Show original equation
        steps.append(self._create_step(step_num, "Original equation", expr))
        step_num += 1

        # Convert to equation form if not already
        if isinstance(expr, Eq):
            eq_to_solve = expr
            # Show as "equation = 0" form
            zero_form = expr.lhs - expr.rhs
        else:
            # Treat expression as "expr = 0"
            eq_to_solve = Eq(expr, 0)
            zero_form = expr
            steps.append(
                self._create_step(step_num, "Set expression equal to zero", eq_to_solve)
            )
            step_num += 1

        # Attempt to solve
        try:
            solutions, elapsed_ms = self._timed_solve(solve, eq_to_solve, target)
        except Exception as e:
            return SolverResult.failure(
                f"SymPy solve() failed: {e}", solver_name=self.name
            )

        # Handle different solution types
        if solutions is None or solutions == []:
            return SolverResult.failure(
                f"No solution found for {target}", solver_name=self.name
            )

        # solve() returns a list for single variable
        if isinstance(solutions, list):
            if len(solutions) == 0:
                return SolverResult.failure(
                    f"No solution found for {target}", solver_name=self.name
                )
            elif len(solutions) == 1:
                solution_expr = solutions[0]
            else:
                # Multiple solutions - show all
                solution_expr = solutions
        elif isinstance(solutions, dict):
            # System of equations returns dict
            solution_expr = solutions.get(target, solutions)
        else:
            solution_expr = solutions

        # Generate steps for simple algebraic manipulation
        if request.show_steps:
            steps.extend(
                self._generate_algebra_steps(expr, target, solution_expr, step_num)
            )

        # Build result
        if isinstance(solution_expr, list):
            # Multiple solutions
            final_expr = solution_expr
            result_step = self._create_step(
                len(steps) + 1, f"Solutions for {target}", sp.FiniteSet(*solution_expr)
            )
        else:
            final_expr = solution_expr
            result_step = self._create_step(
                len(steps) + 1, f"Solve for {target}", Eq(target, solution_expr)
            )
        steps.append(result_step)

        # Create Solution object
        solution = Solution(
            equation=equation,
            target_variable=target,
            symbolic_result=final_expr,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used="sympy.solve",
        )

        # Apply numerical values if provided
        if request.numerical_values:
            try:
                if isinstance(final_expr, list):
                    num_results = [
                        float(e.subs(request.numerical_values).evalf())
                        for e in final_expr
                    ]
                    solution.numerical_result = num_results
                else:
                    solution.numerical_result = float(
                        final_expr.subs(request.numerical_values).evalf()
                    )
            except Exception:
                pass  # Numerical evaluation failed, leave as None

        return SolverResult.from_solution(solution, solver_name=self.name)

    def _generate_algebra_steps(
        self,
        original_expr: sp.Basic,
        target: Symbol,
        solution: sp.Basic,
        start_step: int,
    ) -> List[SolutionStep]:
        """
        Generate intermediate algebraic steps.

        This is a simplified step generator - a more sophisticated version
        would trace the actual transformations SymPy performs.
        """
        steps = []
        step_num = start_step

        if isinstance(original_expr, Eq):
            lhs, rhs = original_expr.args

            # Check if target is already isolated on one side
            if lhs == target:
                # Already solved
                return steps
            if rhs == target:
                steps.append(self._create_step(step_num, "Swap sides", Eq(rhs, lhs)))
                return steps

            # Show simplified form
            simplified_lhs = simplify(lhs - rhs)
            if simplified_lhs != lhs - rhs:
                steps.append(
                    self._create_step(step_num, "Simplify", Eq(simplified_lhs, 0))
                )
                step_num += 1

            # For linear equations, show the isolation step
            if simplified_lhs.is_polynomial(target):
                try:
                    poly = sp.Poly(simplified_lhs, target)
                    if poly.degree() == 1:
                        # Linear equation: ax + b = 0
                        a = poly.nth(1)
                        b = poly.nth(0)
                        if a != 1 and a != 0:
                            steps.append(
                                self._create_step(
                                    step_num,
                                    f"Divide both sides by coefficient of {target}",
                                    Eq(target + b / a, 0),
                                )
                            )
                            step_num += 1
                except Exception:
                    pass

        return steps

    def simplify_expression(self, equation: Equation) -> Equation:
        """
        Simplify an expression without solving.

        Returns a new Equation with simplified expression.
        """
        simplified = simplify(equation.sympy_expr)
        return Equation(
            raw_latex=latex(simplified),
            sympy_expr=simplified,
            classification=equation.classification,
            variables=list(simplified.free_symbols),
            constants=equation.constants,
            ocr_confidence=equation.ocr_confidence,
            timestamp=equation.timestamp,
        )
