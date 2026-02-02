"""
Step-by-step solution generator.

Generates human-readable explanations for solution steps.
"""

from typing import List, Optional
import sympy as sp
from sympy import Eq, Symbol, latex

from ..models import SolutionStep, Equation


class StepGenerator:
    """
    Generate step-by-step explanations for equation solving.

    This class provides pedagogical output by breaking down
    the solution process into understandable steps.
    """

    def __init__(self):
        """Initialize step generator."""
        pass

    def create_step(
        self, step_number: int, operation: str, equation_state: sp.Basic
    ) -> SolutionStep:
        """
        Create a solution step.

        Args:
            step_number: Sequential step number
            operation: Human-readable description of the operation
            equation_state: SymPy expression at this step

        Returns:
            SolutionStep object
        """
        return SolutionStep(
            step_number=step_number,
            operation=operation,
            equation_state=equation_state,
            latex_repr=latex(equation_state),
        )

    def generate_linear_steps(
        self, equation: Equation, target: Symbol, solution: sp.Basic
    ) -> List[SolutionStep]:
        """
        Generate steps for solving a linear equation.

        For equations of form: ax + b = c
        """
        steps = []
        step_num = 1
        expr = equation.sympy_expr

        # Step 1: Original equation
        steps.append(self.create_step(step_num, "Original equation", expr))
        step_num += 1

        if isinstance(expr, Eq):
            lhs, rhs = expr.args

            # Move all terms to one side
            combined = lhs - rhs
            if combined != lhs:
                steps.append(
                    self.create_step(
                        step_num, "Subtract right side from both sides", Eq(combined, 0)
                    )
                )
                step_num += 1

            # Factor out target variable
            try:
                collected = sp.collect(combined, target)
                if collected != combined:
                    steps.append(
                        self.create_step(
                            step_num, f"Collect terms with {target}", Eq(collected, 0)
                        )
                    )
                    step_num += 1
            except Exception:
                pass

        # Final step: Solution
        steps.append(
            self.create_step(step_num, f"Solve for {target}", Eq(target, solution))
        )

        return steps

    def generate_quadratic_steps(
        self, equation: Equation, target: Symbol, solutions: List[sp.Basic]
    ) -> List[SolutionStep]:
        """
        Generate steps for solving a quadratic equation.

        For equations of form: ax² + bx + c = 0
        """
        steps = []
        step_num = 1
        expr = equation.sympy_expr

        # Step 1: Original equation
        steps.append(self.create_step(step_num, "Original equation", expr))
        step_num += 1

        if isinstance(expr, Eq):
            lhs, rhs = expr.args
            combined = lhs - rhs
        else:
            combined = expr

        # Get polynomial form
        try:
            poly = sp.Poly(combined, target)
            a = poly.nth(2)
            b = poly.nth(1)
            c = poly.nth(0)

            # Show standard form
            standard_form = a * target**2 + b * target + c
            steps.append(
                self.create_step(
                    step_num,
                    "Write in standard form ax² + bx + c = 0",
                    Eq(standard_form, 0),
                )
            )
            step_num += 1

            # Show discriminant
            discriminant = b**2 - 4 * a * c
            steps.append(
                self.create_step(
                    step_num,
                    "Calculate discriminant Δ = b² - 4ac",
                    Eq(sp.Symbol("Δ"), discriminant),
                )
            )
            step_num += 1

            # Show quadratic formula
            steps.append(
                self.create_step(
                    step_num,
                    "Apply quadratic formula",
                    Eq(target, (-b + sp.sqrt(discriminant)) / (2 * a)),
                )
            )
            step_num += 1

        except Exception:
            # Fallback if polynomial analysis fails
            pass

        # Show solutions
        if len(solutions) == 2:
            steps.append(
                self.create_step(
                    step_num, "Two solutions", Eq(target, sp.FiniteSet(*solutions))
                )
            )
        else:
            for i, sol in enumerate(solutions):
                steps.append(
                    self.create_step(step_num, f"Solution {i + 1}", Eq(target, sol))
                )
                step_num += 1

        return steps

    def format_step_text(self, step: SolutionStep) -> str:
        """
        Format a step for plain text display.
        """
        return f"Step {step.step_number}: {step.operation}\n    {step.latex_repr}"

    def steps_to_text(self, steps: List[SolutionStep]) -> str:
        """
        Convert all steps to plain text.
        """
        return "\n\n".join(self.format_step_text(s) for s in steps)
