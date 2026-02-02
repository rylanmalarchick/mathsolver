"""
Calculus solver for derivatives, integrals, limits, and series.

Provides step-by-step solutions using SymPy's calculus functions.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
import sympy as sp
from sympy import (
    Symbol,
    Function,
    Derivative,
    Integral,
    Limit,
    Sum,
    Product,
    Eq,
    diff,
    integrate,
    limit,
    series,
    summation,
    sin,
    cos,
    tan,
    exp,
    log,
    sqrt,
    pi,
    E,
    oo,
    latex,
    simplify,
    expand,
    factor,
    trigsimp,
    symbols,
    sympify,
)

from .base import BaseSolver, SolverResult
from ..models import Equation, Solution, SolutionStep, SolveRequest, EquationType


class CalculusSolver(BaseSolver):
    """
    Solver for calculus operations.

    Handles:
    - Derivatives (with chain rule explanation)
    - Integrals (definite and indefinite)
    - Limits
    - Series expansions
    - Summations

    Priority: 25 (specialized for calculus)
    """

    name = "CalculusSolver"
    description = "Calculus solver for derivatives, integrals, limits, and series"

    def can_solve(self, equation: Equation) -> bool:
        """
        Check if equation involves calculus operations.
        """
        expr = equation.sympy_expr

        # Check for unevaluated calculus operations
        if expr.has(Derivative) and not expr.atoms(sp.Function):
            # Has Derivative but not as part of an ODE
            # (ODEs have function symbols like f(x))
            return True

        if expr.has(Integral):
            return True

        if expr.has(Limit):
            return True

        if expr.has(Sum) or expr.has(Product):
            return True

        # Check if it's a request to differentiate/integrate an expression
        # This would come from classification
        eq_type, subtype = equation.classification
        if eq_type == EquationType.CALCULUS:
            return True

        return False

    def solve(self, request: SolveRequest) -> SolverResult:
        """
        Evaluate the calculus expression with step-by-step explanation.
        """
        equation = request.equation
        expr = equation.sympy_expr

        # Determine the type of calculus operation
        if expr.has(Derivative):
            return self._solve_derivative(request)
        elif expr.has(Integral):
            return self._solve_integral(request)
        elif expr.has(Limit):
            return self._solve_limit(request)
        elif expr.has(Sum):
            return self._solve_sum(request)
        else:
            # Try to evaluate as general calculus
            return self._solve_general(request)

    def _solve_derivative(self, request: SolveRequest) -> SolverResult:
        """Solve/evaluate a derivative expression."""
        equation = request.equation
        expr = equation.sympy_expr

        steps = []
        step_num = 1

        # Step 1: Show original expression
        steps.append(
            self._create_step(step_num, "Original derivative expression", expr)
        )
        step_num += 1

        # Find the derivative to evaluate
        derivatives = list(expr.atoms(Derivative))

        if derivatives:
            deriv = derivatives[0]
            inner_expr = deriv.expr
            diff_vars = [(var, count) for var, count in deriv.variable_count]

            # Generate chain rule steps if applicable
            if request.show_steps:
                chain_steps = self._generate_derivative_steps(
                    inner_expr, diff_vars, step_num
                )
                steps.extend(chain_steps)
                step_num += len(chain_steps)

        # Evaluate the derivative
        try:
            result, elapsed_ms = self._timed_solve(lambda e: e.doit(), expr)
        except Exception as e:
            return SolverResult.failure(
                f"Derivative evaluation failed: {e}", solver_name=self.name
            )

        # Simplify result
        try:
            simplified = simplify(result)
            if simplified != result:
                steps.append(self._create_step(step_num, "Simplify", simplified))
                step_num += 1
                result = simplified
        except Exception:
            pass

        # Final step
        steps.append(self._create_step(step_num, "Result", result))

        solution = Solution(
            equation=equation,
            target_variable=None,
            symbolic_result=result,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used="derivative",
        )

        return SolverResult.from_solution(solution, solver_name=self.name)

    def _solve_integral(self, request: SolveRequest) -> SolverResult:
        """Solve/evaluate an integral expression."""
        equation = request.equation
        expr = equation.sympy_expr

        steps = []
        step_num = 1

        # Step 1: Show original expression
        steps.append(self._create_step(step_num, "Original integral expression", expr))
        step_num += 1

        # Find the integral
        integrals = list(expr.atoms(Integral))

        is_definite = False
        if integrals:
            integral = integrals[0]
            integrand = integral.function
            limits = integral.limits

            # Check if definite (has bounds)
            if limits and len(limits[0]) == 3:
                is_definite = True
                var, lower, upper = limits[0]
                steps.append(
                    SolutionStep(
                        step_number=step_num,
                        operation=f"Definite integral from {lower} to {upper}",
                        equation_state=integral,
                        latex_repr=f"\\int_{{{latex(lower)}}}^{{{latex(upper)}}} {latex(integrand)} \\, d{latex(var)}",
                    )
                )
            else:
                var = limits[0][0] if limits else Symbol("x")
                steps.append(
                    SolutionStep(
                        step_number=step_num,
                        operation=f"Indefinite integral with respect to {var}",
                        equation_state=integral,
                        latex_repr=f"\\int {latex(integrand)} \\, d{latex(var)}",
                    )
                )
            step_num += 1

            # Generate integration steps
            if request.show_steps:
                int_steps = self._generate_integral_steps(integrand, var, step_num)
                steps.extend(int_steps)
                step_num += len(int_steps)

        # Evaluate the integral
        try:
            result, elapsed_ms = self._timed_solve(lambda e: e.doit(), expr)
        except Exception as e:
            return SolverResult.failure(
                f"Integration failed: {e}", solver_name=self.name
            )

        # For indefinite integrals, add constant
        if not is_definite and not expr.has(oo):
            C = Symbol("C")
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Add constant of integration",
                    equation_state=result + C,
                    latex_repr=f"{latex(result)} + C",
                )
            )
            step_num += 1
            # Don't actually add C to result for SymPy compatibility

        # Final step
        steps.append(self._create_step(step_num, "Result", result))

        solution = Solution(
            equation=equation,
            target_variable=None,
            symbolic_result=result,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used="integral",
        )

        return SolverResult.from_solution(solution, solver_name=self.name)

    def _solve_limit(self, request: SolveRequest) -> SolverResult:
        """Solve/evaluate a limit expression."""
        equation = request.equation
        expr = equation.sympy_expr

        steps = []
        step_num = 1

        # Step 1: Show original expression
        steps.append(self._create_step(step_num, "Original limit expression", expr))
        step_num += 1

        # Find the limit
        limits = list(expr.atoms(Limit))

        if limits:
            lim = limits[0]
            lim_expr = lim.args[0]
            var = lim.args[1]
            point = lim.args[2]
            direction = lim.args[3] if len(lim.args) > 3 else "+-"

            dir_str = ""
            if direction == "+":
                dir_str = " from the right"
            elif direction == "-":
                dir_str = " from the left"

            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation=f"Take limit as {var} → {point}{dir_str}",
                    equation_state=lim,
                    latex_repr=f"\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(lim_expr)}",
                )
            )
            step_num += 1

            # Check for indeterminate forms
            if request.show_steps:
                try:
                    direct_sub = lim_expr.subs(var, point)
                    if (
                        direct_sub.has(sp.zoo)
                        or direct_sub.has(sp.nan)
                        or direct_sub == sp.zoo
                    ):
                        steps.append(
                            SolutionStep(
                                step_number=step_num,
                                operation="Direct substitution gives indeterminate form",
                                equation_state=Symbol("_"),
                                latex_repr=r"\text{Indeterminate form: apply L'Hôpital's rule or algebraic manipulation}",
                            )
                        )
                        step_num += 1
                except Exception:
                    pass

        # Evaluate the limit
        try:
            result, elapsed_ms = self._timed_solve(lambda e: e.doit(), expr)
        except Exception as e:
            return SolverResult.failure(
                f"Limit evaluation failed: {e}", solver_name=self.name
            )

        # Final step
        steps.append(self._create_step(step_num, "Result", result))

        solution = Solution(
            equation=equation,
            target_variable=None,
            symbolic_result=result,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used="limit",
        )

        return SolverResult.from_solution(solution, solver_name=self.name)

    def _solve_sum(self, request: SolveRequest) -> SolverResult:
        """Solve/evaluate a summation expression."""
        equation = request.equation
        expr = equation.sympy_expr

        steps = []
        step_num = 1

        # Step 1: Show original expression
        steps.append(self._create_step(step_num, "Original summation", expr))
        step_num += 1

        # Evaluate
        try:
            result, elapsed_ms = self._timed_solve(lambda e: e.doit(), expr)
        except Exception as e:
            return SolverResult.failure(f"Summation failed: {e}", solver_name=self.name)

        # Final step
        steps.append(self._create_step(step_num, "Result", result))

        solution = Solution(
            equation=equation,
            target_variable=None,
            symbolic_result=result,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used="summation",
        )

        return SolverResult.from_solution(solution, solver_name=self.name)

    def _solve_general(self, request: SolveRequest) -> SolverResult:
        """Handle general calculus operations based on classification."""
        return SolverResult.failure(
            "Could not identify calculus operation", solver_name=self.name
        )

    def _generate_derivative_steps(
        self, expr: sp.Basic, diff_vars: List[Tuple[Symbol, int]], start_step: int
    ) -> List[SolutionStep]:
        """
        Generate step-by-step derivative explanation.

        Includes chain rule, product rule, quotient rule detection.
        """
        steps = []
        step_num = start_step

        if not diff_vars:
            return steps

        var = diff_vars[0][0]
        order = diff_vars[0][1]

        # Detect composite function (chain rule)
        if self._needs_chain_rule(expr, var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Apply chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)",
                    equation_state=Symbol("_"),
                    latex_repr=r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)",
                )
            )
            step_num += 1

            # Try to identify outer and inner functions
            inner, outer = self._identify_composition(expr, var)
            if inner and outer:
                steps.append(
                    SolutionStep(
                        step_number=step_num,
                        operation=f"Inner function: {latex(inner)}, Outer applied to inner",
                        equation_state=Symbol("_"),
                        latex_repr=f"\\text{{Let }} u = {latex(inner)}",
                    )
                )
                step_num += 1

        # Detect product (product rule)
        elif self._needs_product_rule(expr, var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Apply product rule: d/dx[f·g] = f'·g + f·g'",
                    equation_state=Symbol("_"),
                    latex_repr=r"\frac{d}{dx}[f \cdot g] = f' \cdot g + f \cdot g'",
                )
            )
            step_num += 1

        # Detect quotient (quotient rule)
        elif self._needs_quotient_rule(expr, var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Apply quotient rule: d/dx[f/g] = (f'g - fg')/g²",
                    equation_state=Symbol("_"),
                    latex_repr=r"\frac{d}{dx}\left[\frac{f}{g}\right] = \frac{f'g - fg'}{g^2}",
                )
            )
            step_num += 1

        # Show direct differentiation
        try:
            result = diff(expr, var)
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation=f"Differentiate with respect to {var}",
                    equation_state=result,
                    latex_repr=latex(result),
                )
            )
        except Exception:
            pass

        return steps

    def _generate_integral_steps(
        self, integrand: sp.Basic, var: Symbol, start_step: int
    ) -> List[SolutionStep]:
        """
        Generate step-by-step integration explanation.

        Includes substitution hints, integration by parts detection.
        """
        steps = []
        step_num = start_step

        # Check for common patterns

        # Power rule: x^n
        if integrand.is_Pow and integrand.base == var:
            n = integrand.exp
            if n != -1:
                steps.append(
                    SolutionStep(
                        step_number=step_num,
                        operation="Apply power rule: ∫x^n dx = x^(n+1)/(n+1)",
                        equation_state=Symbol("_"),
                        latex_repr=r"\int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)",
                    )
                )
                step_num += 1

        # Exponential: e^x
        elif integrand == exp(var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Exponential integral: ∫e^x dx = e^x",
                    equation_state=Symbol("_"),
                    latex_repr=r"\int e^x \, dx = e^x + C",
                )
            )
            step_num += 1

        # Trigonometric
        elif integrand == sin(var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Trigonometric integral: ∫sin(x) dx = -cos(x)",
                    equation_state=Symbol("_"),
                    latex_repr=r"\int \sin(x) \, dx = -\cos(x) + C",
                )
            )
            step_num += 1
        elif integrand == cos(var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Trigonometric integral: ∫cos(x) dx = sin(x)",
                    equation_state=Symbol("_"),
                    latex_repr=r"\int \cos(x) \, dx = \sin(x) + C",
                )
            )
            step_num += 1

        # Check if substitution might help
        elif self._might_need_substitution(integrand, var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Consider u-substitution",
                    equation_state=Symbol("_"),
                    latex_repr=r"\text{Let } u = g(x), \quad du = g'(x)\,dx",
                )
            )
            step_num += 1

        # Check if integration by parts might help
        elif self._might_need_parts(integrand, var):
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation="Consider integration by parts: ∫u dv = uv - ∫v du",
                    equation_state=Symbol("_"),
                    latex_repr=r"\int u \, dv = uv - \int v \, du",
                )
            )
            step_num += 1

        return steps

    def _needs_chain_rule(self, expr: sp.Basic, var: Symbol) -> bool:
        """Check if expression likely needs chain rule."""
        # Check for composite functions
        for func in expr.atoms(sp.Function):
            if func.args and any(arg.has(var) and arg != var for arg in func.args):
                return True

        # Check for power of expression
        if expr.is_Pow:
            base, exp_val = expr.as_base_exp()
            if base.has(var) and base != var:
                return True

        return False

    def _identify_composition(
        self, expr: sp.Basic, var: Symbol
    ) -> Tuple[Optional[sp.Basic], Optional[sp.Basic]]:
        """Try to identify inner and outer functions in a composition."""
        # Check for sin(something), cos(something), etc.
        for func in expr.atoms(sp.Function):
            if func.args:
                inner = func.args[0]
                if inner.has(var) and inner != var:
                    return inner, func.func

        # Check for (something)^n
        if expr.is_Pow:
            base, _ = expr.as_base_exp()
            if base.has(var) and base != var:
                return base, None

        return None, None

    def _needs_product_rule(self, expr: sp.Basic, var: Symbol) -> bool:
        """Check if expression is a product needing product rule."""
        if expr.is_Mul:
            # Count factors that depend on var
            var_factors = sum(1 for arg in expr.args if arg.has(var))
            return var_factors >= 2
        return False

    def _needs_quotient_rule(self, expr: sp.Basic, var: Symbol) -> bool:
        """Check if expression is a quotient needing quotient rule."""
        if expr.is_Pow:
            base, exp_val = expr.as_base_exp()
            # Check for negative exponent (division)
            if exp_val.is_negative and base.has(var):
                return True
        return False

    def _might_need_substitution(self, expr: sp.Basic, var: Symbol) -> bool:
        """Heuristic for whether u-substitution might help."""
        # Check for f(g(x)) * g'(x) pattern
        if expr.is_Mul:
            for arg in expr.args:
                if arg.is_Pow or isinstance(arg, sp.Function):
                    # Has a composite function, might benefit from substitution
                    return True
        return False

    def _might_need_parts(self, expr: sp.Basic, var: Symbol) -> bool:
        """Heuristic for whether integration by parts might help."""
        # Classic patterns: x*e^x, x*sin(x), x*log(x), etc.
        if expr.is_Mul:
            has_poly = any(arg.is_polynomial(var) for arg in expr.args)
            has_transcendental = any(
                arg.has(exp) or arg.has(sin) or arg.has(cos) or arg.has(log)
                for arg in expr.args
            )
            return has_poly and has_transcendental
        return False

    # === Convenience methods for direct operations ===

    def differentiate(
        self, expr: sp.Basic, var: Symbol, order: int = 1, show_steps: bool = True
    ) -> SolverResult:
        """
        Differentiate an expression.

        Args:
            expr: Expression to differentiate
            var: Variable to differentiate with respect to
            order: Order of derivative
            show_steps: Whether to generate steps

        Returns:
            SolverResult with derivative
        """
        # Create a Derivative expression
        deriv_expr = Derivative(expr, (var, order))

        equation = Equation(
            raw_latex=latex(deriv_expr),
            sympy_expr=deriv_expr,
            classification=(EquationType.CALCULUS, "derivative"),
        )

        request = SolveRequest(equation=equation, show_steps=show_steps)
        return self._solve_derivative(request)

    def integrate_expr(
        self,
        expr: sp.Basic,
        var: Symbol,
        lower: Optional[sp.Basic] = None,
        upper: Optional[sp.Basic] = None,
        show_steps: bool = True,
    ) -> SolverResult:
        """
        Integrate an expression.

        Args:
            expr: Expression to integrate
            var: Variable to integrate with respect to
            lower: Lower bound (for definite integral)
            upper: Upper bound (for definite integral)
            show_steps: Whether to generate steps

        Returns:
            SolverResult with integral
        """
        if lower is not None and upper is not None:
            int_expr = Integral(expr, (var, lower, upper))
        else:
            int_expr = Integral(expr, var)

        equation = Equation(
            raw_latex=latex(int_expr),
            sympy_expr=int_expr,
            classification=(EquationType.CALCULUS, "integral"),
        )

        request = SolveRequest(equation=equation, show_steps=show_steps)
        return self._solve_integral(request)

    def take_limit(
        self,
        expr: sp.Basic,
        var: Symbol,
        point: sp.Basic,
        direction: str = "+-",
        show_steps: bool = True,
    ) -> SolverResult:
        """
        Take limit of an expression.

        Args:
            expr: Expression
            var: Variable approaching the point
            point: The point being approached
            direction: '+' for right, '-' for left, '+-' for both
            show_steps: Whether to generate steps

        Returns:
            SolverResult with limit
        """
        lim_expr = Limit(expr, var, point, direction)

        equation = Equation(
            raw_latex=latex(lim_expr),
            sympy_expr=lim_expr,
            classification=(EquationType.CALCULUS, "limit"),
        )

        request = SolveRequest(equation=equation, show_steps=show_steps)
        return self._solve_limit(request)

    def expand_series(
        self, expr: sp.Basic, var: Symbol, point: sp.Basic = 0, order: int = 6
    ) -> SolverResult:
        """
        Expand expression as Taylor/Maclaurin series.

        Args:
            expr: Expression to expand
            var: Variable
            point: Point to expand around (0 for Maclaurin)
            order: Number of terms

        Returns:
            SolverResult with series expansion
        """
        steps = []
        step_num = 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation=f"Expand as {'Maclaurin' if point == 0 else 'Taylor'} series around {point}",
                equation_state=expr,
                latex_repr=latex(expr),
            )
        )
        step_num += 1

        try:
            result, elapsed_ms = self._timed_solve(
                lambda: series(expr, var, point, order)
            )
        except Exception as e:
            return SolverResult.failure(
                f"Series expansion failed: {e}", solver_name=self.name
            )

        # Remove O() term for cleaner result
        result_clean = result.removeO()

        steps.append(self._create_step(step_num, f"Series to O({var}^{order})", result))

        equation = Equation(
            raw_latex=latex(expr),
            sympy_expr=expr,
            classification=(EquationType.CALCULUS, "series"),
        )

        solution = Solution(
            equation=equation,
            target_variable=None,
            symbolic_result=result_clean,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used="series",
        )

        return SolverResult.from_solution(solution, solver_name=self.name)
