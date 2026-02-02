"""
Step-by-step solution generator.

Generates human-readable explanations for solution steps.
Enhanced with calculus, polynomial factorization, and physics explanations.
"""

from typing import List, Optional, Tuple, Dict, Any
import sympy as sp
from sympy import (
    Eq,
    Symbol,
    latex,
    Derivative,
    Integral,
    Limit,
    sin,
    cos,
    tan,
    exp,
    log,
    sqrt,
    pi,
    diff,
    integrate,
    factor,
    expand,
    simplify,
    collect,
    Poly,
    degree,
    roots,
    solve,
)

from ..models import SolutionStep, Equation, EquationType


class StepGenerator:
    """
    Generate step-by-step explanations for equation solving.

    This class provides pedagogical output by breaking down
    the solution process into understandable steps.

    Supports:
    - Linear and quadratic equations
    - Polynomial factorization
    - Calculus (derivatives with chain rule, integrals with substitution)
    - Physics formulas with physical explanations
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

    def create_text_step(
        self, step_number: int, operation: str, latex_text: str
    ) -> SolutionStep:
        """
        Create a step with custom LaTeX text (for explanations).
        """
        return SolutionStep(
            step_number=step_number,
            operation=operation,
            equation_state=Symbol("_"),  # Placeholder
            latex_repr=latex_text,
        )

    # === Linear Equations ===

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

    # === Quadratic Equations ===

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

            # Check if factorable
            factored = factor(combined)
            if factored != combined and factored.is_Mul:
                steps.append(
                    self.create_step(step_num, "Factor the quadratic", Eq(factored, 0))
                )
                step_num += 1
                steps.append(
                    self.create_text_step(
                        step_num,
                        "Set each factor equal to zero",
                        r"\text{If } (x - r_1)(x - r_2) = 0 \text{ then } x = r_1 \text{ or } x = r_2",
                    )
                )
                step_num += 1
            else:
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

                # Interpret discriminant
                disc_simplified = sp.simplify(discriminant)
                if disc_simplified.is_positive:
                    steps.append(
                        self.create_text_step(
                            step_num,
                            "Δ > 0: Two distinct real roots",
                            r"\Delta > 0 \Rightarrow \text{two distinct real solutions}",
                        )
                    )
                elif disc_simplified.is_zero:
                    steps.append(
                        self.create_text_step(
                            step_num,
                            "Δ = 0: One repeated root",
                            r"\Delta = 0 \Rightarrow \text{one repeated solution}",
                        )
                    )
                elif disc_simplified.is_negative:
                    steps.append(
                        self.create_text_step(
                            step_num,
                            "Δ < 0: Two complex roots",
                            r"\Delta < 0 \Rightarrow \text{two complex solutions}",
                        )
                    )
                step_num += 1

                # Show quadratic formula
                steps.append(
                    self.create_text_step(
                        step_num,
                        "Apply quadratic formula",
                        r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}",
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

    # === Polynomial Factorization ===

    def generate_factorization_steps(
        self, expr: sp.Basic, target: Symbol
    ) -> List[SolutionStep]:
        """
        Generate steps for polynomial factorization.
        """
        steps = []
        step_num = 1

        steps.append(self.create_step(step_num, "Original polynomial", expr))
        step_num += 1

        try:
            poly = Poly(expr, target)
            deg = poly.degree()

            steps.append(
                self.create_text_step(
                    step_num,
                    f"Identify as degree {deg} polynomial",
                    f"\\text{{Degree: }} {deg}",
                )
            )
            step_num += 1

            # Try to find rational roots
            if deg >= 2:
                rational_roots = list(roots(poly, filter="Q").keys())
                if rational_roots:
                    steps.append(
                        self.create_text_step(
                            step_num,
                            "Find rational roots using Rational Root Theorem",
                            f"\\text{{Possible rational roots: }} \\pm\\frac{{p}}{{q}}",
                        )
                    )
                    step_num += 1

                    for root in rational_roots[:3]:  # Show first 3
                        steps.append(
                            self.create_text_step(
                                step_num,
                                f"Verify root {root}",
                                f"p({latex(root)}) = 0 \\checkmark",
                            )
                        )
                        step_num += 1

            # Factor
            factored = factor(expr)
            if factored != expr:
                steps.append(self.create_step(step_num, "Factor completely", factored))

        except Exception:
            pass

        return steps

    # === Calculus Steps ===

    def generate_derivative_steps(
        self, expr: sp.Basic, var: Symbol, result: sp.Basic
    ) -> List[SolutionStep]:
        """
        Generate detailed steps for differentiation with rule identification.
        """
        steps = []
        step_num = 1

        # Original expression
        deriv_notation = Derivative(expr, var)
        steps.append(self.create_step(step_num, "Find derivative", deriv_notation))
        step_num += 1

        # Identify which rules apply
        rules_used = self._identify_derivative_rules(expr, var)

        for rule_name, rule_latex in rules_used:
            steps.append(
                self.create_text_step(step_num, f"Apply {rule_name}", rule_latex)
            )
            step_num += 1

        # Show intermediate steps if composite
        if self._is_composite(expr, var):
            inner, outer_type = self._decompose_composite(expr, var)
            if inner:
                steps.append(
                    self.create_text_step(
                        step_num,
                        f"Let u = {latex(inner)}",
                        f"u = {latex(inner)}, \\quad \\frac{{du}}{{d{var}}} = {latex(diff(inner, var))}",
                    )
                )
                step_num += 1

        # Simplify if needed
        simplified = simplify(result)
        if simplified != result:
            steps.append(self.create_step(step_num, "Compute derivative", result))
            step_num += 1
            steps.append(self.create_step(step_num, "Simplify", simplified))
        else:
            steps.append(self.create_step(step_num, "Result", result))

        return steps

    def generate_integral_steps(
        self,
        integrand: sp.Basic,
        var: Symbol,
        result: sp.Basic,
        lower: Optional[sp.Basic] = None,
        upper: Optional[sp.Basic] = None,
    ) -> List[SolutionStep]:
        """
        Generate detailed steps for integration.
        """
        steps = []
        step_num = 1

        is_definite = lower is not None and upper is not None

        # Original integral
        if is_definite:
            int_expr = Integral(integrand, (var, lower, upper))
        else:
            int_expr = Integral(integrand, var)

        steps.append(self.create_step(step_num, "Evaluate integral", int_expr))
        step_num += 1

        # Identify integration method
        method = self._identify_integration_method(integrand, var)

        if method == "power":
            steps.append(
                self.create_text_step(
                    step_num,
                    "Apply power rule",
                    r"\int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)",
                )
            )
            step_num += 1
        elif method == "exponential":
            steps.append(
                self.create_text_step(
                    step_num, "Apply exponential rule", r"\int e^x \, dx = e^x + C"
                )
            )
            step_num += 1
        elif method == "substitution":
            # Try to identify substitution
            u = self._suggest_substitution(integrand, var)
            if u:
                steps.append(
                    self.create_text_step(
                        step_num,
                        "Use u-substitution",
                        f"\\text{{Let }} u = {latex(u)}, \\quad du = {latex(diff(u, var))} \\, d{var}",
                    )
                )
                step_num += 1
        elif method == "parts":
            steps.append(
                self.create_text_step(
                    step_num,
                    "Use integration by parts",
                    r"\int u \, dv = uv - \int v \, du",
                )
            )
            step_num += 1
        elif method == "trig":
            steps.append(
                self.create_text_step(
                    step_num,
                    "Apply trigonometric integral formula",
                    r"\int \sin(x) \, dx = -\cos(x), \quad \int \cos(x) \, dx = \sin(x)",
                )
            )
            step_num += 1

        # Show antiderivative
        if is_definite:
            antideriv = integrate(integrand, var)
            steps.append(self.create_step(step_num, "Find antiderivative", antideriv))
            step_num += 1

            # Evaluate at bounds
            steps.append(
                self.create_text_step(
                    step_num,
                    "Apply Fundamental Theorem of Calculus",
                    f"F({latex(upper)}) - F({latex(lower)})",
                )
            )
            step_num += 1
        else:
            steps.append(
                self.create_text_step(
                    step_num, "Add constant of integration", f"{latex(result)} + C"
                )
            )
            step_num += 1

        steps.append(self.create_step(step_num, "Result", result))

        return steps

    # === Physics Steps ===

    def generate_physics_steps(
        self,
        formula_name: str,
        target_var: str,
        solve_template: str,
        constants: Dict[str, Any],
    ) -> List[SolutionStep]:
        """
        Generate steps for physics formula solving with physical explanations.
        """
        steps = []
        step_num = 1

        steps.append(
            self.create_text_step(
                step_num, f"Recognize as {formula_name}", f"\\text{{{formula_name}}}"
            )
        )
        step_num += 1

        steps.append(
            self.create_text_step(
                step_num,
                f"Solve for {target_var}",
                f"\\text{{Rearranging to isolate }} {target_var}",
            )
        )
        step_num += 1

        # List constants used
        if constants:
            const_list = ", ".join(
                f"{name} = {info.get('value', '?')} \\text{{ {info.get('unit', '')}}}"
                for name, info in constants.items()
            )
            steps.append(
                self.create_text_step(step_num, "Physical constants", const_list)
            )
            step_num += 1

        return steps

    # === Helper Methods ===

    def _identify_derivative_rules(
        self, expr: sp.Basic, var: Symbol
    ) -> List[Tuple[str, str]]:
        """Identify which derivative rules are needed."""
        rules = []

        # Chain rule
        if self._is_composite(expr, var):
            rules.append(
                ("Chain Rule", r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)")
            )

        # Product rule
        if expr.is_Mul:
            var_factors = [arg for arg in expr.args if arg.has(var)]
            if len(var_factors) >= 2:
                rules.append(
                    (
                        "Product Rule",
                        r"\frac{d}{dx}[f \cdot g] = f' \cdot g + f \cdot g'",
                    )
                )

        # Quotient rule
        if self._is_quotient(expr, var):
            rules.append(
                (
                    "Quotient Rule",
                    r"\frac{d}{dx}\left[\frac{f}{g}\right] = \frac{f'g - fg'}{g^2}",
                )
            )

        # Power rule
        if expr.is_Pow:
            base, exp_val = expr.as_base_exp()
            if base == var and exp_val.is_number:
                rules.append(("Power Rule", r"\frac{d}{dx}[x^n] = nx^{n-1}"))

        return rules

    def _is_composite(self, expr: sp.Basic, var: Symbol) -> bool:
        """Check if expression is a composite function."""
        for func in expr.atoms(sp.Function):
            if func.args and any(arg.has(var) and arg != var for arg in func.args):
                return True
        if expr.is_Pow:
            base, _ = expr.as_base_exp()
            if base.has(var) and base != var:
                return True
        return False

    def _is_quotient(self, expr: sp.Basic, var: Symbol) -> bool:
        """Check if expression involves division with var."""
        if expr.is_Mul:
            for arg in expr.args:
                if arg.is_Pow:
                    base, exp_val = arg.as_base_exp()
                    if exp_val.is_negative and base.has(var):
                        return True
        return False

    def _decompose_composite(
        self, expr: sp.Basic, var: Symbol
    ) -> Tuple[Optional[sp.Basic], Optional[str]]:
        """Decompose composite function into inner and outer."""
        for func in expr.atoms(sp.Function):
            if func.args:
                inner = func.args[0]
                if inner.has(var) and inner != var:
                    return inner, type(func).__name__
        return None, None

    def _identify_integration_method(self, expr: sp.Basic, var: Symbol) -> str:
        """Identify the best integration method for an expression."""
        # Power rule
        if expr.is_Pow:
            base, exp_val = expr.as_base_exp()
            if base == var and exp_val.is_number:
                return "power"

        # Simple polynomial
        if expr.is_polynomial(var):
            return "power"

        # Exponential
        if expr == exp(var) or (
            expr.is_Mul and any(arg == exp(var) for arg in expr.args)
        ):
            return "exponential"

        # Trigonometric
        if expr in (sin(var), cos(var), tan(var)):
            return "trig"

        # Integration by parts candidate
        if expr.is_Mul:
            has_poly = any(arg.is_polynomial(var) for arg in expr.args)
            has_trans = any(
                arg.has(exp) or arg.has(sin) or arg.has(cos) or arg.has(log)
                for arg in expr.args
            )
            if has_poly and has_trans:
                return "parts"

        # Substitution candidate
        if self._is_composite(expr, var):
            return "substitution"

        return "general"

    def _suggest_substitution(self, expr: sp.Basic, var: Symbol) -> Optional[sp.Basic]:
        """Suggest a u-substitution if applicable."""
        # Look for inner function in composite
        for func in expr.atoms(sp.Function):
            if func.args:
                inner = func.args[0]
                if inner.has(var) and inner != var:
                    # Check if derivative of inner appears
                    inner_deriv = diff(inner, var)
                    if (
                        expr.has(inner_deriv)
                        or (expr / inner_deriv).simplify().is_number
                    ):
                        return inner
        return None

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

    def steps_to_html(self, steps: List[SolutionStep]) -> str:
        """
        Convert steps to HTML with MathJax-compatible LaTeX.
        """
        html_parts = ['<div class="solution-steps">']

        for step in steps:
            html_parts.append(f"""
            <div class="step">
                <div class="step-number">Step {step.step_number}</div>
                <div class="step-operation">{step.operation}</div>
                <div class="step-math">\\[{step.latex_repr}\\]</div>
            </div>
            """)

        html_parts.append("</div>")
        return "\n".join(html_parts)
