"""
Tests for ODE and Calculus solvers.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestODESolver:
    """Tests for ODE solver."""

    def test_can_solve_first_order(self):
        """Test that ODE solver recognizes first-order ODEs."""
        from src.solvers.ode_solver import ODESolver
        from src.models import Equation, EquationType
        import sympy as sp

        solver = ODESolver()

        # dy/dx = y
        x = sp.Symbol("x")
        y = sp.Function("y")
        expr = sp.Derivative(y(x), x) - y(x)

        equation = Equation(
            raw_latex=r"\frac{dy}{dx} = y",
            sympy_expr=expr,
            classification=(EquationType.ODE, "ode_order_1"),
        )

        assert solver.can_solve(equation)

    def test_solve_separable(self):
        """Test solving a separable ODE: dy/dx = y."""
        from src.solvers.ode_solver import ODESolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = ODESolver()

        x = sp.Symbol("x")
        y = sp.Function("y")
        # dy/dx = y  =>  y' - y = 0
        expr = sp.Eq(sp.Derivative(y(x), x), y(x))

        equation = Equation(
            raw_latex=r"\frac{dy}{dx} = y",
            sympy_expr=expr,
            classification=(EquationType.ODE, "ode_order_1"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        assert result.solution is not None
        # Solution should contain exponential
        solution_str = str(result.solution.symbolic_result)
        assert "exp" in solution_str or "C1" in solution_str

    def test_solve_second_order_harmonic(self):
        """Test solving y'' + y = 0 (simple harmonic oscillator)."""
        from src.solvers.ode_solver import ODESolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = ODESolver()

        x = sp.Symbol("x")
        y = sp.Function("y")
        # y'' + y = 0
        expr = sp.Derivative(y(x), x, 2) + y(x)

        equation = Equation(
            raw_latex=r"y'' + y = 0",
            sympy_expr=expr,
            classification=(EquationType.ODE, "ode_order_2"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        # Solution should contain sin and/or cos
        solution_str = str(result.solution.symbolic_result)
        assert "sin" in solution_str or "cos" in solution_str

    def test_generates_steps(self):
        """Test that ODE solver generates step explanations."""
        from src.solvers.ode_solver import ODESolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = ODESolver()

        x = sp.Symbol("x")
        y = sp.Function("y")
        expr = sp.Eq(sp.Derivative(y(x), x), y(x))

        equation = Equation(
            raw_latex=r"\frac{dy}{dx} = y",
            sympy_expr=expr,
            classification=(EquationType.ODE, "ode_order_1"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        assert len(result.solution.steps) > 0
        # Should have classification step
        step_texts = [s.operation for s in result.solution.steps]
        assert any("order" in s.lower() for s in step_texts)

    def test_classification_info(self):
        """Test getting classification info for an ODE."""
        from src.solvers.ode_solver import ODESolver
        from src.models import Equation, EquationType
        import sympy as sp

        solver = ODESolver()

        x = sp.Symbol("x")
        y = sp.Function("y")
        expr = sp.Derivative(y(x), x) - y(x)

        equation = Equation(
            raw_latex=r"y' - y = 0",
            sympy_expr=expr,
            classification=(EquationType.ODE, None),
        )

        info = solver.get_classification_info(equation)

        assert "order" in info
        assert info["order"] == 1
        assert "classifications" in info


class TestCalculusSolver:
    """Tests for Calculus solver."""

    def test_can_solve_derivative(self):
        """Test that calculus solver recognizes derivatives."""
        from src.solvers.calculus_solver import CalculusSolver
        from src.models import Equation, EquationType
        import sympy as sp

        solver = CalculusSolver()

        x = sp.Symbol("x")
        expr = sp.Derivative(x**2, x)

        equation = Equation(
            raw_latex=r"\frac{d}{dx}(x^2)",
            sympy_expr=expr,
            classification=(EquationType.CALCULUS, "derivative"),
        )

        assert solver.can_solve(equation)

    def test_solve_derivative(self):
        """Test evaluating a derivative: d/dx(x^2) = 2x."""
        from src.solvers.calculus_solver import CalculusSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = CalculusSolver()

        x = sp.Symbol("x")
        expr = sp.Derivative(x**2, x)

        equation = Equation(
            raw_latex=r"\frac{d}{dx}(x^2)",
            sympy_expr=expr,
            classification=(EquationType.CALCULUS, "derivative"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        # d/dx(x^2) = 2x
        assert result.solution.symbolic_result == 2 * x

    def test_solve_chain_rule(self):
        """Test derivative with chain rule: d/dx(sin(x^2))."""
        from src.solvers.calculus_solver import CalculusSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = CalculusSolver()

        x = sp.Symbol("x")
        expr = sp.Derivative(sp.sin(x**2), x)

        equation = Equation(
            raw_latex=r"\frac{d}{dx}(\sin(x^2))",
            sympy_expr=expr,
            classification=(EquationType.CALCULUS, "derivative"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        # d/dx(sin(x^2)) = 2x*cos(x^2)
        expected = 2 * x * sp.cos(x**2)
        assert sp.simplify(result.solution.symbolic_result - expected) == 0

    def test_solve_indefinite_integral(self):
        """Test indefinite integral: ∫x^2 dx = x^3/3."""
        from src.solvers.calculus_solver import CalculusSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = CalculusSolver()

        x = sp.Symbol("x")
        expr = sp.Integral(x**2, x)

        equation = Equation(
            raw_latex=r"\int x^2 \, dx",
            sympy_expr=expr,
            classification=(EquationType.CALCULUS, "indefinite_integral"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        # ∫x^2 dx = x^3/3
        expected = x**3 / 3
        assert sp.simplify(result.solution.symbolic_result - expected) == 0

    def test_solve_definite_integral(self):
        """Test definite integral: ∫_0^π sin(x) dx = 2."""
        from src.solvers.calculus_solver import CalculusSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = CalculusSolver()

        x = sp.Symbol("x")
        expr = sp.Integral(sp.sin(x), (x, 0, sp.pi))

        equation = Equation(
            raw_latex=r"\int_0^\pi \sin(x) \, dx",
            sympy_expr=expr,
            classification=(EquationType.CALCULUS, "definite_integral"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        # ∫_0^π sin(x) dx = 2
        assert result.solution.symbolic_result == 2

    def test_solve_limit(self):
        """Test limit: lim_{x->0} sin(x)/x = 1."""
        from src.solvers.calculus_solver import CalculusSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = CalculusSolver()

        x = sp.Symbol("x")
        expr = sp.Limit(sp.sin(x) / x, x, 0)

        equation = Equation(
            raw_latex=r"\lim_{x \to 0} \frac{\sin(x)}{x}",
            sympy_expr=expr,
            classification=(EquationType.CALCULUS, "limit"),
        )

        request = SolveRequest(equation=equation, show_steps=True)
        result = solver.solve(request)

        assert result.success
        # lim sin(x)/x as x->0 = 1
        assert result.solution.symbolic_result == 1

    def test_convenience_differentiate(self):
        """Test differentiate convenience method."""
        from src.solvers.calculus_solver import CalculusSolver
        import sympy as sp

        solver = CalculusSolver()
        x = sp.Symbol("x")

        result = solver.differentiate(x**3, x)

        assert result.success
        assert result.solution.symbolic_result == 3 * x**2

    def test_convenience_integrate(self):
        """Test integrate_expr convenience method."""
        from src.solvers.calculus_solver import CalculusSolver
        import sympy as sp

        solver = CalculusSolver()
        x = sp.Symbol("x")

        result = solver.integrate_expr(sp.cos(x), x)

        assert result.success
        assert result.solution.symbolic_result == sp.sin(x)

    def test_series_expansion(self):
        """Test Taylor series expansion."""
        from src.solvers.calculus_solver import CalculusSolver
        import sympy as sp

        solver = CalculusSolver()
        x = sp.Symbol("x")

        result = solver.expand_series(sp.exp(x), x, 0, 4)

        assert result.success
        # e^x ≈ 1 + x + x^2/2 + x^3/6 + ...
        series_result = result.solution.symbolic_result
        # Check first few terms
        assert series_result.has(x**2)


class TestStepGenerator:
    """Tests for enhanced step generator."""

    def test_linear_steps(self):
        """Test generating steps for linear equation."""
        from src.output.step_generator import StepGenerator
        from src.models import Equation
        import sympy as sp

        gen = StepGenerator()
        x = sp.Symbol("x")

        equation = Equation(
            raw_latex="2x + 3 = 7",
            sympy_expr=sp.Eq(2 * x + 3, 7),
        )

        steps = gen.generate_linear_steps(equation, x, 2)

        assert len(steps) > 0
        assert steps[0].step_number == 1
        assert "Original" in steps[0].operation

    def test_quadratic_steps(self):
        """Test generating steps for quadratic equation."""
        from src.output.step_generator import StepGenerator
        from src.models import Equation
        import sympy as sp

        gen = StepGenerator()
        x = sp.Symbol("x")

        equation = Equation(
            raw_latex="x^2 - 4 = 0",
            sympy_expr=sp.Eq(x**2 - 4, 0),
        )

        steps = gen.generate_quadratic_steps(equation, x, [2, -2])

        assert len(steps) > 0
        # Should include discriminant step
        step_texts = [s.operation for s in steps]
        assert any(
            "discriminant" in s.lower() or "factor" in s.lower() for s in step_texts
        )

    def test_derivative_steps(self):
        """Test generating derivative steps."""
        from src.output.step_generator import StepGenerator
        import sympy as sp

        gen = StepGenerator()
        x = sp.Symbol("x")

        expr = sp.sin(x**2)
        result = 2 * x * sp.cos(x**2)

        steps = gen.generate_derivative_steps(expr, x, result)

        assert len(steps) > 0
        # Should mention chain rule
        step_texts = [s.operation for s in steps]
        assert any("chain" in s.lower() for s in step_texts)

    def test_integral_steps(self):
        """Test generating integral steps."""
        from src.output.step_generator import StepGenerator
        import sympy as sp

        gen = StepGenerator()
        x = sp.Symbol("x")

        integrand = x**2
        result = x**3 / 3

        steps = gen.generate_integral_steps(integrand, x, result)

        assert len(steps) > 0
        # Should mention power rule
        step_texts = [s.operation for s in steps]
        assert any("power" in s.lower() or "constant" in s.lower() for s in step_texts)

    def test_steps_to_html(self):
        """Test converting steps to HTML."""
        from src.output.step_generator import StepGenerator
        from src.models import SolutionStep
        import sympy as sp

        gen = StepGenerator()

        steps = [
            SolutionStep(1, "Original", sp.Symbol("x"), "x"),
            SolutionStep(2, "Result", sp.Integer(5), "5"),
        ]

        html = gen.steps_to_html(steps)

        assert '<div class="solution-steps">' in html
        assert "Step 1" in html
        assert "Step 2" in html


class TestMathJaxRenderer:
    """Tests for MathJax rendering (basic tests, no actual rendering)."""

    def test_render_steps_html(self):
        """Test that renderer produces valid HTML."""
        from src.output.mathjax_widget import MathJaxRenderer
        from src.models import SolutionStep
        import sympy as sp

        renderer = MathJaxRenderer()

        steps = [
            SolutionStep(1, "Original equation", sp.Symbol("x"), "x = 5"),
            SolutionStep(2, "Solution", sp.Integer(5), "x = 5"),
        ]

        html = renderer.render_steps(steps)

        assert "<!DOCTYPE html>" in html
        assert "MathJax" in html
        assert "step-1" in html

    def test_dark_mode(self):
        """Test dark mode theme."""
        from src.output.mathjax_widget import MathJaxRenderer

        renderer = MathJaxRenderer(dark_mode=True)

        assert "#1e1e1e" in str(renderer.theme.values())

    def test_show_final_only(self):
        """Test showing only final result."""
        from src.output.mathjax_widget import MathJaxRenderer
        from src.models import SolutionStep
        import sympy as sp

        renderer = MathJaxRenderer()

        steps = [
            SolutionStep(1, "Step 1", sp.Symbol("x"), "x"),
            SolutionStep(2, "Step 2", sp.Symbol("y"), "y"),
            SolutionStep(3, "Final", sp.Integer(5), "5"),
        ]

        html = renderer.render_steps(steps, show_final_only=True)

        # Should only have the final step
        assert "Step 3" in html
        # Step 1 and 2 should not be rendered as step divs
        # Count actual step div elements (id="step-N")
        import re

        step_divs = re.findall(r'id="step-\d+"', html)
        assert len(step_divs) == 1


class TestClassifierIntegration:
    """Test classifier integration with new solvers."""

    def test_classifies_ode(self):
        """Test that ODEs are classified correctly."""
        from src.classification.classifier import EquationClassifier
        from src.models import Equation, EquationType
        import sympy as sp

        classifier = EquationClassifier()

        x = sp.Symbol("x")
        y = sp.Function("y")
        expr = sp.Derivative(y(x), x) - y(x)

        equation = Equation(
            raw_latex="y' - y = 0",
            sympy_expr=expr,
        )

        eq_type, subtype = classifier.classify(equation)

        assert eq_type == EquationType.ODE
        assert "order_1" in subtype

    def test_classifies_integral(self):
        """Test that integrals are classified correctly."""
        from src.classification.classifier import EquationClassifier
        from src.models import Equation, EquationType
        import sympy as sp

        classifier = EquationClassifier()

        x = sp.Symbol("x")
        expr = sp.Integral(x**2, x)

        equation = Equation(
            raw_latex=r"\int x^2 dx",
            sympy_expr=expr,
        )

        eq_type, subtype = classifier.classify(equation)

        assert eq_type == EquationType.CALCULUS
        assert "integral" in subtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
