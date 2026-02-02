"""
Tests for solution export functionality.
"""

import pytest
import tempfile
from pathlib import Path

import sympy as sp

from src.models import Equation, Solution, SolutionStep
from src.output.exporter import SolutionExporter, ExportOptions


@pytest.fixture
def sample_solution():
    """Create a sample solution for testing exports."""
    x, y = sp.symbols("x y")

    equation = Equation(
        raw_latex=r"x^2 + 2x + 1 = 0",
        sympy_expr=sp.Eq(x**2 + 2 * x + 1, 0),
    )

    steps = [
        SolutionStep(
            step_number=1,
            operation="Original equation",
            equation_state=sp.Eq(x**2 + 2 * x + 1, 0),
            latex_repr=r"x^{2} + 2 x + 1 = 0",
        ),
        SolutionStep(
            step_number=2,
            operation="Factor the left side",
            equation_state=sp.Eq((x + 1) ** 2, 0),
            latex_repr=r"\left(x + 1\right)^{2} = 0",
        ),
        SolutionStep(
            step_number=3,
            operation="Solve for x",
            equation_state=sp.Eq(x, -1),
            latex_repr=r"x = -1",
        ),
    ]

    return Solution(
        equation=equation,
        target_variable=x,
        symbolic_result=sp.Integer(-1),
        steps=steps,
        numerical_result=-1.0,
        solve_time_ms=5,
        method_used="sympy.solve",
    )


class TestPythonExport:
    """Test Python/SymPy code export."""

    def test_generates_valid_python(self, sample_solution):
        """Test that generated Python code is syntactically valid."""
        exporter = SolutionExporter(sample_solution)
        code = exporter.to_python()

        # Should be valid Python (no syntax errors)
        compile(code, "<string>", "exec")

    def test_includes_imports(self, sample_solution):
        """Test that code includes necessary imports."""
        exporter = SolutionExporter(sample_solution)
        code = exporter.to_python()

        assert "from sympy import" in code

    def test_includes_symbols(self, sample_solution):
        """Test that code defines symbols."""
        exporter = SolutionExporter(sample_solution)
        code = exporter.to_python()

        assert "symbols" in code

    def test_includes_solution(self, sample_solution):
        """Test that code includes the solution."""
        exporter = SolutionExporter(sample_solution)
        code = exporter.to_python()

        assert "solution =" in code

    def test_includes_steps_as_comments(self, sample_solution):
        """Test that steps are included as comments."""
        exporter = SolutionExporter(sample_solution)
        code = exporter.to_python()

        assert "# Solution steps:" in code
        assert "# 1." in code

    def test_writes_to_file(self, sample_solution):
        """Test writing to file."""
        exporter = SolutionExporter(sample_solution)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            exporter.to_python(path)
            content = Path(path).read_text()
            assert "from sympy import" in content
        finally:
            Path(path).unlink()


class TestLatexExport:
    """Test LaTeX document export."""

    def test_generates_valid_latex(self, sample_solution):
        """Test that generated LaTeX has proper structure."""
        exporter = SolutionExporter(sample_solution)
        latex = exporter.to_latex()

        assert r"\documentclass" in latex
        assert r"\begin{document}" in latex
        assert r"\end{document}" in latex

    def test_includes_equation(self, sample_solution):
        """Test that LaTeX includes the equation."""
        exporter = SolutionExporter(sample_solution)
        latex = exporter.to_latex()

        assert r"\begin{equation}" in latex
        assert sample_solution.equation.raw_latex in latex

    def test_includes_steps(self, sample_solution):
        """Test that LaTeX includes solution steps."""
        exporter = SolutionExporter(sample_solution)
        latex = exporter.to_latex()

        assert "Step 1" in latex
        assert "Step 2" in latex

    def test_includes_boxed_result(self, sample_solution):
        """Test that result is in a box."""
        exporter = SolutionExporter(sample_solution)
        latex = exporter.to_latex()

        assert r"\boxed{" in latex

    def test_custom_title(self, sample_solution):
        """Test custom title option."""
        options = ExportOptions(title="My Custom Title")
        exporter = SolutionExporter(sample_solution, options)
        latex = exporter.to_latex()

        assert "My Custom Title" in latex

    def test_escapes_special_chars(self, sample_solution):
        """Test that special characters are escaped."""
        exporter = SolutionExporter(sample_solution)

        # Test internal escape function
        escaped = exporter._latex_escape("50% of $100 is #special")
        assert r"\%" in escaped
        assert r"\$" in escaped
        assert r"\#" in escaped

    def test_writes_to_file(self, sample_solution):
        """Test writing to file."""
        exporter = SolutionExporter(sample_solution)

        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
            path = f.name

        try:
            exporter.to_latex(path)
            content = Path(path).read_text()
            assert r"\documentclass" in content
        finally:
            Path(path).unlink()


class TestTextExport:
    """Test plain text export."""

    def test_generates_readable_text(self, sample_solution):
        """Test that text output is readable."""
        exporter = SolutionExporter(sample_solution)
        text = exporter.to_text()

        assert "MathSolver Solution" in text
        assert "Problem:" in text
        assert "Result:" in text

    def test_includes_steps(self, sample_solution):
        """Test that steps are included."""
        exporter = SolutionExporter(sample_solution)
        text = exporter.to_text()

        assert "Steps:" in text
        assert "1." in text

    def test_includes_numerical_result(self, sample_solution):
        """Test that numerical result is included."""
        exporter = SolutionExporter(sample_solution)
        text = exporter.to_text()

        assert "Numerical value:" in text
        assert "-1" in text


class TestExportOptions:
    """Test export options."""

    def test_exclude_steps(self, sample_solution):
        """Test excluding steps from export."""
        options = ExportOptions(include_steps=False)
        exporter = SolutionExporter(sample_solution, options)

        text = exporter.to_text()
        assert "Steps:" not in text

    def test_exclude_timestamp(self, sample_solution):
        """Test excluding timestamp."""
        options = ExportOptions(include_timestamp=False)
        exporter = SolutionExporter(sample_solution, options)

        text = exporter.to_text()
        assert "Date:" not in text

    def test_exclude_method(self, sample_solution):
        """Test excluding method info."""
        options = ExportOptions(include_method=False)
        exporter = SolutionExporter(sample_solution, options)

        text = exporter.to_text()
        assert "Method:" not in text


class TestPDFExport:
    """Test PDF export (requires pdflatex)."""

    def test_pdf_without_pdflatex_raises_error(self, sample_solution, monkeypatch):
        """Test that missing pdflatex raises ExportError."""
        from src.utils.errors import ExportError

        exporter = SolutionExporter(sample_solution)

        # Mock the command check to return False
        monkeypatch.setattr(exporter, "_command_available", lambda cmd: False)

        with pytest.raises(ExportError) as exc_info:
            exporter.to_pdf("/tmp/test.pdf")

        assert "pdflatex" in str(exc_info.value).lower()
