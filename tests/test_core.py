"""
Basic tests for MathSolver core functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLatexParser:
    """Tests for LaTeX parsing."""

    def test_simple_equation(self):
        """Test parsing a simple equation."""
        from src.input.parser import LatexParser
        import sympy as sp

        parser = LatexParser()
        eq = parser.parse(r"x^2 + 2x + 1 = 0")

        assert eq.sympy_expr is not None
        assert len(eq.variables) > 0

    def test_physics_equation(self):
        """Test parsing E = mc^2."""
        from src.input.parser import LatexParser
        import sympy as sp

        parser = LatexParser()
        eq = parser.parse(r"E = mc^2")

        assert eq.sympy_expr is not None
        # Should have E, m, c as variables
        var_names = {str(v) for v in eq.variables}
        assert "E" in var_names or "m" in var_names

    def test_plain_text_parse(self):
        """Test parsing plain text notation."""
        from src.input.parser import LatexParser

        parser = LatexParser()
        eq = parser.parse_plain_text("x^2 = 4")

        assert eq.sympy_expr is not None


class TestClassifier:
    """Tests for equation classification."""

    def test_polynomial_classification(self):
        """Test that polynomial equations are classified correctly."""
        from src.input.parser import LatexParser
        from src.classification.classifier import EquationClassifier
        from src.models import EquationType

        parser = LatexParser()
        classifier = EquationClassifier()

        eq = parser.parse(r"x^2 + 2x + 1 = 0")
        eq_type, subtype = classifier.classify(eq)

        assert eq_type == EquationType.POLYNOMIAL
        assert "degree" in subtype

    def test_general_classification(self):
        """Test fallback to general classification."""
        from src.input.parser import LatexParser
        from src.classification.classifier import EquationClassifier
        from src.models import EquationType

        parser = LatexParser()
        classifier = EquationClassifier()

        eq = parser.parse(r"x + y = z")
        eq_type, _ = classifier.classify(eq)

        # Should be general (no special structure)
        assert eq_type in (EquationType.GENERAL, EquationType.POLYNOMIAL)


class TestGeneralSolver:
    """Tests for general equation solving."""

    def test_linear_equation(self):
        """Test solving a linear equation."""
        from src.input.parser import LatexParser
        from src.solvers.general import GeneralSolver
        from src.models import SolveRequest
        import sympy as sp

        parser = LatexParser()
        solver = GeneralSolver()

        eq = parser.parse(r"2x + 4 = 10")

        # Find x symbol
        x = None
        for var in eq.variables:
            if str(var) == "x":
                x = var
                break

        request = SolveRequest(equation=eq, target_variable=x)
        result = solver.solve(request)

        assert result.success
        assert result.solution is not None
        # 2x + 4 = 10 => x = 3
        assert result.solution.symbolic_result == 3

    def test_quadratic_equation(self):
        """Test solving a quadratic equation."""
        from src.input.parser import LatexParser
        from src.solvers.general import GeneralSolver
        from src.models import SolveRequest
        import sympy as sp

        parser = LatexParser()
        solver = GeneralSolver()

        eq = parser.parse(r"x^2 - 4 = 0")

        # Find x symbol
        x = None
        for var in eq.variables:
            if str(var) == "x":
                x = var
                break

        request = SolveRequest(equation=eq, target_variable=x)
        result = solver.solve(request)

        assert result.success
        # x^2 = 4 => x = ±2
        solutions = result.solution.symbolic_result
        assert -2 in solutions or 2 in solutions


class TestPhysicalConstants:
    """Tests for physical constants library."""

    def test_speed_of_light(self):
        """Test speed of light value."""
        from src.utils.constants import get_constant_value

        c = get_constant_value("c")
        assert c == 299792458  # Exact SI definition

    def test_planck_constant(self):
        """Test Planck constant value."""
        from src.utils.constants import get_constant_value

        h = get_constant_value("h")
        assert abs(h - 6.62607015e-34) < 1e-40


class TestDatabase:
    """Tests for history database."""

    def test_add_and_retrieve(self, tmp_path):
        """Test adding and retrieving history entries."""
        from src.utils.database import HistoryDatabase

        db_path = tmp_path / "test_history.db"
        db = HistoryDatabase(db_path)

        # Add entry
        entry_id = db.add_entry(
            raw_latex=r"x^2 = 4",
            classification="polynomial",
            target_variable="x",
            solution_latex=r"x = \pm 2",
            solve_time_ms=50,
        )

        assert entry_id > 0

        # Retrieve
        entries = db.get_recent(limit=10)
        assert len(entries) == 1
        assert entries[0].raw_latex == r"x^2 = 4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
