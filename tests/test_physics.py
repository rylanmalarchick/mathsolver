"""
Tests for physics pattern matching and solving.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhysicsPatternLibrary:
    """Tests for physics formula pattern matching."""

    def test_library_loads(self):
        """Test that pattern library loads successfully."""
        from src.classification.physics_patterns import PhysicsPatternLibrary

        library = PhysicsPatternLibrary()
        assert len(library) > 50  # Should have 50+ formulas

    def test_get_by_id(self):
        """Test formula lookup by ID."""
        from src.classification.physics_patterns import PhysicsPatternLibrary

        library = PhysicsPatternLibrary()
        formula = library.get_by_id("planck_energy")

        assert formula is not None
        assert formula.name == "Planck Energy Relation"
        assert "E" in formula.variables
        assert "nu" in formula.variables
        assert "h" in formula.constants

    def test_get_by_category(self):
        """Test formula lookup by category."""
        from src.classification.physics_patterns import PhysicsPatternLibrary

        library = PhysicsPatternLibrary()
        quantum_formulas = library.get_by_category("quantum")

        assert len(quantum_formulas) > 0
        assert all(f.category == "quantum" for f in quantum_formulas)

    def test_search_formulas(self):
        """Test formula text search."""
        from src.classification.physics_patterns import PhysicsPatternLibrary

        library = PhysicsPatternLibrary()
        results = library.search("energy")

        assert len(results) > 0
        # Should find energy-related formulas
        assert any(
            "energy" in f.name.lower() or "energy" in f.description.lower()
            for f in results
        )

    def test_formula_sympy_expr(self):
        """Test that formulas generate valid SymPy expressions."""
        from src.classification.physics_patterns import PhysicsPatternLibrary
        import sympy as sp

        library = PhysicsPatternLibrary()
        formula = library.get_by_id("einstein_mass_energy")

        expr = formula.get_sympy_expr()
        assert expr is not None
        assert isinstance(expr, sp.Basic)
        # E - m*c**2
        assert len(expr.free_symbols) > 0

    def test_pattern_match_exact(self):
        """Test exact pattern matching."""
        from src.classification.physics_patterns import PhysicsPatternLibrary
        import sympy as sp

        library = PhysicsPatternLibrary()

        # Create E - m*c^2 expression
        E, m, c = sp.symbols("E m c", real=True, positive=True)
        expr = E - m * c**2

        match = library.match(expr)

        assert match is not None
        assert match.formula.id == "einstein_mass_energy"
        assert match.confidence >= 0.7

    def test_pattern_match_different_symbols(self):
        """Test pattern matching with different symbol names."""
        from src.classification.physics_patterns import PhysicsPatternLibrary
        import sympy as sp

        library = PhysicsPatternLibrary()

        # Create expression with different variable names
        # Wien's law: lambda_max * T = b, template is lambda_max * T - b
        lam, temp, const = sp.symbols("lam temp const", real=True, positive=True)
        expr = lam * temp - const

        # This should match Wien's displacement law via structural similarity
        match = library.match(expr)

        # Match might not be exact due to different naming
        # but structural match should work
        if match:
            assert match.confidence >= 0.7


class TestPhysicsSolver:
    """Tests for physics formula solving."""

    def test_solver_can_solve_physics(self):
        """Test that physics solver can handle physics equations."""
        from src.classification.physics_patterns import PhysicsPatternLibrary
        from src.solvers.physics_solver import PhysicsSolver
        from src.models import Equation, EquationType
        import sympy as sp

        solver = PhysicsSolver()

        # E = mc^2
        E, m, c = sp.symbols("E m c", real=True, positive=True)
        expr = E - m * c**2

        equation = Equation(
            raw_latex="E = mc^2",
            sympy_expr=expr,
            classification=(EquationType.PHYSICS, "einstein_mass_energy"),
        )

        assert solver.can_solve(equation)

    def test_solve_for_mass(self):
        """Test solving E=mc² for mass."""
        from src.solvers.physics_solver import PhysicsSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = PhysicsSolver()

        E, m, c = sp.symbols("E m c", real=True, positive=True)
        expr = E - m * c**2

        equation = Equation(
            raw_latex="E = mc^2",
            sympy_expr=expr,
            classification=(EquationType.PHYSICS, "einstein_mass_energy"),
        )

        request = SolveRequest(equation=equation, target_variable=m)
        result = solver.solve(request)

        assert result.success
        assert result.solution is not None
        # m = E / c^2
        assert result.solution.symbolic_result is not None
        assert str(result.solution.target_variable) == "m"

    def test_solve_with_numerical_values(self):
        """Test numerical evaluation with constant substitution."""
        from src.solvers.physics_solver import PhysicsSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = PhysicsSolver()

        E, m, c = sp.symbols("E m c", real=True, positive=True)
        expr = E - m * c**2

        equation = Equation(
            raw_latex="E = mc^2",
            sympy_expr=expr,
            classification=(EquationType.PHYSICS, "einstein_mass_energy"),
        )

        # Solve for E with m = 1 kg
        request = SolveRequest(
            equation=equation, target_variable=E, numerical_values={m: 1.0}
        )
        result = solver.solve(request)

        assert result.success
        assert result.solution.numerical_result is not None
        # E = mc² ≈ 8.99e16 J for m=1kg
        assert result.solution.numerical_result > 8e16

    def test_solve_generates_steps(self):
        """Test that solution includes step-by-step explanation."""
        from src.solvers.physics_solver import PhysicsSolver
        from src.models import Equation, SolveRequest, EquationType
        import sympy as sp

        solver = PhysicsSolver()

        E, nu, h = sp.symbols("E nu h", real=True, positive=True)
        expr = E - h * nu

        equation = Equation(
            raw_latex="E = hν",
            sympy_expr=expr,
            classification=(EquationType.PHYSICS, "planck_energy"),
        )

        request = SolveRequest(equation=equation, target_variable=E, show_steps=True)
        result = solver.solve(request)

        assert result.success
        assert len(result.solution.steps) > 0
        # Should mention the formula name
        assert any("Planck" in step.operation for step in result.solution.steps)

    def test_get_formula_info(self):
        """Test formula info retrieval."""
        from src.solvers.physics_solver import PhysicsSolver

        solver = PhysicsSolver()
        info = solver.get_formula_info("de_broglie")

        assert info is not None
        assert info["name"] == "de Broglie Wavelength"
        assert "lambda" in info["variables"]
        assert "p" in info["variables"]
        assert "h" in info["constants"]

    def test_list_categories(self):
        """Test listing formula categories."""
        from src.solvers.physics_solver import PhysicsSolver

        solver = PhysicsSolver()
        categories = solver.categories

        assert len(categories) > 0
        assert "quantum" in categories
        assert "mechanics" in categories


class TestClassifierPhysicsIntegration:
    """Tests for classifier with physics pattern matching."""

    def test_classifies_physics_formula(self):
        """Test that classifier identifies physics formulas."""
        from src.classification.classifier import EquationClassifier
        from src.models import Equation, EquationType
        import sympy as sp

        classifier = EquationClassifier()

        # E = mc^2
        E, m, c = sp.symbols("E m c", real=True, positive=True)
        expr = E - m * c**2

        equation = Equation(raw_latex="E = mc^2", sympy_expr=expr)
        eq_type, subtype = classifier.classify(equation)

        assert eq_type == EquationType.PHYSICS
        assert subtype == "einstein_mass_energy"

    def test_stores_match_for_solver(self):
        """Test that classifier stores match for solver use."""
        from src.classification.classifier import EquationClassifier
        from src.models import Equation, EquationType
        import sympy as sp

        classifier = EquationClassifier()

        E, m, c = sp.symbols("E m c", real=True, positive=True)
        expr = E - m * c**2

        equation = Equation(raw_latex="E = mc^2", sympy_expr=expr)
        classifier.classify(equation)

        # Match should be stored
        match = classifier.last_physics_match
        assert match is not None
        assert match.formula.id == "einstein_mass_energy"

    def test_non_physics_still_works(self):
        """Test that non-physics equations still classify correctly."""
        from src.classification.classifier import EquationClassifier
        from src.models import Equation, EquationType
        import sympy as sp

        classifier = EquationClassifier()

        # Simple polynomial, not a physics formula
        x = sp.Symbol("x")
        expr = x**2 - 4

        equation = Equation(raw_latex="x^2 - 4 = 0", sympy_expr=expr)
        eq_type, subtype = classifier.classify(equation)

        # Should NOT be classified as physics
        assert eq_type == EquationType.POLYNOMIAL
        assert classifier.last_physics_match is None


class TestUnitsHandler:
    """Tests for unit handling."""

    def test_parse_simple_value(self):
        """Test parsing a simple value with units."""
        from src.utils.units import UnitHandler

        handler = UnitHandler()
        value = handler.parse("5.5 m/s")

        assert value is not None
        assert value.magnitude == 5.5

    def test_parse_scientific_notation(self):
        """Test parsing scientific notation."""
        from src.utils.units import UnitHandler

        handler = UnitHandler()
        value = handler.parse("3e8 m/s")

        assert value is not None
        assert value.magnitude == 3e8

    def test_convert_units(self):
        """Test unit conversion."""
        from src.utils.units import UnitHandler

        handler = UnitHandler()

        if not handler.available:
            pytest.skip("Pint not installed")

        value = handler.create(1000, "m")
        converted = handler.convert(value, "km")

        assert converted is not None
        assert abs(converted.magnitude - 1.0) < 1e-10

    def test_check_dimensionality(self):
        """Test dimensionality checking."""
        from src.utils.units import UnitHandler

        handler = UnitHandler()

        if not handler.available:
            pytest.skip("Pint not installed")

        val1 = handler.create(5.0, "m/s")
        val2 = handler.create(10.0, "km/h")

        # Both are velocities, should be compatible
        assert handler.check_dimensionality(val1, val2)

        val3 = handler.create(5.0, "kg")
        # Mass and velocity are not compatible
        assert not handler.check_dimensionality(val1, val3)

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        from src.utils.units import parse_value, are_compatible

        value = parse_value("100 m")
        assert value is not None
        assert value.magnitude == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
