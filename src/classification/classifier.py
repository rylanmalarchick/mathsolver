"""
Equation classifier for routing to specialized solvers.

Priority-based classification that checks specialized patterns first.
"""

from typing import Tuple, Optional, List
import sympy as sp
from sympy import (
    Derivative,
    Integral,
    Matrix,
    MatrixSymbol,
    Function,
    Eq,
    Symbol,
    Piecewise,
)

from ..models import Equation, EquationType


class EquationClassifier:
    """
    Classifies equations to route them to appropriate solvers.

    Classification priority (highest first):
    1. Physics formulas (pattern library match)
    2. Differential equations (contains derivatives)
    3. Linear algebra (matrices, systems)
    4. Calculus (integrals, limits)
    5. Polynomial/rational
    6. General (fallback to SymPy solve)

    Usage:
        classifier = EquationClassifier()
        eq_type, subtype = classifier.classify(equation)
    """

    def __init__(self):
        """Initialize classifier with pattern libraries."""
        # Physics patterns will be loaded from physics_patterns.py
        # For now, use a simple placeholder
        self._physics_patterns = None

    def classify(self, equation: Equation) -> Tuple[EquationType, Optional[str]]:
        """
        Classify an equation by type.

        Args:
            equation: Equation object to classify

        Returns:
            Tuple of (EquationType, subtype string or None)
        """
        expr = equation.sympy_expr

        # Try physics patterns first (highest priority)
        physics_match = self._check_physics_patterns(expr)
        if physics_match:
            return (EquationType.PHYSICS, physics_match)

        # Check for differential equations
        ode_type = self._check_differential(expr)
        if ode_type:
            return (EquationType.ODE, ode_type)

        # Check for calculus operations
        calc_type = self._check_calculus(expr)
        if calc_type:
            return (EquationType.CALCULUS, calc_type)

        # Check for linear algebra
        linalg_type = self._check_linear_algebra(expr)
        if linalg_type:
            return (EquationType.LINEAR_ALGEBRA, linalg_type)

        # Check for polynomial expressions
        poly_info = self._check_polynomial(expr)
        if poly_info:
            return (EquationType.POLYNOMIAL, poly_info)

        # Check for trigonometric
        if self._has_trig(expr):
            return (EquationType.TRIGONOMETRIC, None)

        # Fallback to general
        return (EquationType.GENERAL, None)

    def _check_physics_patterns(self, expr: sp.Basic) -> Optional[str]:
        """
        Check if expression matches a known physics formula.

        Returns formula name if matched, None otherwise.
        """
        # TODO: Implement physics pattern matching in Week 2
        # This will use the PhysicsPatternLibrary
        return None

    def _check_differential(self, expr: sp.Basic) -> Optional[str]:
        """Check for differential equations."""
        if not expr.has(Derivative):
            return None

        # Get all derivatives in expression
        derivatives = list(expr.atoms(Derivative))
        if not derivatives:
            return None

        # Determine order (highest derivative order)
        max_order = 0
        for deriv in derivatives:
            # Count derivative orders
            order = sum(count for _, count in deriv.variable_count)
            max_order = max(max_order, order)

        # Check if it's an ODE (single independent variable) or PDE
        indep_vars = set()
        for deriv in derivatives:
            for var, _ in deriv.variable_count:
                indep_vars.add(var)

        if len(indep_vars) > 1:
            return f"pde_order_{max_order}"
        else:
            return f"ode_order_{max_order}"

    def _check_calculus(self, expr: sp.Basic) -> Optional[str]:
        """Check for calculus operations (integrals, limits)."""
        if expr.has(Integral):
            # Check if definite or indefinite
            integrals = list(expr.atoms(Integral))
            for integral in integrals:
                if len(integral.limits[0]) == 3:  # (var, lower, upper)
                    return "definite_integral"
            return "indefinite_integral"

        # Check for limit expressions
        if expr.has(sp.Limit):
            return "limit"

        # Check for series/summation
        if expr.has(sp.Sum):
            return "series"

        return None

    def _check_linear_algebra(self, expr: sp.Basic) -> Optional[str]:
        """Check for linear algebra operations."""
        if isinstance(expr, (Matrix, MatrixSymbol)):
            return "matrix"

        if expr.has(MatrixSymbol):
            return "matrix_expr"

        # Check if it's a system of equations (list of Eq)
        if isinstance(expr, (list, tuple)):
            if all(isinstance(e, Eq) for e in expr):
                return "system"

        return None

    def _check_polynomial(self, expr: sp.Basic) -> Optional[str]:
        """Check if expression is polynomial."""
        # Handle equations
        if isinstance(expr, Eq):
            # Check both sides
            lhs, rhs = expr.args
            combined = lhs - rhs
        else:
            combined = expr

        # Get free symbols
        symbols = list(combined.free_symbols)
        if not symbols:
            return None

        # Check if polynomial in any variable
        for sym in symbols:
            try:
                poly = sp.Poly(combined, sym)
                degree = poly.degree()
                if degree >= 1:
                    return f"degree_{degree}"
            except (sp.PolynomialError, sp.GeneratorsNeeded):
                continue

        return None

    def _has_trig(self, expr: sp.Basic) -> bool:
        """Check if expression contains trigonometric functions."""
        trig_funcs = (
            sp.sin,
            sp.cos,
            sp.tan,
            sp.cot,
            sp.sec,
            sp.csc,
            sp.asin,
            sp.acos,
            sp.atan,
            sp.sinh,
            sp.cosh,
            sp.tanh,
        )
        return any(expr.has(f) for f in trig_funcs)

    def get_variables(self, equation: Equation) -> List[sp.Symbol]:
        """
        Get solvable variables from an equation.

        Filters out known constants (e, pi, etc.)
        """
        expr = equation.sympy_expr

        # Get all free symbols
        all_symbols = expr.free_symbols

        # Filter out known mathematical constants
        constants = {sp.E, sp.pi, sp.I, sp.oo, sp.zoo, sp.nan}

        # Also filter symbols that look like constants (single lowercase letters
        # that are commonly constants: e, i, etc.)
        # This is heuristic and might need refinement

        variables = [s for s in all_symbols if s not in constants]

        return sorted(variables, key=lambda s: str(s))
