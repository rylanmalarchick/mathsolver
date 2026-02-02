"""
Core data structures for MathSolver.

These dataclasses define the contract between layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum, auto

import sympy as sp


class EquationType(Enum):
    """Classification categories for equations."""

    PHYSICS = auto()
    ODE = auto()
    PDE = auto()
    LINEAR_ALGEBRA = auto()
    CALCULUS = auto()
    POLYNOMIAL = auto()
    TRIGONOMETRIC = auto()
    GENERAL = auto()
    NUMERICAL = auto()  # Fallback when symbolic fails


@dataclass
class SolutionStep:
    """A single step in a solution derivation."""

    step_number: int
    operation: str  # Human-readable description, e.g., "Divide both sides by T"
    equation_state: sp.Basic  # SymPy expression at this step
    latex_repr: str  # Pre-rendered LaTeX for display

    def __post_init__(self):
        if not self.latex_repr:
            self.latex_repr = sp.latex(self.equation_state)


@dataclass
class Equation:
    """
    Represents a parsed equation ready for classification and solving.

    This is the primary data object passed through the pipeline.
    """

    raw_latex: str
    sympy_expr: sp.Basic  # Can be Expr, Eq, or Relational
    classification: Tuple[EquationType, Optional[str]] = field(
        default=(EquationType.GENERAL, None)
    )
    variables: List[sp.Symbol] = field(default_factory=list)
    constants: Dict[sp.Symbol, float] = field(default_factory=dict)
    ocr_confidence: float = 1.0  # 1.0 if manually entered
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Auto-extract variables if not provided
        if not self.variables and self.sympy_expr is not None:
            self.variables = list(self.sympy_expr.free_symbols)


@dataclass
class Solution:
    """
    Complete solution for an equation.

    Contains symbolic result, steps, and optional numerical evaluation.
    """

    equation: Equation
    target_variable: Optional[sp.Symbol]
    symbolic_result: sp.Basic  # The solved expression
    steps: List[SolutionStep] = field(default_factory=list)
    numerical_result: Optional[float] = None
    solve_time_ms: int = 0
    method_used: str = ""  # e.g., "sympy.solve", "physics_template", "dsolve"

    @property
    def latex(self) -> str:
        """Get LaTeX representation of the solution."""
        result = self.symbolic_result

        # Handle list of solutions (e.g., from x^2 - 4 = 0 -> [-2, 2])
        if isinstance(result, (list, tuple)):
            if len(result) == 1:
                result = result[0]
            else:
                # Multiple solutions: x = -2 or x = 2
                if self.target_variable:
                    parts = [sp.latex(sp.Eq(self.target_variable, r)) for r in result]
                    return " \\text{ or } ".join(parts)
                return ", ".join(sp.latex(r) for r in result)

        if self.target_variable:
            return sp.latex(sp.Eq(self.target_variable, result))
        return sp.latex(result)


@dataclass
class SolveRequest:
    """
    Request to solve an equation.

    Allows specifying target variable and numerical values.
    """

    equation: Equation
    target_variable: Optional[sp.Symbol] = None
    numerical_values: Dict[sp.Symbol, float] = field(default_factory=dict)
    show_steps: bool = True

    def __post_init__(self):
        # Default to first variable if not specified
        if self.target_variable is None and self.equation.variables:
            self.target_variable = self.equation.variables[0]


@dataclass
class OCRResult:
    """Result from OCR processing."""

    latex: str
    confidence: float
    processing_time_ms: int
    image_path: Optional[str] = None
