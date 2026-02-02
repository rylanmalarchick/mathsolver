"""
Physics formula pattern matching system.

Matches user equations against a database of known physics formulas
using SymPy structural pattern matching.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import sympy as sp
from sympy import Wild, Symbol, Eq, sin, cos, sqrt, pi, log, exp


@dataclass
class PhysicsFormula:
    """A physics formula from the database."""

    id: str
    name: str
    category: str
    subcategory: str
    latex: str
    sympy_template: str
    variables: List[str]
    constants: Dict[str, Dict[str, Any]]
    units: Dict[str, str]
    description: str
    solve_templates: Dict[str, str] = field(default_factory=dict)
    steps: Dict[str, List[str]] = field(default_factory=dict)

    # Cached SymPy expression (built lazily)
    _sympy_expr: Optional[sp.Basic] = field(default=None, repr=False)

    def get_sympy_expr(self) -> sp.Basic:
        """Get the SymPy expression for this formula."""
        if self._sympy_expr is None:
            self._sympy_expr = self._parse_template()
        return self._sympy_expr

    def _parse_template(self) -> sp.Basic:
        """Parse the sympy_template string into a SymPy expression."""
        # Create symbol namespace
        namespace = {}

        # Add all variables
        for var in self.variables:
            namespace[var] = Symbol(var, real=True, positive=True)

        # Add all constants as symbols (we'll substitute values later)
        for const_name in self.constants:
            namespace[const_name] = Symbol(const_name, real=True, positive=True)

        # Add common functions
        namespace.update(
            {"sin": sin, "cos": cos, "sqrt": sqrt, "pi": pi, "log": log, "exp": exp}
        )

        try:
            expr = sp.sympify(self.sympy_template, locals=namespace)
            return expr
        except Exception as e:
            # Fallback to basic parsing
            return sp.sympify(self.sympy_template)

    def get_constant_values(self) -> Dict[Symbol, float]:
        """Get a dict mapping constant symbols to their values."""
        return {Symbol(name): info["value"] for name, info in self.constants.items()}


@dataclass
class PatternMatch:
    """Result of a pattern match attempt."""

    formula: PhysicsFormula
    confidence: float  # 0.0 to 1.0
    variable_mapping: Dict[Symbol, sp.Basic]  # Maps template vars to user vars
    matched_structure: bool  # True if structure matched exactly


class PhysicsPatternLibrary:
    """
    Library of physics formulas with pattern matching.

    Loads formulas from JSON and provides matching against user expressions.

    Usage:
        library = PhysicsPatternLibrary()
        match = library.match(user_expr)
        if match:
            print(f"Matched: {match.formula.name}")
    """

    DEFAULT_PATH = (
        Path(__file__).parent.parent.parent / "config" / "physics_formulas.json"
    )

    def __init__(self, formulas_path: Optional[Path] = None):
        """
        Load formulas from JSON file.

        Args:
            formulas_path: Path to physics_formulas.json
        """
        self.path = formulas_path or self.DEFAULT_PATH
        self.formulas: List[PhysicsFormula] = []
        self._by_id: Dict[str, PhysicsFormula] = {}
        self._by_category: Dict[str, List[PhysicsFormula]] = {}

        self._load_formulas()

    def _load_formulas(self):
        """Load and parse formulas from JSON."""
        if not self.path.exists():
            raise FileNotFoundError(f"Physics formulas not found: {self.path}")

        with open(self.path, "r") as f:
            data = json.load(f)

        for formula_data in data.get("formulas", []):
            formula = PhysicsFormula(
                id=formula_data["id"],
                name=formula_data["name"],
                category=formula_data.get("category", ""),
                subcategory=formula_data.get("subcategory", ""),
                latex=formula_data.get("latex", ""),
                sympy_template=formula_data.get("sympy_template", ""),
                variables=formula_data.get("variables", []),
                constants=formula_data.get("constants", {}),
                units=formula_data.get("units", {}),
                description=formula_data.get("description", ""),
                solve_templates=formula_data.get("solve_templates", {}),
                steps=formula_data.get("steps", {}),
            )

            self.formulas.append(formula)
            self._by_id[formula.id] = formula

            # Index by category
            if formula.category not in self._by_category:
                self._by_category[formula.category] = []
            self._by_category[formula.category].append(formula)

    def match(self, expr: sp.Basic) -> Optional[PatternMatch]:
        """
        Find the best matching physics formula for an expression.

        Args:
            expr: SymPy expression to match

        Returns:
            PatternMatch if found, None otherwise
        """
        best_match: Optional[PatternMatch] = None
        best_confidence = 0.0

        for formula in self.formulas:
            match = self._try_match(expr, formula)
            if match and match.confidence > best_confidence:
                best_match = match
                best_confidence = match.confidence

        # Only return if confidence is high enough
        if best_match and best_match.confidence >= 0.7:
            return best_match

        return None

    def _try_match(
        self, expr: sp.Basic, formula: PhysicsFormula
    ) -> Optional[PatternMatch]:
        """
        Try to match an expression against a single formula.

        Uses multiple matching strategies:
        1. Exact structural match with Wild symbols
        2. Normalized form comparison
        3. Variable count and structure heuristics
        """
        try:
            template = formula.get_sympy_expr()
        except Exception:
            return None

        # Strategy 1: Wild symbol matching
        wild_match = self._wild_match(expr, template, formula)
        if wild_match:
            return wild_match

        # Strategy 2: Structural similarity
        structural_match = self._structural_match(expr, template, formula)
        if structural_match:
            return structural_match

        return None

    def _wild_match(
        self, expr: sp.Basic, template: sp.Basic, formula: PhysicsFormula
    ) -> Optional[PatternMatch]:
        """
        Match using SymPy Wild symbols.

        Creates a pattern where each variable is a Wild symbol,
        then attempts to match the user expression.
        """
        # Create wild pattern
        wild_symbols = {}
        for var_name in formula.variables:
            wild_symbols[var_name] = Wild(var_name)

        # Also create wilds for constants
        for const_name in formula.constants:
            wild_symbols[const_name] = Wild(const_name)

        # Build pattern expression by substituting wilds
        try:
            pattern = template
            for name, wild in wild_symbols.items():
                sym = Symbol(name)
                pattern = pattern.subs(sym, wild)
        except Exception:
            return None

        # Handle equation form (expr should equal 0)
        if isinstance(expr, Eq):
            expr_normalized = expr.lhs - expr.rhs
        else:
            expr_normalized = expr

        # Try to match
        match_result = expr_normalized.match(pattern)

        if match_result is not None:
            # Verify that the match makes sense
            var_mapping = {
                Symbol(name): match_result.get(wild, Symbol(name))
                for name, wild in wild_symbols.items()
            }

            return PatternMatch(
                formula=formula,
                confidence=0.95,
                variable_mapping=var_mapping,
                matched_structure=True,
            )

        # Try matching negative form (pattern might be lhs - rhs = 0)
        match_result = (-expr_normalized).match(pattern)
        if match_result is not None:
            var_mapping = {
                Symbol(name): match_result.get(wild, Symbol(name))
                for name, wild in wild_symbols.items()
            }

            return PatternMatch(
                formula=formula,
                confidence=0.90,
                variable_mapping=var_mapping,
                matched_structure=True,
            )

        return None

    def _structural_match(
        self, expr: sp.Basic, template: sp.Basic, formula: PhysicsFormula
    ) -> Optional[PatternMatch]:
        """
        Match based on structural similarity.

        Compares:
        - Number of free symbols
        - Types of operations (Add, Mul, Pow, etc.)
        - Expression depth
        """
        # Normalize expressions
        if isinstance(expr, Eq):
            expr = expr.lhs - expr.rhs

        # Get structural features
        expr_ops = self._get_operations(expr)
        template_ops = self._get_operations(template)

        # Compare operation types
        op_similarity = self._jaccard_similarity(expr_ops, template_ops)

        # Compare number of free symbols
        expr_vars = len(expr.free_symbols)
        template_vars = len(formula.variables) + len(formula.constants)
        var_ratio = min(expr_vars, template_vars) / max(expr_vars, template_vars, 1)

        # Combined confidence
        confidence = op_similarity * 0.6 + var_ratio * 0.4

        if confidence >= 0.7:
            # Attempt variable mapping heuristically
            var_mapping = self._heuristic_var_mapping(expr, formula)

            return PatternMatch(
                formula=formula,
                confidence=confidence,
                variable_mapping=var_mapping,
                matched_structure=False,
            )

        return None

    def _get_operations(self, expr: sp.Basic) -> set:
        """Extract operation types from an expression."""
        ops = set()

        def traverse(e):
            ops.add(type(e).__name__)
            if hasattr(e, "args"):
                for arg in e.args:
                    traverse(arg)

        traverse(expr)
        return ops

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _heuristic_var_mapping(
        self, expr: sp.Basic, formula: PhysicsFormula
    ) -> Dict[Symbol, sp.Basic]:
        """
        Attempt to map formula variables to expression symbols.

        Uses heuristics like matching symbol names.
        """
        mapping = {}
        expr_symbols = list(expr.free_symbols)

        for var_name in formula.variables:
            template_sym = Symbol(var_name)

            # Try exact name match first
            for sym in expr_symbols:
                if str(sym) == var_name or str(sym).replace(
                    "_", ""
                ) == var_name.replace("_", ""):
                    mapping[template_sym] = sym
                    break

            # If no match, use template symbol itself
            if template_sym not in mapping:
                mapping[template_sym] = template_sym

        return mapping

    def get_by_id(self, formula_id: str) -> Optional[PhysicsFormula]:
        """Get a formula by its ID."""
        return self._by_id.get(formula_id)

    def get_by_category(self, category: str) -> List[PhysicsFormula]:
        """Get all formulas in a category."""
        return self._by_category.get(category, [])

    def search(self, query: str) -> List[PhysicsFormula]:
        """Search formulas by name or description."""
        query_lower = query.lower()
        results = []

        for formula in self.formulas:
            if (
                query_lower in formula.name.lower()
                or query_lower in formula.description.lower()
                or query_lower in formula.category.lower()
            ):
                results.append(formula)

        return results

    @property
    def categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self._by_category.keys())

    def __len__(self) -> int:
        return len(self.formulas)
