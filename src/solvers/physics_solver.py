"""
Physics formula solver with template-based solving.

Uses the PhysicsPatternLibrary to match equations against known physics
formulas and provides step-by-step solutions with physical explanations.
"""

from typing import List, Optional, Dict, Any
import sympy as sp
from sympy import Symbol, Eq, latex, sqrt, sin, cos, pi, log, exp

from .base import BaseSolver, SolverResult
from ..models import Equation, Solution, SolutionStep, SolveRequest, EquationType
from ..classification.physics_patterns import PhysicsPatternLibrary, PatternMatch


class PhysicsSolver(BaseSolver):
    """
    Solver for physics equations using template matching.

    Matches input equations against a database of known physics formulas
    and uses pre-defined solve templates for fast, accurate solutions
    with physics-specific step explanations.

    Priority: 10 (highest - checked before general solvers)
    """

    name = "PhysicsSolver"
    description = "Physics formula solver with template matching"

    def __init__(self, library: Optional[PhysicsPatternLibrary] = None):
        """
        Initialize with physics formula library.

        Args:
            library: PhysicsPatternLibrary instance (creates default if None)
        """
        self._library = library

    @property
    def library(self) -> PhysicsPatternLibrary:
        """Lazy-load the pattern library."""
        if self._library is None:
            self._library = PhysicsPatternLibrary()
        return self._library

    def can_solve(self, equation: Equation) -> bool:
        """
        Check if equation matches a known physics formula.

        Returns True if:
        - Equation is classified as PHYSICS type, OR
        - Expression matches a formula in the library with high confidence
        """
        # If already classified as physics, we can handle it
        eq_type, subtype = equation.classification
        if eq_type == EquationType.PHYSICS:
            return True

        # Try to match against library
        match = self.library.match(equation.sympy_expr)
        return match is not None

    def solve(self, request: SolveRequest) -> SolverResult:
        """
        Solve using physics formula templates.

        Process:
        1. Match equation against formula library
        2. Map user variables to template variables
        3. Apply solve template for target variable
        4. Substitute constants if numerical evaluation requested
        5. Generate physics-specific step explanations
        """
        equation = request.equation
        target = request.target_variable
        expr = equation.sympy_expr

        # Find matching formula
        match = self.library.match(expr)
        if match is None:
            return SolverResult.failure(
                "No matching physics formula found", solver_name=self.name
            )

        formula = match.formula

        # Determine target variable
        target_name = self._resolve_target_variable(target, formula, match)
        if target_name is None:
            return SolverResult.failure(
                f"Cannot solve for {target} - not in formula variables: {formula.variables}",
                solver_name=self.name,
            )

        # Check if we have a solve template for this variable
        if target_name not in formula.solve_templates:
            return SolverResult.failure(
                f"No solve template for {target_name} in {formula.name}",
                solver_name=self.name,
            )

        # Parse the solve template
        try:
            solution_expr, elapsed_ms = self._timed_solve(
                self._apply_template, formula, target_name, match
            )
        except Exception as e:
            return SolverResult.failure(
                f"Template solving failed: {e}", solver_name=self.name
            )

        # Generate solution steps
        steps = self._generate_physics_steps(formula, target_name, match, request)

        # Build the target symbol
        target_sym = Symbol(target_name)

        # Create Solution object
        solution = Solution(
            equation=equation,
            target_variable=target_sym,
            symbolic_result=solution_expr,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used=f"physics_template:{formula.id}",
        )

        # Numerical evaluation
        if request.numerical_values or formula.constants:
            solution.numerical_result = self._evaluate_numerically(
                solution_expr, formula, request.numerical_values, match
            )

        return SolverResult.from_solution(solution, solver_name=self.name)

    def _resolve_target_variable(
        self, target: Optional[Symbol], formula: "PhysicsFormula", match: PatternMatch
    ) -> Optional[str]:
        """
        Resolve target variable to a formula variable name.

        Maps user's target symbol to the corresponding template variable.
        """
        if target is None:
            # Default to first variable in formula
            return formula.variables[0] if formula.variables else None

        target_str = str(target)

        # Direct name match
        if target_str in formula.variables:
            return target_str

        # Check variable mapping
        for template_sym, user_sym in match.variable_mapping.items():
            if str(user_sym) == target_str and str(template_sym) in formula.variables:
                return str(template_sym)

        # Try case-insensitive match
        for var in formula.variables:
            if var.lower() == target_str.lower():
                return var

        return None

    def _apply_template(
        self, formula: "PhysicsFormula", target_name: str, match: PatternMatch
    ) -> sp.Basic:
        """
        Apply the solve template to get the solution expression.
        """
        template_str = formula.solve_templates[target_name]

        # Create namespace with all variables and constants
        namespace = {}

        # Add variables as symbols
        for var in formula.variables:
            if var != target_name:
                namespace[var] = Symbol(var, real=True, positive=True)

        # Add constants as symbols (for symbolic result)
        for const_name in formula.constants:
            namespace[const_name] = Symbol(const_name, real=True, positive=True)

        # Add math functions
        namespace.update(
            {"sin": sin, "cos": cos, "sqrt": sqrt, "pi": pi, "log": log, "exp": exp}
        )

        # Parse the template
        solution_expr = sp.sympify(template_str, locals=namespace)

        return solution_expr

    def _generate_physics_steps(
        self,
        formula: "PhysicsFormula",
        target_name: str,
        match: PatternMatch,
        request: SolveRequest,
    ) -> List[SolutionStep]:
        """
        Generate physics-specific solution steps.

        Uses pre-defined steps if available, otherwise generates generic ones.
        """
        steps = []
        step_num = 1

        # Step 1: Identify the formula
        original_eq = formula.get_sympy_expr()
        steps.append(
            SolutionStep(
                step_number=step_num,
                operation=f"Recognize as {formula.name}",
                equation_state=Eq(original_eq, 0),
                latex_repr=formula.latex,
            )
        )
        step_num += 1

        # Step 2: State what we're solving for
        steps.append(
            SolutionStep(
                step_number=step_num,
                operation=f"Solve for {target_name}",
                equation_state=Symbol(target_name),
                latex_repr=f"\\text{{Solving for }} {target_name}",
            )
        )
        step_num += 1

        # Use pre-defined steps if available
        if target_name in formula.steps:
            for step_text in formula.steps[target_name]:
                steps.append(
                    SolutionStep(
                        step_number=step_num,
                        operation=step_text,
                        equation_state=sp.Symbol("_"),  # Placeholder
                        latex_repr=f"\\text{{{step_text}}}",
                    )
                )
                step_num += 1
        else:
            # Generate generic algebraic step
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation=f"Rearrange to isolate {target_name}",
                    equation_state=sp.Symbol("_"),
                    latex_repr=f"\\text{{Algebraic rearrangement}}",
                )
            )
            step_num += 1

        # Final step: Show the solution template
        template_str = formula.solve_templates[target_name]
        namespace = self._build_namespace(formula, target_name)
        solution_expr = sp.sympify(template_str, locals=namespace)

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation=f"Solution: {target_name} = ",
                equation_state=Eq(Symbol(target_name), solution_expr),
                latex_repr=latex(Eq(Symbol(target_name), solution_expr)),
            )
        )

        return steps

    def _build_namespace(
        self, formula: "PhysicsFormula", exclude_var: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a namespace for sympify with all variables and constants."""
        namespace = {}

        for var in formula.variables:
            if var != exclude_var:
                namespace[var] = Symbol(var, real=True, positive=True)

        for const_name in formula.constants:
            namespace[const_name] = Symbol(const_name, real=True, positive=True)

        namespace.update(
            {"sin": sin, "cos": cos, "sqrt": sqrt, "pi": pi, "log": log, "exp": exp}
        )

        return namespace

    def _evaluate_numerically(
        self,
        solution_expr: sp.Basic,
        formula: "PhysicsFormula",
        user_values: Dict[Symbol, float],
        match: PatternMatch,
    ) -> Optional[float]:
        """
        Evaluate the solution numerically.

        Substitutes:
        1. User-provided numerical values
        2. Physical constants from the formula
        """
        try:
            subs_dict = {}

            # Add physical constants
            for const_name, const_info in formula.constants.items():
                const_sym = Symbol(const_name)
                subs_dict[const_sym] = const_info["value"]

            # Add user-provided values (map to template variables if needed)
            for user_sym, value in user_values.items():
                # Check if this maps to a template variable
                for template_sym, mapped_sym in match.variable_mapping.items():
                    if str(mapped_sym) == str(user_sym):
                        subs_dict[template_sym] = value
                        break
                else:
                    # Direct substitution
                    subs_dict[user_sym] = value

            # Evaluate
            result = solution_expr.subs(subs_dict).evalf()

            # Return float if it's a number
            if result.is_number:
                return float(result)

            return None

        except Exception:
            return None

    def get_formula_info(self, formula_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a formula.

        Useful for UI display and help text.
        """
        formula = self.library.get_by_id(formula_id)
        if formula is None:
            return None

        return {
            "id": formula.id,
            "name": formula.name,
            "category": formula.category,
            "subcategory": formula.subcategory,
            "latex": formula.latex,
            "description": formula.description,
            "variables": formula.variables,
            "constants": {
                name: {
                    "value": info["value"],
                    "unit": info.get("unit", ""),
                    "name": info.get("name", name),
                }
                for name, info in formula.constants.items()
            },
            "units": formula.units,
            "can_solve_for": list(formula.solve_templates.keys()),
        }

    def list_formulas_by_category(self, category: str) -> List[Dict[str, str]]:
        """List all formulas in a category with basic info."""
        formulas = self.library.get_by_category(category)
        return [
            {"id": f.id, "name": f.name, "latex": f.latex, "description": f.description}
            for f in formulas
        ]

    @property
    def categories(self) -> List[str]:
        """Get all available formula categories."""
        return self.library.categories
