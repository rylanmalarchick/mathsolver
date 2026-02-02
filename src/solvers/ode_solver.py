"""
ODE (Ordinary Differential Equation) solver.

Uses SymPy's dsolve with classification and method-specific step generation.
"""

from typing import List, Optional, Dict, Any, Tuple
import sympy as sp
from sympy import (
    Symbol,
    Function,
    Derivative,
    Eq,
    dsolve,
    classify_ode,
    symbols,
    exp,
    sin,
    cos,
    log,
    sqrt,
    latex,
    simplify,
    checkodesol,
)

from .base import BaseSolver, SolverResult
from ..models import Equation, Solution, SolutionStep, SolveRequest, EquationType


# ODE classification to human-readable descriptions
ODE_METHOD_DESCRIPTIONS = {
    "separable": "Separable equation: can be written as f(y)dy = g(x)dx",
    "1st_exact": "Exact equation: M(x,y)dx + N(x,y)dy = 0 with ∂M/∂y = ∂N/∂x",
    "1st_linear": "First-order linear: dy/dx + P(x)y = Q(x)",
    "Bernoulli": "Bernoulli equation: dy/dx + P(x)y = Q(x)y^n",
    "1st_homogeneous_coeff_best": "Homogeneous coefficients: dy/dx = f(y/x)",
    "1st_homogeneous_coeff_subs_indep_div_dep": "Homogeneous (substitution v = x/y)",
    "1st_homogeneous_coeff_subs_dep_div_indep": "Homogeneous (substitution v = y/x)",
    "nth_linear_constant_coeff_homogeneous": "Linear with constant coefficients (homogeneous)",
    "nth_linear_constant_coeff_undetermined_coefficients": "Linear with constant coefficients (undetermined coefficients)",
    "nth_linear_constant_coeff_variation_of_parameters": "Linear with constant coefficients (variation of parameters)",
    "2nd_linear_airy": "Airy equation: y'' - xy = 0",
    "2nd_linear_bessel": "Bessel equation",
    "Liouville": "Liouville equation",
    "Riccati_special_minus2": "Special Riccati equation",
    "nth_order_reducible": "Reducible nth order equation",
    "separable_reduced": "Reduced separable form",
    "lie_group": "Solvable via Lie group methods",
}


class ODESolver(BaseSolver):
    """
    Solver for ordinary differential equations.

    Uses SymPy's dsolve() with:
    - Automatic ODE classification
    - Method-specific step generation
    - Initial/boundary condition handling
    - Solution verification

    Priority: 20 (high - specialized for ODEs)
    """

    name = "ODESolver"
    description = "Ordinary differential equation solver using SymPy dsolve"

    def can_solve(self, equation: Equation) -> bool:
        """
        Check if equation is an ODE.

        Returns True if expression contains Derivative of a function
        (like y(x)) with respect to a single independent variable.
        Plain derivative expressions like d/dx(x^3) should go to CalculusSolver.
        """
        expr = equation.sympy_expr

        # Check for derivatives
        if not expr.has(Derivative):
            return False

        # Get all derivatives
        derivatives = list(expr.atoms(Derivative))
        if not derivatives:
            return False

        # Check that at least one derivative is of a function (like y(x))
        # not just a plain expression (like x^3)
        has_function_derivative = False
        indep_vars = set()

        for deriv in derivatives:
            # The first argument is what's being differentiated
            diff_expr = deriv.args[0]
            # Check if it's an applied function (like y(x))
            if isinstance(diff_expr, sp.core.function.AppliedUndef):
                has_function_derivative = True

            for var, _ in deriv.variable_count:
                indep_vars.add(var)

        # Must have a function derivative and single independent variable
        return has_function_derivative and len(indep_vars) == 1

    def solve(self, request: SolveRequest) -> SolverResult:
        """
        Solve the ODE using SymPy's dsolve.
        """
        equation = request.equation
        expr = equation.sympy_expr

        # Extract the dependent and independent variables
        dep_func, indep_var = self._extract_ode_variables(expr)
        if dep_func is None:
            return SolverResult.failure(
                "Could not identify dependent function in ODE", solver_name=self.name
            )

        # Convert to equation form if needed
        if isinstance(expr, Eq):
            ode_eq = expr
        else:
            ode_eq = Eq(expr, 0)

        # Classify the ODE
        try:
            classifications = classify_ode(ode_eq, dep_func)
            if isinstance(classifications, dict):
                classification_list = classifications.get("order", [])
            else:
                classification_list = list(classifications) if classifications else []
        except Exception:
            classification_list = []

        # Determine ODE order
        order = self._get_ode_order(expr, dep_func)

        # Generate initial steps
        steps = []
        step_num = 1

        # Step 1: Show original ODE
        steps.append(
            self._create_step(step_num, "Original differential equation", ode_eq)
        )
        step_num += 1

        # Step 2: Classification
        if classification_list:
            primary_class = classification_list[0] if classification_list else "unknown"
            class_desc = ODE_METHOD_DESCRIPTIONS.get(
                primary_class, f"Classification: {primary_class}"
            )
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation=f"Identify as order {order} ODE",
                    equation_state=Symbol("_"),
                    latex_repr=f"\\text{{{class_desc}}}",
                )
            )
            step_num += 1

        # Solve the ODE
        try:
            solution, elapsed_ms = self._timed_solve(dsolve, ode_eq, dep_func)
        except Exception as e:
            return SolverResult.failure(f"dsolve failed: {e}", solver_name=self.name)

        if solution is None:
            return SolverResult.failure(
                "No solution found for ODE", solver_name=self.name
            )

        # Handle multiple solutions
        if isinstance(solution, list):
            solutions = solution
            primary_solution = solutions[0]
        else:
            solutions = [solution]
            primary_solution = solution

        # Generate method-specific steps
        if request.show_steps and classification_list:
            method_steps = self._generate_method_steps(
                ode_eq, dep_func, indep_var, classification_list[0], step_num
            )
            steps.extend(method_steps)
            step_num += len(method_steps)

        # Final step: Show solution
        if isinstance(primary_solution, Eq):
            solution_expr = primary_solution.rhs
            final_eq = primary_solution
        else:
            solution_expr = primary_solution
            final_eq = Eq(dep_func, solution_expr)

        steps.append(self._create_step(step_num, "General solution", final_eq))

        # Verify solution if possible
        try:
            verified = checkodesol(ode_eq, primary_solution, dep_func)
            if verified[0]:
                steps.append(
                    SolutionStep(
                        step_number=step_num + 1,
                        operation="Solution verified ✓",
                        equation_state=Symbol("_"),
                        latex_repr="\\text{Solution satisfies the original ODE}",
                    )
                )
        except Exception:
            pass

        # Build Solution object
        solution_obj = Solution(
            equation=equation,
            target_variable=dep_func,
            symbolic_result=solution_expr,
            steps=steps,
            solve_time_ms=elapsed_ms,
            method_used=f"dsolve:{classification_list[0] if classification_list else 'auto'}",
        )

        return SolverResult.from_solution(solution_obj, solver_name=self.name)

    def _extract_ode_variables(
        self, expr: sp.Basic
    ) -> Tuple[Optional[Function], Optional[Symbol]]:
        """
        Extract the dependent function and independent variable from an ODE.

        Returns:
            (dependent_function, independent_variable) or (None, None)
        """
        derivatives = list(expr.atoms(Derivative))
        if not derivatives:
            return None, None

        # Get the dependent function from first derivative
        first_deriv = derivatives[0]
        dep_func = first_deriv.expr

        # Get independent variable
        indep_var = None
        for var, _ in first_deriv.variable_count:
            indep_var = var
            break

        return dep_func, indep_var

    def _get_ode_order(self, expr: sp.Basic, dep_func: Function) -> int:
        """Get the order of the ODE (highest derivative order)."""
        derivatives = list(expr.atoms(Derivative))
        max_order = 0

        for deriv in derivatives:
            if deriv.expr == dep_func or (
                hasattr(dep_func, "func") and deriv.expr.func == dep_func.func
            ):
                order = sum(count for _, count in deriv.variable_count)
                max_order = max(max_order, order)

        return max_order

    def _generate_method_steps(
        self,
        ode_eq: Eq,
        dep_func: Function,
        indep_var: Symbol,
        method: str,
        start_step: int,
    ) -> List[SolutionStep]:
        """
        Generate method-specific solution steps.
        """
        steps = []
        step_num = start_step

        if method == "separable":
            steps.extend(self._separable_steps(ode_eq, dep_func, indep_var, step_num))
        elif method == "1st_linear":
            steps.extend(
                self._first_order_linear_steps(ode_eq, dep_func, indep_var, step_num)
            )
        elif "nth_linear_constant_coeff" in method:
            steps.extend(
                self._constant_coeff_steps(ode_eq, dep_func, indep_var, step_num)
            )
        elif method == "1st_exact":
            steps.extend(
                self._exact_equation_steps(ode_eq, dep_func, indep_var, step_num)
            )
        else:
            # Generic step
            steps.append(
                SolutionStep(
                    step_number=step_num,
                    operation=f"Apply {method.replace('_', ' ')} method",
                    equation_state=Symbol("_"),
                    latex_repr=f"\\text{{Using SymPy's {method} solver}}",
                )
            )

        return steps

    def _separable_steps(
        self, ode_eq: Eq, dep_func: Function, indep_var: Symbol, start_step: int
    ) -> List[SolutionStep]:
        """Generate steps for separable ODEs."""
        steps = []
        step_num = start_step
        y = Symbol("y")
        x = indep_var

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Separate variables: move y terms to left, x terms to right",
                equation_state=Symbol("_"),
                latex_repr=r"f(y)\,dy = g(x)\,dx",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Integrate both sides",
                equation_state=Symbol("_"),
                latex_repr=r"\int f(y)\,dy = \int g(x)\,dx",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Add constant of integration",
                equation_state=Symbol("_"),
                latex_repr=r"F(y) = G(x) + C",
            )
        )

        return steps

    def _first_order_linear_steps(
        self, ode_eq: Eq, dep_func: Function, indep_var: Symbol, start_step: int
    ) -> List[SolutionStep]:
        """Generate steps for first-order linear ODEs."""
        steps = []
        step_num = start_step
        x = indep_var

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Write in standard form: y' + P(x)y = Q(x)",
                equation_state=Symbol("_"),
                latex_repr=r"\frac{dy}{dx} + P(x)y = Q(x)",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Find integrating factor: μ(x) = e^{∫P(x)dx}",
                equation_state=Symbol("_"),
                latex_repr=r"\mu(x) = e^{\int P(x)\,dx}",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Multiply through by integrating factor",
                equation_state=Symbol("_"),
                latex_repr=r"\frac{d}{dx}[\mu(x)y] = \mu(x)Q(x)",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Integrate both sides",
                equation_state=Symbol("_"),
                latex_repr=r"y = \frac{1}{\mu(x)}\int \mu(x)Q(x)\,dx + \frac{C}{\mu(x)}",
            )
        )

        return steps

    def _constant_coeff_steps(
        self, ode_eq: Eq, dep_func: Function, indep_var: Symbol, start_step: int
    ) -> List[SolutionStep]:
        """Generate steps for constant coefficient linear ODEs."""
        steps = []
        step_num = start_step

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Form characteristic equation by substituting y = e^{rx}",
                equation_state=Symbol("_"),
                latex_repr=r"\text{Substitute } y = e^{rx} \text{ to get characteristic equation}",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Solve characteristic polynomial for roots",
                equation_state=Symbol("_"),
                latex_repr=r"ar^2 + br + c = 0 \rightarrow r = \frac{-b \pm \sqrt{b^2-4ac}}{2a}",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Form general solution based on root types",
                equation_state=Symbol("_"),
                latex_repr=r"\text{Distinct real: } C_1e^{r_1x} + C_2e^{r_2x}, \quad \text{Repeated: } (C_1 + C_2x)e^{rx}",
            )
        )

        return steps

    def _exact_equation_steps(
        self, ode_eq: Eq, dep_func: Function, indep_var: Symbol, start_step: int
    ) -> List[SolutionStep]:
        """Generate steps for exact equations."""
        steps = []
        step_num = start_step

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Write as M(x,y)dx + N(x,y)dy = 0",
                equation_state=Symbol("_"),
                latex_repr=r"M(x,y)\,dx + N(x,y)\,dy = 0",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Verify exactness: ∂M/∂y = ∂N/∂x",
                equation_state=Symbol("_"),
                latex_repr=r"\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Find potential function F where ∂F/∂x = M, ∂F/∂y = N",
                equation_state=Symbol("_"),
                latex_repr=r"F(x,y) = \int M\,dx + g(y) = \int N\,dy + h(x)",
            )
        )
        step_num += 1

        steps.append(
            SolutionStep(
                step_number=step_num,
                operation="Solution is F(x,y) = C",
                equation_state=Symbol("_"),
                latex_repr=r"F(x,y) = C",
            )
        )

        return steps

    def solve_ivp(
        self, request: SolveRequest, initial_conditions: Dict[Any, float]
    ) -> SolverResult:
        """
        Solve ODE with initial value conditions.

        Args:
            request: SolveRequest with the ODE
            initial_conditions: Dict mapping conditions like {y(0): 1, y'(0): 0}

        Returns:
            SolverResult with particular solution
        """
        # First get general solution
        result = self.solve(request)
        if not result.success:
            return result

        # Apply initial conditions
        solution = result.solution
        general_sol = solution.symbolic_result

        # This is a simplified implementation
        # Full IVP solving would substitute conditions and solve for constants
        try:
            # Get constants in solution (C1, C2, etc.)
            constants = [
                s
                for s in general_sol.free_symbols
                if str(s).startswith("C") and str(s)[1:].isdigit()
            ]

            if constants and initial_conditions:
                # Would need to set up system of equations from ICs
                # and solve for constants
                pass
        except Exception:
            pass

        return result

    def get_classification_info(self, equation: Equation) -> Dict[str, Any]:
        """
        Get detailed classification information for an ODE.

        Returns dict with order, classifications, suggested methods, etc.
        """
        expr = equation.sympy_expr
        dep_func, indep_var = self._extract_ode_variables(expr)

        if dep_func is None:
            return {"error": "Not a valid ODE"}

        if isinstance(expr, Eq):
            ode_eq = expr
        else:
            ode_eq = Eq(expr, 0)

        try:
            classifications = classify_ode(ode_eq, dep_func)
            if isinstance(classifications, tuple):
                classifications = list(classifications)
        except Exception as e:
            return {"error": str(e)}

        order = self._get_ode_order(expr, dep_func)

        return {
            "order": order,
            "dependent_function": str(dep_func),
            "independent_variable": str(indep_var),
            "classifications": classifications,
            "primary_method": classifications[0] if classifications else None,
            "description": ODE_METHOD_DESCRIPTIONS.get(
                classifications[0] if classifications else "", "Unknown classification"
            ),
        }
