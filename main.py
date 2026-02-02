#!/usr/bin/env python3
"""
MathSolver - Offline equation solver with OCR and symbolic computation.

Entry point for the application with CLI support.

Usage:
    mathsolver                      # Launch GUI
    mathsolver "x^2 + 2x + 1 = 0"   # Solve equation in terminal
    mathsolver -f latex "E = mc^2"  # Solve and output as LaTeX
    mathsolver --from-clipboard     # Solve equation from clipboard
"""

import sys
import os
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mathsolver",
        description="Fast, offline equation solver with OCR and symbolic computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mathsolver                           Launch the GUI
  mathsolver "x^2 - 4 = 0"             Solve equation and print result
  mathsolver "E = mc^2" -v m           Solve for variable m
  mathsolver -f latex "F = ma"         Output solution as LaTeX
  mathsolver -f python "y = mx + b"    Output as Python code
  mathsolver --from-clipboard          Solve from clipboard content
  mathsolver --list-formulas           List available physics formulas
        """,
    )

    # Positional: equation to solve
    parser.add_argument(
        "equation",
        nargs="?",
        help="Equation to solve (LaTeX or plain text)",
    )

    # Output format
    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "latex", "python", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Target variable
    parser.add_argument(
        "-v",
        "--variable",
        help="Variable to solve for (auto-detected if not specified)",
    )

    # Show steps
    parser.add_argument(
        "-s",
        "--steps",
        action="store_true",
        help="Show solution steps",
    )

    # Clipboard input
    parser.add_argument(
        "--from-clipboard",
        action="store_true",
        help="Read equation from clipboard",
    )

    # List formulas
    parser.add_argument(
        "--list-formulas",
        action="store_true",
        help="List available physics formulas",
    )

    # Search formulas
    parser.add_argument(
        "--search",
        metavar="TERM",
        help="Search physics formulas by name or category",
    )

    # Numerical values
    parser.add_argument(
        "-n",
        "--numerical",
        metavar="VAR=VALUE",
        action="append",
        help="Provide numerical values (e.g., -n m=5 -n c=3e8)",
    )

    # GUI mode (explicit)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI mode (default if no equation given)",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.4.0",
    )

    # Verbose
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    return parser


def solve_equation_cli(
    equation: str,
    target_var: str | None,
    output_format: str,
    show_steps: bool,
    numerical_values: dict,
    verbose: bool,
) -> int:
    """Solve an equation and print the result."""
    import sympy as sp
    from src.input.parser import LatexParser, ParseError
    from src.classification.classifier import EquationClassifier
    from src.solvers import get_default_registry
    from src.models import SolveRequest

    # Parse equation
    parser = LatexParser()
    try:
        eq = parser.parse_plain_text(equation)
    except ParseError:
        try:
            eq = parser.parse(equation)
        except ParseError as e:
            print(f"Error: Failed to parse equation: {e}", file=sys.stderr)
            if e.suggestions:
                print(f"Suggestion: {e.suggestions[0]}", file=sys.stderr)
            return 1

    # Classify
    classifier = EquationClassifier()
    eq_type, subtype = classifier.classify(eq)
    eq.classification = (eq_type, subtype)

    if verbose:
        print(f"Classified as: {eq_type.name}", file=sys.stderr)
        if subtype:
            print(f"  Subtype: {subtype}", file=sys.stderr)

    # Get target variable
    variables = classifier.get_variables(eq)
    if target_var:
        target = sp.Symbol(target_var)
        if target not in variables:
            print(f"Warning: Variable '{target_var}' not in equation", file=sys.stderr)
    elif variables:
        target = variables[0]
    else:
        print("Error: No variables found in equation", file=sys.stderr)
        return 1

    # Create solve request
    numerical_subs = {}
    if numerical_values:
        for var_name, value in numerical_values.items():
            numerical_subs[sp.Symbol(var_name)] = float(value)

    request = SolveRequest(
        equation=eq,
        target_variable=target,
        numerical_values=numerical_subs,
        show_steps=show_steps,
    )

    # Solve using appropriate solver from registry
    registry = get_default_registry()
    solver = registry.get_solver(eq)
    if not solver:
        print("Error: No solver available for this equation type", file=sys.stderr)
        return 1

    if verbose:
        print(f"Using solver: {solver.name}", file=sys.stderr)

    result = solver.solve(request)

    if not result.success:
        print(f"Error: {result.error_message}", file=sys.stderr)
        return 1

    # Output result
    solution = result.solution

    if output_format == "json":
        import json

        output = {
            "equation": equation,
            "target": str(target),
            "solution": str(solution.symbolic_result),
            "latex": solution.latex,
            "method": solution.method_used,
            "time_ms": solution.solve_time_ms,
        }
        if solution.numerical_result is not None:
            output["numerical"] = solution.numerical_result
        if show_steps:
            output["steps"] = [
                {"step": s.step_number, "operation": s.operation, "latex": s.latex_repr}
                for s in solution.steps
            ]
        print(json.dumps(output, indent=2))

    elif output_format == "latex":
        from src.output.exporter import SolutionExporter, ExportOptions

        options = ExportOptions(include_steps=show_steps)
        exporter = SolutionExporter(solution, options)
        print(exporter.to_latex())

    elif output_format == "python":
        from src.output.exporter import SolutionExporter, ExportOptions

        options = ExportOptions(include_steps=show_steps)
        exporter = SolutionExporter(solution, options)
        print(exporter.to_python())

    else:  # text
        print(f"Equation: {equation}")
        print(f"Solving for: {target}")
        print()

        if show_steps and solution.steps:
            print("Steps:")
            for step in solution.steps:
                print(f"  {step.step_number}. {step.operation}")
                print(f"     {step.latex_repr}")
            print()

        print(f"Solution: {solution.latex}")

        if solution.numerical_result is not None:
            print(f"Numerical: {solution.numerical_result}")

        if verbose:
            print(
                f"\nSolved in {solution.solve_time_ms}ms using {solution.method_used}"
            )

    return 0


def list_formulas(search_term: str | None = None) -> int:
    """List available physics formulas."""
    from src.classification.physics_patterns import PhysicsPatternLibrary

    library = PhysicsPatternLibrary()

    if search_term:
        formulas = library.search(search_term)
        if not formulas:
            print(f"No formulas found matching '{search_term}'")
            return 0
    else:
        formulas = library.formulas

    # Group by category
    by_category: dict[str, list] = {}
    for formula in formulas:
        cat = formula.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(formula)

    for category, cat_formulas in sorted(by_category.items()):
        print(f"\n{category.upper()}")
        print("-" * len(category))
        for f in cat_formulas:
            print(f"  {f.id}: {f.name}")
            print(f"    {f.latex}")

    print(f"\nTotal: {len(formulas)} formulas")
    return 0


def get_clipboard_text() -> str | None:
    """Get text from system clipboard."""
    try:
        # Try PyQt6 first (most reliable)
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        clipboard = app.clipboard()
        return clipboard.text()
    except ImportError:
        pass

    try:
        # Try xclip
        import subprocess

        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        pass

    try:
        # Try xsel
        import subprocess

        result = subprocess.run(
            ["xsel", "--clipboard", "--output"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        pass

    return None


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # List formulas mode
    if args.list_formulas:
        return list_formulas()

    # Search formulas mode
    if args.search:
        return list_formulas(args.search)

    # Get equation from clipboard if requested
    equation = args.equation
    if args.from_clipboard:
        equation = get_clipboard_text()
        if not equation:
            print("Error: Could not read from clipboard", file=sys.stderr)
            return 1
        equation = equation.strip()
        if not equation:
            print("Error: Clipboard is empty", file=sys.stderr)
            return 1

    # If no equation and not explicit GUI, launch GUI
    if not equation and not args.gui:
        args.gui = True

    # GUI mode
    if args.gui or not equation:
        from src.gui.main_window import run_app

        run_app()
        return 0

    # CLI solve mode
    numerical_values = {}
    if args.numerical:
        for item in args.numerical:
            if "=" not in item:
                print(f"Error: Invalid numerical value format: {item}", file=sys.stderr)
                print("Expected format: VAR=VALUE (e.g., m=5)", file=sys.stderr)
                return 1
            var, val = item.split("=", 1)
            try:
                numerical_values[var.strip()] = float(val.strip())
            except ValueError:
                print(f"Error: Invalid number: {val}", file=sys.stderr)
                return 1

    return solve_equation_cli(
        equation=equation,
        target_var=args.variable,
        output_format=args.format,
        show_steps=args.steps,
        numerical_values=numerical_values,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
