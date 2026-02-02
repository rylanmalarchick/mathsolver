"""
LaTeX to SymPy parser.

Converts LaTeX strings to SymPy expression trees for symbolic manipulation.
"""

import re
from typing import Optional, Tuple, List
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

from ..models import Equation, EquationType
from ..utils.errors import ParseError, UnbalancedBracesError


class LatexParser:
    """
    Parse LaTeX strings into SymPy expressions.

    Handles common OCR errors and provides helpful error messages.

    Usage:
        parser = LatexParser()
        equation = parser.parse(r"E = mc^2")
        print(equation.sympy_expr)  # Eq(E, m*c**2)
    """

    # Common OCR mistakes and their corrections
    OCR_FIXES = [
        # Greek letters sometimes mangled
        (r"\\Iambda", r"\\lambda"),
        (r"\\Iamda", r"\\lambda"),
        (r"\\apha", r"\\alpha"),
        (r"\\betta", r"\\beta"),
        # Operator confusion
        (r"\\tim es", r"\\times"),
        (r"\\cd ot", r"\\cdot"),
        # Space issues
        (r"\\ ", r" "),
        # Common symbol issues
        (r"\\hslash", r"\\hbar"),
    ]

    def __init__(self):
        """Initialize the parser."""
        pass

    def parse(self, latex: str, ocr_confidence: float = 1.0) -> Equation:
        """
        Parse LaTeX string into an Equation object.

        Args:
            latex: LaTeX string (e.g., r"E = mc^2")
            ocr_confidence: Confidence score from OCR (1.0 if manual input)

        Returns:
            Equation object with parsed SymPy expression.

        Raises:
            ParseError: If parsing fails with details about the error.
        """
        # Clean and normalize the LaTeX
        cleaned = self._preprocess(latex)

        # Try parsing
        try:
            expr = parse_latex(cleaned)
        except LaTeXParsingError as e:
            # Try with OCR fixes applied
            fixed = self._apply_ocr_fixes(cleaned)
            if fixed != cleaned:
                try:
                    expr = parse_latex(fixed)
                    cleaned = fixed  # Use the fixed version
                except LaTeXParsingError:
                    raise ParseError(
                        f"Failed to parse LaTeX: {e}",
                        latex=latex,
                        suggestion=self._suggest_fix(latex, str(e)),
                    )
            else:
                raise ParseError(
                    f"Failed to parse LaTeX: {e}",
                    latex=latex,
                    suggestion=self._suggest_fix(latex, str(e)),
                )
        except Exception as e:
            raise ParseError(f"Unexpected parsing error: {e}", latex=latex)

        # Build Equation object
        return Equation(raw_latex=latex, sympy_expr=expr, ocr_confidence=ocr_confidence)

    def _preprocess(self, latex: str) -> str:
        """
        Clean and normalize LaTeX for parsing.
        """
        result = latex.strip()

        # Remove display math delimiters if present
        for delim in [r"\[", r"\]", r"$$", r"$", r"\(", r"\)"]:
            result = result.replace(delim, "")

        # Normalize whitespace
        result = " ".join(result.split())

        return result

    def _apply_ocr_fixes(self, latex: str) -> str:
        """Apply common OCR error corrections."""
        result = latex
        for pattern, replacement in self.OCR_FIXES:
            result = re.sub(pattern, replacement, result)
        return result

    def _suggest_fix(self, latex: str, error_msg: str) -> Optional[str]:
        """
        Suggest a fix based on the error message.
        """
        # Check for unbalanced braces
        open_count = latex.count("{")
        close_count = latex.count("}")
        if open_count != close_count:
            diff = open_count - close_count
            if diff > 0:
                return f"Missing {diff} closing brace(s) '}}'"
            else:
                return f"Missing {-diff} opening brace(s) '{{'"

        # Check for common issues
        if "Expected" in error_msg:
            return "Check for missing operators or malformed commands"

        return None

    def try_parse(self, latex: str) -> Tuple[Optional[Equation], Optional[str]]:
        """
        Attempt to parse, returning None on failure instead of raising.

        Returns:
            Tuple of (Equation or None, error message or None)
        """
        try:
            eq = self.parse(latex)
            return eq, None
        except ParseError as e:
            return None, str(e)

    def parse_plain_text(self, text: str) -> Equation:
        """
        Parse plain text math notation (not LaTeX).

        Converts common notation to SymPy:
        - "E = mc^2" → Eq(E, m*c**2)
        - "x^2 + 2x + 1" → x**2 + 2*x + 1

        Args:
            text: Plain text math expression

        Returns:
            Equation object
        """
        # Use SymPy's sympify with transformations
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )

        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

        # Create local dict to prevent reserved names from being interpreted as constants
        # This allows 'E' to be a symbol instead of Euler's number, 'I' as a symbol not sqrt(-1), etc.
        # We identify single uppercase letters and common physics variable names
        local_dict = {}

        # Find all potential variable names (letters and letter combinations)
        import re

        potential_vars = set(re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", text))

        # Reserved SymPy names that should become symbols in physics context
        reserved_names = {
            "E",
            "I",
            "N",
            "S",
            "O",
            "Q",
            "C",
        }  # Common physics vars that conflict

        for var in potential_vars:
            if var in reserved_names or (len(var) == 1 and var.isupper()):
                local_dict[var] = sp.Symbol(var)

        # Handle equation format "lhs = rhs"
        if "=" in text and text.count("=") == 1:
            lhs, rhs = text.split("=")
            try:
                lhs_expr = parse_expr(
                    lhs.strip(), local_dict=local_dict, transformations=transformations
                )
                rhs_expr = parse_expr(
                    rhs.strip(), local_dict=local_dict, transformations=transformations
                )
                expr = sp.Eq(lhs_expr, rhs_expr)
            except Exception as e:
                raise ParseError(f"Failed to parse plain text: {e}", latex=text)
        else:
            try:
                expr = parse_expr(
                    text.strip(), local_dict=local_dict, transformations=transformations
                )
            except Exception as e:
                raise ParseError(f"Failed to parse plain text: {e}", latex=text)

        return Equation(
            raw_latex=sp.latex(expr),  # Generate LaTeX from parsed expr
            sympy_expr=expr,
            ocr_confidence=1.0,
        )


def latex_to_sympy(latex: str) -> sp.Basic:
    """
    Convenience function: parse LaTeX directly to SymPy expression.

    Raises ParseError on failure.
    """
    parser = LatexParser()
    return parser.parse(latex).sympy_expr
