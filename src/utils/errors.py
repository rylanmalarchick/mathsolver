"""
Centralized error handling for MathSolver.

Provides a hierarchy of custom exceptions with user-friendly messages,
suggestions for fixes, and error recovery hints.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    INFO = auto()  # Informational, operation may have partially succeeded
    WARNING = auto()  # Non-fatal, can continue with degraded functionality
    ERROR = auto()  # Operation failed, but can retry
    CRITICAL = auto()  # Unrecoverable error


@dataclass
class ErrorContext:
    """
    Rich context for error reporting.

    Provides user-friendly information beyond the raw exception.
    """

    title: str  # Short title for dialog/toast
    message: str  # User-friendly message
    technical_details: Optional[str]  # Debug info (shown on expand)
    suggestions: List[str]  # Actionable suggestions
    severity: ErrorSeverity
    recoverable: bool = True  # Can user retry?

    @classmethod
    def from_exception(cls, exc: Exception, context: str = "") -> "ErrorContext":
        """Create ErrorContext from any exception using smart detection."""
        exc_type = type(exc).__name__
        exc_msg = str(exc)

        # Try to match known exception patterns
        if isinstance(exc, MathSolverError):
            return exc.to_context()

        # Parser errors
        if "parse" in exc_type.lower() or "latex" in exc_msg.lower():
            return cls(
                title="Parse Error",
                message="Could not parse the equation.",
                technical_details=f"{exc_type}: {exc_msg}",
                suggestions=[
                    "Check for unbalanced braces { }",
                    "Verify LaTeX command spelling (e.g., \\frac not \\frc)",
                    "Try using plain text input instead",
                ],
                severity=ErrorSeverity.ERROR,
            )

        # Timeout errors
        if "timeout" in exc_msg.lower() or "timed out" in exc_msg.lower():
            return cls(
                title="Timeout",
                message="The operation took too long and was cancelled.",
                technical_details=f"{exc_type}: {exc_msg}",
                suggestions=[
                    "Try a simpler equation",
                    "Reduce complexity (fewer variables)",
                    "Try solving for a different variable",
                ],
                severity=ErrorSeverity.WARNING,
            )

        # Import/dependency errors
        if isinstance(exc, ImportError):
            return cls(
                title="Missing Dependency",
                message="A required component is not installed.",
                technical_details=f"{exc_type}: {exc_msg}",
                suggestions=[
                    "Check that all dependencies are installed",
                    "Run: pip install -e .[dev]",
                    "Restart the application",
                ],
                severity=ErrorSeverity.CRITICAL,
                recoverable=False,
            )

        # Generic fallback
        return cls(
            title="Error",
            message=f"An unexpected error occurred: {exc_msg}",
            technical_details=f"{exc_type}: {exc_msg}\nContext: {context}",
            suggestions=["Try again", "Restart the application"],
            severity=ErrorSeverity.ERROR,
        )


class MathSolverError(Exception):
    """
    Base exception for all MathSolver errors.

    Subclasses provide rich error context for user-friendly reporting.
    """

    default_title = "Error"
    default_suggestions: List[str] = []
    default_severity = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str,
        *,
        suggestions: Optional[List[str]] = None,
        technical_details: Optional[str] = None,
        severity: Optional[ErrorSeverity] = None,
    ):
        super().__init__(message)
        self.user_message = message
        self.suggestions = suggestions or self.default_suggestions.copy()
        self.technical_details = technical_details
        self.severity = severity or self.default_severity

    def to_context(self) -> ErrorContext:
        """Convert to ErrorContext for display."""
        return ErrorContext(
            title=self.default_title,
            message=self.user_message,
            technical_details=self.technical_details,
            suggestions=self.suggestions,
            severity=self.severity,
            recoverable=self.severity != ErrorSeverity.CRITICAL,
        )


# === Input Errors ===


class OCRError(MathSolverError):
    """Raised when OCR processing fails."""

    default_title = "OCR Error"
    default_suggestions = [
        "Ensure the equation is clearly visible",
        "Try capturing a larger area around the equation",
        "Use higher contrast (dark text on light background)",
        "Enter the equation manually instead",
    ]


class ScreenshotError(MathSolverError):
    """Raised when screenshot capture fails."""

    default_title = "Screenshot Error"
    default_suggestions = [
        "Ensure the application has screen capture permissions",
        "Try capturing the equation again",
        "Copy the equation and use Paste instead",
    ]


class ScreenshotCancelledError(ScreenshotError):
    """Raised when user cancels screenshot selection."""

    default_title = "Cancelled"
    default_severity = ErrorSeverity.INFO
    default_suggestions = []

    def __init__(self):
        super().__init__("Screenshot capture was cancelled.")


# === Parser Errors ===


class ParseError(MathSolverError):
    """Raised when LaTeX/text parsing fails."""

    default_title = "Parse Error"
    default_suggestions = [
        "Check for missing or extra braces { }",
        "Verify LaTeX commands are spelled correctly",
        "Try using plain text input (e.g., 'x^2 + 2x + 1')",
    ]

    def __init__(
        self,
        message: str,
        *,
        latex: str = "",
        suggestion: Optional[str] = None,
        **kwargs,
    ):
        # Add specific suggestion to front of list if provided
        suggestions = kwargs.pop("suggestions", None) or self.default_suggestions.copy()
        if suggestion:
            suggestions.insert(0, suggestion)

        super().__init__(message, suggestions=suggestions, **kwargs)
        self.latex = latex


class UnbalancedBracesError(ParseError):
    """Raised when braces are unbalanced."""

    default_title = "Unbalanced Braces"

    def __init__(self, latex: str, open_count: int, close_count: int):
        diff = open_count - close_count
        if diff > 0:
            msg = f"Missing {diff} closing brace(s) '}}'"
        else:
            msg = f"Missing {-diff} opening brace(s) '{{'"

        super().__init__(
            msg,
            latex=latex,
            suggestions=[
                f"Current count: {open_count} opening, {close_count} closing",
                "Add the missing braces to balance the expression",
            ],
        )


# === Solver Errors ===


class SolveError(MathSolverError):
    """Raised when equation solving fails."""

    default_title = "Solve Error"
    default_suggestions = [
        "Try solving for a different variable",
        "Check that the equation is valid",
        "Simplify the equation if possible",
    ]


class NoSolutionError(SolveError):
    """Raised when no solution exists."""

    default_title = "No Solution"
    default_severity = ErrorSeverity.WARNING

    def __init__(self, equation: str, variable: str):
        super().__init__(
            f"No solution found for '{variable}'",
            suggestions=[
                f"The equation may have no solution for {variable}",
                "Try solving for a different variable",
                "Check if the equation is correctly entered",
            ],
            technical_details=f"Equation: {equation}",
        )


class SolveTimeoutError(SolveError):
    """Raised when solving times out."""

    default_title = "Timeout"
    default_severity = ErrorSeverity.WARNING
    default_suggestions = [
        "The equation is too complex to solve quickly",
        "Try a simpler form of the equation",
        "Reduce the number of variables",
        "Try solving for a different variable",
    ]

    def __init__(self, timeout_seconds: float, equation: str = ""):
        super().__init__(
            f"Solving timed out after {timeout_seconds:.1f} seconds",
            technical_details=f"Equation: {equation}" if equation else None,
        )


class UnsupportedEquationError(SolveError):
    """Raised when equation type is not supported."""

    default_title = "Unsupported Equation"
    default_severity = ErrorSeverity.WARNING

    def __init__(self, equation_type: str, suggestions: Optional[List[str]] = None):
        default_sugg = [
            f"Equations of type '{equation_type}' are not yet supported",
            "Try reformulating as a simpler equation",
        ]
        super().__init__(
            f"Cannot solve equations of type '{equation_type}'",
            suggestions=suggestions or default_sugg,
        )


# === Classification Errors ===


class ClassificationError(MathSolverError):
    """Raised when equation classification fails."""

    default_title = "Classification Error"
    default_suggestions = [
        "The equation format was not recognized",
        "Try entering a simpler form",
        "Check for typos in variable names",
    ]


# === Export Errors ===


class ExportError(MathSolverError):
    """Raised when export fails."""

    default_title = "Export Error"
    default_suggestions = [
        "Check that you have write permission to the location",
        "Try exporting to a different location",
        "Ensure enough disk space is available",
    ]


# === Utility functions ===


def format_error_for_user(exc: Exception, context: str = "") -> str:
    """
    Format an exception into a user-friendly string.

    Returns a single string suitable for status bar or simple display.
    """
    ctx = ErrorContext.from_exception(exc, context)

    result = ctx.message
    if ctx.suggestions:
        result += f" Try: {ctx.suggestions[0]}"

    return result


def format_error_for_dialog(exc: Exception, context: str = "") -> dict:
    """
    Format an exception into a dict suitable for QMessageBox.

    Returns dict with 'title', 'text', 'detailed_text', 'icon' keys.
    """
    from PyQt6.QtWidgets import QMessageBox

    ctx = ErrorContext.from_exception(exc, context)

    # Build detailed text with suggestions
    detailed_parts = []
    if ctx.suggestions:
        detailed_parts.append("Suggestions:")
        for i, sugg in enumerate(ctx.suggestions, 1):
            detailed_parts.append(f"  {i}. {sugg}")
    if ctx.technical_details:
        detailed_parts.append("")
        detailed_parts.append("Technical details:")
        detailed_parts.append(ctx.technical_details)

    # Map severity to icon
    icon_map = {
        ErrorSeverity.INFO: QMessageBox.Icon.Information,
        ErrorSeverity.WARNING: QMessageBox.Icon.Warning,
        ErrorSeverity.ERROR: QMessageBox.Icon.Critical,
        ErrorSeverity.CRITICAL: QMessageBox.Icon.Critical,
    }

    return {
        "title": ctx.title,
        "text": ctx.message,
        "detailed_text": "\n".join(detailed_parts) if detailed_parts else None,
        "icon": icon_map.get(ctx.severity, QMessageBox.Icon.Warning),
    }
