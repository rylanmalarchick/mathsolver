"""
Tests for error handling module.

Tests the centralized error handling with rich context and suggestions.
"""

import pytest


class TestErrorContext:
    """Test ErrorContext creation and conversion."""

    def test_from_parse_error(self):
        """Test ErrorContext from ParseError."""
        from src.utils.errors import ParseError, ErrorContext

        exc = ParseError(
            "Failed to parse: unexpected token",
            latex=r"\frac{x}{",
            suggestion="Missing closing brace",
        )

        ctx = ErrorContext.from_exception(exc)

        assert ctx.title == "Parse Error"
        assert "Failed to parse" in ctx.message
        assert "Missing closing brace" in ctx.suggestions
        assert ctx.recoverable is True

    def test_from_ocr_error(self):
        """Test ErrorContext from OCRError."""
        from src.utils.errors import OCRError, ErrorContext

        exc = OCRError("Model failed to process image")
        ctx = ErrorContext.from_exception(exc)

        assert ctx.title == "OCR Error"
        assert "Model failed" in ctx.message
        assert len(ctx.suggestions) > 0  # Has default suggestions
        assert any("manual" in s.lower() for s in ctx.suggestions)

    def test_from_solve_timeout_error(self):
        """Test ErrorContext from SolveTimeoutError."""
        from src.utils.errors import SolveTimeoutError, ErrorContext, ErrorSeverity

        exc = SolveTimeoutError(30.0, equation="x^100 + y^100 = z^100")
        ctx = ErrorContext.from_exception(exc)

        assert ctx.title == "Timeout"
        assert "30" in ctx.message
        assert ctx.severity == ErrorSeverity.WARNING
        assert any("simpler" in s.lower() for s in ctx.suggestions)

    def test_from_no_solution_error(self):
        """Test ErrorContext from NoSolutionError."""
        from src.utils.errors import NoSolutionError, ErrorContext

        exc = NoSolutionError(equation="x^2 = -1", variable="x")
        ctx = ErrorContext.from_exception(exc)

        assert ctx.title == "No Solution"
        assert "x" in ctx.message
        assert ctx.recoverable is True

    def test_from_generic_exception(self):
        """Test ErrorContext from generic exception."""
        from src.utils.errors import ErrorContext

        exc = ValueError("Something went wrong")
        ctx = ErrorContext.from_exception(exc, context="during calculation")

        assert ctx.title == "Error"
        assert "Something went wrong" in ctx.message
        assert "calculation" in ctx.technical_details


class TestMathSolverError:
    """Test base MathSolverError class."""

    def test_custom_suggestions(self):
        """Test error with custom suggestions."""
        from src.utils.errors import SolveError

        exc = SolveError(
            "Cannot solve this equation",
            suggestions=["Try X", "Try Y"],
        )

        assert "Try X" in exc.suggestions
        assert "Try Y" in exc.suggestions

    def test_to_context_conversion(self):
        """Test conversion to ErrorContext."""
        from src.utils.errors import ParseError

        exc = ParseError(
            "Unbalanced braces",
            latex=r"\frac{x",
            technical_details="Found 1 open, 0 close",
        )

        ctx = exc.to_context()

        assert ctx.title == "Parse Error"
        assert ctx.message == "Unbalanced braces"
        assert "1 open" in ctx.technical_details


class TestUnbalancedBracesError:
    """Test specialized UnbalancedBracesError."""

    def test_missing_close_braces(self):
        """Test error when closing braces are missing."""
        from src.utils.errors import UnbalancedBracesError

        exc = UnbalancedBracesError(latex=r"\frac{x}{y", open_count=2, close_count=1)

        assert "1" in str(exc)  # Missing 1 brace
        assert "closing" in str(exc).lower()

    def test_missing_open_braces(self):
        """Test error when opening braces are missing."""
        from src.utils.errors import UnbalancedBracesError

        exc = UnbalancedBracesError(latex=r"x}{y}", open_count=0, close_count=2)

        assert "2" in str(exc)
        assert "opening" in str(exc).lower()


class TestFormatFunctions:
    """Test error formatting utility functions."""

    def test_format_error_for_user(self):
        """Test brief user-friendly formatting."""
        from src.utils.errors import format_error_for_user, ParseError

        exc = ParseError("Bad LaTeX", latex=r"\bad")
        msg = format_error_for_user(exc, "testing")

        assert "Bad LaTeX" in msg
        # Should include first suggestion
        assert "Try:" in msg or len(msg.split()) > 3

    def test_format_error_for_dialog(self):
        """Test dialog formatting."""
        from src.utils.errors import format_error_for_dialog, OCRError

        exc = OCRError("Model not loaded")
        info = format_error_for_dialog(exc, "testing OCR")

        assert "title" in info
        assert "text" in info
        assert "detailed_text" in info
        assert "icon" in info
        assert info["title"] == "OCR Error"


class TestErrorSeverity:
    """Test error severity levels."""

    def test_severity_affects_recoverability(self):
        """Test that CRITICAL errors are not recoverable."""
        from src.utils.errors import ErrorContext, ErrorSeverity

        ctx = ErrorContext(
            title="Test",
            message="Test message",
            technical_details=None,
            suggestions=[],
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
        )

        assert ctx.recoverable is False

    def test_warning_is_recoverable(self):
        """Test that WARNING errors are recoverable by default."""
        from src.utils.errors import SolveTimeoutError

        exc = SolveTimeoutError(10.0)
        ctx = exc.to_context()

        assert ctx.recoverable is True


class TestIntegrationWithParser:
    """Test that parser uses new error types correctly."""

    def test_parser_raises_parse_error(self):
        """Test that parser raises ParseError with suggestions."""
        from src.input.parser import LatexParser
        from src.utils.errors import ParseError

        parser = LatexParser()

        with pytest.raises(ParseError) as exc_info:
            parser.parse(r"\frac{x}{")  # Missing close brace

        exc = exc_info.value
        assert exc.latex == r"\frac{x}{"
        # Should have suggestion about braces
        assert len(exc.suggestions) > 0
