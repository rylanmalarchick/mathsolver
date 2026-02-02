"""
OCR engine wrapper for pix2tex LaTeX-OCR.

Provides a clean interface for image-to-LaTeX conversion with
confidence estimation and error handling.
"""

import time
from typing import Tuple, Optional
from PIL import Image

from ..models import OCRResult


class OCRError(Exception):
    """Raised when OCR processing fails."""

    pass


class OCREngine:
    """
    Wrapper around pix2tex LaTeX-OCR model.

    Lazy-loads the model on first use to avoid slow startup.
    The model is ~500MB and takes 2-5 seconds to load.

    Usage:
        ocr = OCREngine()
        result = ocr.image_to_latex(pil_image)
        print(result.latex, result.confidence)
    """

    def __init__(self, lazy_load: bool = True):
        """
        Initialize OCR engine.

        Args:
            lazy_load: If True, defer model loading until first use.
                      Set False to load immediately (blocks for 2-5s).
        """
        self._model = None
        self._model_loaded = False

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load the pix2tex model. Called lazily on first use."""
        if self._model_loaded:
            return

        try:
            from pix2tex.cli import LatexOCR

            self._model = LatexOCR()
            self._model_loaded = True
        except ImportError as e:
            raise OCRError(
                "pix2tex not installed. Run: pip install 'pix2tex[gui]'\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise OCRError(f"Failed to load OCR model: {e}")

    def image_to_latex(self, image: Image.Image) -> OCRResult:
        """
        Convert a PIL Image to LaTeX string.

        Args:
            image: PIL Image containing a mathematical equation.

        Returns:
            OCRResult with latex string, confidence score, and timing.

        Raises:
            OCRError: If OCR processing fails.
        """
        # Lazy load model
        if not self._model_loaded:
            self._load_model()

        start_time = time.perf_counter()

        try:
            # pix2tex expects PIL Image directly
            latex = self._model(image)
        except Exception as e:
            raise OCRError(f"OCR processing failed: {e}")

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Estimate confidence based on output characteristics
        confidence = self._estimate_confidence(latex)

        return OCRResult(
            latex=latex, confidence=confidence, processing_time_ms=elapsed_ms
        )

    def _estimate_confidence(self, latex: str) -> float:
        """
        Estimate confidence score for OCR output.

        pix2tex doesn't provide confidence directly, so we use heuristics:
        - Very short output → likely failed
        - Unusual characters → OCR confusion
        - Unbalanced braces → parse errors likely

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not latex or len(latex) < 3:
            return 0.3

        confidence = 1.0

        # Penalize very short outputs
        if len(latex) < 5:
            confidence -= 0.2

        # Penalize unusual/garbage characters
        garbage_chars = latex.count("?") + latex.count("�") + latex.count("□")
        confidence -= garbage_chars * 0.15

        # Penalize unbalanced braces (common OCR issue)
        open_braces = latex.count("{")
        close_braces = latex.count("}")
        if open_braces != close_braces:
            confidence -= 0.2

        # Penalize unbalanced parens
        open_parens = latex.count("(")
        close_parens = latex.count(")")
        if open_parens != close_parens:
            confidence -= 0.1

        # Bonus for common LaTeX commands (likely valid)
        common_commands = ["\\frac", "\\sqrt", "\\int", "\\sum", "\\partial", "="]
        for cmd in common_commands:
            if cmd in latex:
                confidence += 0.05

        # Clamp to [0.3, 1.0] - never say 0% or truly 100%
        return max(0.3, min(1.0, confidence))

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model_loaded

    def preload(self) -> None:
        """
        Explicitly load the model.

        Call this during app startup to avoid delay on first OCR.
        """
        self._load_model()


class MockOCREngine:
    """
    Mock OCR engine for testing without pix2tex installed.

    Returns predefined responses based on image size.
    """

    def __init__(self):
        self._model_loaded = True

    def image_to_latex(self, image: Image.Image) -> OCRResult:
        """Return a mock LaTeX string."""
        # Use image dimensions to vary output (for testing)
        w, h = image.size

        if w > h:
            latex = r"E = mc^2"
        else:
            latex = r"\frac{d}{dx} x^2 = 2x"

        return OCRResult(latex=latex, confidence=0.95, processing_time_ms=50)

    @property
    def is_loaded(self) -> bool:
        return True

    def preload(self) -> None:
        pass
