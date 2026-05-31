"""Input layer: screenshot capture, OCR, and LaTeX parsing."""

from .ocr import OCREngine
from .parser import LatexParser
from .screenshot import ScreenshotCapture

__all__ = ["ScreenshotCapture", "OCREngine", "LatexParser"]
