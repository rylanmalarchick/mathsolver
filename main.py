#!/usr/bin/env python3
"""
MathSolver - Offline equation solver with OCR and symbolic computation.

Entry point for the application.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))


def main():
    """Main entry point."""
    from src.gui.main_window import run_app

    run_app()


if __name__ == "__main__":
    main()
