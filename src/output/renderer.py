"""
Math rendering for the GUI.

Provides LaTeX rendering via matplotlib or MathJax (QtWebEngine).
"""

from typing import Optional
import sympy as sp


class MathRenderer:
    """
    Render LaTeX equations for display.

    Supports two backends:
    - matplotlib: Native rendering, lighter weight
    - mathjax: Web-based, more features, requires QtWebEngine

    Usage:
        renderer = MathRenderer(backend='matplotlib')
        pixmap = renderer.render_latex(r"E = mc^2")
    """

    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize renderer with specified backend.

        Args:
            backend: 'matplotlib' or 'mathjax'
        """
        self.backend = backend
        self._figure = None

    def render_latex(self, latex: str, fontsize: int = 14) -> "QPixmap":
        """
        Render LaTeX string to a QPixmap.

        Args:
            latex: LaTeX string to render
            fontsize: Font size in points

        Returns:
            QPixmap containing rendered equation
        """
        if self.backend == "matplotlib":
            return self._render_matplotlib(latex, fontsize)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' not implemented")

    def _render_matplotlib(self, latex: str, fontsize: int) -> "QPixmap":
        """Render using matplotlib's mathtext."""
        import matplotlib

        matplotlib.use("Agg")  # Non-GUI backend
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io

        # Create figure
        fig = plt.figure(figsize=(8, 1))
        fig.patch.set_facecolor("white")

        # Render LaTeX
        fig.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, ha="center", va="center")

        # Convert to bytes
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="white",
        )
        plt.close(fig)
        buf.seek(0)

        # Convert to QPixmap
        try:
            from PyQt6.QtGui import QPixmap

            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            return pixmap
        except ImportError:
            # Return raw bytes if PyQt6 not available
            return buf.getvalue()

    def render_html(self, latex: str) -> str:
        """
        Generate HTML with MathJax for rendering in QWebEngineView.

        Args:
            latex: LaTeX string to render

        Returns:
            Complete HTML document with MathJax
        """
        return f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <style>
        body {{
            font-family: 'Latin Modern Roman', 'Computer Modern', serif;
            font-size: 16px;
            padding: 10px;
            margin: 0;
            background: white;
        }}
    </style>
</head>
<body>
    <div id="equation">
        \\[{latex}\\]
    </div>
</body>
</html>
"""

    def render_steps_html(self, steps: list, title: str = "Solution") -> str:
        """
        Generate HTML for step-by-step solution display.

        Args:
            steps: List of SolutionStep objects
            title: Title for the solution section

        Returns:
            Complete HTML document with all steps
        """
        steps_html = ""
        for step in steps:
            steps_html += f"""
            <div class="step">
                <div class="step-header">Step {step.step_number}: {step.operation}</div>
                <div class="step-equation">\\[{step.latex_repr}\\]</div>
            </div>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            padding: 15px;
            margin: 0;
            background: #fafafa;
            color: #333;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .step {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .step-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .step-equation {{
            text-align: center;
            padding: 10px 0;
        }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    {steps_html}
</body>
</html>
"""


def sympy_to_latex(expr: sp.Basic) -> str:
    """Convert SymPy expression to LaTeX string."""
    return sp.latex(expr)
