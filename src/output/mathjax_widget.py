"""
MathJax rendering widget for PyQt6.

Provides beautiful mathematical rendering using MathJax with QtWebEngine.
Falls back to matplotlib rendering if WebEngine is not available.
"""

from typing import List, Optional
from pathlib import Path
import html

# Try to import PyQt6 WebEngine
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebChannel import QWebChannel

    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False
    QWebEngineView = None

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QPushButton,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QFont

from ..models import SolutionStep


# MathJax HTML template (loads from CDN, or can be bundled locally)
MATHJAX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
            margin: 10px;
            padding: 0;
            background: {bg_color};
            color: {text_color};
        }}
        .solution-container {{
            padding: 10px;
        }}
        .step {{
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            background: {step_bg};
            border-left: 3px solid {accent_color};
        }}
        .step-header {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }}
        .step-number {{
            font-weight: bold;
            color: {accent_color};
            margin-right: 10px;
            min-width: 60px;
        }}
        .step-operation {{
            color: {text_color};
            font-style: italic;
        }}
        .step-math {{
            margin-top: 8px;
            padding: 5px;
            overflow-x: auto;
        }}
        .final-result {{
            background: {result_bg};
            border-left-color: {result_accent};
            padding: 15px;
        }}
        .final-result .step-number {{
            color: {result_accent};
            font-size: 1.1em;
        }}
        .collapsible {{
            cursor: pointer;
        }}
        .collapsible:hover {{
            background: {hover_bg};
        }}
        .collapsed .step-math {{
            display: none;
        }}
        .equation-box {{
            border: 1px solid {border_color};
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
            background: {eq_bg};
        }}
        mjx-container {{
            margin: 0 !important;
        }}
    </style>
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true
            }},
            svg: {{
                fontCache: 'global'
            }},
            options: {{
                renderActions: {{
                    addMenu: []
                }}
            }}
        }};
    </script>
    <script id="MathJax-script" async 
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
    </script>
</head>
<body>
    <div class="solution-container">
        {content}
    </div>
    <script>
        function toggleStep(stepId) {{
            var step = document.getElementById(stepId);
            if (step) {{
                step.classList.toggle('collapsed');
            }}
        }}
        
        function collapseAll() {{
            var steps = document.querySelectorAll('.step:not(.final-result)');
            steps.forEach(function(step) {{
                step.classList.add('collapsed');
            }});
        }}
        
        function expandAll() {{
            var steps = document.querySelectorAll('.step');
            steps.forEach(function(step) {{
                step.classList.remove('collapsed');
            }});
        }}
    </script>
</body>
</html>
"""

# Color themes
LIGHT_THEME = {
    "bg_color": "#ffffff",
    "text_color": "#333333",
    "step_bg": "#f8f9fa",
    "accent_color": "#007bff",
    "result_bg": "#e8f5e9",
    "result_accent": "#28a745",
    "hover_bg": "#e9ecef",
    "border_color": "#dee2e6",
    "eq_bg": "#ffffff",
}

DARK_THEME = {
    "bg_color": "#1e1e1e",
    "text_color": "#d4d4d4",
    "step_bg": "#2d2d2d",
    "accent_color": "#569cd6",
    "result_bg": "#1e3a1e",
    "result_accent": "#4ec9b0",
    "hover_bg": "#3c3c3c",
    "border_color": "#3c3c3c",
    "eq_bg": "#252526",
}


class MathJaxRenderer:
    """
    Renders LaTeX to HTML with MathJax.
    """

    def __init__(self, dark_mode: bool = False):
        """Initialize renderer with theme."""
        self.theme = DARK_THEME if dark_mode else LIGHT_THEME

    def render_steps(
        self,
        steps: List[SolutionStep],
        collapsible: bool = True,
        show_final_only: bool = False,
    ) -> str:
        """
        Render solution steps to HTML.

        Args:
            steps: List of SolutionStep objects
            collapsible: Make intermediate steps collapsible
            show_final_only: Only show the final result

        Returns:
            Complete HTML document with MathJax
        """
        if show_final_only and steps:
            steps = [steps[-1]]

        content_parts = []

        for i, step in enumerate(steps):
            is_final = i == len(steps) - 1
            step_html = self._render_step(step, i, is_final, collapsible)
            content_parts.append(step_html)

        content = "\n".join(content_parts)

        return MATHJAX_TEMPLATE.format(content=content, **self.theme)

    def _render_step(
        self, step: SolutionStep, index: int, is_final: bool, collapsible: bool
    ) -> str:
        """Render a single step to HTML."""
        step_id = f"step-{index}"
        classes = ["step"]

        if is_final:
            classes.append("final-result")
        if collapsible and not is_final:
            classes.append("collapsible")

        onclick = (
            f"onclick=\"toggleStep('{step_id}')\""
            if collapsible and not is_final
            else ""
        )

        # Escape operation text for HTML
        operation = html.escape(step.operation)

        # The latex_repr should be wrapped for display math
        latex_content = step.latex_repr
        if not latex_content.startswith("\\["):
            latex_content = f"\\[{latex_content}\\]"

        return f'''
        <div id="{step_id}" class="{" ".join(classes)}" {onclick}>
            <div class="step-header">
                <span class="step-number">Step {step.step_number}</span>
                <span class="step-operation">{operation}</span>
            </div>
            <div class="step-math">
                {latex_content}
            </div>
        </div>
        '''

    def render_equation(self, latex: str, display: bool = True) -> str:
        """
        Render a single equation.

        Args:
            latex: LaTeX string
            display: Use display mode (centered, larger)

        Returns:
            HTML document with equation
        """
        if display:
            math_content = f"\\[{latex}\\]"
        else:
            math_content = f"\\({latex}\\)"

        content = f'<div class="equation-box">{math_content}</div>'

        return MATHJAX_TEMPLATE.format(content=content, **self.theme)

    def set_dark_mode(self, enabled: bool):
        """Toggle dark mode theme."""
        self.theme = DARK_THEME if enabled else LIGHT_THEME


class MathJaxWidget(QWidget):
    """
    Widget for displaying MathJax-rendered content.

    Uses QWebEngineView if available, falls back to plain text.
    """

    # Signal emitted when content is loaded
    contentLoaded = pyqtSignal()

    def __init__(self, parent=None, dark_mode: bool = False):
        super().__init__(parent)

        self.renderer = MathJaxRenderer(dark_mode=dark_mode)
        self._use_webengine = WEBENGINE_AVAILABLE

        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self._use_webengine:
            self.web_view = QWebEngineView()
            self.web_view.loadFinished.connect(self._on_load_finished)
            layout.addWidget(self.web_view)
        else:
            # Fallback to scrollable label
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)

            self.fallback_label = QLabel()
            self.fallback_label.setWordWrap(True)
            self.fallback_label.setTextFormat(Qt.TextFormat.RichText)
            self.fallback_label.setFont(QFont("Monospace", 11))
            self.fallback_label.setAlignment(Qt.AlignmentFlag.AlignTop)

            scroll.setWidget(self.fallback_label)
            layout.addWidget(scroll)

    def display_steps(
        self,
        steps: List[SolutionStep],
        collapsible: bool = True,
        show_final_only: bool = False,
    ):
        """
        Display solution steps.

        Args:
            steps: List of SolutionStep objects
            collapsible: Make intermediate steps collapsible
            show_final_only: Only show the final result
        """
        if self._use_webengine:
            html_content = self.renderer.render_steps(
                steps, collapsible, show_final_only
            )
            self.web_view.setHtml(html_content)
        else:
            # Fallback: plain text display
            text_parts = []
            for step in steps:
                text_parts.append(f"<b>Step {step.step_number}:</b> {step.operation}")
                text_parts.append(f"<pre>  {step.latex_repr}</pre>")
            self.fallback_label.setText("<br>".join(text_parts))

    def display_equation(self, latex: str, display: bool = True):
        """Display a single equation."""
        if self._use_webengine:
            html_content = self.renderer.render_equation(latex, display)
            self.web_view.setHtml(html_content)
        else:
            self.fallback_label.setText(f"<pre>{latex}</pre>")

    def display_html(self, html_content: str):
        """Display raw HTML content."""
        if self._use_webengine:
            self.web_view.setHtml(html_content)
        else:
            self.fallback_label.setText(html_content)

    def clear(self):
        """Clear the display."""
        if self._use_webengine:
            self.web_view.setHtml("")
        else:
            self.fallback_label.clear()

    def set_dark_mode(self, enabled: bool):
        """Toggle dark mode."""
        self.renderer.set_dark_mode(enabled)

    def collapse_all(self):
        """Collapse all intermediate steps."""
        if self._use_webengine:
            self.web_view.page().runJavaScript("collapseAll();")

    def expand_all(self):
        """Expand all steps."""
        if self._use_webengine:
            self.web_view.page().runJavaScript("expandAll();")

    def _on_load_finished(self, ok: bool):
        """Handle page load completion."""
        if ok:
            self.contentLoaded.emit()

    @property
    def using_webengine(self) -> bool:
        """Check if using WebEngine rendering."""
        return self._use_webengine


class CollapsibleStepsWidget(QWidget):
    """
    Widget with collapsible steps and show all/final answer toggle.
    """

    def __init__(self, parent=None, dark_mode: bool = False):
        super().__init__(parent)

        self._steps: List[SolutionStep] = []
        self._show_final_only = False

        self._init_ui(dark_mode)

    def _init_ui(self, dark_mode: bool):
        """Initialize UI with controls and display."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Control bar
        controls = QHBoxLayout()

        self.expand_btn = QPushButton("Expand All")
        self.expand_btn.clicked.connect(self._on_expand_all)
        controls.addWidget(self.expand_btn)

        self.collapse_btn = QPushButton("Collapse All")
        self.collapse_btn.clicked.connect(self._on_collapse_all)
        controls.addWidget(self.collapse_btn)

        controls.addStretch()

        self.toggle_btn = QPushButton("Show Final Only")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.toggled.connect(self._on_toggle_final)
        controls.addWidget(self.toggle_btn)

        layout.addLayout(controls)

        # MathJax display
        self.mathjax_widget = MathJaxWidget(dark_mode=dark_mode)
        layout.addWidget(self.mathjax_widget)

    def set_steps(self, steps: List[SolutionStep]):
        """Set the solution steps to display."""
        self._steps = steps
        self._refresh_display()

    def clear(self):
        """Clear the display."""
        self._steps = []
        self.mathjax_widget.clear()

    def set_dark_mode(self, enabled: bool):
        """Toggle dark mode."""
        self.mathjax_widget.set_dark_mode(enabled)
        self._refresh_display()

    def _refresh_display(self):
        """Refresh the steps display."""
        self.mathjax_widget.display_steps(
            self._steps, collapsible=True, show_final_only=self._show_final_only
        )

    def _on_expand_all(self):
        """Handle expand all button."""
        self.mathjax_widget.expand_all()

    def _on_collapse_all(self):
        """Handle collapse all button."""
        self.mathjax_widget.collapse_all()

    def _on_toggle_final(self, checked: bool):
        """Handle show final only toggle."""
        self._show_final_only = checked
        self.toggle_btn.setText("Show All Steps" if checked else "Show Final Only")
        self._refresh_display()
