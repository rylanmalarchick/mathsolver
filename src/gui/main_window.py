"""
Main application window for MathSolver.

PyQt6-based GUI with input, classification, solving, and output panels.
"""

import sys
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QGroupBox,
    QScrollArea,
    QStatusBar,
    QToolBar,
    QMessageBox,
    QApplication,
    QProgressBar,
    QSplitter,
    QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QFont, QClipboard


class OCRWorker(QThread):
    """Background thread for OCR processing."""

    finished = pyqtSignal(str, float)  # latex, confidence
    error = pyqtSignal(str)

    def __init__(self, screenshot_capture, ocr_engine):
        super().__init__()
        self.screenshot_capture = screenshot_capture
        self.ocr_engine = ocr_engine

    def run(self):
        try:
            # Capture screenshot
            image = self.screenshot_capture.capture_area()

            # Run OCR
            result = self.ocr_engine.image_to_latex(image)

            self.finished.emit(result.latex, result.confidence)
        except Exception as e:
            self.error.emit(str(e))


class SolveWorker(QThread):
    """Background thread for equation solving."""

    finished = pyqtSignal(object)  # SolverResult
    error = pyqtSignal(str)

    def __init__(self, solver, request):
        super().__init__()
        self.solver = solver
        self.request = request

    def run(self):
        try:
            result = self.solver.solve(self.request)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main application window for MathSolver.

    Layout:
    - Toolbar: Screenshot, Paste, History buttons
    - Input Panel: LaTeX input with preview
    - Classification Panel: Detected equation type
    - Solver Panel: Variable selector and solution display
    - Status Bar: Processing status and timing
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MathSolver")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(600, 400)

        # Initialize components (lazy-loaded)
        self._screenshot_capture = None
        self._ocr_engine = None
        self._parser = None
        self._classifier = None
        self._solver = None

        # Current state
        self._current_equation = None
        self._current_solution = None

        # Setup UI
        self._init_ui()
        self._init_toolbar()
        self._init_statusbar()

        # Show ready status
        self.statusBar().showMessage("Ready. Enter LaTeX or capture screenshot.")

    def _init_ui(self):
        """Initialize the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Input Panel
        input_group = self._create_input_panel()
        splitter.addWidget(input_group)

        # Classification Panel
        class_group = self._create_classification_panel()
        splitter.addWidget(class_group)

        # Solution Panel
        solution_group = self._create_solution_panel()
        splitter.addWidget(solution_group)

        # Set initial sizes (input smaller, solution larger)
        splitter.setSizes([150, 80, 400])

        main_layout.addWidget(splitter)

        # Export buttons at bottom
        export_layout = self._create_export_buttons()
        main_layout.addLayout(export_layout)

    def _create_input_panel(self) -> QGroupBox:
        """Create the input panel with LaTeX entry."""
        group = QGroupBox("Input")
        layout = QVBoxLayout(group)

        # LaTeX input field
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("LaTeX:"))

        self.latex_input = QLineEdit()
        self.latex_input.setPlaceholderText(r"Enter LaTeX (e.g., E = mc^2)")
        self.latex_input.setFont(QFont("Monospace", 11))
        self.latex_input.returnPressed.connect(self._on_solve_clicked)
        input_layout.addWidget(self.latex_input)

        layout.addLayout(input_layout)

        # Plain text input option
        plain_layout = QHBoxLayout()
        plain_layout.addWidget(QLabel("Or plain text:"))

        self.plain_input = QLineEdit()
        self.plain_input.setPlaceholderText("E = mc^2")
        self.plain_input.returnPressed.connect(self._on_plain_text_solve)
        plain_layout.addWidget(self.plain_input)

        layout.addLayout(plain_layout)

        # Solve button and confidence indicator
        button_layout = QHBoxLayout()

        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._on_solve_clicked)
        self.solve_btn.setStyleSheet("font-weight: bold; padding: 5px 20px;")
        button_layout.addWidget(self.solve_btn)

        button_layout.addStretch()

        self.confidence_label = QLabel("")
        button_layout.addWidget(self.confidence_label)

        layout.addLayout(button_layout)

        return group

    def _create_classification_panel(self) -> QGroupBox:
        """Create the classification display panel."""
        group = QGroupBox("Classification")
        layout = QVBoxLayout(group)

        self.classification_label = QLabel("No equation loaded")
        self.classification_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.classification_label)

        # Variable selector
        var_layout = QHBoxLayout()
        var_layout.addWidget(QLabel("Solve for:"))

        self.var_selector = QComboBox()
        self.var_selector.setMinimumWidth(100)
        self.var_selector.currentTextChanged.connect(self._on_variable_changed)
        var_layout.addWidget(self.var_selector)

        var_layout.addStretch()
        layout.addLayout(var_layout)

        return group

    def _create_solution_panel(self) -> QGroupBox:
        """Create the solution display panel."""
        group = QGroupBox("Solution")
        layout = QVBoxLayout(group)

        # Solution display (scrollable text area for now)
        # TODO: Replace with QWebEngineView for MathJax rendering
        self.solution_display = QTextEdit()
        self.solution_display.setReadOnly(True)
        self.solution_display.setFont(QFont("Monospace", 11))
        self.solution_display.setPlaceholderText("Solution will appear here...")
        layout.addWidget(self.solution_display)

        return group

    def _create_export_buttons(self) -> QHBoxLayout:
        """Create export buttons."""
        layout = QHBoxLayout()

        copy_btn = QPushButton("Copy LaTeX")
        copy_btn.clicked.connect(self._on_copy_latex)
        layout.addWidget(copy_btn)

        copy_python_btn = QPushButton("Copy Python")
        copy_python_btn.clicked.connect(self._on_copy_python)
        layout.addWidget(copy_python_btn)

        layout.addStretch()

        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._on_new_clicked)
        layout.addWidget(new_btn)

        return layout

    def _init_toolbar(self):
        """Initialize the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Screenshot action
        screenshot_action = QAction("📷 Screenshot", self)
        screenshot_action.setStatusTip("Capture equation from screen")
        screenshot_action.triggered.connect(self._on_screenshot_clicked)
        toolbar.addAction(screenshot_action)

        toolbar.addSeparator()

        # Paste action
        paste_action = QAction("📋 Paste", self)
        paste_action.setStatusTip("Paste LaTeX from clipboard")
        paste_action.triggered.connect(self._on_paste_clicked)
        toolbar.addAction(paste_action)

        toolbar.addSeparator()

        # History action
        history_action = QAction("📂 History", self)
        history_action.setStatusTip("View solve history")
        history_action.triggered.connect(self._on_history_clicked)
        toolbar.addAction(history_action)

    def _init_statusbar(self):
        """Initialize the status bar."""
        self.statusBar().showMessage("Ready")

    # === Component Lazy Loading ===

    def _get_screenshot_capture(self):
        """Lazy-load screenshot capture."""
        if self._screenshot_capture is None:
            from ..input.screenshot import ScreenshotCapture

            self._screenshot_capture = ScreenshotCapture()
        return self._screenshot_capture

    def _get_ocr_engine(self):
        """Lazy-load OCR engine."""
        if self._ocr_engine is None:
            from ..input.ocr import OCREngine

            self._ocr_engine = OCREngine(lazy_load=True)
        return self._ocr_engine

    def _get_parser(self):
        """Lazy-load LaTeX parser."""
        if self._parser is None:
            from ..input.parser import LatexParser

            self._parser = LatexParser()
        return self._parser

    def _get_classifier(self):
        """Lazy-load equation classifier."""
        if self._classifier is None:
            from ..classification.classifier import EquationClassifier

            self._classifier = EquationClassifier()
        return self._classifier

    def _get_solver(self):
        """Lazy-load general solver."""
        if self._solver is None:
            from ..solvers.general import GeneralSolver

            self._solver = GeneralSolver()
        return self._solver

    # === Event Handlers ===

    def _on_screenshot_clicked(self):
        """Handle screenshot button click."""
        self.statusBar().showMessage("Select area to capture...")

        try:
            screenshot = self._get_screenshot_capture()
            ocr = self._get_ocr_engine()

            # Run in background thread
            self.ocr_worker = OCRWorker(screenshot, ocr)
            self.ocr_worker.finished.connect(self._on_ocr_finished)
            self.ocr_worker.error.connect(self._on_ocr_error)
            self.ocr_worker.start()

        except Exception as e:
            QMessageBox.warning(self, "Screenshot Error", str(e))
            self.statusBar().showMessage("Screenshot failed")

    def _on_ocr_finished(self, latex: str, confidence: float):
        """Handle OCR completion."""
        self.latex_input.setText(latex)

        # Show confidence
        conf_pct = int(confidence * 100)
        if confidence < 0.7:
            self.confidence_label.setText(f"⚠️ Confidence: {conf_pct}%")
            self.confidence_label.setStyleSheet("color: orange;")
        else:
            self.confidence_label.setText(f"✓ Confidence: {conf_pct}%")
            self.confidence_label.setStyleSheet("color: green;")

        self.statusBar().showMessage(f"OCR complete ({conf_pct}% confidence)")

        # Auto-solve if high confidence
        if confidence >= 0.8:
            self._on_solve_clicked()

    def _on_ocr_error(self, error: str):
        """Handle OCR error."""
        if "cancelled" in error.lower():
            self.statusBar().showMessage("Screenshot cancelled")
        else:
            QMessageBox.warning(self, "OCR Error", error)
            self.statusBar().showMessage("OCR failed")

    def _on_solve_clicked(self):
        """Handle solve button click."""
        latex = self.latex_input.text().strip()
        if not latex:
            self.statusBar().showMessage("Enter an equation first")
            return

        self._solve_latex(latex)

    def _on_plain_text_solve(self):
        """Handle plain text input solve."""
        text = self.plain_input.text().strip()
        if not text:
            return

        try:
            parser = self._get_parser()
            equation = parser.parse_plain_text(text)
            self.latex_input.setText(equation.raw_latex)
            self._solve_equation(equation)
        except Exception as e:
            QMessageBox.warning(self, "Parse Error", str(e))

    def _solve_latex(self, latex: str):
        """Parse and solve a LaTeX string."""
        try:
            parser = self._get_parser()
            equation = parser.parse(latex)
            self._solve_equation(equation)
        except Exception as e:
            QMessageBox.warning(self, "Parse Error", str(e))
            self.statusBar().showMessage("Failed to parse equation")

    def _solve_equation(self, equation):
        """Solve a parsed equation."""
        from ..models import SolveRequest

        self._current_equation = equation

        # Classify
        classifier = self._get_classifier()
        eq_type, subtype = classifier.classify(equation)
        equation.classification = (eq_type, subtype)

        # Update classification display
        type_str = eq_type.name
        if subtype:
            type_str += f" ({subtype})"
        self.classification_label.setText(f"Type: {type_str}")

        # Update variable selector
        self.var_selector.clear()
        variables = classifier.get_variables(equation)
        for var in variables:
            self.var_selector.addItem(str(var))

        # Get selected target variable
        target = None
        if variables:
            import sympy as sp

            target = variables[0]

        # Create solve request
        request = SolveRequest(
            equation=equation, target_variable=target, show_steps=True
        )

        # Solve
        solver = self._get_solver()
        self.statusBar().showMessage("Solving...")

        result = solver.solve(request)
        self._on_solve_finished(result)

    def _on_solve_finished(self, result):
        """Handle solve completion."""
        if result.success:
            self._current_solution = result.solution

            # Display solution
            output = []
            output.append(f"Solution ({result.solver_name}):")
            output.append(f"  {result.solution.latex}")
            output.append("")

            if result.solution.steps:
                output.append("Steps:")
                for step in result.solution.steps:
                    output.append(f"  {step.step_number}. {step.operation}")
                    output.append(f"     {step.latex_repr}")

            self.solution_display.setText("\n".join(output))

            time_ms = result.solution.solve_time_ms
            self.statusBar().showMessage(f"Solved in {time_ms}ms")
        else:
            self.solution_display.setText(f"Failed: {result.error_message}")
            self.statusBar().showMessage("Solve failed")

    def _on_variable_changed(self, var_text: str):
        """Handle variable selector change."""
        if not self._current_equation or not var_text:
            return

        import sympy as sp
        from ..models import SolveRequest

        target = sp.Symbol(var_text)
        request = SolveRequest(
            equation=self._current_equation, target_variable=target, show_steps=True
        )

        solver = self._get_solver()
        result = solver.solve(request)
        self._on_solve_finished(result)

    def _on_paste_clicked(self):
        """Handle paste button click."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if text:
            self.latex_input.setText(text)
            self.statusBar().showMessage("Pasted from clipboard")

    def _on_history_clicked(self):
        """Handle history button click."""
        # TODO: Implement history dialog
        QMessageBox.information(self, "History", "History feature coming soon!")

    def _on_copy_latex(self):
        """Copy solution LaTeX to clipboard."""
        if self._current_solution:
            clipboard = QApplication.clipboard()
            clipboard.setText(self._current_solution.latex)
            self.statusBar().showMessage("LaTeX copied to clipboard")

    def _on_copy_python(self):
        """Copy solution as Python/SymPy code."""
        if self._current_solution:
            import sympy as sp

            code = f"# {self._current_solution.latex}\n"
            code += f"from sympy import *\n"
            code += f"result = {repr(self._current_solution.symbolic_result)}"

            clipboard = QApplication.clipboard()
            clipboard.setText(code)
            self.statusBar().showMessage("Python code copied to clipboard")

    def _on_new_clicked(self):
        """Clear current equation and start fresh."""
        self.latex_input.clear()
        self.plain_input.clear()
        self.solution_display.clear()
        self.classification_label.setText("No equation loaded")
        self.var_selector.clear()
        self.confidence_label.clear()
        self._current_equation = None
        self._current_solution = None
        self.statusBar().showMessage("Ready")


def run_app():
    """Run the MathSolver application."""
    app = QApplication(sys.argv)
    app.setApplicationName("MathSolver")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
