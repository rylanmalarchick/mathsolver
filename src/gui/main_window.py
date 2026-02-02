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

from ..utils.errors import (
    format_error_for_dialog,
    format_error_for_user,
    MathSolverError,
    ParseError,
    OCRError,
    ScreenshotCancelledError,
    SolveError,
    SolveTimeoutError,
    ErrorSeverity,
)


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

        # Numerical Evaluation Panel
        numerical_group = self._create_numerical_panel()
        splitter.addWidget(numerical_group)

        # Solution Panel
        solution_group = self._create_solution_panel()
        splitter.addWidget(solution_group)

        # Set initial sizes (input smaller, solution larger)
        splitter.setSizes([150, 80, 100, 400])

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

    def _create_numerical_panel(self) -> QGroupBox:
        """Create the numerical evaluation panel."""
        group = QGroupBox("Numerical Evaluation")
        layout = QVBoxLayout(group)

        # Variable input fields container
        self.var_inputs_widget = QWidget()
        self.var_inputs_layout = QVBoxLayout(self.var_inputs_widget)
        self.var_inputs_layout.setContentsMargins(0, 0, 0, 0)
        self.var_inputs_layout.setSpacing(5)

        # Dict to track variable input fields
        self.var_input_fields = {}

        # Placeholder message
        self.var_inputs_placeholder = QLabel(
            "Solve an equation to enter variable values"
        )
        self.var_inputs_placeholder.setStyleSheet("color: gray; font-style: italic;")
        self.var_inputs_layout.addWidget(self.var_inputs_placeholder)

        layout.addWidget(self.var_inputs_widget)

        # Evaluate button and result
        eval_layout = QHBoxLayout()

        self.eval_btn = QPushButton("Evaluate Numerically")
        self.eval_btn.clicked.connect(self._on_evaluate_clicked)
        self.eval_btn.setEnabled(False)
        eval_layout.addWidget(self.eval_btn)

        eval_layout.addStretch()
        layout.addLayout(eval_layout)

        # Numerical result display
        result_layout = QHBoxLayout()
        result_layout.addWidget(QLabel("Result:"))

        self.numerical_result_label = QLabel("")
        self.numerical_result_label.setFont(QFont("Monospace", 12))
        self.numerical_result_label.setStyleSheet("font-weight: bold;")
        result_layout.addWidget(self.numerical_result_label)

        result_layout.addStretch()
        layout.addLayout(result_layout)

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

    # === Error Handling ===

    def _show_error(self, exc: Exception, context: str = "") -> None:
        """
        Show a rich error dialog with suggestions.

        Uses the centralized error handling system to provide
        user-friendly error messages with actionable suggestions.
        """
        error_info = format_error_for_dialog(exc, context)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(error_info["title"])
        msg_box.setText(error_info["text"])
        msg_box.setIcon(error_info["icon"])

        if error_info["detailed_text"]:
            msg_box.setDetailedText(error_info["detailed_text"])

        msg_box.exec()

        # Update status bar with brief message
        self.statusBar().showMessage(format_error_for_user(exc, context))

    def _show_info(self, title: str, message: str) -> None:
        """Show an informational message."""
        self.statusBar().showMessage(message)

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
            self._show_error(e, "initializing screenshot capture")

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
        # Check for cancellation (not a real error)
        if "cancelled" in error.lower() or "cancel" in error.lower():
            self.statusBar().showMessage("Screenshot cancelled")
            return

        # Create appropriate error and show
        ocr_error = OCRError(error)
        self._show_error(ocr_error, "during OCR processing")

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
        except ParseError as e:
            self._show_error(e, "parsing plain text")
        except Exception as e:
            self._show_error(e, "parsing plain text")

    def _solve_latex(self, latex: str):
        """Parse and solve a LaTeX string."""
        try:
            parser = self._get_parser()
            equation = parser.parse(latex)
            self._solve_equation(equation)
        except ParseError as e:
            self._show_error(e, "parsing LaTeX")
        except Exception as e:
            self._show_error(e, "parsing LaTeX")

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

            # Update numerical evaluation panel
            self._update_numerical_panel()
        else:
            self.solution_display.setText(f"Failed: {result.error_message}")
            self.statusBar().showMessage("Solve failed")
            self._clear_numerical_panel()

    def _update_numerical_panel(self):
        """Update the numerical evaluation panel with variable input fields."""
        # Clear existing inputs
        self._clear_numerical_panel()

        if not self._current_equation:
            return

        # Get variables that aren't the target
        classifier = self._get_classifier()
        all_vars = classifier.get_variables(self._current_equation)

        if not all_vars:
            return

        # Hide placeholder
        self.var_inputs_placeholder.hide()

        # Get target variable (selected in var_selector)
        target_text = self.var_selector.currentText()

        # Create input field for each non-target variable
        for var in all_vars:
            var_name = str(var)
            if var_name == target_text:
                continue  # Skip target variable

            # Create row for this variable
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"{var_name} =")
            label.setMinimumWidth(50)
            row_layout.addWidget(label)

            input_field = QLineEdit()
            input_field.setPlaceholderText("Enter value (e.g., 5.5 or 3e8)")
            input_field.returnPressed.connect(self._on_evaluate_clicked)
            row_layout.addWidget(input_field)

            # Unit hint (if we have physics formula info)
            unit_label = QLabel("")
            unit_label.setStyleSheet("color: gray;")
            unit_label.setMinimumWidth(50)
            row_layout.addWidget(unit_label)

            self.var_inputs_layout.addWidget(row_widget)
            self.var_input_fields[var_name] = (input_field, unit_label, row_widget)

        # Enable evaluate button if we have input fields
        self.eval_btn.setEnabled(len(self.var_input_fields) > 0)

        # Try to set unit hints from physics formula
        self._update_unit_hints()

    def _update_unit_hints(self):
        """Update unit hints from physics formula if available."""
        if not self._current_equation:
            return

        # Check if this is a physics equation
        eq_type, subtype = self._current_equation.classification
        if subtype:
            try:
                from ..solvers.physics_solver import PhysicsSolver

                solver = PhysicsSolver()
                info = solver.get_formula_info(subtype)

                if info and "units" in info:
                    for var_name, (
                        input_field,
                        unit_label,
                        _,
                    ) in self.var_input_fields.items():
                        if var_name in info["units"]:
                            unit_label.setText(f"({info['units'][var_name]})")
            except Exception:
                pass  # Not a physics formula or couldn't get info

    def _clear_numerical_panel(self):
        """Clear the numerical evaluation panel."""
        # Remove all variable input widgets
        for var_name, (_, _, widget) in self.var_input_fields.items():
            widget.deleteLater()

        self.var_input_fields.clear()
        self.numerical_result_label.clear()
        self.eval_btn.setEnabled(False)
        self.var_inputs_placeholder.show()

    def _on_evaluate_clicked(self):
        """Handle numerical evaluation button click."""
        if not self._current_solution:
            return

        import sympy as sp

        # Gather numerical values from input fields
        numerical_values = {}
        for var_name, (input_field, _, _) in self.var_input_fields.items():
            value_text = input_field.text().strip()
            if value_text:
                try:
                    # Parse the value (handle scientific notation)
                    # Remove unit suffix if present
                    value_str = (
                        value_text.split()[0] if " " in value_text else value_text
                    )
                    value = float(value_str)
                    numerical_values[sp.Symbol(var_name)] = value
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid Value",
                        f"Cannot parse value for {var_name}: {value_text}",
                    )
                    return

        if not numerical_values:
            self.statusBar().showMessage("Enter at least one value to evaluate")
            return

        # Try to evaluate the solution numerically
        try:
            result_expr = self._current_solution.symbolic_result

            # Check if it's a physics formula - get constants
            eq_type, subtype = self._current_equation.classification
            if subtype:
                try:
                    from ..solvers.physics_solver import PhysicsSolver

                    solver = PhysicsSolver()
                    formula = solver.library.get_by_id(subtype)
                    if formula:
                        # Add physical constants
                        for const_name, const_info in formula.constants.items():
                            numerical_values[sp.Symbol(const_name)] = const_info[
                                "value"
                            ]
                except Exception:
                    pass

            # Substitute and evaluate
            result = result_expr.subs(numerical_values).evalf()

            # Format the result
            if result.is_number:
                result_float = float(result)
                if abs(result_float) > 1e4 or (
                    abs(result_float) < 1e-3 and result_float != 0
                ):
                    result_str = f"{result_float:.4e}"
                else:
                    result_str = f"{result_float:.6g}"

                # Add unit if available
                target_var = str(self._current_solution.target_variable)
                if subtype:
                    try:
                        from ..solvers.physics_solver import PhysicsSolver

                        physics_solver = PhysicsSolver()
                        info = physics_solver.get_formula_info(subtype)
                        if info and "units" in info and target_var in info["units"]:
                            result_str += f" {info['units'][target_var]}"
                    except Exception:
                        pass

                self.numerical_result_label.setText(result_str)
                self.numerical_result_label.setStyleSheet(
                    "font-weight: bold; color: green;"
                )
                self.statusBar().showMessage("Numerical evaluation complete")
            else:
                # Result still has free symbols
                remaining = result.free_symbols
                self.numerical_result_label.setText(
                    f"Need values for: {', '.join(str(s) for s in remaining)}"
                )
                self.numerical_result_label.setStyleSheet(
                    "font-weight: bold; color: orange;"
                )

        except Exception as e:
            self.numerical_result_label.setText(f"Error: {str(e)}")
            self.numerical_result_label.setStyleSheet("font-weight: bold; color: red;")
            self.statusBar().showMessage("Numerical evaluation failed")

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
        from .history_dialog import HistoryDialog
        from ..utils.database import HistoryDatabase

        dialog = HistoryDialog(database=HistoryDatabase(), parent=self)
        dialog.re_solve_requested.connect(self._on_re_solve_from_history)
        dialog.exec()

    def _on_re_solve_from_history(self, raw_latex: str):
        """Handle re-solve request from history dialog."""
        self.latex_input.setText(raw_latex)
        self.plain_input.clear()
        self._on_solve_clicked()

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
        self._clear_numerical_panel()
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
