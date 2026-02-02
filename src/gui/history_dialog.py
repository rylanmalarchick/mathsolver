"""
History dialog for browsing and re-solving past equations.

Provides search, filtering, and export capabilities.
"""

from typing import Optional, List
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QMessageBox,
    QHeaderView,
    QAbstractItemView,
    QComboBox,
    QWidget,
    QSplitter,
    QTextEdit,
    QGroupBox,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..utils.database import HistoryDatabase, HistoryEntry


class HistoryDialog(QDialog):
    """
    Dialog for browsing solve history.

    Features:
    - Search by equation content
    - Filter by classification type
    - Preview equation and solution
    - Re-solve selected equation
    - Export history to file
    - Delete entries

    Signals:
        re_solve_requested: Emitted when user wants to re-solve an equation
    """

    re_solve_requested = pyqtSignal(str)  # Emits the raw LaTeX to solve

    def __init__(self, database: Optional[HistoryDatabase] = None, parent=None):
        super().__init__(parent)
        self.db = database or HistoryDatabase()
        self.current_entries: List[HistoryEntry] = []

        self._setup_ui()
        self._load_history()

    def _setup_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Solve History")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)

        # Search and filter bar
        search_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search equations...")
        self.search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(self.search_input, stretch=3)

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Types", "")
        self.filter_combo.addItem("Physics", "physics")
        self.filter_combo.addItem("Calculus", "calculus")
        self.filter_combo.addItem("ODE", "ode")
        self.filter_combo.addItem("Polynomial", "polynomial")
        self.filter_combo.addItem("General", "general")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        search_layout.addWidget(QLabel("Filter:"))
        search_layout.addWidget(self.filter_combo)

        layout.addLayout(search_layout)

        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: History table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            ["Date", "Equation", "Type", "Solution", "Time (ms)"]
        )

        # Table settings
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.history_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.history_table.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self.history_table.doubleClicked.connect(self._on_double_click)

        table_layout.addWidget(self.history_table)

        # Entry count label
        self.count_label = QLabel("0 entries")
        table_layout.addWidget(self.count_label)

        splitter.addWidget(table_widget)

        # Right side: Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # Equation preview
        eq_group = QGroupBox("Equation")
        eq_layout = QVBoxLayout(eq_group)
        self.equation_preview = QTextEdit()
        self.equation_preview.setReadOnly(True)
        self.equation_preview.setFont(QFont("Courier", 11))
        self.equation_preview.setMaximumHeight(80)
        eq_layout.addWidget(self.equation_preview)
        preview_layout.addWidget(eq_group)

        # Solution preview
        sol_group = QGroupBox("Solution")
        sol_layout = QVBoxLayout(sol_group)
        self.solution_preview = QTextEdit()
        self.solution_preview.setReadOnly(True)
        self.solution_preview.setFont(QFont("Courier", 11))
        sol_layout.addWidget(self.solution_preview)
        preview_layout.addWidget(sol_group)

        # Details
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout(details_group)
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)
        preview_layout.addWidget(details_group)

        splitter.addWidget(preview_widget)
        splitter.setSizes([500, 300])

        layout.addWidget(splitter, stretch=1)

        # Button row
        button_layout = QHBoxLayout()

        self.re_solve_btn = QPushButton("Re-solve")
        self.re_solve_btn.clicked.connect(self._on_re_solve)
        self.re_solve_btn.setEnabled(False)
        button_layout.addWidget(self.re_solve_btn)

        self.copy_btn = QPushButton("Copy LaTeX")
        self.copy_btn.clicked.connect(self._on_copy)
        self.copy_btn.setEnabled(False)
        button_layout.addWidget(self.copy_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        self.delete_btn.setEnabled(False)
        button_layout.addWidget(self.delete_btn)

        button_layout.addStretch()

        self.export_btn = QPushButton("Export All...")
        self.export_btn.clicked.connect(self._on_export)
        button_layout.addWidget(self.export_btn)

        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self._on_clear)
        button_layout.addWidget(self.clear_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def _load_history(self, search_query: str = "", type_filter: str = ""):
        """Load history entries from database."""
        if search_query:
            entries = self.db.search(search_query, limit=100)
        else:
            entries = self.db.get_recent(limit=100)

        # Apply type filter
        if type_filter:
            entries = [
                e for e in entries if type_filter.lower() in e.classification.lower()
            ]

        self.current_entries = entries
        self._populate_table()

    def _populate_table(self):
        """Populate the table with current entries."""
        self.history_table.setRowCount(len(self.current_entries))

        for row, entry in enumerate(self.current_entries):
            # Date
            date_item = QTableWidgetItem(entry.timestamp.strftime("%Y-%m-%d %H:%M"))
            self.history_table.setItem(row, 0, date_item)

            # Equation (truncated)
            eq_text = entry.raw_latex[:50]
            if len(entry.raw_latex) > 50:
                eq_text += "..."
            eq_item = QTableWidgetItem(eq_text)
            eq_item.setToolTip(entry.raw_latex)
            self.history_table.setItem(row, 1, eq_item)

            # Type
            type_item = QTableWidgetItem(entry.classification)
            self.history_table.setItem(row, 2, type_item)

            # Solution (truncated)
            sol_text = entry.solution_latex[:40]
            if len(entry.solution_latex) > 40:
                sol_text += "..."
            sol_item = QTableWidgetItem(sol_text)
            sol_item.setToolTip(entry.solution_latex)
            self.history_table.setItem(row, 3, sol_item)

            # Time
            time_item = QTableWidgetItem(str(entry.solve_time_ms))
            time_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.history_table.setItem(row, 4, time_item)

        self.count_label.setText(f"{len(self.current_entries)} entries")

        # Clear preview if no entries
        if not self.current_entries:
            self._clear_preview()

    def _clear_preview(self):
        """Clear the preview panel."""
        self.equation_preview.clear()
        self.solution_preview.clear()
        self.details_label.setText("")
        self.re_solve_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)

    def _get_selected_entry(self) -> Optional[HistoryEntry]:
        """Get the currently selected entry."""
        selection = self.history_table.selectionModel().selectedRows()
        if selection:
            row = selection[0].row()
            if 0 <= row < len(self.current_entries):
                return self.current_entries[row]
        return None

    def _on_search(self, text: str):
        """Handle search input change."""
        type_filter = self.filter_combo.currentData() or ""
        self._load_history(search_query=text, type_filter=type_filter)

    def _on_filter_changed(self, index: int):
        """Handle filter combo change."""
        type_filter = self.filter_combo.currentData() or ""
        search_text = self.search_input.text()
        self._load_history(search_query=search_text, type_filter=type_filter)

    def _on_selection_changed(self, selected, deselected):
        """Handle table selection change."""
        entry = self._get_selected_entry()
        if entry:
            self.equation_preview.setText(entry.raw_latex)
            self.solution_preview.setText(entry.solution_latex)
            self.details_label.setText(
                f"<b>Type:</b> {entry.classification}<br>"
                f"<b>Target Variable:</b> {entry.target_variable}<br>"
                f"<b>Solve Time:</b> {entry.solve_time_ms}ms<br>"
                f"<b>Date:</b> {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.re_solve_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
        else:
            self._clear_preview()

    def _on_double_click(self, index):
        """Handle double-click on table row."""
        self._on_re_solve()

    def _on_re_solve(self):
        """Emit signal to re-solve the selected equation."""
        entry = self._get_selected_entry()
        if entry:
            self.re_solve_requested.emit(entry.raw_latex)
            self.accept()  # Close dialog

    def _on_copy(self):
        """Copy the equation LaTeX to clipboard."""
        entry = self._get_selected_entry()
        if entry:
            from PyQt6.QtWidgets import QApplication

            clipboard = QApplication.clipboard()
            clipboard.setText(entry.raw_latex)

    def _on_delete(self):
        """Delete the selected entry."""
        entry = self._get_selected_entry()
        if entry:
            reply = QMessageBox.question(
                self,
                "Delete Entry",
                f"Delete this entry?\n\n{entry.raw_latex[:60]}...",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.db.delete_entry(entry.id)
                self._load_history(
                    search_query=self.search_input.text(),
                    type_filter=self.filter_combo.currentData() or "",
                )

    def _on_export(self):
        """Export history to a file."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export History",
            "mathsolver_history.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;LaTeX Files (*.tex)",
        )

        if not file_path:
            return

        entries = self.db.get_recent(limit=1000)  # Get all

        try:
            if file_path.endswith(".csv"):
                self._export_csv(file_path, entries)
            elif file_path.endswith(".tex"):
                self._export_latex(file_path, entries)
            else:
                self._export_text(file_path, entries)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(entries)} entries to:\n{file_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def _export_text(self, file_path: str, entries: List[HistoryEntry]):
        """Export to plain text."""
        with open(file_path, "w") as f:
            f.write("MathSolver History Export\n")
            f.write("=" * 60 + "\n\n")

            for entry in entries:
                f.write(f"Date: {entry.timestamp}\n")
                f.write(f"Type: {entry.classification}\n")
                f.write(f"Equation: {entry.raw_latex}\n")
                f.write(f"Target: {entry.target_variable}\n")
                f.write(f"Solution: {entry.solution_latex}\n")
                f.write(f"Solve Time: {entry.solve_time_ms}ms\n")
                f.write("-" * 40 + "\n\n")

    def _export_csv(self, file_path: str, entries: List[HistoryEntry]):
        """Export to CSV."""
        import csv

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Classification",
                    "Equation",
                    "Target Variable",
                    "Solution",
                    "Solve Time (ms)",
                ]
            )

            for entry in entries:
                writer.writerow(
                    [
                        entry.timestamp.isoformat(),
                        entry.classification,
                        entry.raw_latex,
                        entry.target_variable,
                        entry.solution_latex,
                        entry.solve_time_ms,
                    ]
                )

    def _export_latex(self, file_path: str, entries: List[HistoryEntry]):
        """Export to LaTeX document."""
        with open(file_path, "w") as f:
            f.write(r"\documentclass{article}" + "\n")
            f.write(r"\usepackage{amsmath}" + "\n")
            f.write(r"\usepackage{longtable}" + "\n")
            f.write(r"\title{MathSolver History}" + "\n")
            f.write(r"\date{\today}" + "\n")
            f.write(r"\begin{document}" + "\n")
            f.write(r"\maketitle" + "\n\n")

            for entry in entries:
                f.write(
                    r"\subsection*{"
                    + entry.timestamp.strftime("%Y-%m-%d %H:%M")
                    + "}\n"
                )
                f.write(f"\\textbf{{Type:}} {entry.classification}\n\n")
                f.write(f"\\textbf{{Equation:}}\n")
                f.write(f"\\[ {entry.raw_latex} \\]\n\n")
                f.write(f"\\textbf{{Solution:}}\n")
                f.write(f"\\[ {entry.solution_latex} \\]\n\n")
                f.write(r"\hrule" + "\n\n")

            f.write(r"\end{document}" + "\n")

    def _on_clear(self):
        """Clear all history."""
        reply = QMessageBox.warning(
            self,
            "Clear History",
            "This will delete ALL history entries.\n\nAre you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            count = self.db.clear_all()
            self._load_history()
            QMessageBox.information(
                self, "History Cleared", f"Deleted {count} entries."
            )
