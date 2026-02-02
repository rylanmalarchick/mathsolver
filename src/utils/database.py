"""
SQLite database for solution history.

Stores solved equations for later reference and analysis.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class HistoryEntry:
    """A single entry in the solve history."""

    id: int
    timestamp: datetime
    raw_latex: str
    classification: str
    target_variable: str
    solution_latex: str
    solve_time_ms: int


class HistoryDatabase:
    """
    SQLite database for storing solve history.

    Usage:
        db = HistoryDatabase()
        db.add_entry(latex, classification, target, solution, time_ms)
        entries = db.get_recent(limit=10)
    """

    DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "history.db"

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Uses default if None.
        """
        self.db_path = db_path or self.DEFAULT_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS solve_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    raw_latex TEXT NOT NULL,
                    classification TEXT,
                    target_variable TEXT,
                    solution_latex TEXT,
                    solve_time_ms INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON solve_history(timestamp DESC)
            """)
            conn.commit()

    def add_entry(
        self,
        raw_latex: str,
        classification: str,
        target_variable: str,
        solution_latex: str,
        solve_time_ms: int,
    ) -> int:
        """
        Add a solve entry to history.

        Returns:
            ID of the new entry
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO solve_history 
                (raw_latex, classification, target_variable, solution_latex, solve_time_ms)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    raw_latex,
                    classification,
                    target_variable,
                    solution_latex,
                    solve_time_ms,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent(self, limit: int = 20) -> List[HistoryEntry]:
        """Get the most recent entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM solve_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (limit,),
            )

            return [
                HistoryEntry(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    raw_latex=row["raw_latex"],
                    classification=row["classification"] or "",
                    target_variable=row["target_variable"] or "",
                    solution_latex=row["solution_latex"] or "",
                    solve_time_ms=row["solve_time_ms"] or 0,
                )
                for row in cursor.fetchall()
            ]

    def search(self, query: str, limit: int = 20) -> List[HistoryEntry]:
        """Search history by LaTeX content."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM solve_history 
                WHERE raw_latex LIKE ? OR solution_latex LIKE ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", limit),
            )

            return [
                HistoryEntry(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    raw_latex=row["raw_latex"],
                    classification=row["classification"] or "",
                    target_variable=row["target_variable"] or "",
                    solution_latex=row["solution_latex"] or "",
                    solve_time_ms=row["solve_time_ms"] or 0,
                )
                for row in cursor.fetchall()
            ]

    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM solve_history WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Clear all history. Returns number of entries deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM solve_history")
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> dict:
        """Get statistics about the history database."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM solve_history").fetchone()[0]

            avg_time = (
                conn.execute("SELECT AVG(solve_time_ms) FROM solve_history").fetchone()[
                    0
                ]
                or 0
            )

            by_type = conn.execute("""
                SELECT classification, COUNT(*) as count 
                FROM solve_history 
                GROUP BY classification
                ORDER BY count DESC
            """).fetchall()

            return {
                "total_entries": total,
                "average_solve_time_ms": round(avg_time, 2),
                "by_classification": dict(by_type),
            }
