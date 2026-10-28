"""Shared helpers for the tests: make src/ importable and build a tiny dataset."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402


def tiny_clean_students():
    """Six hand-made, already-clean students with easy-to-check numbers."""
    return pd.DataFrame({
        "student_id": [f"T{i}" for i in range(1, 7)],
        "name": ["A", "B", "C", "D", "E", "F"],
        "age": [19, 20, 21, 22, 20, 19],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "department": ["Computer Science", "Computer Science", "Data Science",
                       "Data Science", "Data Science", "Computer Science"],
        "semester": [1, 2, 3, 4, 5, 6],
        "attendance_pct": [95.0, 60.0, 88.0, 70.0, 90.0, 80.0],
        "assignment_marks": [80, 60, 100, 40, 90, 70],
        "quiz_marks": [70, 50, 100, 30, 80, 60],
        "midterm_marks": [60, 40, 100, 20, 70, 50],
        "final_marks": [90, 30, 100, 10, 80, 40],
        "study_hours_per_week": [10.0, 4.0, 20.0, 2.0, 14.0, 8.0],
    })


def tiny_messy_students():
    """A small dataset with one example of every problem the cleaner handles."""
    return pd.DataFrame({
        "student_id": ["M1", "M2", "M2", "M3", "M4", "M5", "M6"],
        "name": ["Ann", "Bob", "Bob", "Cat", "Dan", "Eve", "Fay"],
        "age": [20, 21, 21, np.nan, 22, 5, 23],                       # M3 missing, M5 invalid
        "gender": ["F", " male ", " male ", "FEMALE", "m", "female", "f"],
        "department": [" cs ", "COMPUTER SCIENCE", "COMPUTER SCIENCE", "Data Science",
                       "ds", "Data Science", None],                    # M6 has no department
        "semester": [1, 2, 2, 3, 4, 5, 6],
        "attendance_pct": [90.0, 80.0, 80.0, 70.0, 120.0, 85.0, 75.0],  # M4 invalid
        "assignment_marks": [70.0, 60.0, 60.0, np.nan, 80.0, 90.0, 50.0],  # M3 missing
        "quiz_marks": [70.0, 60.0, 60.0, 50.0, 80.0, 90.0, 50.0],
        "midterm_marks": [60.0, 50.0, 50.0, 40.0, 70.0, 80.0, 40.0],
        "final_marks": [80.0, 70.0, 70.0, 60.0, 105.0, 95.0, 60.0],     # M4 final invalid -> removed
        "study_hours_per_week": [10.0, 8.0, 8.0, 6.0, 12.0, 15.0, 5.0],
    })
