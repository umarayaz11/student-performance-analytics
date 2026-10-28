"""Central settings for the whole project.

Every "magic number" lives here (weights, thresholds, file paths, colours),
so you can change the behaviour of the application by editing ONE file.

IMPORTANT: The grading rules below are EXAMPLE rules for practice.
They are NOT an official NUST (or any university) grading policy.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (pathlib works on Windows, macOS and Linux without changes)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"
RAW_CSV = DATA_DIR / "students.csv"

# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------
RANDOM_SEED = 42          # fixed seed -> the same "random" data every run
NUM_STUDENTS = 150        # unique students (a few duplicates are added on top)

DEPARTMENTS = [
    "Computer Science",
    "Software Engineering",
    "Data Science",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Aerospace Engineering",
]

# Used to demonstrate DataFrame.merge() in pandas_analysis.py
DEPARTMENT_FACULTY = {
    "Computer Science": "Computing",
    "Software Engineering": "Computing",
    "Data Science": "Computing",
    "Electrical Engineering": "Engineering",
    "Mechanical Engineering": "Engineering",
    "Aerospace Engineering": "Engineering",
}

# Messy spellings we expect to find in "real-world" data -> clean label.
# Keys must be lower-case, without full stops.
DEPARTMENT_ALIASES = {
    "cs": "Computer Science",
    "comp sci": "Computer Science",
    "computer sci": "Computer Science",
    "se": "Software Engineering",
    "software eng": "Software Engineering",
    "ds": "Data Science",
    "ee": "Electrical Engineering",
    "electrical eng": "Electrical Engineering",
    "me": "Mechanical Engineering",
    "mech eng": "Mechanical Engineering",
    "mechanical eng": "Mechanical Engineering",
    "aero": "Aerospace Engineering",
    "aerospace eng": "Aerospace Engineering",
}

GENDER_ALIASES = {
    "m": "Male", "male": "Male", "man": "Male",
    "f": "Female", "female": "Female", "woman": "Female",
}

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
COLUMNS = [
    "student_id", "name", "age", "gender", "department", "semester",
    "attendance_pct", "assignment_marks", "quiz_marks", "midterm_marks",
    "final_marks", "study_hours_per_week",
]
MARK_COLUMNS = ["assignment_marks", "quiz_marks", "midterm_marks", "final_marks"]

# Human-friendly labels for tables and charts
COLUMN_LABELS = {
    "assignment_marks": "Assignments",
    "quiz_marks": "Quizzes",
    "midterm_marks": "Midterm",
    "final_marks": "Final Exam",
}

# Labels used by the correlation heatmap and the component charts
FEATURE_LABELS = {
    "study_hours_per_week": "Study hours",
    "attendance_pct": "Attendance",
    **COLUMN_LABELS,
}

# --------------------------------------------------------------------------
# Valid ranges (used by the cleaning step to flag impossible values)
# --------------------------------------------------------------------------
MAX_MARKS = 100.0            # every mark is stored on a 0-100 scale
MIN_MARKS = 0.0
AGE_RANGE = (16, 40)
SEMESTER_RANGE = (1, 8)
ATTENDANCE_RANGE = (0.0, 100.0)
STUDY_HOURS_RANGE = (0.0, 80.0)

# --------------------------------------------------------------------------
# Grading rules (EXAMPLE rules - change them freely)
# --------------------------------------------------------------------------
# Weights must add up to 1.0 (i.e. 100 %).
GRADE_WEIGHTS = {
    "assignment_marks": 0.10,
    "quiz_marks": 0.10,
    "midterm_marks": 0.30,
    "final_marks": 0.50,
}

# (minimum total, letter) - checked from the top down.
GRADE_BANDS = [(85, "A"), (75, "B"), (65, "C"), (50, "D"), (0, "F")]

# (minimum total, category) - used for the "performance category" chart.
PERFORMANCE_BANDS = [
    (85, "Excellent"),
    (70, "Good"),
    (50, "Satisfactory"),
    (0, "At Risk"),
]
PERFORMANCE_ORDER = ["Excellent", "Good", "Satisfactory", "At Risk"]

PASS_THRESHOLD = 50.0             # total marks needed to pass
LOW_ATTENDANCE_THRESHOLD = 75.0   # below this = low attendance
HIGH_ATTENDANCE_THRESHOLD = 85.0  # at or above this = high attendance
LOW_PERFORMANCE_THRESHOLD = 55.0  # below this = low performance
TOP_N = 10

# --------------------------------------------------------------------------
# Visual theme (charts + terminal)
# --------------------------------------------------------------------------
THEME = {
    "navy": "#1B2A49",
    "blue": "#2F6DB5",
    "teal": "#2A9D8F",
    "amber": "#E9B44C",
    "coral": "#E4572E",
    "grey": "#6B7280",
    "light": "#F3F5F9",
    "grid": "#D9DEE7",
}
CATEGORY_COLOURS = {
    "Excellent": THEME["teal"],
    "Good": THEME["blue"],
    "Satisfactory": THEME["amber"],
    "At Risk": THEME["coral"],
}
