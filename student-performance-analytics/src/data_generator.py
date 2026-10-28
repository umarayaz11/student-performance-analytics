"""Create a synthetic student dataset (no real or confidential data).

The generator uses a *fixed random seed*, so everyone who runs it gets exactly
the same dataset. That makes results reproducible and easy to discuss.

By default the data is deliberately "messy" (missing values, impossible marks,
inconsistent spellings, duplicate rows) because real-world data is messy and
the cleaning module needs something to clean.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config

FIRST_NAMES_MALE = [
    "Ahmed", "Ali", "Hamza", "Usman", "Bilal", "Hassan", "Zain", "Faisal",
    "Omar", "Saad", "Talha", "Daniyal", "Abdullah", "Hashir", "Rayyan",
]
FIRST_NAMES_FEMALE = [
    "Ayesha", "Fatima", "Zainab", "Maryam", "Hira", "Sana", "Noor", "Areeba",
    "Iqra", "Laiba", "Mahnoor", "Rabia", "Sidra", "Amna", "Eman",
]
LAST_NAMES = [
    "Khan", "Malik", "Chaudhry", "Sheikh", "Butt", "Raza", "Qureshi", "Siddiqui",
    "Ansari", "Hussain", "Javed", "Iqbal", "Mirza", "Abbasi", "Farooq", "Aslam",
]


def generate_students(n=config.NUM_STUDENTS, seed=config.RANDOM_SEED, messy=True):
    """Return a DataFrame of synthetic students.

    Parameters
    ----------
    n : number of unique students.
    seed : random seed (same seed -> same dataset).
    messy : if True, inject missing values, invalid values, label
        inconsistencies and a few duplicate rows.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    # default_rng is NumPy's modern random generator. Using one generator
    # object (instead of the global np.random) keeps results reproducible.
    rng = np.random.default_rng(seed)

    # A hidden "ability" score makes the data realistic: students with higher
    # ability tend to study more, attend more and score higher.
    ability = rng.normal(0, 1, n)

    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    names = []
    for g in gender:
        pool = FIRST_NAMES_MALE if g == "Male" else FIRST_NAMES_FEMALE
        names.append(f"{rng.choice(pool)} {rng.choice(LAST_NAMES)}")

    semester = rng.integers(1, 9, size=n)
    age = 18 + (semester - 1) // 2 + rng.integers(0, 3, size=n)
    department = rng.choice(config.DEPARTMENTS, size=n)

    study_hours = np.clip(rng.normal(12, 4.5, n) + 2.5 * ability, 1, 40).round(1)
    attendance = np.clip(rng.normal(82, 9, n) + 3 * ability, 40, 100).round(1)

    base = 67 + 12 * ability + 0.35 * (study_hours - 12) + 0.12 * (attendance - 82)

    def marks(offset, noise):
        return np.clip(base + offset + rng.normal(0, noise, n), 0, 100).round(1)

    df = pd.DataFrame({
        "student_id": [f"STU-{i:04d}" for i in range(1, n + 1)],
        "name": names,
        "age": age.astype(float),          # float so NaN can be injected later
        "gender": list(gender),
        "department": list(department),
        "semester": semester,
        "attendance_pct": attendance,
        "assignment_marks": marks(+6, 7),
        "quiz_marks": marks(+2, 9),
        "midterm_marks": marks(0, 9),
        "final_marks": marks(-2, 10),
        "study_hours_per_week": study_hours,
    })[config.COLUMNS]

    if messy:
        df = _inject_messiness(df, rng)
    return df


def _inject_messiness(df, rng):
    """Add realistic data-quality problems (all reproducible via `rng`)."""
    df = df.copy()
    n = len(df)

    # 1) Missing numeric values (~3 % per column)
    for col in ["age", "attendance_pct", "assignment_marks", "quiz_marks",
                "midterm_marks", "final_marks", "study_hours_per_week"]:
        rows = rng.choice(n, size=max(1, round(0.03 * n)), replace=False)
        df.loc[rows, col] = np.nan

    # 2) Impossible values (one of each kind)
    invalid = {
        "final_marks": 105.0,
        "midterm_marks": -8.0,
        "quiz_marks": 130.0,
        "assignment_marks": 250.0,
        "attendance_pct": 112.0,
        "study_hours_per_week": -4.0,
        "age": 7.0,
    }
    for col, bad_value in invalid.items():
        row = int(rng.integers(0, n))
        df.loc[row, col] = bad_value

    # 3) Inconsistent department labels ("cs", "COMPUTER SCIENCE", " Data Science ")
    reverse_alias = {}
    for alias, official in config.DEPARTMENT_ALIASES.items():
        reverse_alias.setdefault(official, []).append(alias)

    dept_values = df["department"].tolist()
    for row in rng.choice(n, size=int(0.15 * n), replace=False):
        label = dept_values[row]
        style = int(rng.integers(0, 4))
        if style == 0:
            dept_values[row] = label.lower()
        elif style == 1:
            dept_values[row] = label.upper()
        elif style == 2:
            dept_values[row] = f"  {label} "
        else:
            dept_values[row] = str(rng.choice(reverse_alias[label]))
    for row in rng.choice(n, size=2, replace=False):      # unrecoverable labels
        dept_values[row] = None
    df["department"] = dept_values

    # 4) Inconsistent gender labels
    gender_values = df["gender"].tolist()
    for row in rng.choice(n, size=int(0.15 * n), replace=False):
        label = gender_values[row]
        options = ["M", "m", "male", " MALE"] if label == "Male" else ["F", "f", "female", " FEMALE"]
        gender_values[row] = str(rng.choice(options))
    df["gender"] = gender_values

    # 5) Duplicate rows (exact copies, as if a record was entered twice)
    duplicates = df.iloc[rng.choice(n, size=5, replace=False)]
    df = pd.concat([df, duplicates], ignore_index=True)
    df = df.sort_values("student_id", kind="stable").reset_index(drop=True)
    return df


def save_students_csv(df, path=config.RAW_CSV):
    """Write the DataFrame to CSV (creating the folder if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    data = generate_students()
    saved = save_students_csv(data)
    print(f"Created {len(data)} rows -> {saved}")
