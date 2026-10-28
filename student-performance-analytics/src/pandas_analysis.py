"""Data analysis with Pandas.

Pandas is built for TABLES: named columns, mixed data types, grouping, joining
and missing-value handling. NumPy is perfect for pure numbers; pandas adds the
labels and the "group by department" style questions that analysts ask daily.

Every function takes the *performance DataFrame* (the cleaned data plus the
columns made by grading.add_performance_columns) and returns a new DataFrame
or dictionary. Nothing here modifies its input.
"""

import pandas as pd

import config

REPORT_COLUMNS = [
    "rank", "student_id", "name", "department", "semester", "attendance_pct",
    "assignment_marks", "quiz_marks", "midterm_marks", "final_marks",
    "total_marks", "grade", "performance_category", "result",
]


def _check_threshold(value, name):
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100 (got {value}).")


def performance_report(df):
    """The full student performance report, best student first."""
    return (df.sort_values(["total_marks", "student_id"], ascending=[False, True])
              [REPORT_COLUMNS].reset_index(drop=True))


def top_students(df, n=config.TOP_N):
    return (df.sort_values(["total_marks", "student_id"], ascending=[False, True])
              .head(n)[REPORT_COLUMNS].reset_index(drop=True))


def bottom_students(df, n=config.TOP_N):
    return (df.sort_values(["total_marks", "student_id"], ascending=[True, True])
              .head(n)[REPORT_COLUMNS].reset_index(drop=True))


def department_stats(df, pass_threshold=config.PASS_THRESHOLD):
    """One row per department: size, average/median/min/max marks, pass rate..."""
    _check_threshold(pass_threshold, "Pass threshold")
    stats = df.groupby("department").agg(
        students=("student_id", "count"),
        avg_total=("total_marks", "mean"),
        median_total=("total_marks", "median"),
        min_total=("total_marks", "min"),
        max_total=("total_marks", "max"),
        pass_rate_pct=("total_marks", lambda s: (s >= pass_threshold).mean() * 100),
        avg_attendance=("attendance_pct", "mean"),
        avg_study_hours=("study_hours_per_week", "mean"),
    ).reset_index()

    # merge(): join a second table (department -> faculty) onto the statistics.
    faculty = pd.DataFrame(list(config.DEPARTMENT_FACULTY.items()),
                           columns=["department", "faculty"])
    stats = stats.merge(faculty, on="department", how="left")
    stats = stats[["department", "faculty"] + [c for c in stats.columns
                                               if c not in ("department", "faculty")]]
    return stats.sort_values("avg_total", ascending=False).round(2).reset_index(drop=True)


def semester_stats(df, pass_threshold=config.PASS_THRESHOLD):
    _check_threshold(pass_threshold, "Pass threshold")
    stats = df.groupby("semester").agg(
        students=("student_id", "count"),
        avg_total=("total_marks", "mean"),
        avg_attendance=("attendance_pct", "mean"),
        pass_rate_pct=("total_marks", lambda s: (s >= pass_threshold).mean() * 100),
    ).reset_index()
    return stats.round(2)


def department_counts(df):
    """Students per department using value_counts()."""
    counts = df["department"].value_counts().rename_axis("department").reset_index(name="students")
    counts["share_pct"] = (counts["students"] / counts["students"].sum() * 100).round(1)
    return counts


def pass_fail_summary(df, pass_threshold=config.PASS_THRESHOLD):
    _check_threshold(pass_threshold, "Pass threshold")
    total = len(df)
    if total == 0:
        raise ValueError("There are no students to analyse.")
    passed = int((df["total_marks"] >= pass_threshold).sum())
    return {
        "threshold": pass_threshold, "students": total,
        "passed": passed, "failed": total - passed,
        "pass_pct": passed / total * 100, "fail_pct": (total - passed) / total * 100,
    }


def low_attendance_students(df, threshold=config.LOW_ATTENDANCE_THRESHOLD):
    """Students whose attendance is below `threshold` (uses DataFrame.query)."""
    _check_threshold(threshold, "Attendance threshold")
    cols = ["student_id", "name", "department", "semester", "attendance_pct",
            "total_marks", "grade", "result"]
    return (df.query("attendance_pct < @threshold")[cols]
              .sort_values("attendance_pct").reset_index(drop=True))


def high_attendance_low_performance(df, attendance_min=config.HIGH_ATTENDANCE_THRESHOLD,
                                    performance_max=config.LOW_PERFORMANCE_THRESHOLD):
    """Students who attend a lot but still score low - worth a conversation."""
    _check_threshold(attendance_min, "Attendance threshold")
    _check_threshold(performance_max, "Performance threshold")
    cols = ["student_id", "name", "department", "attendance_pct",
            "study_hours_per_week", "total_marks", "grade"]
    mask = (df["attendance_pct"] >= attendance_min) & (df["total_marks"] < performance_max)
    return df.loc[mask, cols].sort_values("total_marks").reset_index(drop=True)


def assessment_averages(df, weights=None):
    """Average mark per component, next to how much each component counts."""
    weights = config.GRADE_WEIGHTS if weights is None else weights
    means = df[config.MARK_COLUMNS].mean()
    out = pd.DataFrame({
        "component": [config.COLUMN_LABELS[c] for c in config.MARK_COLUMNS],
        "average_mark": means.to_numpy(),
        "std_dev": df[config.MARK_COLUMNS].std().to_numpy(),
        "weight_pct": [weights[c] * 100 for c in config.MARK_COLUMNS],
    })
    return out.round(2)


def performance_category_counts(df):
    counts = (df["performance_category"].value_counts()
              .reindex(config.PERFORMANCE_ORDER, fill_value=0))
    return counts.rename_axis("category").reset_index(name="students")


def grade_distribution(df):
    order = [letter for _, letter in config.GRADE_BANDS]
    counts = df["grade"].value_counts().reindex(order, fill_value=0)
    return counts.rename_axis("grade").reset_index(name="students")


def _strength(r):
    r = abs(r)
    if r >= 0.7:
        return "strong"
    if r >= 0.4:
        return "moderate"
    if r >= 0.2:
        return "weak"
    return "very weak or none"


def study_hours_association(df):
    """Is more study time associated with higher marks? (association != causation)"""
    pearson = df["study_hours_per_week"].corr(df["total_marks"])
    # Spearman = Pearson correlation of the RANKS. Computing it this way needs no extra
    # library (method="spearman" would require SciPy on some pandas versions).
    spearman = df["study_hours_per_week"].rank().corr(df["total_marks"].rank())

    bins = [0, 5, 10, 15, 20, float("inf")]
    labels = ["0-5 h", "5-10 h", "10-15 h", "15-20 h", "20+ h"]
    bucket = pd.cut(df["study_hours_per_week"], bins=bins, labels=labels, include_lowest=True)
    by_bucket = (df.groupby(bucket, observed=True)
                   .agg(students=("student_id", "count"), avg_total=("total_marks", "mean"))
                   .round(2).reset_index().rename(columns={"study_hours_per_week": "study_hours"}))

    direction = "positive" if pearson > 0 else "negative"
    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "strength": _strength(pearson),
        "direction": direction,
        "by_bucket": by_bucket,
        "caution": ("Correlation measures how two columns move together. It does NOT prove that "
                    "studying more CAUSES higher marks - other factors (prior knowledge, "
                    "attendance, motivation) may influence both."),
    }


def correlation_matrix(df):
    """Pearson correlations between study hours, attendance and the four marks.

    total_marks is left out on purpose: it is CALCULATED from the four marks,
    so its correlation with them is built in and would teach us nothing.
    """
    columns = list(config.FEATURE_LABELS)
    return df[columns].corr().rename(index=config.FEATURE_LABELS,
                                     columns=config.FEATURE_LABELS).round(2)


def department_grade_table(df):
    """How many students of each department got each letter grade (crosstab)."""
    grades = [letter for _, letter in config.GRADE_BANDS]
    table = pd.crosstab(df["department"], df["grade"])
    return table.reindex(columns=grades, fill_value=0)


def components_long(df):
    """Reshape the four mark columns from WIDE to LONG format with pd.melt.

    wide: one row per student, one column per component
    long: one row per (student, component) pair - the shape seaborn prefers
    """
    long = df[["student_id"] + config.MARK_COLUMNS].melt(
        id_vars="student_id", var_name="component", value_name="marks")
    long["component"] = long["component"].map(config.COLUMN_LABELS)
    return long
