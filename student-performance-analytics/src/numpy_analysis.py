"""Numerical analysis with NumPy.

Why NumPy? A NumPy array stores numbers in one compact block of memory and
runs calculations in fast compiled code. `marks.mean()` or `marks >= 50`
processes every student at once ("vectorisation"), with no Python for-loop.

Functions here take plain arrays/Series and return plain numbers or arrays, so
they are easy to test. `build_numpy_report()` assembles everything for the menu.
"""

import time

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def descriptive_stats(values, ddof=0):
    """Mean, median, min, max, range, standard deviation and variance.

    ddof ("delta degrees of freedom") controls the variance formula:
        ddof=0 -> divide by n     (the whole class IS the population; NumPy's default)
        ddof=1 -> divide by n - 1 (our students are a sample; pandas' default)
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot calculate statistics on an empty array.")
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.ptp(arr)),
        "std": float(np.std(arr, ddof=ddof)),
        "variance": float(np.var(arr, ddof=ddof)),
    }


def percentile_summary(values, percentiles=(10, 25, 50, 75, 90)):
    """np.percentile: the value below which p % of the data falls."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot calculate percentiles on an empty array.")
    if np.any((np.asarray(percentiles) < 0) | (np.asarray(percentiles) > 100)):
        raise ValueError("Percentiles must be between 0 and 100.")
    return dict(zip(percentiles, np.percentile(arr, percentiles)))


def unique_with_counts(values):
    """np.unique(..., return_counts=True) -> (sorted distinct values, counts)."""
    labels, counts = np.unique(np.asarray(values), return_counts=True)
    return labels, counts


def score_change(before, after):
    """Element-by-element subtraction of two arrays (e.g. final - midterm)."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError("Both arrays must have the same length.")
    return after - before


def z_scores(values):
    """How many standard deviations each value is from the mean."""
    arr = np.asarray(values, dtype=float)
    spread = arr.std()
    if spread == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / spread


def timing_demo(size=1_000_000, seed=config.RANDOM_SEED):
    """Compare a Python loop with NumPy for summing `size` numbers."""
    data = np.random.default_rng(seed).random(size)
    as_list = data.tolist()

    start = time.perf_counter()
    total = 0.0
    for x in as_list:
        total += x
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_total = data.sum()
    numpy_time = time.perf_counter() - start

    return {
        "size": size, "loop_seconds": loop_time, "numpy_seconds": numpy_time,
        "speedup": loop_time / numpy_time if numpy_time > 0 else float("inf"),
        "same_answer": bool(np.isclose(total, numpy_total)),
    }


# ---------------------------------------------------------------------------
# Report used by the menu
# ---------------------------------------------------------------------------
def build_numpy_report(df):
    """Return a list of sections. Each section explains one NumPy operation."""
    sections = []

    # 1. Descriptive statistics ------------------------------------------
    rows = []
    for col in config.MARK_COLUMNS + ["total_marks"]:
        stats = descriptive_stats(df[col].to_numpy())
        label = config.COLUMN_LABELS.get(col, "Weighted total")
        rows.append({"component": label, **{k: v for k, v in stats.items() if k != "count"}})
    sections.append({
        "title": "1. Descriptive statistics (mean, median, min, max, std, variance)",
        "input": "One array of marks per component (one number per student).",
        "output": "A single number per statistic, per component.",
        "purpose": "Summarise a whole column of marks with a few meaningful numbers.",
        "why": "np.mean / np.median / np.std work on the whole array at once, with no loop.",
        "table": pd.DataFrame(rows),
        "lines": ["std/variance use ddof=0 (population). pandas .std() uses ddof=1, "
                  "so tiny differences between the two libraries are expected."],
    })

    # 2. Percentiles ----------------------------------------------------
    pct = percentile_summary(df["total_marks"].to_numpy())
    sections.append({
        "title": "2. Percentiles of total marks",
        "input": "Array of total marks.",
        "output": "The mark below which 10 %, 25 %, 50 %, 75 % and 90 % of students fall.",
        "purpose": "Show how scores are spread, and where a student stands relative to the class.",
        "why": "np.percentile sorts and interpolates internally - one call, many percentiles.",
        "table": pd.DataFrame({"percentile": [f"P{int(p)}" for p in pct],
                               "total_marks": list(pct.values())}),
        "lines": [],
    })

    # 3. Boolean masking -----------------------------------------------
    total = df["total_marks"].to_numpy()
    attendance = df["attendance_pct"].to_numpy()
    mask_high = (total >= 85) & (attendance >= 85)
    mask_risk = total < config.PASS_THRESHOLD
    mask_hidden = (attendance < config.LOW_ATTENDANCE_THRESHOLD) & (total >= 70)
    n = len(df)
    sections.append({
        "title": "3. Boolean masking (selecting students that match a rule)",
        "input": "Arrays of total marks and attendance.",
        "output": "An array of True/False values; .sum() counts the Trues, .mean() gives the share.",
        "purpose": "Find students meeting specific criteria without writing a loop.",
        "why": "A comparison like `total >= 85` is evaluated for every student in one step; "
               "& combines conditions element by element.",
        "table": df.loc[mask_high, ["student_id", "name", "department", "total_marks", "attendance_pct"]].head(8),
        "lines": [
            f"Total >= 85 AND attendance >= 85 : {mask_high.sum():>3} students ({mask_high.mean():.1%})",
            f"Total below pass mark ({config.PASS_THRESHOLD:g})        : {mask_risk.sum():>3} students ({mask_risk.mean():.1%})",
            f"Attendance < {config.LOW_ATTENDANCE_THRESHOLD:g}% yet total >= 70   : {mask_hidden.sum():>3} students ({mask_hidden.mean():.1%})",
            f"(Table shows up to 8 of the {n} students who meet the first rule.)",
        ],
    })

    # 4. np.unique ------------------------------------------------------
    dept_labels, dept_counts = unique_with_counts(df["department"].to_numpy())
    sem_labels, sem_counts = unique_with_counts(df["semester"].to_numpy())
    sections.append({
        "title": "4. np.unique with return_counts=True",
        "input": "Arrays of department names and semester numbers.",
        "output": "The sorted distinct values and how many times each appears.",
        "purpose": "Discover the categories in a column and how large each group is.",
        "why": "It replaces a manual loop with a dictionary of counters.",
        "table": pd.DataFrame({"department": dept_labels, "students": dept_counts}),
        "lines": ["Semesters: " + ", ".join(f"S{s}={c}" for s, c in zip(sem_labels, sem_counts))],
    })

    # 5. Array arithmetic ----------------------------------------------
    change = score_change(df["midterm_marks"], df["final_marks"])
    best = int(np.argmax(change))
    worst = int(np.argmin(change))
    z = z_scores(total)
    sections.append({
        "title": "5. Array arithmetic: final exam compared with midterm",
        "input": "Two arrays of the same length (midterm and final marks).",
        "output": "A new array where each element is final - midterm for that student.",
        "purpose": "Spot improvement or decline between assessments, and unusual totals (z-scores).",
        "why": "Subtracting arrays works element by element, so one line replaces a loop.",
        "table": None,
        "lines": [
            f"Average change (final - midterm): {change.mean():+.2f} marks",
            f"Improved: {(change > 0).sum()}   Declined: {(change < 0).sum()}   Unchanged: {(change == 0).sum()}",
            f"Biggest improvement: {df['name'].iloc[best]} ({change[best]:+.1f})",
            f"Biggest decline    : {df['name'].iloc[worst]} ({change[worst]:+.1f})",
            f"Unusual totals (|z-score| > 2): {(np.abs(z) > 2).sum()} student(s)",
        ],
    })

    # 6. Speed demo -----------------------------------------------------
    timing = timing_demo()
    sections.append({
        "title": "6. Why NumPy? A quick speed test",
        "input": f"{timing['size']:,} random numbers.",
        "output": "Time taken to add them up with a Python loop versus np.sum.",
        "purpose": "See the benefit of vectorisation with your own eyes.",
        "why": "NumPy loops in compiled C code on a compact array; Python loops run one object at a time.",
        "table": None,
        "lines": [
            f"Python for-loop : {timing['loop_seconds'] * 1000:8.2f} ms",
            f"NumPy .sum()    : {timing['numpy_seconds'] * 1000:8.3f} ms",
            f"NumPy was about {timing['speedup']:.0f}x faster (same answer: {timing['same_answer']}).",
        ],
    })
    return sections
