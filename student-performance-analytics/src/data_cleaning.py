"""Clean the raw student data - and explain every decision.

Golden rule: NEVER change data silently. Every step appends a sentence to a
`log` list so you (and anyone reviewing your work) can see what happened and why.

Decision guide used in this module
----------------------------------
REMOVE a row when the missing/invalid information cannot be honestly guessed:
    * identity fields (student_id, name, department, semester)
    * core exam marks (midterm, final) - inventing an exam score would
      change a student's grade based on a guess
    * exact duplicate records (keeping both would count a student twice)

FILL a missing value when a reasonable estimate exists and the field is not
part of a student's identity or core grade:
    * assignments, quizzes, attendance, study hours, age
    * we use the MEDIAN of similar students (robust to outliers, unlike the mean)
    * every filled row is flagged in the `had_imputed_values` column
"""

import numpy as np
import pandas as pd

import config

TEXT_COLUMNS = ["student_id", "name", "department", "gender"]
NUMERIC_COLUMNS = [
    "age", "semester", "attendance_pct", "assignment_marks", "quiz_marks",
    "midterm_marks", "final_marks", "study_hours_per_week",
]

VALID_RANGES = {col: (config.MIN_MARKS, config.MAX_MARKS) for col in config.MARK_COLUMNS}
VALID_RANGES.update({
    "age": config.AGE_RANGE,
    "semester": config.SEMESTER_RANGE,
    "attendance_pct": config.ATTENDANCE_RANGE,
    "study_hours_per_week": config.STUDY_HOURS_RANGE,
})

# column to fill -> "group by" column whose median we use
FILL_RULES = {
    "assignment_marks": "department",
    "quiz_marks": "department",
    "attendance_pct": "department",
    "study_hours_per_week": "department",
    "age": "semester",
}

# A row missing any of these is removed (see the decision guide above).
REQUIRED_COLUMNS = ["student_id", "name", "department", "semester",
                    "midterm_marks", "final_marks"]


# ---------------------------------------------------------------------------
# Inspection helpers (do not change the data)
# ---------------------------------------------------------------------------
def invalid_mask(series, low, high):
    """True where a value exists but lies outside [low, high]."""
    values = pd.to_numeric(series, errors="coerce")
    return (values < low) | (values > high)      # NaN comparisons are False


def count_invalid(df):
    """Number of out-of-range values per column (only columns with problems)."""
    counts = {}
    for col, (low, high) in VALID_RANGES.items():
        if col in df.columns:
            n_bad = int(invalid_mask(df[col], low, high).sum())
            if n_bad:
                counts[col] = n_bad
    return counts


def _label_key(value):
    """'  Comp. Sci ' -> 'comp sci'  (lower-case, no full stops, single spaces)."""
    return " ".join(str(value).lower().replace(".", " ").split())


def dataset_summary(df):
    """Key data-quality numbers, used for the before/after comparison."""
    return {
        "Rows": len(df),
        "Missing values (all cells)": int(df.isnull().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Duplicate student IDs": int(df["student_id"].duplicated().sum()),
        "Out-of-range values": int(sum(count_invalid(df).values())),
        "Distinct department labels": int(df["department"].nunique()),
        "Distinct gender labels": int(df["gender"].nunique()),
    }


def before_after_table(before_df, after_df):
    """Side-by-side DataFrame of dataset_summary() for two versions of the data."""
    before, after = dataset_summary(before_df), dataset_summary(after_df)
    return pd.DataFrame({
        "Check": list(before.keys()),
        "Before": list(before.values()),
        "After": list(after.values()),
    })


def missing_report(df):
    """Missing values per column (only columns that have any)."""
    counts = df.isnull().sum()
    counts = counts[counts > 0]
    return counts.rename_axis("column").reset_index(name="missing")


# ---------------------------------------------------------------------------
# The cleaning pipeline
# ---------------------------------------------------------------------------
def clean_students(df):
    """Clean a raw student DataFrame.

    Returns a dictionary with:
        "data"    - the cleaned DataFrame (a NEW object; `df` is untouched)
        "log"     - list of sentences explaining every decision
        "removed" - DataFrame of the rows that were removed, with a reason
    """
    log = []
    removed_parts = []
    out = df.copy()          # work on a copy: the original stays unchanged

    # --- Step 1: tidy text ------------------------------------------------
    for col in TEXT_COLUMNS:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace("", pd.NA)
    log.append("Step 1: Removed leading/trailing spaces from text columns.")

    # --- Step 2: standardise category labels ------------------------------
    dept_lookup = {_label_key(d): d for d in config.DEPARTMENTS}
    dept_lookup.update(config.DEPARTMENT_ALIASES)
    original = out["department"].astype(object)
    mapped = original.map(lambda v: dept_lookup.get(_label_key(v)), na_action="ignore")
    changed = int((original.notna() & mapped.notna() & (original != mapped)).sum())
    unmapped = original[original.notna() & mapped.isna()]
    out["department"] = mapped
    log.append(
        f"Step 2: Standardised department labels - {changed} value(s) changed "
        f"(e.g. 'cs' / 'COMPUTER SCIENCE' -> 'Computer Science')."
    )
    if len(unmapped):
        log.append(
            f"        {len(unmapped)} department label(s) were not recognised "
            f"({', '.join(sorted(set(unmapped)))}) and are treated as missing."
        )

    gender_original = out["gender"].astype(object)
    gender_mapped = gender_original.map(
        lambda v: config.GENDER_ALIASES.get(_label_key(v)), na_action="ignore"
    )
    gender_changed = int((gender_original.notna() & gender_mapped.notna()
                          & (gender_original != gender_mapped)).sum())
    out["gender"] = gender_mapped.fillna("Unknown")
    log.append(f"        Standardised gender labels - {gender_changed} value(s) changed; "
               "unrecognised or missing genders set to 'Unknown'.")

    # --- Step 3: make numeric columns numeric -----------------------------
    for col in NUMERIC_COLUMNS:
        before_missing = int(out[col].isna().sum())
        out[col] = pd.to_numeric(out[col], errors="coerce")
        extra = int(out[col].isna().sum()) - before_missing
        if extra:
            log.append(f"Step 3: {extra} non-numeric value(s) in '{col}' could not be "
                       "converted and are now missing.")
    log.append("Step 3: Converted numeric columns with pd.to_numeric().")

    # --- Step 4: remove duplicate records ---------------------------------
    out = out.dropna(subset=["student_id"])
    exact = int(out.duplicated().sum())
    dup_mask = out.duplicated(subset="student_id", keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        removed_parts.append(out[dup_mask].assign(removal_reason="Duplicate student_id"))
        out = out[~dup_mask]
        log.append(
            f"Step 4: Removed {n_dup} duplicate record(s) ({exact} exact copies). "
            "We keep the FIRST occurrence because a student must be counted once."
        )
    else:
        log.append("Step 4: No duplicate records found.")

    # --- Step 5: invalid values -> missing --------------------------------
    for col, (low, high) in VALID_RANGES.items():
        bad = invalid_mask(out[col], low, high)
        if bad.any():
            examples = ", ".join(
                f"{sid}={val:g}" for sid, val in
                zip(out.loc[bad, "student_id"].head(3), out.loc[bad, col].head(3))
            )
            out.loc[bad, col] = np.nan
            log.append(
                f"Step 5: {int(bad.sum())} impossible value(s) in '{col}' "
                f"(valid range {low:g}-{high:g}; e.g. {examples}) were treated as missing. "
                "We do not clip them because we cannot know the true value."
            )
    if not any(line.startswith("Step 5") for line in log):
        log.append("Step 5: No out-of-range values found.")

    # --- Step 6: remove rows that cannot be repaired ----------------------
    for col in REQUIRED_COLUMNS:
        n_missing = int(out[col].isna().sum())
        if n_missing:
            log.append(f"Step 6: {n_missing} row(s) have no usable '{col}'.")
    unrepairable = out[REQUIRED_COLUMNS].isna().any(axis=1)
    if unrepairable.any():
        reason = out.loc[unrepairable, REQUIRED_COLUMNS].isna().apply(
            lambda row: "Missing: " + ", ".join(row.index[row]), axis=1
        )
        removed_parts.append(out[unrepairable].assign(removal_reason=reason))
        ids = ", ".join(out.loc[unrepairable, "student_id"].head(8))
        log.append(
            f"        Removed {int(unrepairable.sum())} row(s) in total (IDs: {ids}"
            f"{'...' if unrepairable.sum() > 8 else ''}). Identity fields and exam marks "
            "cannot be guessed without affecting a student's result."
        )
        out = out[~unrepairable]
    else:
        log.append("Step 6: No rows needed removal for missing key fields.")

    # --- Step 7: fill remaining gaps with group medians -------------------
    out = out.copy()
    needs_fill = out[list(FILL_RULES)].isna()
    out["had_imputed_values"] = needs_fill.any(axis=1)
    for col, group_col in FILL_RULES.items():
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue
        group_median = out.groupby(group_col)[col].transform("median")
        out[col] = out[col].fillna(group_median).fillna(out[col].median())
        log.append(
            f"Step 7: Filled {n_missing} missing '{col}' value(s) with the median of "
            f"the student's {group_col} group (median ignores extreme values)."
        )
    log.append(f"        {int(out['had_imputed_values'].sum())} student(s) are flagged "
               "in the 'had_imputed_values' column.")

    # --- Step 8: final data types -----------------------------------------
    out["age"] = out["age"].round().astype(int)
    out["semester"] = out["semester"].astype(int)
    for col in ["attendance_pct", "study_hours_per_week"] + config.MARK_COLUMNS:
        out[col] = out[col].astype(float).round(1)
    for col in TEXT_COLUMNS:
        out[col] = out[col].astype(str)
    log.append("Step 8: Set data types (age & semester -> int, marks -> float, text -> str).")

    out = out.sort_values("student_id").reset_index(drop=True)
    removed = (pd.concat(removed_parts) if removed_parts
               else pd.DataFrame(columns=list(df.columns) + ["removal_reason"]))
    return {"data": out, "log": log, "removed": removed.reset_index(drop=True)}
