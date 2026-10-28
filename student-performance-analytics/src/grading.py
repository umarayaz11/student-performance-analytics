"""Configurable weighted grading system.

WEIGHTED AVERAGE in one sentence: each component's mark is multiplied by how
much it "counts", and the results are added together.

    total = 0.10 * assignments + 0.10 * quizzes + 0.30 * midterm + 0.50 * final

Because every mark is on the same 0-100 scale and the weights add up to 1.0,
the total is also on a 0-100 scale. If a component were out of 20 and another
out of 100, we would have to convert them to a common scale FIRST.

The bands and thresholds in config.py are EXAMPLE rules, not an official policy.
"""

import numpy as np
import pandas as pd

import config


def validate_weights(weights):
    """Raise ValueError unless the weights are non-negative and sum to 1."""
    if not weights:
        raise ValueError("Grading weights are empty.")
    values = np.array(list(weights.values()), dtype=float)
    if np.any(values < 0):
        raise ValueError("Grading weights cannot be negative.")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError(f"Grading weights must add up to 1.0 (100%), but they add up to {values.sum():.2f}.")


def validate_marks(df, columns, max_marks=config.MAX_MARKS):
    """Raise ValueError if any mark is missing or outside 0..max_marks."""
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the data.")
        values = df[col]
        if values.isna().any():
            raise ValueError(f"'{col}' still contains missing values - clean the data first.")
        if ((values < 0) | (values > max_marks)).any():
            raise ValueError(f"'{col}' has marks outside the allowed range 0-{max_marks:g}.")


def weighted_total(df, weights=None, max_marks=config.MAX_MARKS):
    """Return a Series with each student's weighted total (0-100)."""
    weights = config.GRADE_WEIGHTS if weights is None else weights
    validate_weights(weights)
    columns = list(weights)
    validate_marks(df, columns, max_marks)

    marks = df[columns].to_numpy(dtype=float)        # shape: (students, components)
    weight_vector = np.array([weights[c] for c in columns])
    totals = marks @ weight_vector                    # one matrix product does every student at once
    return pd.Series(totals, index=df.index, name="total_marks").round(2)


def _validate_bands(bands):
    minimums = [low for low, _ in bands]
    if minimums != sorted(minimums, reverse=True):
        raise ValueError("Bands must be listed from the highest minimum to the lowest.")
    if minimums[-1] > 0:
        raise ValueError("The last band must start at 0 so that every score gets a label.")


def assign_labels(totals, bands):
    """Turn numeric totals into labels using (minimum, label) bands."""
    _validate_bands(bands)
    values = np.asarray(totals, dtype=float)
    conditions = [values >= low for low, _ in bands]      # checked top-down
    labels = np.select(conditions, [label for _, label in bands], default=bands[-1][1])
    index = totals.index if isinstance(totals, pd.Series) else None
    return pd.Series(labels, index=index)


def add_performance_columns(df, weights=None, pass_threshold=config.PASS_THRESHOLD,
                            grade_bands=None, performance_bands=None):
    """Return a COPY of `df` with total, grade, category, result and rank columns."""
    if not 0 <= pass_threshold <= config.MAX_MARKS:
        raise ValueError(f"Pass threshold must be between 0 and {config.MAX_MARKS:g}.")
    grade_bands = config.GRADE_BANDS if grade_bands is None else grade_bands
    performance_bands = config.PERFORMANCE_BANDS if performance_bands is None else performance_bands

    out = df.copy()
    out["total_marks"] = weighted_total(out, weights)
    out["grade"] = assign_labels(out["total_marks"], grade_bands)
    out["performance_category"] = assign_labels(out["total_marks"], performance_bands)
    out["result"] = np.where(out["total_marks"] >= pass_threshold, "Pass", "Fail")
    out["rank"] = out["total_marks"].rank(ascending=False, method="min").astype(int)
    return out
