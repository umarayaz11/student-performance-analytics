"""Statistical charts with seaborn.

Matplotlib draws everything from basic shapes: you decide each bar and line.
seaborn sits ON TOP of Matplotlib and understands DataFrames, so one line can
produce a statistical chart that would take dozens of Matplotlib lines
(box plots, violin plots, annotated heatmaps).

    Matplotlib = the drawing engine      seaborn = statistical charts, built on it

Every function returns a Matplotlib Figure, exactly like visualizations.py, and
all charts reuse the same colour theme (visualizations.RC) so the project looks
consistent.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import config
import pandas_analysis as pa
from visualizations import FOOTER, RC, T, _finish

try:
    import seaborn as sns
except ImportError as problem:       # pragma: no cover
    raise ImportError(
        "seaborn is not installed. Activate your virtual environment and run:\n"
        "    pip install -r requirements.txt"
    ) from problem


# seaborn 0.13 still passes Matplotlib's old `vert=` option, which Matplotlib 3.10
# flags as "pending deprecation". It is harmless and comes from inside the libraries,
# so we hide ONLY this exact message (every other warning stays visible).
warnings.filterwarnings("ignore", message="vert: bool will be deprecated",
                        category=PendingDeprecationWarning)


def apply_theme():
    """seaborn's theming call, with our shared settings layered on top."""
    sns.set_theme(style="whitegrid", rc=RC,
                  palette=[T["blue"], T["teal"], T["amber"], T["coral"], T["navy"]])


def department_boxplot(df):
    """Box plot + individual students: the SPREAD of marks per department."""
    order = (df.groupby("department")["total_marks"].median()
               .sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sns.boxplot(data=df, x="total_marks", y="department", order=order, orient="h",
                color=T["blue"], width=0.55, fliersize=0, linewidth=1.2,
                boxprops={"alpha": 0.55}, ax=ax)
    sns.stripplot(data=df, x="total_marks", y="department", order=order, orient="h",
                  color=T["navy"], size=3.2, alpha=0.45, jitter=0.18, ax=ax)
    ax.axvline(config.PASS_THRESHOLD, color=T["coral"], linestyle="--", linewidth=1.4)
    ax.text(config.PASS_THRESHOLD, 1.01, f"Pass mark ({config.PASS_THRESHOLD:g})",
            transform=ax.get_xaxis_transform(), color=T["coral"], fontsize=9,
            ha="center", va="bottom")
    ax.set_xlim(20, 100)
    ax.grid(axis="y", visible=False)
    _finish(fig, ax, "Spread of Total Marks by Department",
            "Total marks (box = middle 50 % of students, line = median)", "Department",
            title_pad=26)
    return fig


def correlation_heatmap(df):
    """Heatmap: how strongly do study time, attendance and each mark move together?"""
    corr = pa.correlation_matrix(df)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)     # hide the duplicate upper half
    cmap = sns.diverging_palette(20, 240, s=75, l=50, as_cmap=True)   # coral (-) to blue (+)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                center=0, square=True, linewidths=1, linecolor="white",
                cbar_kws={"label": "Correlation (-1 to +1)", "shrink": 0.8},
                annot_kws={"fontsize": 10, "color": T["navy"]}, ax=ax)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _finish(fig, ax, "Correlation Between Study Habits and Marks", "", "")
    fig.text(0.99, 0.005, "Association, not proof of cause", fontsize=8, color=T["grey"],
             ha="right", va="bottom")
    return fig


def department_grade_heatmap(df):
    """Heatmap: how many students of each department earned each letter grade."""
    table = pa.department_grade_table(df)
    cmap = sns.light_palette(T["blue"], as_cmap=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.heatmap(table, annot=True, fmt="d", cmap=cmap, linewidths=1, linecolor="white",
                cbar_kws={"label": "Number of students", "shrink": 0.85},
                annot_kws={"fontsize": 11, "color": T["navy"]}, ax=ax)
    ax.grid(False)
    plt.setp(ax.get_yticklabels(), rotation=0)
    _finish(fig, ax, "Grades by Department", "Letter grade (example bands)", "Department")
    return fig


def assessment_violin(df):
    """Violin plot: the whole distribution of marks for each assessment component."""
    long = pa.components_long(df)
    order = [config.COLUMN_LABELS[c] for c in config.MARK_COLUMNS]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    sns.violinplot(data=long, x="component", y="marks", order=order, color=T["blue"],
                   inner="quartile", cut=0, linewidth=1.1, saturation=0.8, ax=ax)
    ax.axhline(config.PASS_THRESHOLD, color=T["coral"], linestyle="--", linewidth=1.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 105)
    _finish(fig, ax, "Distribution of Marks by Assessment Component",
            "Assessment component (dashed line = pass mark; inner lines = quartiles)",
            "Marks (out of 100)")
    return fig


CHARTS = {
    "department_boxplot.png": department_boxplot,
    "correlation_heatmap.png": correlation_heatmap,
    "department_grade_heatmap.png": department_grade_heatmap,
    "assessment_distributions.png": assessment_violin,
}
