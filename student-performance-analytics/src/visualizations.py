"""Charts with Matplotlib.

Matplotlib vocabulary (useful in interviews):
    Figure - the whole picture/canvas.
    Axes   - one plot inside the figure (with its x-axis, y-axis, title...).
    fig, ax = plt.subplots() creates both at once.

Every chart function returns a Figure; `save_all_charts()` saves them as PNG
files and can optionally open them in windows.
"""

from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

import config

try:
    import matplotlib.pyplot as plt
except ImportError as problem:       # pragma: no cover
    raise ImportError(
        "Matplotlib is not installed. Activate your virtual environment and run:\n"
        "    pip install -r requirements.txt"
    ) from problem

T = config.THEME
FOOTER = "Synthetic practice data  |  Example grading rules, not an official policy"


# One look for every chart. Seaborn charts reuse this same dictionary.
RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": T["grid"],
    "axes.labelcolor": T["navy"],
    "axes.titlecolor": T["navy"],
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": T["grid"],
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": T["grey"],
    "ytick.color": T["grey"],
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
    "savefig.dpi": 150,
}


def apply_theme():
    """Apply the shared look to Matplotlib."""
    plt.rcParams.update(RC)


def _finish(fig, ax, title, xlabel, ylabel, title_pad=14):
    ax.set_title(title, pad=title_pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.text(0.01, 0.005, FOOTER, fontsize=8, color=T["grey"], ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))


def department_average_chart(df):
    """Bar chart: average total marks per department."""
    means = df.groupby("department")["total_marks"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(means.index, means.to_numpy(), color=T["blue"], height=0.62)
    bars[-1].set_color(T["teal"])                       # highlight the best department
    ax.axvline(config.PASS_THRESHOLD, color=T["coral"], linestyle="--", linewidth=1.4)
    ax.text(config.PASS_THRESHOLD, 1.01, f"Pass mark ({config.PASS_THRESHOLD:g})",
            transform=ax.get_xaxis_transform(), color=T["coral"], fontsize=9,
            ha="center", va="bottom")
    for bar, value in zip(bars, means):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                va="center", fontsize=10, color=T["navy"], fontweight="bold")
    ax.set_xlim(0, max(100, means.max() + 8))
    ax.grid(axis="y", visible=False)
    _finish(fig, ax, "Average Total Marks by Department",
            "Average total marks (out of 100)", "Department", title_pad=26)
    return fig


def total_marks_histogram(df):
    """Histogram: how total marks are distributed."""
    totals = df["total_marks"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(totals, bins=np.arange(0, 105, 5), color=T["blue"], edgecolor="white")
    ax.axvline(np.mean(totals), color=T["teal"], linewidth=2, label=f"Mean = {np.mean(totals):.1f}")
    ax.axvline(np.median(totals), color=T["amber"], linewidth=2, linestyle="-.",
               label=f"Median = {np.median(totals):.1f}")
    ax.axvline(config.PASS_THRESHOLD, color=T["coral"], linewidth=1.6, linestyle="--",
               label=f"Pass mark = {config.PASS_THRESHOLD:g}")
    ax.set_xlim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))     # counts are whole numbers
    ax.legend(loc="upper right")
    _finish(fig, ax, "Distribution of Total Marks", "Total marks (out of 100)", "Number of students")
    return fig


def study_hours_scatter(df):
    """Scatter plot: study hours vs total marks, with a trend line."""
    x = df["study_hours_per_week"].to_numpy()
    y = df["total_marks"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for category in config.PERFORMANCE_ORDER:
        part = df[df["performance_category"] == category]
        ax.scatter(part["study_hours_per_week"], part["total_marks"], s=42, alpha=0.85,
                   color=config.CATEGORY_COLOURS[category], edgecolor="white",
                   linewidth=0.6, label=category)
    slope, intercept = np.polyfit(x, y, 1)              # straight line y = slope*x + intercept
    line_x = np.array([x.min(), x.max()])
    ax.plot(line_x, slope * line_x + intercept, color=T["navy"], linewidth=1.8, linestyle="--")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.02, 0.96, f"Correlation r = {r:.2f}\n(association, not proof of cause)",
            transform=ax.transAxes, va="top", fontsize=9.5, color=T["navy"],
            bbox={"boxstyle": "round,pad=0.4", "facecolor": T["light"], "edgecolor": T["grid"]})
    ax.set_ylim(0, 100)
    ax.legend(title="Performance", loc="lower right", ncol=2)
    _finish(fig, ax, "Study Hours vs Total Marks", "Study hours per week", "Total marks (out of 100)")
    return fig


def performance_category_chart(df):
    """Bar chart: number of students in each performance category."""
    counts = df["performance_category"].value_counts().reindex(config.PERFORMANCE_ORDER, fill_value=0)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(counts.index, counts.to_numpy(), width=0.6,
                  color=[config.CATEGORY_COLOURS[c] for c in counts.index])
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.008,
                f"{value}  ({value / total:.0%})", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=T["navy"])
    ax.set_ylim(0, counts.max() * 1.15 + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    _finish(fig, ax, "Students by Performance Category", "Performance category", "Number of students")
    return fig


CHARTS = {
    "department_average_marks.png": department_average_chart,
    "total_marks_distribution.png": total_marks_histogram,
    "study_hours_vs_marks.png": study_hours_scatter,
    "performance_categories.png": performance_category_chart,
}


def save_all_charts(df, out_dir=config.CHARTS_DIR, show=False, include_seaborn=True):
    """Create every chart, save it as PNG, optionally show it. Returns saved paths.

    The four core charts use plain Matplotlib. With include_seaborn=True, four
    more statistical charts (see statistical_charts.py) are added.
    """
    if len(df) == 0:
        raise ValueError("There is no data to plot.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_theme()

    jobs = list(CHARTS.items())
    if include_seaborn:
        import statistical_charts          # raises a helpful ImportError if seaborn is missing
        statistical_charts.apply_theme()
        jobs += list(statistical_charts.CHARTS.items())

    saved, figures = [], []
    for filename, make_chart in jobs:
        fig = make_chart(df)
        path = out_dir / filename
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
        figures.append(fig)

    if show:
        plt.show()               # opens one window per figure; blocks until closed
    for fig in figures:
        plt.close(fig)           # free memory
    return saved
