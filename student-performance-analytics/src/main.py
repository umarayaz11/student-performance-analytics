"""Student Performance & Data Analytics System - command-line application.

Run from the project folder:
    python src/main.py                 # interactive menu
    python src/main.py --pipeline      # run everything once, no questions
    python src/main.py --regenerate    # create a fresh synthetic dataset first
    python src/main.py --debug         # show full error tracebacks

This file only handles the MENU and printing. All real work happens in the
other modules (data_loader, data_cleaning, grading, numpy_analysis, ...).
"""

import argparse
import sys
import textwrap

import config
import data_cleaning as dc
import data_generator as dg
import data_loader as dl
import grading
import numpy_analysis as na
import pandas_analysis as pa
import report_export as rx
import ui

DEBUG = False

# Shorter column titles so tables fit in a normal terminal window.
SHORT = {
    "student_id": "id", "semester": "sem", "attendance_pct": "att_%",
    "assignment_marks": "assign", "quiz_marks": "quiz", "midterm_marks": "mid",
    "final_marks": "final", "study_hours_per_week": "study_h", "total_marks": "total",
    "performance_category": "category", "avg_total": "avg", "median_total": "median",
    "min_total": "min", "max_total": "max", "pass_rate_pct": "pass_%",
    "avg_attendance": "att_%", "avg_study_hours": "study_h", "average_mark": "average",
    "weight_pct": "weight_%", "share_pct": "share_%", "age": "age",
}
LEGEND = "att_% = attendance %, study_h = study hours per week, sem = semester"


def new_state():
    return {
        "raw": None, "clean": None, "log": [], "removed": None, "performance": None,
        "settings": {
            "weights": dict(config.GRADE_WEIGHTS),
            "pass_threshold": config.PASS_THRESHOLD,
            "low_attendance": config.LOW_ATTENDANCE_THRESHOLD,
            "high_attendance": config.HIGH_ATTENDANCE_THRESHOLD,
            "low_performance": config.LOW_PERFORMANCE_THRESHOLD,
        },
    }


def show(df, columns=None, digits=2, **kwargs):
    """Print a DataFrame using short column titles."""
    if columns is not None:
        df = df[columns]
    ui.show_table(df.rename(columns=SHORT), float_digits=digits, **kwargs)


# ---------------------------------------------------------------------------
# Getting data into the program
# ---------------------------------------------------------------------------
def generate_dataset(state):
    data = dg.generate_students()
    path = dg.save_students_csv(data)
    ui.success(f"Generated {len(data)} synthetic student records (seed={config.RANDOM_SEED}).")
    ui.info(f"Saved to {path}")
    state.update(raw=data, clean=None, log=[], removed=None, performance=None)


def ensure_raw(state, interactive=True):
    """Make sure the raw CSV is loaded. Offer to create it if it is missing."""
    if state["raw"] is not None:
        return True
    try:
        state["raw"] = dl.load_students()
        ui.success(f"Loaded {len(state['raw'])} rows from {config.RAW_CSV.name}")
        return True
    except FileNotFoundError as problem:
        ui.warning(str(problem))
        if not interactive or ui.ask("Generate the dataset now? (y/n)", ["y", "n"], "y").lower() == "y":
            generate_dataset(state)
            state["raw"] = dl.load_students()
            return True
        return False


def run_cleaning(state):
    result = dc.clean_students(state["raw"])
    state["clean"], state["log"], state["removed"] = result["data"], result["log"], result["removed"]
    state["performance"] = None


def ensure_clean(state):
    if not ensure_raw(state):
        return False
    if state["clean"] is None:
        ui.info("The data has not been cleaned yet - cleaning it first (see option 3 for details).")
        run_cleaning(state)
    return True


def ensure_performance(state):
    if not ensure_clean(state):
        return False
    if state["performance"] is None:
        s = state["settings"]
        state["performance"] = grading.add_performance_columns(
            state["clean"], weights=s["weights"], pass_threshold=s["pass_threshold"])
    return True


def print_log(lines):
    hang = " " * 10                      # lines up with the text after "Step N: "
    for line in lines:
        start = "  " if line.startswith("Step") else hang
        print(textwrap.fill(line.strip(), width=ui.WIDTH, initial_indent=start,
                            subsequent_indent=hang))


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------
def action_dataset_info(state):
    ui.header("1. Dataset information")
    if not ensure_raw(state):
        return
    df = state["raw"]
    facts = dl.overview(df)
    ui.show_key_values([
        ("File", config.RAW_CSV),
        ("Rows x columns", f"{facts['rows']} x {facts['columns']}"),
        ("Memory used", f"{facts['memory_kb']} KB"),
        ("Column names", ", ".join(facts["column_names"])),
    ], key_width=16)

    ui.subheader("First five records (shown in two blocks to fit the screen)")
    show(df.head(), ["student_id", "name", "age", "gender", "department", "semester"], digits=0)
    print()
    show(df.head(), ["student_id", "attendance_pct", "assignment_marks", "quiz_marks",
                     "midterm_marks", "final_marks", "study_hours_per_week"], digits=1)

    ui.subheader("Data types and missing values per column")
    ui.show_table(dl.dtypes_table(df))

    ui.subheader("Summary statistics (numeric columns)")
    stats = dl.summary_statistics(df)
    stats["count"] = stats["count"].astype(int)
    ui.show_table(stats)
    ui.note("count < 150 means some values are missing in that column.")

    ui.subheader("Series vs DataFrame")
    one, table = dl.series_vs_dataframe_demo(df)
    ui.show_key_values([
        ("df['final_marks']", f"{type(one).__name__}, shape {one.shape}   <- ONE column"),
        ("df[['name','final_marks']]", f"{type(table).__name__}, shape {table.shape}   <- a TABLE"),
    ], key_width=28)
    ui.note("A Series is one labelled column (like a 1-D NumPy array with an index).")
    ui.note("A DataFrame is a table: many Series that share the same row index.")
    ui.note("Single brackets give a Series; a list inside brackets gives a DataFrame.")


def action_view_records(state):
    ui.header("2. View student records")
    if state["performance"] is not None:
        df, columns = state["performance"], ["student_id", "name", "department", "semester",
                                             "attendance_pct", "total_marks", "grade", "result"]
        ui.info("Showing the CLEANED data with calculated totals.")
    elif state["clean"] is not None:
        df, columns = state["clean"], ["student_id", "name", "department", "semester",
                                       "attendance_pct", "midterm_marks", "final_marks"]
        ui.info("Showing the CLEANED data.")
    else:
        if not ensure_raw(state):
            return
        df, columns = state["raw"], ["student_id", "name", "department", "semester",
                                     "attendance_pct", "midterm_marks", "final_marks"]
        ui.warning("Showing the RAW data (not cleaned yet) - expect messy labels and gaps.")

    ui.menu("View by", [(1, "First N records"), (2, "Search by name or student ID"),
                        (3, "Filter by department"), (4, "Filter by semester")])
    choice = ui.ask("Choose 1-4", ["1", "2", "3", "4"], "1")

    if choice == "1":
        n = ui.ask_int("How many records", 1, 100, 10)
        result = df.head(n)
    elif choice == "2":
        text = ui.ask("Name or ID contains").strip()
        if not text:
            ui.warning("Nothing to search for.")
            return
        mask = (df["name"].astype(str).str.contains(text, case=False, regex=False)
                | df["student_id"].astype(str).str.contains(text, case=False, regex=False))
        result = df[mask]
    elif choice == "3":
        departments = sorted(df["department"].dropna().astype(str).str.strip().unique())
        for i, name in enumerate(departments, 1):
            print(f"  {i:>2}. {name}")
        pick = ui.ask_int("Department number", 1, len(departments))
        chosen = departments[pick - 1]
        result = df[df["department"].astype(str).str.strip() == chosen]
    else:
        result = df[df["semester"] == ui.ask_int("Semester", 1, 8)]

    ui.subheader(f"{len(result)} record(s) found" + (" - showing the first 25" if len(result) > 25 else ""))
    show(result.head(25), columns, digits=2 if "total_marks" in columns else 1)
    ui.note(LEGEND)


def action_clean(state):
    ui.header("3. Clean the dataset")
    if not ensure_raw(state):
        return
    raw = state["raw"]
    run_cleaning(state)
    ui.subheader("What happened, step by step")
    print_log(state["log"])

    ui.subheader("Before vs after")
    ui.show_table(dc.before_after_table(raw, state["clean"]))

    removed = state["removed"]
    if len(removed):
        ui.subheader(f"Removed records ({len(removed)}) - nothing is deleted without a reason")
        show(removed, ["student_id", "name", "removal_reason"])
    ui.success("Cleaning finished. The original CSV file and the raw data in memory were NOT modified.")


def action_numpy(state):
    ui.header("4. NumPy statistical analysis")
    if not ensure_performance(state):
        return
    for section in na.build_numpy_report(state["performance"]):
        ui.subheader(section["title"])
        for label in ("input", "output", "purpose", "why"):
            ui.note(f"{label.capitalize():<8}: {section[label]}", hang=10)
        print()
        if section["table"] is not None:
            show(section["table"])
        for line in section["lines"]:
            print(textwrap.fill(line, width=ui.WIDTH, initial_indent="  ", subsequent_indent="    "))


def action_reports(state):
    while True:
        ui.menu("5. Student performance reports", [
            (1, f"Top {config.TOP_N} students"), (2, f"Bottom {config.TOP_N} students"),
            (3, "Pass / fail summary and grade distribution"),
            (4, "Students with low attendance"),
            (5, "High attendance but low performance"),
            (6, "Compare assessment components"),
            (7, "Study hours and marks (association)"),
            (8, "Settings: grading weights and thresholds"),
            (0, "Back to main menu"),
        ])
        choice = ui.ask("Choose 0-8", [str(i) for i in range(9)], "0")
        if choice in ("0", "exit"):
            return
        if choice == "8":
            action_settings(state)
            continue
        if not ensure_performance(state):
            return
        df, s = state["performance"], state["settings"]
        guard(report_view, choice, df, s)
        ui.pause()


def report_view(choice, df, s):
    if choice == "1":
        ui.subheader(f"Top {config.TOP_N} students by total marks")
        show(pa.top_students(df), ["rank", "student_id", "name", "department", "total_marks", "grade"])
    elif choice == "2":
        ui.subheader(f"Bottom {config.TOP_N} students by total marks")
        show(pa.bottom_students(df), ["rank", "student_id", "name", "department", "total_marks", "grade"])
    elif choice == "3":
        summary = pa.pass_fail_summary(df, s["pass_threshold"])
        ui.subheader(f"Pass / fail (example pass mark: {summary['threshold']:g})")
        ui.show_key_values([
            ("Students", summary["students"]),
            ("Passed", f"{summary['passed']}  ({summary['pass_pct']:.1f}%)"),
            ("Failed", f"{summary['failed']}  ({summary['fail_pct']:.1f}%)"),
        ])
        ui.subheader("Grade distribution")
        show(pa.grade_distribution(df))
        ui.subheader("Performance categories")
        show(pa.performance_category_counts(df))
        ui.note("Grade bands and pass mark are example rules, not an official policy.")
    elif choice == "4":
        low = pa.low_attendance_students(df, s["low_attendance"])
        ui.subheader(f"{len(low)} student(s) with attendance below {s['low_attendance']:g}%")
        show(low.head(25), ["student_id", "name", "department", "attendance_pct", "total_marks", "result"])
    elif choice == "5":
        hits = pa.high_attendance_low_performance(df, s["high_attendance"], s["low_performance"])
        ui.subheader(f"{len(hits)} student(s) with attendance >= {s['high_attendance']:g}% "
                     f"but total < {s['low_performance']:g}")
        show(hits, ["student_id", "name", "department", "attendance_pct", "study_hours_per_week", "total_marks"])
        ui.note("Possible reasons to explore: study methods, prior knowledge, or other support needs.")
    elif choice == "6":
        ui.subheader("Average mark per assessment component")
        show(pa.assessment_averages(df, s["weights"]))
        ui.note("weight_% shows how much each component counts towards the total.")
    elif choice == "7":
        result = pa.study_hours_association(df)
        ui.subheader("Study hours vs total marks")
        ui.show_key_values([
            ("Pearson correlation", f"{result['pearson']:+.3f}"),
            ("Spearman correlation", f"{result['spearman']:+.3f}"),
            ("Reading", f"a {result['strength']} {result['direction']} association"),
        ])
        print()
        show(result["by_bucket"])
        ui.subheader("Correlation matrix (every pair of study habits and marks)")
        corr = pa.correlation_matrix(df).rename_axis("feature").reset_index()
        show(corr)
        ui.note("The same matrix is drawn as a seaborn heatmap in menu option 7.")
        ui.warning(result["caution"])
    ui.note(LEGEND)


def action_settings(state):
    s = state["settings"]
    ui.header("Settings: grading weights and thresholds (example rules)")
    ui.show_key_values(
        [(config.COLUMN_LABELS[c] + " weight", f"{w * 100:g}%") for c, w in s["weights"].items()]
        + [("Pass mark", s["pass_threshold"]), ("Low attendance below", s["low_attendance"]),
           ("High attendance from", s["high_attendance"]), ("Low performance below", s["low_performance"])])
    ui.menu("Change", [(1, "Grading weights"), (2, "Thresholds"), (3, "Reset to defaults"), (0, "Back")])
    choice = ui.ask("Choose 0-3", ["0", "1", "2", "3"], "0")
    if choice == "1":
        new_weights = {}
        for col in config.MARK_COLUMNS:
            new_weights[col] = ui.ask_float(f"{config.COLUMN_LABELS[col]} weight in %", 0, 100,
                                            s["weights"][col] * 100) / 100
        try:
            grading.validate_weights(new_weights)
        except ValueError as problem:
            ui.error(f"{problem} Weights were not changed.")
            return
        s["weights"] = new_weights
        state["performance"] = None
        ui.success("Weights updated. Totals will be recalculated.")
    elif choice == "2":
        s["pass_threshold"] = ui.ask_float("Pass mark", 0, 100, s["pass_threshold"])
        s["low_attendance"] = ui.ask_float("Low attendance below (%)", 0, 100, s["low_attendance"])
        s["high_attendance"] = ui.ask_float("High attendance from (%)", 0, 100, s["high_attendance"])
        s["low_performance"] = ui.ask_float("Low performance below (marks)", 0, 100, s["low_performance"])
        state["performance"] = None
        ui.success("Thresholds updated.")
    elif choice == "3":
        state["settings"] = new_state()["settings"]
        state["performance"] = None
        ui.success("Settings reset to the defaults in config.py.")


def action_departments(state):
    ui.header("6. Department performance")
    if not ensure_performance(state):
        return
    df, s = state["performance"], state["settings"]
    dept = pa.department_stats(df, s["pass_threshold"])
    ui.subheader("Department statistics (best average first)")
    show(dept, ["department", "students", "avg_total", "median_total", "pass_rate_pct",
                "avg_attendance", "avg_study_hours"])
    best, worst = dept.iloc[0], dept.iloc[-1]
    print()
    ui.info(f"Highest average: {best['department']} ({best['avg_total']:.1f}); "
            f"lowest: {worst['department']} ({worst['avg_total']:.1f}).")
    ui.note("Small groups can differ by chance - do not over-interpret small gaps.")
    ui.subheader("Students per department")
    show(pa.department_counts(df), digits=1)
    ui.subheader("Grades by department (pd.crosstab)")
    ui.show_table(pa.department_grade_table(df).reset_index(), float_digits=0)
    ui.subheader("Statistics by semester")
    show(pa.semester_stats(df, s["pass_threshold"]))
    ui.note(LEGEND)


def action_visualizations(state):
    ui.header("7. Visualizations")
    if not ensure_performance(state):
        return
    import visualizations as viz      # imported here so the menu still works without Matplotlib
    show_windows = ui.ask("Also open the charts in windows? (y/n)", ["y", "n"], "n").lower() == "y"
    try:
        paths = viz.save_all_charts(state["performance"], show=show_windows)
    except ImportError as problem:
        ui.error(str(problem))
        return
    ui.success(f"Saved {len(paths)} charts to {config.CHARTS_DIR}")
    ui.note("First 4: Matplotlib (visualizations.py). Last 4: seaborn (statistical_charts.py).")
    for path in paths:
        print(f"    • {path.name}")


def action_export(state):
    ui.header("8. Export reports")
    if not ensure_performance(state):
        return
    s = state["settings"]
    written = rx.export_reports(state["clean"], state["performance"], state["log"],
                                pass_threshold=s["pass_threshold"],
                                attendance_threshold=s["low_attendance"])
    ui.success(f"Wrote {len(written)} files to {config.REPORTS_DIR}")
    for name, path in written.items():
        size = path.stat().st_size / 1024
        print(f"    • {name:<34} {size:>6.1f} KB")
        ui.note(f"      {rx.EXPORT_DESCRIPTIONS[name]}")
    ui.note("Open the CSV files in Excel or Google Sheets, or reload them with pandas.read_csv().")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def guard(function, *args):
    """Run a menu action; show friendly messages for expected problems."""
    try:
        function(*args)
    except (FileNotFoundError, ValueError, PermissionError, OSError, ImportError) as problem:
        if DEBUG:
            raise
        ui.error(str(problem))
    except Exception as problem:     # noqa: BLE001 - last line of defence for a menu app
        if DEBUG:
            raise
        ui.error(f"Unexpected problem ({type(problem).__name__}): {problem}")
        ui.note("Run with  python src/main.py --debug  to see the full traceback.")


MENU = [
    ("1", "Display dataset information", action_dataset_info),
    ("2", "View student records", action_view_records),
    ("3", "Clean the dataset", action_clean),
    ("4", "Perform NumPy statistical analysis", action_numpy),
    ("5", "Generate student performance reports", action_reports),
    ("6", "Analyze department performance", action_departments),
    ("7", "Display visualizations", action_visualizations),
    ("8", "Export reports", action_export),
    ("9", "Exit", None),
]


def run_menu(state):
    ui.banner()
    while True:
        ui.menu("Main menu", [(key, label) for key, label, _ in MENU])
        ui.note("First run? Try 1, 3, 4, 5, 7, 8 in order.  (g = new dataset)")
        choice = ui.ask("Choose 1-9", [key for key, _, _ in MENU] + ["g"]).lower()
        if choice in ("9", "exit"):
            ui.success("Goodbye - keep practising!")
            return
        if choice == "g":
            guard(generate_dataset, state)
        else:
            action = next(func for key, _, func in MENU if key == choice)
            guard(action, state)
        ui.pause()


def run_pipeline(state):
    """Run every stage once without asking questions (useful for a quick check)."""
    import visualizations as viz
    ui.banner()
    ui.header("Automatic pipeline")
    if not config.RAW_CSV.exists():
        generate_dataset(state)
    ensure_raw(state, interactive=False)
    run_cleaning(state)
    ui.success(f"Cleaned: {len(state['raw'])} raw rows -> {len(state['clean'])} clean rows")
    ensure_performance(state)
    df = state["performance"]
    summary = pa.pass_fail_summary(df, state["settings"]["pass_threshold"])
    ui.success(f"Average total {df['total_marks'].mean():.1f}; pass rate {summary['pass_pct']:.1f}%")
    charts = viz.save_all_charts(df)
    ui.success(f"Saved {len(charts)} charts -> {config.CHARTS_DIR}")
    written = rx.export_reports(state["clean"], df, state["log"])
    ui.success(f"Exported {len(written)} files -> {config.REPORTS_DIR}")


def main():
    global DEBUG
    parser = argparse.ArgumentParser(description="Student Performance & Data Analytics System")
    parser.add_argument("--pipeline", action="store_true", help="run all stages once, then exit")
    parser.add_argument("--regenerate", action="store_true", help="create a fresh synthetic dataset first")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    args = parser.parse_args()
    DEBUG = args.debug

    state = new_state()
    if args.regenerate:
        generate_dataset(state)
    if args.pipeline:
        run_pipeline(state)
    else:
        run_menu(state)


if __name__ == "__main__":
    sys.exit(main())
