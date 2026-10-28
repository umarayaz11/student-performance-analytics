"""Export results to CSV files.

CSV is the "universal" table format: Excel, Google Sheets, Power BI, pandas
and almost every database can open it. We write with `utf-8-sig` so Excel on
Windows displays all characters correctly, and `index=False` so pandas does not
add an extra (unwanted) row-number column.

Each file is built from a COPY of the data, so exporting can never change the
DataFrames used elsewhere in the program (no "data leakage" between steps).
"""

from pathlib import Path

import config
import pandas_analysis as pa

EXPORT_DESCRIPTIONS = {
    "cleaned_students.csv": "Cleaned dataset - use it as the trusted input for any further analysis.",
    "student_performance_report.csv": "One row per student with total, grade, category, result and rank.",
    "department_statistics.csv": "Department comparison: averages, pass rate, attendance, study hours.",
    "low_attendance_students.csv": "Students below the attendance threshold - for follow-up by advisors.",
    "top_students.csv": f"The top {config.TOP_N} students by total marks.",
    "cleaning_log.txt": "A plain-text record of every cleaning decision (for transparency).",
}


def export_reports(clean_df, performance_df, cleaning_log=None, out_dir=config.REPORTS_DIR,
                   pass_threshold=config.PASS_THRESHOLD,
                   attendance_threshold=config.LOW_ATTENDANCE_THRESHOLD):
    """Write all report files and return {filename: path}."""
    if len(performance_df) == 0:
        raise ValueError("There is no data to export.")
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as problem:
        raise OSError(f"Could not create the output folder '{out_dir}': {problem}") from problem

    tables = {
        "cleaned_students.csv": clean_df.copy(),
        "student_performance_report.csv": pa.performance_report(performance_df),
        "department_statistics.csv": pa.department_stats(performance_df, pass_threshold),
        "low_attendance_students.csv": pa.low_attendance_students(performance_df, attendance_threshold),
        "top_students.csv": pa.top_students(performance_df),
    }

    written = {}
    try:
        for filename, table in tables.items():
            path = out_dir / filename
            table.to_csv(path, index=False, encoding="utf-8-sig")
            written[filename] = path
        if cleaning_log:
            path = out_dir / "cleaning_log.txt"
            path.write_text("\n".join(cleaning_log) + "\n", encoding="utf-8")
            written["cleaning_log.txt"] = path
    except PermissionError:
        raise PermissionError(
            "Could not write a report file. If it is open in Excel, close it and try again."
        ) from None
    return written
