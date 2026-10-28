import tempfile
import unittest
from pathlib import Path

import pandas as pd

import tests.helpers as h          # noqa: F401
import config
import data_cleaning as dc
import data_generator as dg
import data_loader as dl
import grading
import pandas_analysis as pa
import report_export as rx


def tiny_performance():
    return grading.add_performance_columns(h.tiny_clean_students())


class TestPandasAnalysis(unittest.TestCase):
    def setUp(self):
        self.df = tiny_performance()

    def test_top_and_bottom_students(self):
        top, bottom = pa.top_students(self.df, 2), pa.bottom_students(self.df, 2)
        self.assertEqual(top["student_id"].tolist()[0], "T3")        # perfect scores
        self.assertEqual(bottom["student_id"].tolist()[0], "T4")      # lowest scores
        self.assertEqual(len(top), 2)

    def test_department_stats(self):
        stats = pa.department_stats(self.df).set_index("department")
        self.assertEqual(stats.loc["Computer Science", "students"], 3)
        self.assertEqual(stats.loc["Data Science", "students"], 3)
        expected = self.df.loc[self.df["department"] == "Data Science", "total_marks"].mean()
        self.assertAlmostEqual(stats.loc["Data Science", "avg_total"], round(expected, 2))
        self.assertIn("faculty", stats.columns)                       # produced by merge()

    def test_pass_fail_percentages_add_up(self):
        summary = pa.pass_fail_summary(self.df, 50)
        self.assertEqual(summary["passed"] + summary["failed"], len(self.df))
        self.assertAlmostEqual(summary["pass_pct"] + summary["fail_pct"], 100.0)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            pa.pass_fail_summary(self.df, 120)

    def test_low_attendance(self):
        low = pa.low_attendance_students(self.df, 75)
        self.assertListEqual(sorted(low["student_id"]), ["T2", "T4"])  # 60 % and 70 %

    def test_high_attendance_low_performance(self):
        # T1..T6 attendance >= 85 are T1 (95), T3 (88), T5 (90) - none score below 55
        self.assertEqual(len(pa.high_attendance_low_performance(self.df, 85, 55)), 0)
        self.assertGreater(len(pa.high_attendance_low_performance(self.df, 85, 95)), 0)

    def test_category_counts_cover_all_students(self):
        self.assertEqual(pa.performance_category_counts(self.df)["students"].sum(), len(self.df))

    def test_study_association_is_positive_for_sample(self):
        result = pa.study_hours_association(self.df)
        self.assertGreater(result["pearson"], 0.5)
        self.assertIn("NOT prove", result["caution"])


class TestSeabornHelpers(unittest.TestCase):
    def setUp(self):
        self.df = tiny_performance()

    def test_correlation_matrix_is_square_symmetric_with_unit_diagonal(self):
        corr = pa.correlation_matrix(self.df)
        self.assertEqual(corr.shape, (6, 6))
        self.assertTrue((corr.values.diagonal() == 1.0).all())
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_correlation_matrix_excludes_calculated_total(self):
        self.assertNotIn("total_marks", pa.correlation_matrix(self.df).columns)

    def test_department_grade_table_counts_every_student_once(self):
        table = pa.department_grade_table(self.df)
        self.assertEqual(int(table.values.sum()), len(self.df))
        self.assertListEqual(list(table.columns), ["A", "B", "C", "D", "F"])

    def test_components_long_has_one_row_per_student_and_component(self):
        long = pa.components_long(self.df)
        self.assertEqual(len(long), len(self.df) * 4)
        self.assertSetEqual(set(long["component"]),
                            {"Assignments", "Quizzes", "Midterm", "Final Exam"})


class TestGeneratorAndLoader(unittest.TestCase):
    def test_same_seed_same_data(self):
        pd.testing.assert_frame_equal(dg.generate_students(seed=7), dg.generate_students(seed=7))

    def test_different_seed_different_data(self):
        self.assertFalse(dg.generate_students(seed=1).equals(dg.generate_students(seed=2)))

    def test_at_least_100_records_and_expected_columns(self):
        df = dg.generate_students()
        self.assertGreaterEqual(len(df), 100)
        self.assertListEqual(list(df.columns), config.COLUMNS)

    def test_messy_data_really_is_messy(self):
        summary = dc.dataset_summary(dg.generate_students())
        self.assertGreater(summary["Missing values (all cells)"], 0)
        self.assertGreater(summary["Duplicate rows"], 0)
        self.assertGreater(summary["Out-of-range values"], 0)

    def test_load_missing_file_has_helpful_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            dl.load_students(Path("definitely_not_here.csv"))
        self.assertIn("generate", str(ctx.exception).lower())

    def test_load_empty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            empty = Path(folder) / "empty.csv"
            empty.write_text("")
            with self.assertRaises(ValueError):
                dl.load_students(empty)

    def test_load_file_with_missing_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.csv"
            bad.write_text("student_id,name\nS1,Ann\n")
            with self.assertRaises(ValueError) as ctx:
                dl.load_students(bad)
            self.assertIn("missing required column", str(ctx.exception))

    def test_save_and_reload_round_trip(self):
        df = dg.generate_students(n=30, seed=3)
        with tempfile.TemporaryDirectory() as folder:
            path = dg.save_students_csv(df, Path(folder) / "nested" / "students.csv")
            reloaded = dl.load_students(path)
        self.assertEqual(reloaded.shape, df.shape)


class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_and_export(self):
        raw = dg.generate_students()
        raw_copy = raw.copy()
        result = dc.clean_students(raw)
        performance = grading.add_performance_columns(result["data"])

        pd.testing.assert_frame_equal(raw, raw_copy)                 # no leakage into raw data
        self.assertEqual(int(performance["total_marks"].isna().sum()), 0)
        self.assertTrue(performance["total_marks"].between(0, 100).all())

        with tempfile.TemporaryDirectory() as folder:
            written = rx.export_reports(result["data"], performance, result["log"], out_dir=folder)
            self.assertEqual(set(written), set(rx.EXPORT_DESCRIPTIONS))
            for path in written.values():
                self.assertTrue(path.exists())
            reloaded = pd.read_csv(written["student_performance_report.csv"])
            self.assertEqual(len(reloaded), len(performance))


if __name__ == "__main__":
    unittest.main()
