import unittest

import tests.helpers as h          # noqa: F401  (makes src/ importable)
import config
import grading


class TestWeights(unittest.TestCase):
    def test_default_weights_sum_to_one(self):
        grading.validate_weights(config.GRADE_WEIGHTS)      # should not raise

    def test_weights_not_summing_to_one_are_rejected(self):
        with self.assertRaises(ValueError):
            grading.validate_weights({"a": 0.5, "b": 0.6})

    def test_negative_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            grading.validate_weights({"a": 1.5, "b": -0.5})


class TestWeightedTotal(unittest.TestCase):
    def test_known_value(self):
        # 0.10*80 + 0.10*70 + 0.30*60 + 0.50*90 = 8 + 7 + 18 + 45 = 78
        df = h.tiny_clean_students()
        self.assertAlmostEqual(grading.weighted_total(df).iloc[0], 78.0)

    def test_perfect_and_zero_students(self):
        df = h.tiny_clean_students()
        totals = grading.weighted_total(df)
        self.assertAlmostEqual(totals.iloc[2], 100.0)        # student C scored 100 everywhere

    def test_custom_weights(self):
        df = h.tiny_clean_students()
        only_final = {"assignment_marks": 0, "quiz_marks": 0, "midterm_marks": 0, "final_marks": 1}
        self.assertListEqual(grading.weighted_total(df, only_final).tolist(), df["final_marks"].tolist())

    def test_out_of_range_marks_raise(self):
        df = h.tiny_clean_students()
        df.loc[0, "final_marks"] = 101
        with self.assertRaises(ValueError):
            grading.weighted_total(df)

    def test_missing_marks_raise(self):
        df = h.tiny_clean_students().astype({"quiz_marks": float})
        df.loc[1, "quiz_marks"] = float("nan")
        with self.assertRaises(ValueError):
            grading.weighted_total(df)


class TestLabels(unittest.TestCase):
    def test_grade_boundaries(self):
        totals = [100, 85, 84.99, 75, 74.99, 65, 64.99, 50, 49.99, 0]
        expected = ["A", "A", "B", "B", "C", "C", "D", "D", "F", "F"]
        self.assertListEqual(grading.assign_labels(totals, config.GRADE_BANDS).tolist(), expected)

    def test_bands_must_be_descending(self):
        with self.assertRaises(ValueError):
            grading.assign_labels([50], [(0, "F"), (50, "D")])

    def test_last_band_must_start_at_zero(self):
        with self.assertRaises(ValueError):
            grading.assign_labels([50], [(85, "A"), (50, "D")])


class TestPerformanceColumns(unittest.TestCase):
    def test_columns_added_and_input_untouched(self):
        df = h.tiny_clean_students()
        columns_before = list(df.columns)
        result = grading.add_performance_columns(df)
        self.assertListEqual(list(df.columns), columns_before)       # original unchanged
        for col in ("total_marks", "grade", "performance_category", "result", "rank"):
            self.assertIn(col, result.columns)

    def test_pass_threshold_is_configurable(self):
        df = h.tiny_clean_students()
        strict = grading.add_performance_columns(df, pass_threshold=90)
        lenient = grading.add_performance_columns(df, pass_threshold=10)
        self.assertLess((strict["result"] == "Pass").sum(), (lenient["result"] == "Pass").sum())

    def test_invalid_pass_threshold(self):
        with self.assertRaises(ValueError):
            grading.add_performance_columns(h.tiny_clean_students(), pass_threshold=150)

    def test_rank_one_is_best_student(self):
        result = grading.add_performance_columns(h.tiny_clean_students())
        self.assertEqual(result.loc[result["rank"] == 1, "student_id"].iloc[0], "T3")


if __name__ == "__main__":
    unittest.main()
