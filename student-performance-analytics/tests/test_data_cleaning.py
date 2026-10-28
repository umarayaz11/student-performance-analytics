import unittest

import pandas as pd

import tests.helpers as h          # noqa: F401
import data_cleaning as dc


class TestCleaning(unittest.TestCase):
    def setUp(self):
        self.raw = h.tiny_messy_students()
        self.result = dc.clean_students(self.raw)
        self.clean = self.result["data"]

    def test_original_is_not_modified(self):
        pd.testing.assert_frame_equal(self.raw, h.tiny_messy_students())

    def test_duplicates_removed(self):
        self.assertEqual(self.clean["student_id"].is_unique, True)
        self.assertEqual((self.clean["student_id"] == "M2").sum(), 1)

    def test_department_labels_standardised(self):
        self.assertSetEqual(set(self.clean["department"]), {"Computer Science", "Data Science"})

    def test_gender_labels_standardised(self):
        self.assertSetEqual(set(self.clean["gender"]), {"Male", "Female"})

    def test_invalid_core_mark_removes_row(self):
        # M4 has final_marks = 105 -> treated as missing -> row cannot be repaired
        self.assertNotIn("M4", self.clean["student_id"].tolist())

    def test_missing_department_removes_row(self):
        self.assertNotIn("M6", self.clean["student_id"].tolist())

    def test_missing_assignment_is_filled_and_flagged(self):
        row = self.clean[self.clean["student_id"] == "M3"].iloc[0]
        self.assertFalse(pd.isna(row["assignment_marks"]))
        self.assertTrue(row["had_imputed_values"])

    def test_invalid_age_becomes_filled_value(self):
        row = self.clean[self.clean["student_id"] == "M5"].iloc[0]
        self.assertTrue(16 <= row["age"] <= 40)

    def test_no_missing_values_remain(self):
        self.assertEqual(int(self.clean.isnull().sum().sum()), 0)

    def test_no_invalid_values_remain(self):
        self.assertEqual(dc.count_invalid(self.clean), {})

    def test_data_types(self):
        self.assertTrue(pd.api.types.is_integer_dtype(self.clean["age"]))
        self.assertTrue(pd.api.types.is_integer_dtype(self.clean["semester"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.clean["final_marks"]))

    def test_every_decision_is_logged(self):
        text = "\n".join(self.result["log"])
        for step in ("Step 2", "Step 4", "Step 5", "Step 6", "Step 7"):
            self.assertIn(step, text)

    def test_removed_rows_have_reasons(self):
        removed = self.result["removed"]
        self.assertTrue((removed["removal_reason"].str.len() > 0).all())
        self.assertGreaterEqual(len(removed), 3)         # 1 duplicate + M4 + M6


class TestInspectionHelpers(unittest.TestCase):
    def test_invalid_mask(self):
        mask = dc.invalid_mask(pd.Series([-1, 0, 50, 100, 101, None]), 0, 100)
        self.assertListEqual(mask.tolist(), [True, False, False, False, True, False])

    def test_dataset_summary_counts_problems(self):
        summary = dc.dataset_summary(h.tiny_messy_students())
        self.assertEqual(summary["Duplicate rows"], 1)
        self.assertEqual(summary["Out-of-range values"], 3)   # age 5, attendance 120, final 105

    def test_clean_data_is_stable(self):
        """Cleaning already-clean data should not remove anything."""
        again = dc.clean_students(self.make_clean())["data"]
        self.assertEqual(len(again), 6)

    @staticmethod
    def make_clean():
        return h.tiny_clean_students()


if __name__ == "__main__":
    unittest.main()
