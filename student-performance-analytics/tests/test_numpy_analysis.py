import unittest

import numpy as np

import tests.helpers as h          # noqa: F401
import numpy_analysis as na


class TestDescriptiveStats(unittest.TestCase):
    def setUp(self):
        self.values = [10, 20, 30, 40, 50]

    def test_basic_statistics(self):
        stats = na.descriptive_stats(self.values)
        self.assertEqual(stats["mean"], 30)
        self.assertEqual(stats["median"], 30)
        self.assertEqual(stats["min"], 10)
        self.assertEqual(stats["max"], 50)
        self.assertEqual(stats["range"], 40)

    def test_population_vs_sample_variance(self):
        self.assertAlmostEqual(na.descriptive_stats(self.values, ddof=0)["variance"], 200)
        self.assertAlmostEqual(na.descriptive_stats(self.values, ddof=1)["variance"], 250)

    def test_std_is_sqrt_of_variance(self):
        stats = na.descriptive_stats(self.values)
        self.assertAlmostEqual(stats["std"], np.sqrt(stats["variance"]))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            na.descriptive_stats([])


class TestPercentiles(unittest.TestCase):
    def test_known_percentiles(self):
        result = na.percentile_summary([10, 20, 30, 40, 50], percentiles=(0, 50, 100))
        self.assertEqual(result[0], 10)
        self.assertEqual(result[50], 30)
        self.assertEqual(result[100], 50)

    def test_invalid_percentile(self):
        with self.assertRaises(ValueError):
            na.percentile_summary([1, 2, 3], percentiles=(101,))


class TestUniqueAndArithmetic(unittest.TestCase):
    def test_unique_with_counts(self):
        labels, counts = na.unique_with_counts(["CS", "DS", "CS", "EE", "CS"])
        self.assertListEqual(labels.tolist(), ["CS", "DS", "EE"])
        self.assertListEqual(counts.tolist(), [3, 1, 1])

    def test_score_change(self):
        self.assertListEqual(na.score_change([50, 60], [70, 40]).tolist(), [20, -20])

    def test_score_change_length_mismatch(self):
        with self.assertRaises(ValueError):
            na.score_change([1, 2, 3], [1, 2])

    def test_z_scores_have_mean_zero(self):
        self.assertAlmostEqual(float(na.z_scores([1, 2, 3, 4, 5]).mean()), 0.0)

    def test_z_scores_constant_data(self):
        self.assertListEqual(na.z_scores([5, 5, 5]).tolist(), [0.0, 0.0, 0.0])

    def test_boolean_mask_counts(self):
        marks = np.array([45, 50, 72, 90])
        self.assertEqual(int((marks >= 50).sum()), 3)
        self.assertAlmostEqual(float((marks >= 50).mean()), 0.75)


if __name__ == "__main__":
    unittest.main()
