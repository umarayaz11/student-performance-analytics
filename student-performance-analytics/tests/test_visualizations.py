import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # draw to files only: no window needed for tests

import matplotlib.pyplot as plt     # noqa: E402

import tests.helpers as h          # noqa: E402,F401
import data_cleaning as dc          # noqa: E402
import data_generator as dg         # noqa: E402
import grading                      # noqa: E402
import statistical_charts as sc     # noqa: E402
import visualizations as viz        # noqa: E402


def performance():
    clean = dc.clean_students(dg.generate_students())["data"]
    return grading.add_performance_columns(clean)


class TestCharts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = performance()

    def tearDown(self):
        plt.close("all")

    def test_all_eight_charts_are_saved_and_not_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = viz.save_all_charts(self.df, folder)
            self.assertEqual(len(paths), 8)
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 5_000)

    def test_core_charts_work_without_seaborn_charts(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(viz.save_all_charts(self.df, folder, include_seaborn=False)), 4)

    def test_every_chart_has_a_title(self):
        viz.apply_theme()
        sc.apply_theme()
        for make_chart in list(viz.CHARTS.values()) + list(sc.CHARTS.values()):
            ax = make_chart(self.df).axes[0]
            self.assertTrue(ax.get_title(loc="left"))

    def test_non_heatmap_charts_have_axis_labels(self):
        viz.apply_theme()
        sc.apply_theme()
        for make_chart in list(viz.CHARTS.values()) + [sc.department_boxplot, sc.assessment_violin]:
            ax = make_chart(self.df).axes[0]
            self.assertTrue(ax.get_xlabel())
            self.assertTrue(ax.get_ylabel())

    def test_empty_data_is_rejected(self):
        with self.assertRaises(ValueError):
            viz.save_all_charts(self.df.iloc[0:0], tempfile.gettempdir())


if __name__ == "__main__":
    unittest.main()
