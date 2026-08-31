import csv
import tempfile
import unittest
from pathlib import Path

from src.experiment import METRICS_COLUMNS, append_metrics


class MetricsLoggingTests(unittest.TestCase):
    def test_metrics_are_appended_with_one_header(self):
        row = {column: index for index, column in enumerate(METRICS_COLUMNS)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_metrics(path, row)
            append_metrics(path, row)
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual(tuple(rows[0]), METRICS_COLUMNS)


if __name__ == "__main__":
    unittest.main()
