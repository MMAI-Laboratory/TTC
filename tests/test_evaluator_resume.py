import csv
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from ttc.config import Config
from ttc.engine import evaluator


def result_row(seed=3407, accuracy="76.5000"):
    # A zero-shot result from the existing writer has an empty CoOp seed.
    return ["ttc", "3", "16", "0.03", "ViT-B16", "zeroshot", "zeroshot", "dtd", "0", "", str(seed), "0", accuracy]


class EvaluateResumeTest(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.out = Path(directory) / "results.csv"
        self.cfg = Config()
        self.cfg.run.output_dir = directory
        self.cfg.run.seeds = [3407, 3408]
        self.cfg.data.datasets = ["dtd"]
        self.run_one = self.enterContext(patch.object(evaluator, "run_one", side_effect=[76.5, 1.23456]))

    def write_rows(self, *rows, header=evaluator.FIELDS):
        with self.out.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return self.out.read_bytes()

    def assert_rejected(self, original, row_number, reason):
        message = rf"{re.escape(str(self.out))}.*row {row_number}\b.*{reason}"
        with self.assertRaisesRegex(SystemExit, message):
            evaluator.evaluate(self.cfg)
        self.run_one.assert_not_called()
        self.assertEqual(self.out.read_bytes(), original)

    def test_complete_row_is_skipped_without_rewriting(self):
        self.cfg.run.seeds = [3407]
        original = self.write_rows(result_row())
        self.assertEqual(evaluator.evaluate(self.cfg), self.out)
        self.run_one.assert_not_called()
        self.assertEqual(self.out.read_bytes(), original)

    def test_truncated_row_missing_final_accuracy(self):
        self.cfg.run.seeds = [3407]
        self.assert_rejected(self.write_rows(result_row()[:-1]), 2, "expected .*columns")

    def test_empty_accuracy(self):
        self.assert_rejected(self.write_rows(result_row(accuracy="")), 2, "accuracy")

    def test_whitespace_accuracy(self):
        self.assert_rejected(self.write_rows(result_row(accuracy=" \t")), 2, "accuracy")

    def test_non_numeric_accuracy(self):
        self.assert_rejected(self.write_rows(result_row(accuracy="not a number")), 2, "accuracy")

    def test_nan_accuracy(self):
        self.assert_rejected(self.write_rows(result_row(accuracy="NaN")), 2, "accuracy")

    def test_positive_infinite_accuracy(self):
        self.assert_rejected(self.write_rows(result_row(accuracy="+inf")), 2, "accuracy")

    def test_negative_infinite_accuracy(self):
        self.assert_rejected(self.write_rows(result_row(accuracy="-Infinity")), 2, "accuracy")

    def test_extra_csv_field(self):
        self.assert_rejected(self.write_rows([*result_row(), "extra"]), 2, "expected .*columns")

    def test_missing_identity_field(self):
        row = result_row()
        row[evaluator.FIELDS.index("dataset")] = ""
        self.assert_rejected(self.write_rows(row), 2, "dataset")

    def test_incorrect_header(self):
        self.assert_rejected(self.write_rows(result_row(), header=evaluator.FIELDS[:-1]), 1, "expected")

    def test_invalid_csv_quoting(self):
        original = self.write_rows(result_row())
        original += b'"unfinished record\r\n'
        self.out.write_bytes(original)
        self.assert_rejected(original, 3, "CSV")

    def test_multiple_valid_rows_keep_existing_finite_formats(self):
        original = self.write_rows(result_row(accuracy="1.5e2"), result_row(3408, "-0.0000"))
        self.assertEqual(evaluator.evaluate(self.cfg), self.out)
        self.run_one.assert_not_called()
        self.assertEqual(self.out.read_bytes(), original)

    def test_mixed_rows_are_fully_validated_before_any_trial(self):
        original = self.write_rows(result_row(), result_row(9999, "NaN"))
        self.assert_rejected(original, 3, "accuracy")

    def test_fresh_csv_runs_two_trials_with_existing_header_and_format(self):
        self.assertEqual(evaluator.evaluate(self.cfg), self.out)
        self.assertEqual(self.run_one.call_args_list, [
            call(self.cfg, "zeroshot", "dtd", 0, None, 3407),
            call(self.cfg, "zeroshot", "dtd", 0, None, 3408),
        ])
        with self.out.open(newline="") as stream:
            self.assertEqual(list(csv.reader(stream)), [
                ["method", "topk", "memory_size", "lam", "backbone", "task", "subtask", "dataset", "shots",
                 "coop_seed", "seed", "aug", "accuracy"],
                result_row(),
                result_row(3408, "1.2346"),
            ])
        # A second call loads the exact bytes just emitted by the writer.
        original = self.out.read_bytes()
        self.run_one.reset_mock()
        self.assertEqual(evaluator.evaluate(self.cfg), self.out)
        self.run_one.assert_not_called()
        self.assertEqual(self.out.read_bytes(), original)

    def test_valid_resume_appends_only_the_missing_trial(self):
        original = self.write_rows(result_row(3408, "65.2500"))
        self.assertEqual(evaluator.evaluate(self.cfg), self.out)
        self.run_one.assert_called_once_with(self.cfg, "zeroshot", "dtd", 0, None, 3407)
        self.assertTrue(self.out.read_bytes().startswith(original))
        with self.out.open(newline="") as stream:
            self.assertEqual(list(csv.reader(stream))[1:], [result_row(3408, "65.2500"), result_row()])
