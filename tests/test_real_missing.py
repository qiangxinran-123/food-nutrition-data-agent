import unittest
from pathlib import Path

from data_agent.checks.missing import check_missing
from data_agent.io import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "config" / "profiles" / "off_52k_v1.json"


@unittest.skipUnless(PROFILE_PATH.exists(), "base dataset profile is unavailable")
class RealDatasetMissingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(PROFILE_PATH, project_root=PROJECT_ROOT)
        cls.result = check_missing(cls.dataset)

    def test_dataset_shape_and_identifier_integrity(self) -> None:
        self.assertEqual(self.dataset.frame.shape, (52000, 20))
        code_finding = next(
            (
                finding
                for finding in self.result.findings
                if finding.columns == ["code"]
            ),
            None,
        )
        self.assertIsNone(code_finding)

    def test_declared_unknown_tokens_are_counted(self) -> None:
        findings = {
            finding.columns[0]: finding for finding in self.result.findings
        }
        self.assertEqual(findings["nutriscore_grade"].affected_rows, 27611)
        self.assertEqual(
            findings["nutriscore_grade"].evidence["missing_token_counts"],
            {"unknown": 27610},
        )
        self.assertEqual(findings["categories_en"].affected_rows, 22904)
        self.assertEqual(findings["categories"].affected_rows, 33360)

    def test_global_summary_is_consistent(self) -> None:
        summary = self.result.summary
        self.assertEqual(summary["row_count"], 52000)
        self.assertEqual(summary["column_count"], 20)
        self.assertEqual(summary["missing_cells"], 297441)
        self.assertEqual(summary["rows_with_missing"], 48372)
        self.assertEqual(summary["complete_rows"], 3628)


if __name__ == "__main__":
    unittest.main()
