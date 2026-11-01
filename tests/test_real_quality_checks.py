import unittest
from pathlib import Path

from data_agent.checks.conflicts import check_conflicts
from data_agent.checks.duplicates import check_duplicates
from data_agent.checks.imbalance import check_imbalance
from data_agent.checks.labels import check_labels
from data_agent.checks.outliers import check_outliers
from data_agent.checks.types_formats import check_types_formats
from data_agent.io import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "config" / "profiles" / "off_52k_v1.json"


@unittest.skipUnless(PROFILE_PATH.exists(), "base dataset profile is unavailable")
class RealDatasetQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(PROFILE_PATH, project_root=PROJECT_ROOT)

    def test_duplicate_counts(self) -> None:
        result = check_duplicates(self.dataset)
        summaries = {item["rule_id"]: item for item in result.summary["rules"]}
        self.assertEqual(
            summaries["primary_key"]["duplicate_group_count"],
            2000,
        )
        self.assertEqual(summaries["primary_key"]["redundant_rows"], 2000)
        self.assertEqual(result.summary["rows_flagged_union"], 2000)

    def test_range_and_format_counts(self) -> None:
        outliers = check_outliers(self.dataset)
        formats = check_types_formats(self.dataset)
        self.assertEqual(outliers.summary["range_flagged_cells"], 1063)
        findings = {finding.finding_id: finding for finding in formats.findings}
        self.assertEqual(findings["format.regex.code"].affected_rows, 207)

    def test_conflict_counts(self) -> None:
        result = check_conflicts(self.dataset)
        summaries = {
            item["constraint_id"]: item for item in result.summary["constraints"]
        }
        self.assertEqual(
            summaries["energy_kcal_kj_consistency"]["violated_rows"], 3064
        )
        self.assertEqual(
            summaries["salt_sodium_consistency"]["violated_rows"], 203
        )
        self.assertEqual(
            summaries["sugars_not_above_carbohydrates"]["violated_rows"], 28
        )
        self.assertEqual(
            summaries["saturated_fat_not_above_total_fat"]["violated_rows"], 20
        )

    def test_label_availability_and_imbalance(self) -> None:
        labels = check_labels(self.dataset)
        imbalance = check_imbalance(self.dataset)
        label_summary = labels.summary["labels"][0]
        self.assertEqual(label_summary["valid_rows"], 22799)
        self.assertEqual(label_summary["unavailable_rows"], 29201)
        self.assertEqual(label_summary["invalid_rows"], 0)
        self.assertTrue(imbalance.summary["has_imbalance"])
        self.assertEqual(imbalance.findings[0].affected_rows, 2497)


if __name__ == "__main__":
    unittest.main()
