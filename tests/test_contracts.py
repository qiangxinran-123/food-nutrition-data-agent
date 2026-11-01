import json
import unittest

from pydantic import ValidationError

from data_agent.contracts import (
    Artifact,
    ArtifactKind,
    ChangeRecord,
    ChangeSet,
    ChangeType,
    CheckDelta,
    CleaningAction,
    ColumnSpec,
    ConstraintKind,
    ConstraintSpec,
    DataFormat,
    DatasetRef,
    Finding,
    FindingScope,
    IssueType,
    LogicalType,
    RiskLevel,
    SchemaProfile,
    Severity,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
    ValidationReport,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def dataset_ref(
    dataset_id: str = "off_52k_v1",
    rows: int = 52000,
    sha256: str = SHA_A,
    parent_id: str | None = None,
) -> DatasetRef:
    return DatasetRef(
        dataset_id=dataset_id,
        version="1",
        uri=f"data/base/{dataset_id}.parquet",
        format=DataFormat.PARQUET,
        sha256=sha256,
        row_count=rows,
        column_count=20,
        parent_id=parent_id,
    )


def schema_profile() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="off_52k_v1",
        domain_pack="food_nutrition",
        columns=[
            ColumnSpec(
                name="code",
                role="entity_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
                nullable=False,
                unique=True,
            ),
            ColumnSpec(
                name="energy-kcal_100g",
                role="energy_kcal",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="float64",
                unit="kcal/100g",
            ),
            ColumnSpec(
                name="__row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
                nullable=False,
                unique=True,
            ),
            ColumnSpec(
                name="__source_row",
                role="source_row",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="int64",
                nullable=False,
                unique=True,
            ),
        ],
        primary_key_candidates=[["code"]],
        row_id_column="__row_id",
        source_row_column="__source_row",
    )


class ContractTests(unittest.TestCase):
    def test_dataset_ref_json_round_trip(self) -> None:
        reference = dataset_ref()
        payload = reference.model_dump_json()
        loaded = DatasetRef.model_validate_json(payload)
        self.assertEqual(loaded, reference)
        self.assertEqual(json.loads(payload)["contract_version"], "0.1")

    def test_schema_profile_rejects_duplicate_columns(self) -> None:
        column = ColumnSpec(
            name="code",
            logical_type=LogicalType.IDENTIFIER,
            physical_dtype="str",
        )
        with self.assertRaises(ValidationError):
            SchemaProfile(dataset_id="test", columns=[column, column])

    def test_schema_profile_rejects_unknown_key_column(self) -> None:
        profile = schema_profile().model_dump()
        profile["primary_key_candidates"] = [["missing_column"]]
        with self.assertRaises(ValidationError):
            SchemaProfile.model_validate(profile)

    def test_constraint_requires_expression_for_formula(self) -> None:
        with self.assertRaises(ValidationError):
            ConstraintSpec(
                constraint_id="bad_formula",
                kind=ConstraintKind.FORMULA,
                columns=["salt_100g"],
                message="invalid",
            )

    def test_finding_and_tool_result_round_trip(self) -> None:
        finding = Finding(
            finding_id="missing.fiber_100g",
            issue_type=IssueType.MISSING,
            severity=Severity.HIGH,
            scope=FindingScope.COLUMN,
            columns=["fiber_100g"],
            affected_rows=29098,
            affected_rate=0.58196,
            evidence={"missing_rate": 0.58196},
            sample_row_ids=["off52k_000012"],
            suggested_actions=["keep_missing", "group_median_impute"],
        )
        artifact = Artifact(
            artifact_id="fiber_missing_rows",
            kind=ArtifactKind.ROW_LIST,
            uri="data/processed/fiber_missing_rows.parquet",
            media_type="application/parquet",
            sha256=SHA_B,
            row_count=29098,
        )
        result = ToolResult(
            tool_name="check_missing",
            dataset_id="off_52k_v1",
            findings=[finding],
            summary={"finding_count": 1},
            artifacts=[artifact],
        )
        loaded = ToolResult.model_validate_json(result.model_dump_json())
        self.assertEqual(loaded, result)

    def test_failed_tool_result_requires_message(self) -> None:
        with self.assertRaises(ValidationError):
            ToolResult(
                tool_name="check_missing",
                dataset_id="off_52k_v1",
                status=ToolStatus.FAILED,
            )

    def test_dangerous_action_requires_confirmation(self) -> None:
        with self.assertRaises(ValidationError):
            CleaningAction(
                action_id="drop_rows",
                tool_name="drop_rows",
                target_columns=[],
                reason="test",
                risk_level=RiskLevel.DANGEROUS,
            )

        action = CleaningAction(
            action_id="drop_rows",
            tool_name="drop_rows",
            target_columns=[],
            reason="test",
            risk_level=RiskLevel.DANGEROUS,
            requires_confirmation=True,
        )
        self.assertTrue(action.requires_confirmation)

    def test_change_set_row_accounting(self) -> None:
        before = dataset_ref(rows=10)
        after = dataset_ref(
            dataset_id="off_52k_v2",
            rows=9,
            sha256=SHA_B,
            parent_id=before.dataset_id,
        )
        action = CleaningAction(
            action_id="remove_exact_duplicates",
            tool_name="remove_duplicates",
            reason="exact duplicate",
            risk_level=RiskLevel.CAUTION,
        )
        change = ChangeRecord(
            row_id="off52k_000010",
            change_type=ChangeType.DROP,
            reason="exact duplicate",
        )
        change_set = ChangeSet(
            change_id="change_001",
            before_dataset=before,
            after_dataset=after,
            actions=[action],
            rows_removed=1,
            cells_changed=0,
            sample_changes=[change],
        )
        self.assertEqual(change_set.rows_removed, 1)

        with self.assertRaises(ValidationError):
            ChangeSet(
                change_id="bad_change",
                before_dataset=before,
                after_dataset=after,
            )

    def test_validation_report_retention(self) -> None:
        before = dataset_ref(rows=10)
        after = dataset_ref(dataset_id="off_52k_v2", rows=9, sha256=SHA_B)
        report = ValidationReport(
            validation_id="validation_001",
            before_dataset=before,
            after_dataset=after,
            checks=[
                CheckDelta(
                    check_type=IssueType.MISSING,
                    before_count=4,
                    after_count=2,
                    passed=True,
                )
            ],
            passed=True,
            retained_rows=9,
            retained_rate=0.9,
        )
        self.assertTrue(report.passed)

        with self.assertRaises(ValidationError):
            ValidationReport(
                validation_id="bad_validation",
                before_dataset=before,
                after_dataset=after,
                passed=True,
                retained_rows=9,
                retained_rate=1.0,
            )

    def test_tool_spec_and_json_schema(self) -> None:
        tool = ToolSpec(
            name="check_missing",
            category=ToolCategory.DETECTION,
            description="Detect missing values without modifying data.",
            input_schema={"type": "object"},
            output_schema=ToolResult.model_json_schema(),
        )
        self.assertFalse(tool.mutates_data)
        self.assertEqual(tool.risk_level, RiskLevel.SAFE)

        for model in (
            DatasetRef,
            ColumnSpec,
            SchemaProfile,
            ConstraintSpec,
            Finding,
            Artifact,
            ToolResult,
            CleaningAction,
            ChangeRecord,
            ChangeSet,
            CheckDelta,
            ValidationReport,
            ToolSpec,
        ):
            self.assertIsInstance(model.model_json_schema(), dict)


if __name__ == "__main__":
    unittest.main()
