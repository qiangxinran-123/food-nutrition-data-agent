from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "openfoodfacts_nutrition_sample_52000.xlsx"
BASE_DIR = PROJECT_ROOT / "data" / "base"
DATASET_ID = "off_52k_v1"
PARQUET_PATH = BASE_DIR / f"{DATASET_ID}.parquet"
MANIFEST_PATH = BASE_DIR / f"{DATASET_ID}.manifest.json"

ROLE_MAP = {
    "code": "entity_id",
    "product_name": "product_name",
    "brands": "brand",
    "categories": "category",
    "categories_en": "category_en",
    "energy-kcal_100g": "energy_kcal",
    "energy_100g": "energy_kj",
    "proteins_100g": "protein",
    "fat_100g": "fat",
    "saturated-fat_100g": "saturated_fat",
    "carbohydrates_100g": "carbohydrates",
    "sugars_100g": "sugars",
    "fiber_100g": "fiber",
    "salt_100g": "salt",
    "sodium_100g": "sodium",
    "nutriscore_grade": "label",
    "serving_size": "serving_size",
    "serving_quantity": "serving_quantity",
}

UNIT_MAP = {
    "energy-kcal_100g": "kcal/100g",
    "energy_100g": "kJ/100g",
    "proteins_100g": "g/100g",
    "fat_100g": "g/100g",
    "saturated-fat_100g": "g/100g",
    "carbohydrates_100g": "g/100g",
    "sugars_100g": "g/100g",
    "fiber_100g": "g/100g",
    "salt_100g": "g/100g",
    "sodium_100g": "g/100g",
    "serving_quantity": "source_defined_unit",
}

LOGICAL_TYPES = {
    "code": "identifier",
    "product_name": "text",
    "brands": "categorical",
    "categories": "categorical",
    "categories_en": "categorical",
    "nutriscore_grade": "categorical",
    "serving_size": "text",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def describe_column(series: pd.Series, name: str) -> dict:
    logical_type = LOGICAL_TYPES.get(name)
    if logical_type is None:
        logical_type = "numeric" if pd.api.types.is_numeric_dtype(series) else "text"

    description = {
        "name": name,
        "role": ROLE_MAP.get(name),
        "logical_type": logical_type,
        "physical_dtype": str(series.dtype),
        "unit": UNIT_MAP.get(name),
        "nullable": bool(series.isna().any()),
        "missing_count": int(series.isna().sum()),
        "missing_rate": round(float(series.isna().mean()), 6),
        "unique_count": int(series.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(series):
        description.update(
            {
                "min": json_value(series.min()),
                "median": json_value(series.median()),
                "max": json_value(series.max()),
            }
        )
    else:
        description["examples"] = series.dropna().astype(str).head(3).tolist()

    return description


def load_source() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source workbook not found: {SOURCE_PATH}")

    dataframe = pd.read_excel(SOURCE_PATH, dtype={"code": str})
    if "code" not in dataframe.columns:
        raise ValueError("Required column 'code' was not found in the source workbook")

    dataframe["code"] = dataframe["code"].astype("string").str.strip()
    if dataframe["code"].isna().any():
        raise ValueError("The 'code' column contains missing values")
    if not dataframe["code"].str.fullmatch(r"\d+").all():
        raise ValueError("The 'code' column contains non-digit values")

    return dataframe


def prepare_base(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    original_columns = dataframe.columns.tolist()
    dataframe = dataframe.copy()
    dataframe["__row_id"] = [
        f"off52k_{index:06d}" for index in range(1, len(dataframe) + 1)
    ]
    dataframe["__source_row"] = range(2, len(dataframe) + 2)

    for column in dataframe.columns:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            dataframe[column] = [
                None if pd.isna(value) else str(value)
                for value in dataframe[column]
            ]
    dataframe["code"] = [str(value) for value in dataframe["code"]]
    return dataframe, original_columns


def write_parquet(dataframe: pd.DataFrame) -> None:
    temporary_path = PARQUET_PATH.with_suffix(".parquet.tmp")
    dataframe.to_parquet(
        temporary_path,
        engine="fastparquet",
        compression="snappy",
        index=False,
    )
    os.replace(temporary_path, PARQUET_PATH)


def build_manifest(dataframe: pd.DataFrame, original_columns: list[str]) -> dict:
    code_series = dataframe["code"].astype(str)
    code_lengths = {
        str(length): int(count)
        for length, count in code_series.str.len().value_counts().sort_index().items()
    }

    columns = [describe_column(dataframe[column], column) for column in original_columns]
    columns.extend(
        [
            {
                "name": "__row_id",
                "role": "row_id",
                "logical_type": "identifier",
                "physical_dtype": str(dataframe["__row_id"].dtype),
                "unit": None,
                "nullable": False,
                "missing_count": 0,
                "missing_rate": 0.0,
                "unique_count": int(dataframe["__row_id"].nunique()),
                "examples": dataframe["__row_id"].head(3).tolist(),
            },
            {
                "name": "__source_row",
                "role": "source_row",
                "logical_type": "identifier",
                "physical_dtype": str(dataframe["__source_row"].dtype),
                "unit": None,
                "nullable": False,
                "missing_count": 0,
                "missing_rate": 0.0,
                "unique_count": int(dataframe["__source_row"].nunique()),
                "min": 2,
                "max": len(dataframe) + 1,
            },
        ]
    )

    return {
        "contract_version": "0.1",
        "dataset_id": DATASET_ID,
        "status": "synthetic" if int(code_series.duplicated().sum()) else "frozen",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source": {
            "path": SOURCE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "file_size_bytes": SOURCE_PATH.stat().st_size,
            "sha256": sha256_file(SOURCE_PATH),
            "source_url": "https://world.openfoodfacts.org/data",
            "retrieved_date": "2026-10-28",
            "license": None,
        },
        "base_file": {
            "path": PARQUET_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "format": "parquet",
            "compression": "snappy",
            "engine": "fastparquet",
            "sha256": sha256_file(PARQUET_PATH),
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "original_column_count": len(original_columns),
        },
        "row_identity": {
            "row_id_column": "__row_id",
            "source_row_column": "__source_row",
            "source_row_range": f"2-{len(dataframe) + 1}",
            "row_id_unique": bool(dataframe["__row_id"].is_unique),
            "source_row_unique": bool(dataframe["__source_row"].is_unique),
        },
        "identifier_policy": {
            "column": "code",
            "stored_as": "string",
            "leading_zeros_preserved": True,
            "leading_zero_count": int(code_series.str.startswith("0").sum()),
            "missing_count": int(dataframe["code"].isna().sum()),
            "duplicate_count": int(code_series.duplicated().sum()),
            "length_counts": code_lengths,
        },
        "schema": {
            "original_columns": original_columns,
            "added_columns": ["__row_id", "__source_row"],
            "columns": columns,
        },
        "processing": {
            "cleaning_applied": False,
            "deduplication_applied": False,
            "row_filtering_applied": False,
            "synthetic_duplicate_injection_applied": bool(
                int(code_series.duplicated().sum())
            ),
            "notes": "Base snapshot built from the current source workbook. Cleaning must produce a new dataset version under data/processed.",
        },
        "generated_by": "scripts/build_base_dataset.py",
    }


def write_manifest(manifest: dict) -> None:
    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, MANIFEST_PATH)


def build_dataset(force: bool = False) -> None:
    if not force and (PARQUET_PATH.exists() or MANIFEST_PATH.exists()):
        raise FileExistsError(
            f"Base dataset already exists. Use --force to rebuild: {PARQUET_PATH}"
        )

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = load_source()
    dataframe, original_columns = prepare_base(dataframe)
    write_parquet(dataframe)
    write_manifest(build_manifest(dataframe, original_columns))
    print(f"Wrote {PARQUET_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Rows={len(dataframe)}, columns={len(dataframe.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_dataset(force=args.force)


if __name__ == "__main__":
    main()
