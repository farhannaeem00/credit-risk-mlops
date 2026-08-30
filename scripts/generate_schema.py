"""
Infers config/schema.yaml from the raw Home Credit CSV you already
downloaded. Run this once, before Phase 2's demo.py run.

Usage:
    python scripts/generate_schema.py
"""
import sys

import pandas as pd

from src.constants import RAW_DATA_FILE_PATH, SCHEMA_FILE_PATH, TARGET_COLUMN
from src.utils.main_utils import write_yaml_file

MAX_CATEGORY_CARDINALITY = 30  # enumerate allowed values only below this


def infer_dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    return "object"


def main():
    df = pd.read_csv(RAW_DATA_FILE_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    columns_schema = {}
    numerical_columns = []
    categorical_columns = []

    for col in df.columns:
        dtype_label = infer_dtype_label(df[col])
        entry = {"dtype": dtype_label, "nullable": bool(df[col].isnull().any())}

        if dtype_label == "object":
            categorical_columns.append(col)
            uniques = df[col].dropna().unique().tolist()
            if len(uniques) <= MAX_CATEGORY_CARDINALITY:
                entry["allowed_values"] = sorted(str(u) for u in uniques)
        else:
            numerical_columns.append(col)
            entry["min"] = float(df[col].min())
            entry["max"] = float(df[col].max())

        columns_schema[col] = entry

    schema = {
        "target_column": TARGET_COLUMN,
        "id_column": "SK_ID_CURR",
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "columns": columns_schema,
    }

    write_yaml_file(SCHEMA_FILE_PATH, schema, replace=True)
    print(f"Wrote schema for {len(columns_schema)} columns to {SCHEMA_FILE_PATH}")


if __name__ == "__main__":
    sys.exit(main())