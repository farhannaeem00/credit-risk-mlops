import sys

import pandas as pd

from src.constants import SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_column_count(self, dataframe: pd.DataFrame) -> bool:
        expected = len(self._schema["columns"])
        actual = len(dataframe.columns)
        status = expected == actual
        logging.info(f"Column count check: expected {expected}, got {actual} -> {status}")
        return status

    def validate_missing_columns(self, dataframe: pd.DataFrame) -> list:
        missing = [col for col in self._schema["columns"] if col not in dataframe.columns]
        if missing:
            logging.info(f"Missing columns: {missing}")
        return missing

    def validate_dtypes(self, dataframe: pd.DataFrame) -> dict:
        mismatches = {}
        for col, spec in self._schema["columns"].items():
            if col not in dataframe.columns:
                continue
            expected = spec["dtype"]
            series = dataframe[col]

            if expected == "int64":
                ok = pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)
            elif expected == "float64":
                ok = pd.api.types.is_numeric_dtype(series)
            else:
                ok = True  # object columns tolerate NaN / mixed types

            if not ok:
                mismatches[col] = [expected, str(series.dtype)]

        if mismatches:
            logging.info(f"Dtype mismatches: {mismatches}")
        return mismatches

    def validate_ranges(self, dataframe: pd.DataFrame, tolerance: float = 0.05) -> dict:
        """Flags numeric columns where new data falls meaningfully outside the
        min/max seen when the schema was generated (small tolerance band
        allows for normal data drift)."""
        out_of_range = {}
        for col in self._schema.get("numerical_columns", []):
            if col not in dataframe.columns:
                continue
            spec = self._schema["columns"][col]
            if "min" not in spec or "max" not in spec:
                continue

            col_min, col_max = spec["min"], spec["max"]
            span = max(col_max - col_min, 1e-9)
            allowed_min = col_min - tolerance * span
            allowed_max = col_max + tolerance * span

            actual_min = dataframe[col].min()
            actual_max = dataframe[col].max()

            if actual_min < allowed_min or actual_max > allowed_max:
                out_of_range[col] = {
                    "expected_range": [col_min, col_max],
                    "actual_range": [float(actual_min), float(actual_max)],
                }

        if out_of_range:
            logging.info(f"Out-of-range columns: {list(out_of_range.keys())}")
        return out_of_range

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            report = {}
            overall_status = True

            for split_name, df in [("train", train_df), ("test", test_df)]:
                col_count_ok = self.validate_column_count(df)
                missing_cols = self.validate_missing_columns(df)
                dtype_mismatches = self.validate_dtypes(df)
                out_of_range = self.validate_ranges(df)

                split_status = col_count_ok and not missing_cols and not dtype_mismatches
                overall_status = overall_status and split_status

                report[split_name] = {
                    "column_count_ok": col_count_ok,
                    "missing_columns": missing_cols,
                    "dtype_mismatches": dtype_mismatches,
                    "out_of_range_columns": out_of_range,
                    "status": split_status,
                }

            message = "Validation passed." if overall_status else "Validation failed - see report."
            write_yaml_file(self.data_validation_config.validation_report_file_path, report, replace=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=overall_status,
                message=message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)