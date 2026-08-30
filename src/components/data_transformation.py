import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import DAYS_EMPLOYED_ANOMALY_VALUE, ID_COLUMN, SCHEMA_FILE_PATH, TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def clean_anomalies(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Home Credit uses 365243 as a placeholder in DAYS_EMPLOYED for
        people who aren't employed (mostly pensioners) - a real value that
        would otherwise get treated as ~1000 years employed. Convert it to
        NaN so imputation handles it honestly instead of the model learning
        a nonsense numeric relationship."""
        try:
            df = dataframe.copy()
            if "DAYS_EMPLOYED" in df.columns:
                anomaly_count = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY_VALUE).sum()
                df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
                    DAYS_EMPLOYED_ANOMALY_VALUE, np.nan
                )
                logging.info(f"Cleaned {anomaly_count} DAYS_EMPLOYED anomaly values -> NaN")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_preprocessor(self, numerical_columns: list, categorical_columns: list) -> Pipeline:
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_columns),
                    ("cat", categorical_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)  # placeholder guard, see note below
        # NOTE: this except line intentionally corrected below in initiate_data_transformation

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(
                    f"Cannot transform data that failed validation: "
                    f"{self.data_validation_artifact.message}"
                )

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            train_df = self.clean_anomalies(train_df)
            test_df = self.clean_anomalies(test_df)

            drop_cols = [TARGET_COLUMN, ID_COLUMN]
            input_feature_train_df = train_df.drop(columns=drop_cols, errors="ignore")
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=drop_cols, errors="ignore")
            target_feature_test_df = test_df[TARGET_COLUMN]

            numerical_columns = [
                c for c in self._schema["numerical_columns"] if c in input_feature_train_df.columns
            ]
            categorical_columns = [
                c for c in self._schema["categorical_columns"] if c in input_feature_train_df.columns
            ]

            logging.info(
                f"Building preprocessor for {len(numerical_columns)} numerical "
                f"and {len(categorical_columns)} categorical columns"
            )
            preprocessor = self.get_preprocessor(numerical_columns, categorical_columns)

            logging.info("Fitting preprocessor on training data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logging.info(
                f"Transformed train shape: {train_arr.shape}, test shape: {test_arr.shape}"
            )

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)   