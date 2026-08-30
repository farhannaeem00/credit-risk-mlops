import os
import sys

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """Read the raw Kaggle CSV and copy it into this run's feature-store path.

        Originally pulled from MongoDB Atlas; switched to a direct CSV read
        after confirming Atlas's TLS handshake is blocked on this network
        (raw openssl tests failed identically, ruling out Python/pymongo).
        """
        try:
            raw_path = self.data_ingestion_config.raw_data_file_path
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"{raw_path} not found. Download it first via the Kaggle "
                    f"CLI, e.g.: kaggle competitions download -c "
                    f"home-credit-default-risk -f application_train.csv -p data/"
                )

            logging.info(f"Reading raw data from {raw_path}")
            dataframe = pd.read_csv(raw_path)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            logging.info(f"Saving to feature store: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )
            logging.info("Performed train/test split on the dataframe")

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.training_file_path),
                exist_ok=True,
            )

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test CSVs")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)