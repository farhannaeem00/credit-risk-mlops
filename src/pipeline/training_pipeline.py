import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
)
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            print(data_ingestion_artifact)

            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            print(data_validation_artifact)

            if not data_validation_artifact.validation_status:
                raise Exception(
                    f"Data validation failed: {data_validation_artifact.message} "
                    f"See {data_validation_artifact.validation_report_file_path}"
                )

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact, data_validation_artifact
            )
            print(data_transformation_artifact)
            # Model training, evaluation, and pusher stages get wired in
            # here in the next phases.
        except Exception as e:
            raise CustomException(e, sys)