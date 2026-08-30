import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
)
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

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

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            artifact = model_trainer.initiate_model_trainer()
            logging.info("Model trainer completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(
        self, model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation")
            model_evaluation = ModelEvaluation(
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=self.model_evaluation_config,
            )
            artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Model evaluation completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ) -> ModelPusherArtifact:
        try:
            logging.info("Starting model pusher")
            model_pusher = ModelPusher(
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            artifact = model_pusher.initiate_model_pusher()
            logging.info("Model pusher completed")
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

            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            print(model_trainer_artifact)

            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact)
            print(model_evaluation_artifact)

            model_pusher_artifact = self.start_model_pusher(
                model_trainer_artifact, model_evaluation_artifact
            )
            print(model_pusher_artifact)
            # Prediction pipeline + FastAPI serving get wired in next.
        except Exception as e:
            raise CustomException(e, sys)