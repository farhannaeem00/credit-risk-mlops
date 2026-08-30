import sys

import mlflow
import mlflow.pyfunc

from src.configuration.mlflow_connection import init_mlflow_tracking
from src.constants import PRODUCTION_ALIAS
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelPusherConfig
from src.entity.estimator import CreditRiskModel, MLflowCreditRiskModel
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_object


class ModelPusher:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted by evaluation - skipping push to registry")
                return ModelPusherArtifact(
                    is_model_pushed=False,
                    registered_model_name=self.model_pusher_config.registered_model_name,
                    model_version=None,
                )

            tracking_enabled = init_mlflow_tracking()
            if not tracking_enabled:
                logging.info("MLflow tracking unavailable - cannot push to registry")
                return ModelPusherArtifact(
                    is_model_pushed=False,
                    registered_model_name=self.model_pusher_config.registered_model_name,
                    model_version=None,
                )

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)

            bundled_model = CreditRiskModel(
                preprocessing_object=preprocessor, trained_model_object=trained_model
            )
            pyfunc_wrapper = MLflowCreditRiskModel(credit_risk_model=bundled_model)

            registered_name = self.model_pusher_config.registered_model_name

            with mlflow.start_run(run_name=f"push-{self.model_trainer_artifact.best_model_name}"):
                mlflow.log_metric("roc_auc", self.model_evaluation_artifact.best_model_metrics.roc_auc)
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=pyfunc_wrapper,
                    registered_model_name=registered_name,
                )

            client = mlflow.tracking.MlflowClient()
            latest_version = max(
                client.search_model_versions(f"name='{registered_name}'"),
                key=lambda v: int(v.version),
            )

            client.set_registered_model_alias(registered_name, PRODUCTION_ALIAS, latest_version.version)
            logging.info(
                f"Promoted {registered_name} v{latest_version.version} to '{PRODUCTION_ALIAS}' alias "
                f"(bundled: preprocessor + {self.model_trainer_artifact.best_model_name})"
            )

            return ModelPusherArtifact(
                is_model_pushed=True,
                registered_model_name=registered_name,
                model_version=latest_version.version,
            )
        except Exception as e:
            raise CustomException(e, sys)