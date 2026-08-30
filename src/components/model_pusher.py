import sys

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient

from src.configuration.mlflow_connection import init_mlflow_tracking
from src.constants import PRODUCTION_ALIAS
from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_object


class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
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

            model = load_object(self.model_trainer_artifact.trained_model_file_path)
            registered_name = self.model_pusher_config.registered_model_name

            # LightGBM's Booster isn't a plain sklearn estimator under the
            # hood, so mlflow.sklearn's newer skops-based serializer refuses
            # to save it as an "untrusted type". Use LightGBM's own MLflow
            # flavor instead, which handles this natively.
            if isinstance(model, LGBMClassifier):
                log_model_fn = mlflow.lightgbm.log_model
                log_model_kwargs = {"lgb_model": model}
            else:
                log_model_fn = mlflow.sklearn.log_model
                log_model_kwargs = {"sk_model": model}

            with mlflow.start_run(run_name=f"push-{self.model_trainer_artifact.best_model_name}"):
                mlflow.log_metric("roc_auc", self.model_evaluation_artifact.best_model_metrics.roc_auc)
                log_model_fn(
                    **log_model_kwargs,
                    artifact_path="model",
                    registered_model_name=registered_name,
                )

            client = MlflowClient()
            latest_version = max(
                client.search_model_versions(f"name='{registered_name}'"),
                key=lambda v: int(v.version),
            )

            client.set_registered_model_alias(registered_name, PRODUCTION_ALIAS, latest_version.version)
            logging.info(
                f"Promoted {registered_name} v{latest_version.version} to '{PRODUCTION_ALIAS}' alias"
            )

            return ModelPusherArtifact(
                is_model_pushed=True,
                registered_model_name=registered_name,
                model_version=latest_version.version,
            )
        except Exception as e:
            raise CustomException(e, sys)