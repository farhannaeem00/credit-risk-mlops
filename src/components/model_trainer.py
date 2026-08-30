import sys

import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# from src.constants import DAGSHUB_REPO_NAME, DAGSHUB_REPO_OWNER, MLFLOW_EXPERIMENT_NAME
from src.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.configuration.mlflow_connection import init_mlflow_tracking
from src.entity.config_entity import ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object


# def _init_mlflow_tracking():
#     """Wires up MLflow tracking against DagsHub. If dagshub.init() has
#     already authenticated once (browser OAuth flow, cached locally), this
#     is silent and instant on subsequent runs."""
#     try:
#         import dagshub

#         dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
#         mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
#         return True
#     except Exception as e:
#         logging.info(f"MLflow/DagsHub tracking unavailable, continuing without it: {e}")
#         return False


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def get_candidate_models() -> dict:
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=300,
                max_depth=-1,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
            ),
        }

    @staticmethod
    def evaluate(model, x_test, y_test) -> ClassificationMetricArtifact:
        try:
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)[:, 1]

            return ClassificationMetricArtifact(
                roc_auc=roc_auc_score(y_test, y_proba),
                pr_auc=average_precision_score(y_test, y_proba),
                f1_score=f1_score(y_test, y_pred),
                recall=recall_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred),
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            tracking_enabled = init_mlflow_tracking()

            all_metrics = {}
            fitted_models = {}

            for name, model in self.get_candidate_models().items():
                logging.info(f"Training {name}")

                run_ctx = mlflow.start_run(run_name=name) if tracking_enabled else None
                try:
                    model.fit(x_train, y_train)
                    metrics = self.evaluate(model, x_test, y_test)

                    logging.info(f"{name} -> roc_auc={metrics.roc_auc:.4f}, pr_auc={metrics.pr_auc:.4f}")

                    if tracking_enabled:
                        mlflow.log_params(model.get_params())
                        mlflow.log_metric("roc_auc", metrics.roc_auc)
                        mlflow.log_metric("pr_auc", metrics.pr_auc)
                        mlflow.log_metric("f1_score", metrics.f1_score)
                        mlflow.log_metric("recall", metrics.recall)
                        mlflow.log_metric("precision", metrics.precision)

                    all_metrics[name] = metrics
                    fitted_models[name] = model
                finally:
                    if run_ctx is not None:
                        mlflow.end_run()

            best_name = max(all_metrics, key=lambda n: all_metrics[n].roc_auc)
            best_model = fitted_models[best_name]
            best_metrics = all_metrics[best_name]

            logging.info(f"Best model: {best_name} (roc_auc={best_metrics.roc_auc:.4f})")

            if best_metrics.roc_auc < self.model_trainer_config.expected_roc_auc:
                raise Exception(
                    f"Best model roc_auc {best_metrics.roc_auc:.4f} is below the "
                    f"expected floor {self.model_trainer_config.expected_roc_auc}"
                )

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                best_model_name=best_name,
                all_models_metrics=all_metrics,
            )
        except Exception as e:
            raise CustomException(e, sys)