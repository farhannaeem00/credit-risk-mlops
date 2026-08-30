import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.configuration.mlflow_connection import init_mlflow_tracking
from src.constants import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, ROC_AUC_TIE_BREAK_DELTA
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import write_yaml_file


class ModelEvaluation:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config

    @staticmethod
    def select_best_candidate(all_models_metrics: dict) -> str:
        """Applies the agreed selection criteria explicitly (independent of
        whatever the trainer's internal max() picked) so the decision is
        auditable on its own: primary = ROC-AUC; if the top two candidates
        are within ROC_AUC_TIE_BREAK_DELTA of each other, prefer the one
        with higher PR-AUC (more informative than raw ROC-AUC under class
        imbalance)."""
        ranked = sorted(all_models_metrics.items(), key=lambda item: item[1].roc_auc, reverse=True)
        top_name, top_metrics = ranked[0]

        if len(ranked) > 1:
            second_name, second_metrics = ranked[1]
            if (top_metrics.roc_auc - second_metrics.roc_auc) < ROC_AUC_TIE_BREAK_DELTA:
                if second_metrics.pr_auc > top_metrics.pr_auc:
                    logging.info(
                        f"{top_name} and {second_name} within tie-break delta on ROC-AUC; "
                        f"{second_name} wins on PR-AUC instead"
                    )
                    return second_name

        return top_name

    @staticmethod
    def get_previous_production_roc_auc() -> float:
        """Looks up whatever model currently holds the 'production' alias
        in the MLflow Model Registry and returns its logged roc_auc metric.
        Returns None if no model has been promoted yet (first-ever run)."""
        try:
            client = MlflowClient()
            model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
            run = client.get_run(model_version.run_id)
            return run.data.metrics.get("roc_auc")
        except Exception as e:
            logging.info(f"No existing Production model found (expected on first run): {e}")
            return None

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            tracking_enabled = init_mlflow_tracking()

            all_metrics = self.model_trainer_artifact.all_models_metrics
            best_name = self.select_best_candidate(all_metrics)
            best_metrics = all_metrics[best_name]

            previous_roc_auc = self.get_previous_production_roc_auc() if tracking_enabled else None
            changed_roc_auc = None if previous_roc_auc is None else best_metrics.roc_auc - previous_roc_auc

            meets_floor = best_metrics.roc_auc >= self.model_evaluation_config.expected_roc_auc
            beats_previous = previous_roc_auc is None or best_metrics.roc_auc >= previous_roc_auc
            is_accepted = meets_floor and beats_previous

            logging.info(
                f"Evaluation: best={best_name}, roc_auc={best_metrics.roc_auc:.4f}, "
                f"previous_production_roc_auc={previous_roc_auc}, accepted={is_accepted}"
            )

            report = {
                "best_model_name": best_name,
                "best_model_roc_auc": best_metrics.roc_auc,
                "best_model_pr_auc": best_metrics.pr_auc,
                "previous_production_roc_auc": previous_roc_auc,
                "changed_roc_auc": changed_roc_auc,
                "meets_floor_threshold": meets_floor,
                "beats_previous_production": beats_previous,
                "is_model_accepted": is_accepted,
                "all_candidates": {
                    name: {"roc_auc": m.roc_auc, "pr_auc": m.pr_auc, "f1_score": m.f1_score}
                    for name, m in all_metrics.items()
                },
            }
            write_yaml_file(self.model_evaluation_config.report_file_path, report, replace=True)

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                best_model_name=best_name,
                best_model_metrics=best_metrics,
                previous_production_roc_auc=previous_roc_auc,
                changed_roc_auc=changed_roc_auc,
            )
        except Exception as e:
            raise CustomException(e, sys)