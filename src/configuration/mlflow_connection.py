import mlflow

from src.constants import DAGSHUB_REPO_NAME, DAGSHUB_REPO_OWNER, MLFLOW_EXPERIMENT_NAME
from src.logger import logging


def init_mlflow_tracking() -> bool:
    """Wires up MLflow tracking against DagsHub. If dagshub.init() has
    already authenticated once (browser OAuth flow, cached locally), this
    is silent and instant on subsequent runs. Returns False (rather than
    raising) if tracking is unreachable, so training/evaluation can still
    run locally without DagsHub as a hard dependency."""
    try:
        import dagshub

        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return True
    except Exception as e:
        logging.info(f"MLflow/DagsHub tracking unavailable, continuing without it: {e}")
        return False