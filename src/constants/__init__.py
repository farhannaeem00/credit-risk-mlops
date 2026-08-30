import os

# --- pipeline-wide ---
PIPELINE_NAME: str = "credit_risk"
ARTIFACT_DIR: str = "artifact"
TARGET_COLUMN: str = "TARGET"

FILE_NAME: str = "application_data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")
MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# --- Data Ingestion ---
# Reads directly from the Kaggle CSV on disk. We originally routed this
# through MongoDB Atlas (matching the reference project's pattern), but
# Atlas's TLS handshake is being blocked/interfered with on this network
# (confirmed via raw `openssl s_client` tests entirely outside Python/
# pymongo). Reading the CSV directly removes that dependency for the
# core pipeline.
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
RAW_DATA_FILE_PATH: str = os.path.join("data", "application_train.csv")

# --- Optional: MongoDB kept for the later prediction-logging /
# monitoring phase (projectflow.txt steps 41-42), not for core ingestion. ---
DATABASE_NAME: str = "credit_risk_db"
PREDICTION_LOG_COLLECTION_NAME: str = "prediction_logs"
MONGODB_URL_KEY: str = "MONGODB_URL"

# --- Data Transformation ---
ID_COLUMN: str = "SK_ID_CURR"
DAYS_EMPLOYED_ANOMALY_VALUE: int = 365243  # Home Credit's placeholder for "not employed"

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
TRANSFORMED_TRAIN_FILE_NAME: str = "train.npy"
TRANSFORMED_TEST_FILE_NAME: str = "test.npy"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.joblib"

# --- Model Trainer ---
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.joblib"
MODEL_TRAINER_EXPECTED_ROC_AUC: float = 0.65  # sanity floor, not the bar we're aiming for

DAGSHUB_REPO_OWNER: str = "farhannaeem00"
DAGSHUB_REPO_NAME: str = "credit-risk-mlops"
MLFLOW_EXPERIMENT_NAME: str = "home-credit-default-risk"

# --- Model Evaluation & Pusher ---
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "report.yaml"
ROC_AUC_TIE_BREAK_DELTA: float = 0.01  # within this, prefer higher PR-AUC over raw ROC-AUC

REGISTERED_MODEL_NAME: str = "home-credit-default-risk-model"
PRODUCTION_ALIAS: str = "production"