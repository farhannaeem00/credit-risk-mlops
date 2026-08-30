from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str     

@dataclass
class ClassificationMetricArtifact:
    roc_auc: float
    pr_auc: float
    f1_score: float
    recall: float
    precision: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    best_model_name: str
    all_models_metrics: dict  # {model_name: ClassificationMetricArtifact}

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    best_model_name: str
    best_model_metrics: ClassificationMetricArtifact
    previous_production_roc_auc: float  # None if no prior Production model
    changed_roc_auc: float  # None if no prior Production model


@dataclass
class ModelPusherArtifact:
    is_model_pushed: bool
    registered_model_name: str
    model_version: str