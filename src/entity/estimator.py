import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import mlflow.pyfunc


class CreditRiskModel:
    """
    Bundles the fitted preprocessing pipeline with a trained model so
    prediction_pipeline.py only needs to load one object. Keeps the
    "which transform ran on this model" question from ever coming up.
    """

    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame):
        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, dataframe: DataFrame):
        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict_proba(transformed_feature)
        except Exception as e:
            raise CustomException(e, sys)

    def __repr__(self):
        return f"CreditRiskModel(model={type(self.trained_model_object).__name__})"

    def __str__(self):
        return f"CreditRiskModel(model={type(self.trained_model_object).__name__})"

    import mlflow.pyfunc


class MLflowCreditRiskModel(mlflow.pyfunc.PythonModel):
    """
    Thin adapter so CreditRiskModel (preprocessor + trained model bundled
    together) can be logged as a single MLflow pyfunc artifact. This is
    what makes `models:/<name>@production` a self-contained prediction
    unit - load one thing, call .predict(), done. No separate preprocessor
    file to track down at serving time.
    """

    def __init__(self, credit_risk_model: CreditRiskModel):
        self.credit_risk_model = credit_risk_model

    def predict(self, context, model_input):
        # Returns probability of default (class 1), which is what the
        # prediction API and demo form actually want to show.
        return self.credit_risk_model.predict_proba(model_input)[:, 1]