"""
template.py

Run this once to scaffold the entire project's folder/file structure.
Mirrors the workflow: `python template.py` -> creates every empty
file/dir the project needs -> you fill them in phase by phase.

Usage:
    python template.py
"""
import os
from pathlib import Path

PROJECT_NAME = "src"

list_of_files = [
    ".github/workflows/ci.yaml",
    ".github/workflows/cd.yaml",

    "config/model.yaml",
    "config/schema.yaml",

    f"{PROJECT_NAME}/__init__.py",

    f"{PROJECT_NAME}/cloud_storage/__init__.py",
    f"{PROJECT_NAME}/cloud_storage/dagshub_storage.py",

    f"{PROJECT_NAME}/components/__init__.py",
    f"{PROJECT_NAME}/components/data_ingestion.py",
    f"{PROJECT_NAME}/components/data_validation.py",
    f"{PROJECT_NAME}/components/data_transformation.py",
    f"{PROJECT_NAME}/components/model_trainer.py",
    f"{PROJECT_NAME}/components/model_evaluation.py",
    f"{PROJECT_NAME}/components/model_pusher.py",

    f"{PROJECT_NAME}/configuration/__init__.py",
    f"{PROJECT_NAME}/configuration/mongo_db_connection.py",
    f"{PROJECT_NAME}/configuration/mlflow_connection.py",

    f"{PROJECT_NAME}/constants/__init__.py",

    f"{PROJECT_NAME}/data_access/__init__.py",
    f"{PROJECT_NAME}/data_access/credit_data.py",

    f"{PROJECT_NAME}/entity/__init__.py",
    f"{PROJECT_NAME}/entity/config_entity.py",
    f"{PROJECT_NAME}/entity/artifact_entity.py",
    f"{PROJECT_NAME}/entity/estimator.py",
    f"{PROJECT_NAME}/entity/mlflow_estimator.py",

    f"{PROJECT_NAME}/exception/__init__.py",
    f"{PROJECT_NAME}/logger/__init__.py",

    f"{PROJECT_NAME}/pipeline/__init__.py",
    f"{PROJECT_NAME}/pipeline/training_pipeline.py",
    f"{PROJECT_NAME}/pipeline/prediction_pipeline.py",

    f"{PROJECT_NAME}/utils/__init__.py",
    f"{PROJECT_NAME}/utils/main_utils.py",

    "static/css/style.css",
    "templates/loan_default_form.html",

    "tests/__init__.py",
    "tests/test_data_validation.py",
    "tests/test_api.py",

    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "app.py",
    "crashcourse.txt",
    "demo.py",
    "projectflow.txt",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
]


def create_project_structure():
    for filepath in list_of_files:
        path = Path(filepath)
        filedir = path.parent

        if str(filedir) != "":
            os.makedirs(filedir, exist_ok=True)

        if (not path.exists()) or (path.stat().st_size == 0):
            with open(path, "w") as f:
                pass
            print(f"created: {filepath}")
        else:
            print(f"skipped (already exists & non-empty): {filepath}")


if __name__ == "__main__":
    create_project_structure()
    print("\nProject structure created. Next: fill in setup.py, pyproject.toml, "
          "requirements.txt, then follow projectflow.txt phase by phase.")
