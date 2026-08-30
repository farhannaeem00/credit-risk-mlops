"""
Sanity-check script for each phase as we build it. Run this after every
phase to confirm the new pieces work before moving on.
"""
import os
import sys

from src.logger import logging
from src.exception import CustomException

logging.info("demo.py started")

# --- 1. logger + exception sanity check ---
try:
    1 / 0
except Exception as e:
    logging.info("Caught a deliberate error to sanity-check CustomException")
    print(CustomException(e, sys))

logging.info("Logger + exception check passed.")

# --- 2. data ingestion + validation sanity check ---
from src.constants import RAW_DATA_FILE_PATH, SCHEMA_FILE_PATH

if not os.path.exists(RAW_DATA_FILE_PATH):
    logging.info(f"{RAW_DATA_FILE_PATH} not found. Download it via the Kaggle CLI first.")
elif not os.path.exists(SCHEMA_FILE_PATH):
    logging.info(
        f"{SCHEMA_FILE_PATH} not found. Run 'python scripts/generate_schema.py' first."
    )
else:
    from src.pipeline.training_pipeline import TrainPipeline

    logging.info("Running full pipeline (ingestion + validation)")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()