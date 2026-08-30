"""
One-off script: push a locally-downloaded Home Credit CSV into MongoDB
Atlas as the raw data store (projectflow.txt steps 10-12).

1. Get application_train.csv, e.g.:
     kaggle competitions download -c home-credit-default-risk \
         -f application_train.csv -p data/
   (unzip if needed so you end up with data/application_train.csv)

2. export MONGODB_URL="mongodb+srv://<user>:<pass>@..."

3. python scripts/push_data_to_mongodb.py
"""
import os
import sys

import certifi
import pandas as pd
import pymongo

DATABASE_NAME = "credit_risk_db"
COLLECTION_NAME = "home_credit_application"
DATA_FILE_PATH = "data/application_train.csv"


def main() -> None:
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        print("MONGODB_URL is not set. See projectflow.txt steps 5-9.")
        sys.exit(1)

    if not os.path.exists(DATA_FILE_PATH):
        print(
            f"{DATA_FILE_PATH} not found. Download it first, e.g.:\n"
            "  kaggle competitions download -c home-credit-default-risk "
            "-f application_train.csv -p data/"
        )
        sys.exit(1)

    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {DATA_FILE_PATH}")

    records = df.to_dict(orient="records")

    client = pymongo.MongoClient(mongo_url, tlsCAFile=certifi.where())
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # Idempotent: wipe and reinsert, so re-running this script during
    # development doesn't duplicate documents.
    deleted = collection.delete_many({}).deleted_count
    if deleted:
        print(f"Cleared {deleted} existing documents from {DATABASE_NAME}.{COLLECTION_NAME}")

    collection.insert_many(records)
    print(f"Inserted {len(records)} documents into {DATABASE_NAME}.{COLLECTION_NAME}")


if __name__ == "__main__":
    main()
