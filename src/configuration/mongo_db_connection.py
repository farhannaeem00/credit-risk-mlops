import os
import sys

import certifi
import pymongo

from src.constants import DATABASE_NAME, MONGODB_URL_KEY
from src.exception import CustomException
from src.logger import logging

# certifi gives us a trusted CA bundle so TLS to Atlas works consistently
# across OSes (this bites people on some Windows/conda setups otherwise).
_ca = certifi.where()


class MongoDBClient:
    """
    Thin wrapper around pymongo.MongoClient. Reads the connection string
    from the MONGODB_URL env var — never hardcode credentials in source.

    The class-level `client` means repeated instantiations within one
    process reuse the same underlying connection instead of opening a new
    one every time.
    """

    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(
                        f"Environment variable '{MONGODB_URL_KEY}' is not set. "
                        f"See projectflow.txt steps 5-9 for how to get your "
                        f"Atlas connection string."
                    )
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=_ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection established successfully.")
        except Exception as e:
            raise CustomException(e, sys)
