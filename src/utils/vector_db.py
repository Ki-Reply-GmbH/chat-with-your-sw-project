import certifi
import os
from pymongo import MongoClient
from typing import List

class VectorDB:
    def __init__(self, host: str = os.getenv("VECTOR_DB_HOST", "localhost")):
        self.mongo_client = MongoClient(host, tlsCAFile=certifi.where())
        self.mongo_db = self.client["chatbot"]
        self.mongo_collection = self.db["project_files"]
        