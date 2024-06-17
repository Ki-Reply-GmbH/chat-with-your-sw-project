import certifi
import os
from pymongo import MongoClient
from typing import List
from milvus import Milvus, MetricType, IndexType

class VectorDB:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.collection = None

    def connect(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def insert_vector(self, vector, metadata):
        raise NotImplementedError("Subclasses must implement this method.")

    def find_similar_vectors(self, vector, threshold=0.5):
        raise NotImplementedError("Subclasses must implement this method.")
    

class MongoDBAtlasVectorDB(VectorDB):
    def __init__(
            self,
            connection_string = os.getenv("VECTOR_DB_HOST", "localhost"),
            database_name = "chatbot",
            collection_name = "project_files"
            ):
        super().__init__(connection_string)
        self.database_name = database_name
        self.collection_name = collection_name
        self.db_client = None
        self.database = None
        self.collection = None 

    def connect(self):
        self.db_client = MongoClient(self.connection_string)
        self.database = self.db_client[self.database_name]
        self.collection = self.database[self.collection_name]

    def insert_vector(self, vector, metadata):
        document = {"vector": vector, "metadata": metadata}
        result = self.collection.insert_one(document)
        return result.inserted_id

    def find_similar_vectors(self, vector, threshold=0.5):
        pass  # Implementierung für die Suche nach ähnlichen Vektoren in MongoDB


class MilvusVectorDB(VectorDB):
    def __init__(self, host, port, collection_name, dimension, index_file_size=1024, metric_type=MetricType.L2):
        super().__init__(None)  # Milvus benötigt keinen connection_string, stattdessen host und port
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_file_size = index_file_size
        self.metric_type = metric_type
        self.client = Milvus(host=self.host, port=self.port)
        self._create_collection()

    def _create_collection(self):
        param = {
            'collection_name': self.collection_name,
            'dimension': self.dimension,
            'index_file_size': self.index_file_size,
            'metric_type': self.metric_type
        }
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(param)

    def insert_vector(self, vector, metadata):
        if not isinstance(vector, list):
            vector = [vector]
        status, ids = self.client.insert(collection_name=self.collection_name, records=vector, partition_tag=None)
        return ids

    def find_similar_vectors(self, vector, top_k=10, search_params={"nprobe": 16}):
        if not isinstance(vector, list):
            vector = [vector]
        status, results = self.client.search(collection_name=self.collection_name, query_records=vector, top_k=top_k, params=search_params)
        return results

    def connect(self):
        pass  # Verbindung wird im Konstruktor hergestellt