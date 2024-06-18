import certifi
import os
from pymongo import MongoClient
from typing import List

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
            database_name = "embeddings",
            collection_name = "document_embeddings"
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

    def insert_document(self, document):
        result = self.collection.insert_one(document)
        return result.inserted_id

    def find_similar_vectors(self, vector, k=5):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",  # Name des Suchindexes
                    "path": "embeddings",  # Feldname, in dem die Embedding-Vektoren gespeichert sind
                    "queryVector": vector,  # Ihr 384-dimensionaler Vektor
                    "numCandidates": k * 2,  # Anzahl der Kandidaten, die für die Suche berücksichtigt werden sollen, hier doppelt so viele wie k, um die Genauigkeit zu erhöhen
                    "limit": k,  # Anzahl der zurückzugebenden nächsten Nachbarn
                    "filter": {}  # Optional, falls Sie die Ergebnisse basierend auf bestimmten Kriterien filtern möchten
                }
            }
        ]

        results = list(self.collection.aggregate(pipeline))
        return results
