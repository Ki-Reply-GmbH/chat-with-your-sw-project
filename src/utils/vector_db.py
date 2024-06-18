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

    def insert_vector(self, vector, metadata):
        document = {"vector": vector, "metadata": metadata}
        result = self.collection.insert_one(document)
        return result.inserted_id

    def find_similar_vectors(self, vector, k=5):
        pipeline = [
            {
                "$search": {
                    "index": "vectorIndex",  # Der Name Ihres Vektorindexes
                    "compound": {
                        "should": [
                            {
                                "vector": {
                                    "path": "embeddings",  # Der Pfad zum Vektorfeld in Ihren Dokumenten
                                    "query": vector,  # Der Vektor, zu dem die Ähnlichkeit berechnet werden soll
                                    "cosineSimilarityField": "score"  # Feld, in dem die Ähnlichkeitsbewertung gespeichert wird
                                }
                            }
                        ]
                    }
                }
            },
            {"$limit": k},  # Begrenzt die Ergebnisse auf die Top-5
            {"$project": {"_id": 0, "document": "$$ROOT", "score": 1}}  # Passt die zurückgegebenen Felder an
        ]

        results = list(self.collection.aggregate(pipeline))
        return results

