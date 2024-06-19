import os
import torch
import pandas as pd
import numpy as np
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class OpenAIEmbeddingAgent:
    def __init__(
            self,
            documents: list,
            mode: str = "full_text",
            model: str = "text-embedding-3-large"
            ):
        self.client = OpenAI()
        self.documents = documents
        self.model = model
        if mode == "full_text":
            self.chunks = self.use_full_text()
        elif mode == "chunks":
            self.chunks = self.split_text()
        elif mode == "user_query":
            self.chunks = documents
        self.document_embeddings = {}
        self.df = None

    def use_full_text(self) -> list:
        chunks = []
        for doc in self.documents:
            chunks.append(
                {
                    "document_id": doc.metadata["source"],
                    "document_name": os.path.basename(doc.metadata["source"]),
                    "text": doc.page_content.replace("\n", " ")
                }
            )
        return chunks

    def split_text(self) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=16000,
            chunk_overlap=2000
        )
        chunks = []
        for doc in self.documents:
            print("Creating chunks for document:", doc.metadata["source"])
            doc_chunks = splitter.split_text(doc.page_content)
            for i, chunk in enumerate(doc_chunks):
                print("Chunk", i, "created")
                chunks.append(
                    {
                        "document_id": doc.metadata["source"],
                        "document_name": os.path.basename(
                            "".join([doc.metadata["source"], "_chunk", str(i)])
                            ),
                        "text": chunk.replace("\n", " ")
                    }
                )
        return chunks
    
    def get_embedding(self, text: str):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(
            input = [text],
            model=self.model
            ).data[0].embedding
    
    def make_embeddings(self):
        for chunk in self.chunks:
            embedding = self.get_embedding(chunk["text"])
            self.document_embeddings[chunk["document_id"]] = {
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "embeddings": embedding
            }

    def load_from_csv(self, filename="document_embeddings.csv"):
        # Lesen der CSV-Datei in einen DataFrame
        self.df = pd.read_csv(filename)
        
        # Konvertieren der Embeddings von Strings zurück in Listen von Floats
        self.df['Embeddings'] = self.df['Embeddings'].apply(lambda x: [float(num) for num in x.split(",")])
        
        return self.df
    
    def write_to_csv(self, filename="document_embeddings.csv"):
        df_data = []
        for key, values in self.document_embeddings.items():
            # Konvertieren der Embeddings-Liste in einen String mit kommagetrennten Werten
            embeddings_str = ",".join(map(str, values["embeddings"]))
            # Hinzufügen der Zeile (Embeddings, Document Name, Document ID) zur Liste
            df_data.append([embeddings_str, values["document_name"], values["document_id"]])
        
        self.df = pd.DataFrame(df_data, columns=["Embeddings", "Document Name", "Document ID"])
        self.df.to_csv(filename, index=False)

    def similarity_search(self, text_input, top_n=5):
        # Embedding für den Text-Input erhalten
        text_embedding = np.array(self.get_embedding(text_input)).reshape(1, -1) # cosine_similarity erwartet 2d array
        
        # Berechnen der Kosinusähnlichkeit zwischen dem Text-Input und allen Embeddings im DataFrame
        self.df['Similarity'] = self.df['Embeddings'].apply(
            lambda x: cosine_similarity(
                np.array([float(num) for num in x.split(",")]).reshape(1, -1),
                text_embedding
            )[0][0])
        
        # Sortieren der Ergebnisse nach Ähnlichkeit und Auswahl der Top-N-Ergebnisse
        top_results = self.df.sort_values(by='Similarity', ascending=False).head(top_n)
        
        return top_results
