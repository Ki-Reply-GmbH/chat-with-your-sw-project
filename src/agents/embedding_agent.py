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
            documents: list = None,
            mode: str = None,
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
        else:
            self.chunks = None
        self.document_embeddings = {}
        self.df = None

    def use_full_text(self) -> list:
        chunks = []
        for doc in self.documents:
            chunks.append(
                {
                    "document_path": doc.metadata["source"],
                    "chunk_id": os.path.basename(doc.metadata["source"]),
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
            doc_chunks = splitter.split_text(doc.page_content)
            for i, chunk in enumerate(doc_chunks):
                chunks.append(
                    {
                        "document_path": doc.metadata["source"],
                        "chunk_id": os.path.basename(
                            "".join([doc.metadata["source"], "_chunk", str(i)])
                            ),
                        "text": chunk.replace("\n", " ")
                    }
                )
        return chunks

    def get_chunk(self, document_path: str, chunk_id: str):
        # Durchsuchen der Chunks nach dem gegebenen Dokumentennamen
        for chunk in self.chunks:
            if chunk["document_path"] == document_path \
                and chunk["chunk_id"] == chunk_id:
                return chunk
        # Wenn kein entsprechender Chunk gefunden wurde, Rückgabe von None
        return None

    def make_embedding(self, text: str):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(
            input = [text],
            model=self.model
            ).data[0].embedding
    
    def make_embeddings(self):
        for chunk in self.chunks:
            embedding = self.make_embedding(chunk["text"])
            self.document_embeddings[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                "document_path": chunk["document_path"],
                "embeddings": embedding
            }

    def load_from_csv(self, file_paths=["src/models/document_embeddings.csv", "src/models/chunks.csv"]):
        # Laden der Embeddings aus einer CSV-Datei
        self.df = pd.read_csv(file_paths[0])

        # Laden der Chunks aus einer CSV-Datei
        chunks_df = pd.read_csv(file_paths[1])
        self.chunks = [
            {
                "document_path": row["document_path"],
                "chunk_id": row["chunk_id"],
                "text": row["text"]
            }
            for _, row in chunks_df.iterrows()
        ]
            
    def write_to_csv(self, file_paths=["src/models/document_embeddings.csv", "src/models/chunks.csv"]):
        # Schreiben der Embeddings in eine CSV-Datei
        df_data = []
        for _, values in self.document_embeddings.items():
            # Konvertieren der Embeddings-Liste in einen String mit kommagetrennten Werten
            embeddings_str = ",".join(map(str, values["embeddings"]))
            # Hinzufügen der Zeile (Embeddings, Document Name, Document ID) zur Liste
            df_data.append([embeddings_str, values["document_path"], values["chunk_id"]])
        
        self.df = pd.DataFrame(df_data, columns=["Embeddings", "Document Path", "Chunk ID"])
        self.df.to_csv(file_paths[0], index=False)

        # Schreiben der Chunks in eine CSV-Datei
        chunks_df = pd.DataFrame(self.chunks)
        chunks_df.to_csv(file_paths[1], index=False)

    def similarity_search(self, text_input, top_n=5):
        # Embedding für den Text-Input erhalten
        text_embedding = np.array(self.make_embedding(text_input)).reshape(1, -1)  # cosine_similarity erwartet 2d array
        
        # Erstellen einer Kopie von self.df, um Original-DataFrame nicht zu verändern
        df_copy = self.df.copy()
        
        # Berechnen der Kosinusähnlichkeit zwischen dem Text-Input und allen Embeddings im kopierten DataFrame
        df_copy["Similarity"] = df_copy["Embeddings"].apply(
            lambda x: cosine_similarity(
                np.array([float(num) for num in x.split(",")]).reshape(1, -1),
                text_embedding
            )[0][0])
        
        # Sortieren der Ergebnisse nach Ähnlichkeit und Auswahl der Top-N-Ergebnisse
        top_results = df_copy.sort_values(by="Similarity", ascending=False).head(top_n)
        
        return top_results
