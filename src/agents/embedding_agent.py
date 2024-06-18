import os
import torch
import pandas as pd
from openai import OpenaAI
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
        self.client = OpenaAI()
        self.documents = documents
        self.model = model
        if mode == "full_text":
            self.chunks = self.use_full_text()
        elif mode == "chunks":
            self.chunks = self.split_text()
        elif mode == "user_query":
            self.chunks = documents
        self.document_embeddings = {}

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
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = []
        for doc in self.documents:
            doc_chunks = splitter.split_text(doc.page_content)
            for chunk in doc_chunks:
                chunks.append(
                    {
                        "document_id": doc.metadata["source"],
                        "document_name": os.path.basename(doc.metadata["source"]),
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

    def write_to_csv(self):
        df_data = []
        for key, values in self.document_embeddings.items():
            # Konvertieren der Embeddings-Liste in einen String mit kommagetrennten Werten
            embeddings_str = ",".join(map(str, values["embeddings"]))
            # Hinzufügen der Zeile (Embeddings, Document Name, Document ID) zur Liste
            df_data.append([embeddings_str, values["document_name"], values["document_id"]])
        
        df = pd.DataFrame(df_data, columns=["Embeddings", "Document Name", "Document ID"])
        df.to_csv("document_embeddings.csv", index=False)

    def similarity_search(self, text_input, top_n=5):
        # Embedding für den Text-Input erhalten
        text_embedding = self.get_embedding(text_input).reshape(1, -1)
        
        # Berechnen der Kosinusähnlichkeit zwischen dem Text-Input und allen Embeddings im DataFrame
        self.df["Similarity"] = self.df["Embeddings"].apply(lambda x: cosine_similarity(x.reshape(1, -1), text_embedding)[0][0])
        
        # Sortieren der Ergebnisse nach Ähnlichkeit und Auswahl der Top-N-Ergebnisse
        top_results = self.df.sort_values(by="Similarity", ascending=False).head(top_n)
        
        return top_results
    
class EmbeddingAgent:
    def __init__(
            self,
            documents: list,
            mode: str = "full_text",
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
            ):
        self.documents = documents
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if mode == "full_text":
            self.chunks = self.use_full_text()
        elif mode == "chunks":
            self.chunks = self.split_text()
        elif mode == "user_query":
            self.chunks = documents
        self.document_embeddings = {}   # Enthält die Embeddings der chunks bzw documents

    def use_full_text(self) -> list:
        chunks = []
        for doc in self.documents:
            chunks.append(
                {
                    "document_id": doc.metadata["source"],
                    "document_name": os.path.basename(doc.metadata["source"]),
                    "text": doc.page_content  # Verwende den gesamten Inhalt des Dokuments
                }
            )
        return chunks

    def split_text(self) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = []
        for doc in self.documents:
            doc_chunks = splitter.split_text(doc.page_content)
            for chunk in doc_chunks:
                chunks.append(
                    {
                        "document_id": doc.metadata["source"],
                        "document_name": os.path.basename(doc.metadata["source"]),
                        "text": chunk
                    }
                )
        return chunks
    
    def embed_text(self,text: str):
        preprocessed_text = self._remove_unknown_words(text)
        inputs = self.tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # average pooling over tokens
        return embeddings

    def make_embeddings(self):
        for chunk in self.chunks:
            embedding = self.embed_text(chunk["text"])
            embedding_list = embedding.squeeze().tolist()  # convert tensor to list
            
            # Aggregate embeddings for each document
            if chunk["document_id"] not in self.document_embeddings:
                self.document_embeddings[chunk["document_id"]] = {
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "embeddings": []
                }
            self.document_embeddings[chunk["document_id"]]["embeddings"] = embedding_list # war vorher append


    def _remove_unknown_words(self, text):
        """
        Entfernt unbekannte Wörter aus dem Text basierend auf dem Vokabular des Tokenizers.
        """
        known_vocabulary = self.tokenizer.get_vocab()
        tokens = text.split() 
        filtered_tokens = [token for token in tokens if token in known_vocabulary]
        return "" "".join(filtered_tokens)