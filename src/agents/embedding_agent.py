import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

class EmbeddingAgent:
    def __init__(self, documents: list, use_full_text: bool = False):
        self.documents = documents
        if use_full_text:
            self.chunks = self.use_full_text()
        else:
            self.chunks = self.split_text()
        self.document_embeddings = {}

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
    
    def embed_text(
            self,
            text,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
            ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
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
            self.document_embeddings[chunk["document_id"]]["embeddings"].append(embedding_list)

