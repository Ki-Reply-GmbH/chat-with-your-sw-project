import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

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