import pandas as pd
import numpy as np
import os
from src.config import PromptConfig
from src.models import LLModel
from sklearn.metrics.pairwise import cosine_similarity
from src.agents.embedding_agent import OpenAIEmbeddingAgent

class ChatAgent:
    def __init__(
            self,
            embedding_agent: OpenAIEmbeddingAgent, # Pfad zur CSV-Datei mit den Embeddings
            prompts: PromptConfig,
            model: LLModel
            ):
        self.embedding_agent = embedding_agent
        self._prompts = prompts
        self._model = model

    
    def chat(self):
        while True:
            user_input = input("How can I help you with information about the software project (or 'x' to exit): ")
            if user_input == 'x':
                print("Exiting...")
                break
            top_results = self.embedding_agent.similarity_search(user_input)
            chunk_id = top_results.iloc[0]["Chunk ID"]
            document_path = top_results.iloc[0]["Document Path"]
            chunk = self.embedding_agent.get_chunk(document_path, chunk_id)
            chunk_text = chunk["text"]
            response = self.call_gpt(user_input, document_path, chunk_text)
            print(response)
    
    def call_gpt(self, user_query, document_path, chunk):
        prompt = self._prompts.get_chat_with_your_sw_project_prompt()
        completion = self._model._get_llm_completion(
            prompt.format(
                user_query=user_query,
                source_name=os.path.basename(document_path),
                chunk=chunk
            )
        )
        return f"{completion}\n\nSource: {document_path}\n\n\n"