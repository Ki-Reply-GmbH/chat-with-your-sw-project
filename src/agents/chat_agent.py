import pandas as pd
import numpy as np
from src.config import PromptConfig
from src.models import LLModel
from sklearn.metrics.pairwise import cosine_similarity
from src.agents.embedding_agent import OpenAIEmbeddingAgent

class ChatAgent:
    def __init__(
            self,
            embedding_agent: OpenAIEmbeddingAgent, # Pfad zur CSV-Datei mit den Embeddings
            config: PromptConfig,
            model: LLModel
            ):
        self.embedding_agent = embedding_agent
        self.config = config
        self.model = model

    
    def chat(self):
        while True:
            user_input = input("How can I help you with information about the software project (or 'x' to exit): ")
            if user_input == 'x':
                print("Exiting...")
                break
            # TODO ...
            top_results = self.embedding_agent.similarity_search(user_input)
    
    def call_gpt(self, document_path, chunk):
        prompt = self._prompts.get_chat_with_your_sw_project_prompt()
        return self.model._get_llm_completion(
            prompt.format(
                document_path=document_path,
                chunk=chunk
            )
        ).join(["\n\n", "Source: ", document_path])