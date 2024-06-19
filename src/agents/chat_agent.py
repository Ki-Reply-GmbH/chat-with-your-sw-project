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
            self.embedding_agent.similarity_search(user_input)