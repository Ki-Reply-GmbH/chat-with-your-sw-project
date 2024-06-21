import sys
import io
import json
import os
import openai
from src.utils.git_handler import GitHandler
from src.utils.cache import DisabledCache, SimpleCache
from src.utils.directory_loader import DirectoryLoader
from src.utils.vector_db import VectorDB, MongoDBAtlasVectorDB
from src.agents.docstring_agent import DocstringAgent
from src.agents.chat_agent import ChatAgent
from src.agents.embedding_agent import OpenAIEmbeddingAgent
from src.config import load_config, LOGGER
from src.models import LLModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from pprint import pprint

def main():
    # Allow prinint utf-8 characters in console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")

    config = load_config()
    cache = SimpleCache(tmp_path="./.tmp")

    LOGGER.info("Loading relevant documents...")
    dir_loader = DirectoryLoader(directory="./resources/cookiecutter")
    # Ermitteln des absoluten Pfads
    absolute_path = os.path.abspath(dir_loader.directory)
    documents = dir_loader.load()

    py_file_paths = []
    for document in documents:
        print(document.metadata["source"])
        if document.metadata["source"].endswith(".py"):
            py_file_paths.append(document.metadata["source"])

    LOGGER.info("Creating the embeddings (this takes some time)...")
    emb_agent = OpenAIEmbeddingAgent(documents, mode="chunks")
    emb_agent.make_embeddings()
    emb_agent.write_to_csv()

    chat_agent = ChatAgent(emb_agent, config.prompts, LLModel(config))
    chat_agent.chat()

if __name__ == "__main__":
    main()