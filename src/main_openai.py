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
from src.agents.embedding_agent import EmbeddingAgent, OpenAIEmbeddingAgent
from src.config import load_config
from src.models import LLModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from pprint import pprint

def main():
    # Allow prinint utf-8 characters in console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")

    config = load_config()
    print("Model name: ")
    print(config.LLM_MODEL_NAME)
    cache = SimpleCache(tmp_path="./.tmp")


    dir_loader = DirectoryLoader(directory="./resources/cookiecutter")
    # Ermitteln des absoluten Pfads
    absolute_path = os.path.abspath(dir_loader.directory)
    print("Absoluter Pfad:", absolute_path)
    documents = dir_loader.load()

    py_file_paths = []
    print("Loaded files in total: ", len(documents))
    print("File paths:")
    for document in documents:
        print(document.metadata["source"])
        if document.metadata["source"].endswith(".py"):
            py_file_paths.append(document.metadata["source"])
    """
    print("Python files:")
    print(py_file_paths)

    doc_agent = DocstringAgent(
        config.WORKING_DIR,
        py_file_paths,
        config.prompts,
        LLModel(config, cache)
    )
    doc_agent.make_docstrings()
    doc_agent.make_module_descriptions()
    """
    module_responses = json.loads(open("module_responses.json").read())

    emb_agent = OpenAIEmbeddingAgent(documents, mode="chunks")
    emb_agent.make_embeddings()
    emb_agent.write_to_csv()
    
    print("----- Similarity Search -----")
    while True:
        user_input = input("Geben Sie Ihre Anfrage ein (oder 'x' zum Beenden): ")
        if user_input == 'x':
            print("Beenden...")
            break

        # Verwenden Sie den Benutzerinput für die similarity_search
        result = emb_agent.similarity_search(user_input, 5)
        #emb_agent.write_to_csv("document_embeddings_sim.csv")
        pprint(result)

if __name__ == "__main__":
    main()