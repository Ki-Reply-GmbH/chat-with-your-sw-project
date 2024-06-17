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
from src.agents.embedding_agent import EmbeddingAgent
from src.config import load_config
from src.models import LLModel

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
    
    with open("responses.json", "w", encoding="utf-8") as f:
        json.dump(doc_agent.responses, f, ensure_ascii=False, indent=4)

    with open("module_responses.json", "w", encoding="utf-8") as f:
        json.dump(doc_agent.module_responses, f, ensure_ascii=False, indent=4)

    
    """
    docstr_agent = DocstringAgent(
        config.WORKING_DIR,
        py_filepaths,
        config.prompts,
        LLModel(config, cache)
    )
    print("Creating documentation for python code...")
    docstr_agent.make_in_code_docs()
    print("Found these python files:") 
    keys = [key for key in docstr_agent.responses.keys()]
    print(keys)

    with open("responses.json", "w") as f:
        json.dump(docstr_agent.responses, f, indent=4)
    """

def main_embedding():
    module_responses = json.loads(open("module_responses.json").read())

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

    #print("Loaded files in total: ", len(documents))
    #print("File paths:")
    for document in documents:
        print(document.metadata["source"] + ":")
        if document.metadata["source"].endswith(".py"):
            #print(module_responses[document.metadata["source"]])
            #print("---------------------------------------------------------")
            document.page_content = module_responses[document.metadata["source"]] # Ersetze Python Code mit den textuellen Beschreibungen vom Code
    
    print(documents)

    # Make the embedding
    emb_agent = EmbeddingAgent(documents, use_full_text=True)
    emb_agent.make_embeddings()

    print("Embeddings:")
    print(emb_agent.document_embeddings)
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(emb_agent.document_embeddings, f, ensure_ascii=False, indent=4)

def main_db():
    mongo_db = MongoDBAtlasVectorDB()
    mongo_db.connect()
    

if __name__ == "__main__":
    main_embedding()