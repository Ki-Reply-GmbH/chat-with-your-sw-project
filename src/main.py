import sys
import io
import json
import os
from src.utils.git_handler import GitHandler
from src.utils.cache import DisabledCache, SimpleCache
from src.utils.directory_loader import DirectoryLoader
from src.agents.docstring_agent import DocstringAgent
from src.agents.chat_agent import ChatAgent
from src.config import load_config
from src.models import LLModel

def main():
    # Allow prinint utf-8 characters in console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

    config = load_config()
    cache = SimpleCache(tmp_path="./.tmp")

    dir_loader = DirectoryLoader(directory="./resources/cookiecutter")
    # Ermitteln des absoluten Pfads
    absolute_path = os.path.abspath(dir_loader.directory)
    print("Absoluter Pfad:", absolute_path)
    documents = dir_loader.load()

    print("Loaded files in total: ", len(documents))
    print("File paths:")
    for document in documents:
        print(document.metadata["source"])

    
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

    with open('responses.json', 'w') as f:
        json.dump(docstr_agent.responses, f, indent=4)
    """

if __name__ == "__main__":
    main()