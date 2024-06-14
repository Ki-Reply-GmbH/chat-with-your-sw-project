import sys
import io
import json
from src.utils.git_handler import GitHandler
from src.utils.cache import DisabledCache, SimpleCache
from src.utils.file_retriever import FileRetriever
from src.agents.docstring_agent import DocstringAgent
from src.agents.chat_agent import ChatAgent
from src.config import load_config
from src.models import LLModel

def main():
    pass

if __name__ == "__main__":
    # Allow prinint utf-8 characters in console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

    config = load_config()
    cache = SimpleCache(tmp_path="./.tmp")

    py_filepaths = FileRetriever("resources/code").file_mapping["py"]
    print(py_filepaths)
    
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