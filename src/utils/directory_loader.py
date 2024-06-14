import os
from langchain.document_loaders import PythonLoader, UnstructuredRSTLoader, UnstructuredMarkdownLoader, TextLoader
from typing import Type
from pathlib import Path

class DirectoryLoader:
    def __init__(self, directory: str, ignored_dirs: list = [], ignored_files: list = []):
        self.directory = directory
        self.ignored_files = ignored_dirs
        self.ignored_dirs = ignored_files

    def load(self) -> list:
        documents = []
        for file_path in self._list_files():
            file_extension = Path(file_path).suffix
            loader_class = self._get_loader_for_extension(file_extension)
            if loader_class:
                loader = loader_class(file_path)
                documents.extend(loader.load())
        return documents

    def _list_files(self) -> list:
        file_list = []

        # Durchlaufen des Verzeichnisses und seiner Unterverzeichnisse
        for root, dirs, files in os.walk(self.directory):
            dirs[:] = [d for d in dirs if not d in self.ignored_dirs]

            for file in files:
                if not file in self.ignored_files:
                    file_list.append(os.path.abspath(os.path.join(root, file)))

        return file_list

    def _get_loader_for_extension(self, extension: str) -> Type[TextLoader]:
        loaders = {
            #".txt": TextLoader,
            #".pdf": CustomPDFLoader,
            ".rst": UnstructuredRSTLoader,
            ".py": PythonLoader,
            ".md": UnstructuredMarkdownLoader
            # Add loaders for other file types as necessary
        }
        return loaders.get(extension.lower())