import os

class FileRetriever:
    def __init__(self, directory):
        self.directory = directory
        self.ignored_dirs = [
            "__pycache__",
            "venv",
            "node_modules",
            "dist",
            "build",
            "out",
            "target",
            "bin",
            "obj",
            "lib",
            "include",
            "logs"
        ] #TODO files ignorieren, ggf. Einträge aus .gitignore verwenden, falls vorhanden
        self.ignored_files = [
            "Thumbs.db",
            "desktop.ini"
        ]
        self.file_list = []
        self.file_mapping = {} # Keys: File extensions, Values: List of file paths

        self._find_files()
        self._file_mapping()

    def _find_files(self):
        # Liste zum Speichern der gefundenen Dateien
        file_list = []

        # Durchlaufen des Verzeichnisses und seiner Unterverzeichnisse
        for root, dirs, files in os.walk(self.directory):
            # Ignorieren von versteckten Verzeichnissen
            dirs[:] = [d for d in dirs if not d[0] == '.' and not d in self.ignored_dirs]

            # Hinzufügen aller gefundenen Dateien zur Liste, die nicht mit . beginnen.
            for file in files:
                if not file[0] == '.' and not file in self.ignored_files:
                    file_list.append(os.path.abspath(os.path.join(root, file)))

        # Zurückgeben der Liste mit den gefundenen Dateien
        self.file_list = file_list

    def _file_mapping(self):
        # Dictionary zum Speichern des Mappings
        mapping = {}

        # Durchlaufen der Liste mit Dateinamen
        for file in self.file_list:
            # Extrahieren der Dateiendung
            _, extension = os.path.splitext(file)

            # Entfernen des Punkts von der Dateiendung
            extension = extension[1:]

            # Hinzufügen der Datei zum entsprechenden Eintrag im Dictionary
            if extension in mapping:
                mapping[extension].append(file)
            else:
                mapping[extension] = [file]

        # Zurückgeben des Mappings
        self.file_mapping = mapping
    
    def get_mapping(self):
        return self.file_mapping
    
    def __str__(self):
        ret = "Directory: " + self.directory + "\n"
        for key, value in self.file_mapping.items():
            ret += key + ":\n"
            for file in value:
                ret += "  " + file + "\n"
        return ret