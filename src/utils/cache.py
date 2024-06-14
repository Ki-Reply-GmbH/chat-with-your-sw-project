"""
This module provides a Cache class for caching prompts and their corresponding answers.
The Cache class includes methods to update the cache with a new prompt and answer, delete a prompt
and its answer from the cache, lookup a prompt in the cache, and get the answer for a prompt from
the cache.

The cache is stored as a CSV file in a cache folder. Each row in the CSV file represents a prompt
and the index of the row is used to name a separate CSV file that stores the answer for the prompt.
"""

import logging
import os

import pandas as pd


class Cache:
    """
    A class used to cache prompts and their corresponding answers.

    The Cache class provides methods to update the cache with a new prompt and answer, delete a
    prompt and its answer from the cache, lookup a prompt in the cache, and get the answer for a
    prompt from the cache.

    The cache is stored as a CSV file in a cache folder. Each row in the CSV file represents a
    prompt and the index of the row is used to name a separate CSV file that stores the answer for
    the prompt.
    """

    def __init__(self, tmp_path: str) -> None:
        """
        Initializes the Cache with the specified cache folder and cache file.

        This method sets the cache folder and cache file paths. If the cache folder does not exist,
        it creates it. If the cache file does not exist, it creates a new CSV file with a "prompt"
        column.

        Args:
            tmp_path (str): The path to the temporary directory.
        """
        self._tmp_path = tmp_path
        self._logger = logging.getLogger(__name__)

    def update(self, prompt: str, answer: str) -> bool:
        """
        Updates the cache with a new prompt and answer.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt already exists. If the prompt exists, it returns 1. If the prompt does not exist, it
        appends a new row with the prompt to the DataFrame and writes it to the cache file.

        If the cache file does not exist, it creates a new DataFrame with the prompt and writes it to
        the cache file.

        In both cases, it saves the answer to a new CSV file in the cache folder.

        Args:
            prompt (str): The prompt to be added to the cache.
            answer (str): The answer to be added to the cache.

        Returns:
            Optional[str]: 1 if the prompt already exists in the cache, 0 otherwise.
        """
        raise NotImplementedError

    def delete(self, prompt: str) -> None:
        """
        Deletes a prompt and its answer from the cache.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt exists. If the prompt exists, it deletes the row with the prompt from the DataFrame and
        writes the DataFrame back to the cache file. It also deletes the corresponding answer file
        from the cache folder.

        Args:
            prompt (str): The prompt to be deleted from the cache.
        """
        raise NotImplementedError

    def lookup(self, prompt: str) -> bool:
        """
        Checks if a prompt is in the cache.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt exists. If the prompt exists, it returns True. If the cache file does not exist or the
        prompt does not exist in the cache, it returns False.

        Args:
            prompt (str): The prompt to be looked up in the cache.

        Returns:
            bool: True if the prompt exists in the cache, False otherwise.
        """
        raise NotImplementedError

    def get_answer(self, prompt: str):
        """
        Gets the answer for a prompt from the cache.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt exists. If the prompt exists, it reads the corresponding answer file from the cache
        folder and returns the answer.

        If the cache file does not exist or the prompt does not exist in the cache, it returns None.

        Args:
            prompt (str): The prompt for which to get the answer.

        Returns:
            The answer for the prompt if it exists in the cache, None otherwise.
        """
        raise NotImplementedError


class DisabledCache(Cache):
    """
    A class used to disable caching.

    The DisabledCache class is a subclass of the Cache class that disables caching. It overrides the
    update, delete, lookup, and get_answer methods to return None.
    """

    def lookup(self, prompt: str) -> bool:
        """
        Disables the lookup method.

        This method always returns False.

        Args:
            prompt (str): The prompt to be looked up in the cache.

        Returns:
            bool: False.
        """
        _ = prompt
        return False

    def update(self, prompt: str, answer: str) -> bool:
        """
        Disables the update method.

        This method always returns False.

        Args:
            prompt (str): The prompt to be added to the cache.
            answer (str): The answer to be added to the cache.

        Returns:
            bool: False.
        """
        _ = prompt, answer
        return False

    def delete(self, prompt: str) -> None:
        """
        Disables the delete method.

        This method does nothing.

        Args:
            prompt (str): The prompt to be deleted from the cache.
        """
        _ = prompt

    def get_answer(self, prompt: str):
        """
        Disables the get_answer method.

        This method always returns None.

        Args:
            prompt (str): The prompt for which to get the answer.

        Returns:
            None.
        """
        _ = prompt


class SimpleCache(Cache):
    """
    A class used to cache prompts and their corresponding answers.

    The Cache class provides methods to update the cache with a new prompt and answer, delete a
    prompt and its answer from the cache, lookup a prompt in the cache, and get the answer for a
    prompt from the cache.

    The cache is stored as a CSV file in a cache folder. Each row in the CSV file represents a
    prompt and the index of the row is used to name a separate CSV file that stores the answer for
    the prompt.
    """

    def __init__(
        self,
        tmp_path: str,
        cache_folder: str = ".cache",
        cache_file: str = "prompts.csv",
    ) -> None:
        """
        Initializes the Cache with the specified cache folder and cache file.

        This method sets the cache folder and cache file paths. If the cache folder does not exist,
        it creates it. If the cache file does not exist, it creates a new CSV file with a "prompt"
        column.

        Args:
            tmp_path (str): The path to the temporary directory.
            cache_folder (str, optional): The name of the cache folder. Defaults to ".cache".
            cache_file (str, optional): The name of the cache file. Defaults to "prompts.csv".
        """
        super().__init__(tmp_path)
        self.cache_folder = os.path.join(tmp_path, cache_folder)
        self.cache_file = os.path.join(self.cache_folder, cache_file)

        # Create cache directory if it doesn"t exist
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # Create template cache file if it doesn"t exist
        if not os.path.exists(self.cache_file):
            df = pd.DataFrame(columns=["prompt"])
            df.to_csv(self.cache_file, index=False)

    def update(self, prompt: str, answer: str) -> bool:
        """
        Updates the cache with a new prompt and answer.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt already exists. If the prompt exists, it returns 1. If the prompt does not exist, it
        appends a new row with the prompt to the DataFrame and writes it to the cache file.

        If the cache file does not exist, it creates a new DataFrame with the prompt and writes it to
        the cache file.

        In both cases, it saves the answer to a new CSV file in the cache folder.

        Args:
            prompt (str): The prompt to be added to the cache.
            answer (str): The answer to be added to the cache.

        Returns:
            Optional[str]: 1 if the prompt already exists in the cache, 0 otherwise.
        """
        if os.path.exists(self.cache_file):
            # Read existing CSV file
            df = pd.read_csv(self.cache_file)

            # Check if prompt already exists
            if prompt in df["prompt"].values:
                # If prompt exists, return the answer from the respective CSV file
                row_index = df.index[df["prompt"] == prompt].tolist()[0]
                pd.read_csv(f"{self.cache_folder}/{row_index}.csv")
                return False

            # Append new row
            new_row = pd.DataFrame({"prompt": [prompt]})
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Create new DataFrame
            df = pd.DataFrame({"prompt": [prompt]})

        # Write DataFrame to CSV file
        df.to_csv(self.cache_file, index=False)

        # Save the answer to a new CSV file in the cache folder
        answer_df = pd.DataFrame({"answer": [answer]})
        answer_df.to_csv(f"{self.cache_folder}/{len(df)-1}.csv", index=False)

        return True

    def delete(self, prompt: str) -> None:
        """
        Deletes a prompt and its answer from the cache.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt exists. If the prompt exists, it deletes the row with the prompt from the DataFrame and
        writes the DataFrame back to the cache file. It also deletes the corresponding answer file
        from the cache folder.

        Args:
            prompt (str): The prompt to be deleted from the cache.
        """
        if os.path.exists(self.cache_file):
            # Read existing CSV file
            df = pd.read_csv(self.cache_file)

            # Check if prompt exists
            if prompt in df["prompt"].values:
                # If prompt exists, delete it and its corresponding answer file
                row_index = df.index[df["prompt"] == prompt].tolist()[0]
                df = df.drop(row_index)
                df.to_csv(self.cache_file, index=False)

                answer_file = f"{self.cache_folder}/{row_index}.csv"
                if os.path.exists(answer_file):
                    os.remove(answer_file)

    def lookup(self, prompt: str) -> bool:
        """
        Checks if a prompt is in the cache.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt exists. If the prompt exists, it returns True. If the cache file does not exist or the
        prompt does not exist in the cache, it returns False.

        Args:
            prompt (str): The prompt to be looked up in the cache.

        Returns:
            bool: True if the prompt exists in the cache, False otherwise.
        """
        if os.path.exists(self.cache_file):
            # Read existing CSV file
            df = pd.read_csv(self.cache_file)

            # Check if prompt already exists
            if prompt in df["prompt"].values:
                self._logger.info("Cache hit!")
                return True

        self._logger.info("Cache miss!")
        return False

    def get_answer(self, prompt: str):
        """
        Gets the answer for a prompt from the cache.

        This method checks if the cache file exists. If it does, it reads the file and checks if the
        prompt exists. If the prompt exists, it reads the corresponding answer file from the cache
        folder and returns the answer.

        If the cache file does not exist or the prompt does not exist in the cache, it returns None.

        Args:
            prompt (str): The prompt for which to get the answer.

        Returns:
            The answer for the prompt if it exists in the cache, None otherwise.
        """
        if os.path.exists(self.cache_file):
            # Read existing CSV file
            df = pd.read_csv(self.cache_file)

            # Check if prompt already exists
            if prompt in df["prompt"].values:
                # If prompt exists, return the answer from the respective CSV file
                row_index = df.index[df["prompt"] == prompt].tolist()[0]
                answer_df = pd.read_csv(f"{self.cache_folder}/{row_index}.csv")
                return answer_df["answer"].values[0]

        return None
