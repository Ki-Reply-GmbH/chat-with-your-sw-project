"""
This module contains the LLModel class which is used to interact with the 
OpenAI API.It uses the provided configuration and cache to manage requests and 
responses.
The LLModel class can send prompts to the OpenAI API and return the AI's 
response in different formats.
"""
import ast
import json
import logging

import openai

from config import Config
from utils.cache import Cache
from utils.functions import decode_from_base64, encode_to_base64

LOGGER = logging.getLogger(__name__)


class LLModel:
    """
    The LLModel class is used to interact with the OpenAI API. It uses the 
    provided configuration and cache to manage requests.

    Attributes:
        _cache (Cache): An instance of the Cache class used to store and 
                        retrieve data.
        _model_name (str): The name of the model to use for the OpenAI API.
        _temperature (float): The temperature to use for the OpenAI API.
    """
    def __init__(self, config: Config, cache: Cache = None):
        openai.api_key = config.OPENAI_API_KEY
        self._cache = cache
        self._model_name = config.LLM_MODEL_NAME
        self._temperature = config.LLM_TEMPERATURE

    def _get_llm_completion(self, prompt, resp_fmt_type="text"):
        """
        Sends a prompt to the OpenAI API and returns the AI's response.

        Parameters:
            prompt (str): The prompt to send to the OpenAI API.
            resp_fmt_type (str, optional): The format of the response. Defaults
            to "text".

        Returns:
            str: The response from the OpenAI API.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a system designed to find and correct errors in a CI/CD pipeline.",
            },
            {"role": "user", "content": prompt},
        ]
        response = openai.OpenAI().chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,  # this is the degree of randomness of the model's output,
            response_format={"type": resp_fmt_type},
        )
        return response.choices[0].message.content

    def get_completion(self, prompt, resp_fmt_type: str = "json_object"):
        """
        Gets a completion from the OpenAI API.

        Parameters:
            prompt (str): The prompt to send to the OpenAI API.
            resp_fmt_type (str, optional): The format of the response. Defaults
                                           to "json_object".

        Returns:
            str: The completion from the OpenAI API.
        """
        LOGGER.debug("Getting completion for prompt: %s", prompt)
        base64_prompt = encode_to_base64(prompt)

        if resp_fmt_type == "json_object":
            return self._get_completion_json(prompt, base64_prompt)

        return self._get_completion_text(prompt, base64_prompt)

    def _get_completion_json(self, prompt, base64_prompt: str):
        """
        Gets a completion from the OpenAI API in JSON format.

        Parameters:
            prompt (str): The prompt to send to the OpenAI API.
            base64_prompt (str): The base64 encoded prompt.

        Returns:
            str: The completion from the OpenAI API in JSON format.
        """
        if self._cache.lookup(base64_prompt):
            cache_content = self._cache.get_answer(base64_prompt)
            response = decode_from_base64(cache_content)
            # Prevent json.loads from throwing an error
            response = ast.literal_eval(response)
        else:
            response = json.loads(self._get_llm_completion(
                prompt,
                "json_object"
                ))
            self._cache.update(base64_prompt, encode_to_base64(response))

        return response

    def _get_completion_text(self, prompt, base64_prompt: str):
        """
        Gets a completion from the OpenAI API in text format.

        Parameters:
            prompt (str): The prompt to send to the OpenAI API.
            base64_prompt (str): The base64 encoded prompt.

        Returns:
            str: The completion from the OpenAI API in text format.
        """
        if self._cache.lookup(base64_prompt):
            cache_content = self._cache.get_answer(base64_prompt)
            response = decode_from_base64(cache_content)
        else:
            response = self._get_llm_completion(prompt, "text")
            self._cache.update(base64_prompt, encode_to_base64(response))

        return response
