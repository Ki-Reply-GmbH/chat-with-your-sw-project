"""
This module provides the Config class which is used to manage configuration 
settings.

The Config class loads configuration settings from environment variables and 
provides methods to access these settings.
"""

import logging
import os
import sys
import tempfile
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@dataclass
class PromptConfig:
    """
    This class is used to manage the prompts used in the application.

    Attributes:
        log_prompt (str): The prompt used for logging.
        repair_prompt (str): The prompt used for repairing.
        commit_prompt (str): The prompt used for committing.
    """
    def __init__(self):
        self.log_prompt = log_prompt
        self.repair_prompt = repair_prompt
        self.commit_prompt = commit_prompt

log_prompt = """
You will receive the log of a failed CI/CD pipeline. The log contains \
unnecessary information. You'll be given a workflow to follow along with some \
examples:
1. Search for the relevant information that describes the cause of the error.
2. Identify the file in which the error is caused and which need to be \
corrected.
3. Return a Json file with the keys "explanation", "error_area" and "file".
The cause of the error (str), as described in 1., should be in "Explanation".
The lines of code that are causing the error (str), should be in "error_area".
The file (str) that caused the error should be in "file".

Log:
{log}
"""

repair_prompt = """\
You will receive the complete source code of a file that has errors. In a \
previous LLM call, the cause of the error was analyzed and the area of the \
error was narrowed down. You will receive all this information separated from \
each other with ####. Correct the source code using the additional information.\
Return only the source code and no other information. Do not use formatting \
characters in your answer. Your answer should be written directly to a file \
and consist exclusively of interpretable or compilable source code.

####
Explanation:
{explanation}
####
Error Area:
{error_area}
####
Source Code:
{source_code}
####
"""

commit_prompt = """\
You will receive relevant information about the error. The error description \
and the source code that triggered the error. The error was already corrected \
in a previous LLM call. Compose a GitHub commit message based on this \
information.
Do not use formatting characters in your answer. Your answer should be used \
directly as a GitHub commit message without any further formatting.

Relevant information:
{information}
"""


@dataclass
class Config:
    """
    The Config class is responsible for managing configuration settings.

    This class loads configuration settings from environment variables and 
    provides methods to access these settings.

    Attributes:
        OPENAI_API_KEY (str): The API key for OpenAI.
        LLM_MODEL_NAME (str): The name of the language model to use.
        LLM_TEMPERATURE (float): The temperature setting for the language model.
        GITHUB_TOKEN (str): The access token for GitHub.
        GIT_USER (str): The username for GitHub.
        GIT_EMAIL (str): The email for GitHub.
        REPO_OWNER (str): The owner of the repository.
        REPO_NAME (str): The name of the repository.
        MAIN_BRANCH (str): The main branch of the repository.
    """
    _instance = None

    def __init__(self):
        """
        Initializes a Config and loads settings from environment variables.
        """
        ####################
        # Cache CONFIG
        ####################
        self.USE_CACHE = self._read_bool_value("USE_CACHE", "False")

        ####################
        # AGI CONFIG
        ####################
        self.prompts = PromptConfig()
        self.AGI_VERBOSE = True
        self.LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4-1106-preview")
        self.LLM_TEMPERATURE = os.environ.get("LLM_TEMPERATURE", 0.0)
        self.LLM_MAX_LENGTH = os.environ.get("LLM_MAX_LENGTH", 512)

        #######################
        # AZURE OPENAI CONFIG
        #######################
        self.OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
        self.OPENAI_DEPLOYMENT_NAME = os.environ.get("OPENAI_DEPLOYMENT_NAME")
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        self.OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2023-05-15")
        self.OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE", "azure")

        ####################
        # WORKING DIR CONFIG
        ####################

        if os.environ.get("WORKING_DIR"):
            self.WORKING_DIR = os.environ.get("WORKING_DIR")
        else:
            temp_dir = os.path.join(tempfile.gettempdir(), "cdaas-merge")
            os.makedirs(temp_dir, exist_ok=True)
            self.WORKING_DIR = temp_dir
        LOGGER.info("Working directory: %s", self.WORKING_DIR)

    def _read_bool_value(self, env_name, default_value: bool) -> bool:
        """
        Read a boolean value from an environment variable.
        Args:
            env_value (str): The environment variable value.
        Returns:
            bool: The boolean value.
        """
        env_value: str | bool = os.environ.get(env_name, default_value)
        if env_value is None:
            return default_value

        if isinstance(env_value, bool):
            return env_value
        return env_value.lower() in ["true", "1"]

    @classmethod
    def instance(cls):
        """Create a singleton instance of the Config class.
        Returns:
            Config: The singleton instance of the Config class.
        """
        if cls._instance is None:
            logging.info("Creating new Config instance")
            cls._instance = Config()
            # cls._instance.validate_github_user()
            # cls._instance.validate_llm_setup()
            # Put any initialization here.
        return cls._instance

    def validate_llm_setup(self):
        """Validate the LLM setup.
        This function checks if all required environment variables are set.

        Raises:
            SystemExit: If any required environment variable is not set.
        """
        if not self.OPENAI_API_BASE:
            sys.stderr.write(
                "ERROR: OPENAI_API_BASE is not set. Please check your environment variables."
            )
            sys.exit(3)

        if not self.OPENAI_DEPLOYMENT_NAME:
            sys.stderr.write(
                "ERROR: OPENAI_DEPLOYMENT_NAME is not set. Please check your environment variables."
            )
            sys.exit(3)

        if not self.OPENAI_API_KEY:
            sys.stderr.write(
                "ERROR: OPENAI_API_KEY is not set. Please check your environment variables."
            )
            sys.exit(3)

        if not self.OPENAI_API_TYPE:
            sys.stderr.write(
                "ERROR: OPENAI_API_TYPE is not set. Please check your environment variables."
            )
            sys.exit(3)


def load_config():
    return Config.instance()
