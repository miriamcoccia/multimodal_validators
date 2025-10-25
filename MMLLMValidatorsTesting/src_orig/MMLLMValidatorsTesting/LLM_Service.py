import os
import requests
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError
from .TraitList import TraitList
from huggingface_hub import InferenceClient
from ollama import chat
from config import settings


class LLM_Service:
    """
    A service class to interact with various Large Language Model (LLM) APIs,
    including OpenAI, Hugging Face, and local Ollama instances.
    """

    def __init__(self) -> None:
        """
        Initializes the LLM_Service, loading all configuration from the central settings.
        """
        print("🔧 Initializing LLM_Service with configuration...")

        # --- Instance attributes with type hints ---
        self.supported_models: Dict[str, str] = {}
        self.defaults: Dict[str, str] = settings.get("defaults", {})
        self.TIMEOUT: int = settings.get("settings", {}).get("default_timeout", 600)

        self.hf_api_key: Optional[str] = os.environ.get(
            settings.get("api", {}).get("huggingface_api_key_env")
        )
        self.openai_api_key: Optional[str] = os.environ.get(
            settings.get("api", {}).get("openai_api_key_env")
        )

        self.gpt_client: Optional[OpenAI] = None
        if self.openai_api_key:
            self.gpt_client = OpenAI(api_key=self.openai_api_key, timeout=self.TIMEOUT)
            print("✅ OpenAI client initialized.")
        else:
            print("⚠️ OpenAI client not initialized: API key not found.")

        # --- Runtime state tracking ---
        self.cold_models: List[str] = []
        self.error_models: List[str] = []
        self.timeout_models: List[str] = []

        # --- Load all models from different groups into one dictionary ---
        model_groups: Dict[str, Dict[str, str]] = settings.get("models", {})
        for group_name, models_in_group in model_groups.items():
            self.supported_models.update(models_in_group)

    def print_supported_llms(self) -> None:
        """Prints a list of all supported model friendly names and their API identifiers."""
        print("List of supported LLMs:")
        for llm, url in self.supported_models.items():
            print(f"{llm} : {url}")

    def get_all_models(self) -> List[str]:
        """Returns a list of all supported model friendly names."""
        return list(self.supported_models.keys())

    def get_model_url(self, model_id: str) -> str:
        """
        Retrieves the API model identifier for a given friendly name (case-insensitive).
        Returns an empty string if not found.
        """
        for key, value in self.supported_models.items():
            if key.upper() == model_id.upper():
                return value
        return ""

    def get_model_ids_startswith(self, prefix: str) -> List[str]:
        """Returns a list of model friendly names that start with a given prefix."""
        model_ids: List[str] = []
        for key in self.supported_models.keys():
            if key.upper().startswith(prefix.upper()):
                model_ids.append(key)
        return model_ids

    def execute_prompt(
        self, model_id: str, prompt: str, format_instructions: str = ""
    ) -> Optional[Any]:
        """
        Dispatches a prompt to the appropriate LLM service based on the model ID's prefix.
        """
        response: Optional[Any] = None
        if model_id == "GPT35TurboInstruct":
            response = self.gpt_old_execute_prompt(
                model_id, prompt, format_instructions
            )
        elif model_id.startswith("GPT"):
            response = self.gpt_execute_prompt(model_id, prompt, format_instructions)
        elif model_id.startswith("Llama32"):
            response = self.hf_execute_prompt_Llama32(
                model_id, prompt, format_instructions
            )
        elif model_id.startswith("L_"):
            response = self.ollama_execute_prompt(model_id, prompt, format_instructions)
        else:  # Assume model is from Hugging Face
            response = self.hf_execute_prompt(model_id, prompt, format_instructions)
        return response

    def gpt_execute_prompt(
        self, model_id: str, prompt: str, format_instructions: str = ""
    ) -> Optional[TraitList]:
        """Executes a prompt against the OpenAI Chat Completions API."""
        if not self.gpt_client:
            print("gpt_execute_prompt: Aborting, OpenAI client not initialized.")
            return None

        default_model_id = self.defaults.get("openai_default", "GPT4oMini")
        model_url = self.get_model_url(model_id) or self.get_model_url(default_model_id)

        parser = PydanticOutputParser(pydantic_object=TraitList)
        if not format_instructions:
            format_instructions = parser.get_format_instructions()
        try:
            messages = [{"role": "user", "content": prompt + format_instructions}]
            completion = self.gpt_client.chat.completions.create(
                model=model_url, messages=messages
            )
            gpt_response = completion.choices[0].message
            if gpt_response.refusal:
                print(f"gpt_execute_prompt: Refusal: {gpt_response.refusal}")
                return None

            return parser.invoke(gpt_response.content)
        except ValidationError as err:
            print(f"gpt_execute_prompt: ValidationError: {err}")
            return None
        except Exception as exc:
            print(f"gpt_execute_prompt: General exception: {exc}")
            return None

    def gpt_old_execute_prompt(
        self, model_id: str, prompt: str, format_instructions: str = ""
    ) -> Optional[TraitList]:
        """Executes a prompt against the legacy OpenAI Completions API."""
        if not self.gpt_client:
            print("gpt_old_execute_prompt: Aborting, OpenAI client not initialized.")
            return None

        default_model_id = self.defaults.get(
            "openai_legacy_default", "GPT35TurboInstruct"
        )
        model_url = self.get_model_url(model_id) or self.get_model_url(default_model_id)

        parser = PydanticOutputParser(pydantic_object=TraitList)
        if not format_instructions:
            format_instructions = parser.get_format_instructions()
        try:
            messages = prompt + format_instructions
            completion = self.gpt_client.completions.create(
                model=model_url, prompt=messages
            )
            gpt_response = completion.choices[0].text

            if "error" in gpt_response:
                print(f"gpt_execute_prompt: API Error: {gpt_response}")
                return None

            return parser.invoke(gpt_response)
        except ValidationError as err:
            print(f"gpt_execute_prompt: ValidationError: {err}")
            return None
        except Exception as exc:
            print(f"gpt_execute_prompt: General exception: {exc}")
            return None

    def hf_execute_prompt(
        self, model_id: str, prompt: str, format_instructions: str = ""
    ) -> Optional[Union[TraitList, str]]:
        """Executes a prompt against the Hugging Face Inference API."""
        if not self.hf_api_key:
            print("hf_execute_prompt: Aborting, HuggingFace API key not found.")
            return None

        default_model_id = self.defaults.get("huggingface_default", "Llama3170Instruct")
        model_url = self.get_model_url(model_id) or self.get_model_url(default_model_id)

        base_url = (
            settings.get("services", {}).get("huggingface", {}).get("base_url", "")
        )
        api_url = f"{base_url}{model_url}"

        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        parameters = {}
        if not model_id.startswith(("Flan", "MT5")):
            parameters["return_full_text"] = False

        parser = PydanticOutputParser(pydantic_object=TraitList)
        if not format_instructions:
            format_instructions = parser.get_format_instructions()

        payload = {"inputs": prompt + format_instructions, "parameters": parameters}
        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=self.TIMEOUT
            )

            if not response.ok:
                if response.status_code == 503:  # Model is cold
                    self.cold_models.append(model_id)
                elif response.status_code == 504:  # Model timeout
                    self.timeout_models.append(model_id)
                elif response.status_code == 429:  # Rate limit reached
                    print(f"hf_execute_prompt: Rate limit reached: {response.text}")
                    return "error=429"
                else:
                    self.error_models.append(model_id)
                print(f"hf_execute_prompt: Request failed: {response.text}")
                return None

            generated_text = response.json()[0]["generated_text"]
            return parser.invoke(generated_text)
        except ValidationError as err:
            print(f"hf_execute_prompt: ValidationError: {err}")
            return None
        except Exception as exc:
            print(f"hf_execute_prompt: General exception: {exc}")
            return None

    def ollama_execute_prompt(
        self, model_id: str, prompt: str, format_instructions: str = ""
    ) -> Optional[TraitList]:
        """Executes a prompt against a local Ollama service."""
        default_model_id = self.defaults.get("ollama_default", "L_Phi4")
        model_url = self.get_model_url(model_id) or self.get_model_url(default_model_id)

        if not format_instructions:
            parser = PydanticOutputParser(pydantic_object=TraitList)
            format_instructions = parser.get_format_instructions()

        messages = [{"role": "user", "content": prompt + format_instructions}]
        try:
            response = chat(
                messages=messages,
                model=model_url,
                format="json",  # Use Ollama's built-in JSON mode
            )
            # Ollama with format='json' should return a valid JSON string
            parsed_response = TraitList.model_validate_json(
                response["message"]["content"]
            )
            return parsed_response
        except ValidationError as err:
            print(f"ollama_execute_prompt: ValidationError: {err}")
            return None
        except Exception as exc:
            print(f"ollama_execute_prompt: General exception: {exc}")
            self.error_models.append(model_id)
            return None

    def hf_execute_prompt_Llama32(
        self, model_id: str, prompt: str, format_instructions: str = ""
    ) -> Optional[TraitList]:
        """Executes a prompt against the HF API using the dedicated InferenceClient."""
        if not self.hf_api_key:
            print("hf_execute_prompt_Llama32: Aborting, HuggingFace API key not found.")
            return None

        default_model_id = self.defaults.get(
            "huggingface_llama32_default", "Llama32Vision11B"
        )
        model_url = self.get_model_url(model_id) or self.get_model_url(default_model_id)

        parser = PydanticOutputParser(pydantic_object=TraitList)
        if not format_instructions:
            format_instructions = parser.get_format_instructions()
        try:
            client = InferenceClient(
                model_url, token=self.hf_api_key, timeout=self.TIMEOUT
            )
            messages = [{"role": "user", "content": prompt + format_instructions}]
            completion = client.chat.completions.create(
                model=model_url, messages=messages
            )
            generated_text = completion.choices[0].message.content

            if "error" in generated_text:
                print(f"hf_llama32: API error: {generated_text}")
                return None

            return parser.invoke(generated_text)
        except ValidationError as err:
            print(f"hf_llama32: ValidationError: {err}")
            return None
        except Exception as exc:
            print(f"hf_llama32: General exception: {exc}")
            return None
