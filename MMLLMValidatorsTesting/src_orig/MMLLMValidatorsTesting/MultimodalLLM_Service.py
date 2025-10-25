import os
import requests
import base64
import io
from typing import List, Dict, Any, Optional

from ollama import chat
from pydantic import ValidationError
from PIL import Image

from config import settings
from .TraitList import TraitList


class MultimodalLLM_Service:
    """
    A service class to interact with multimodal LLMs.
    It handles prompt execution, image encoding, and model management.
    """

    def __init__(self) -> None:
        """
        Initializes the MultimodalLLM_Service with settings from the config file.
        """
        print("🔧 Initializing MultimodalLLM_Service with configuration...")

        # --- Load settings and models from the config file ---
        self.supported_models: Dict[str, str] = {}
        model_groups: Dict[str, Dict[str, str]] = settings.get("models", {})
        for group_name, models_in_group in model_groups.items():
            self.supported_models.update(models_in_group)

        self.defaults: Dict[str, str] = settings.get("defaults", {})
        self.params: Dict[str, Any] = settings.get("parameters", {})
        self.TIMEOUT: int = settings.get("settings", {}).get("default_timeout", 600)

        # --- Load API key using the name specified in the config ---
        self.openai_api_key: Optional[str] = os.environ.get(
            settings.get("api", {}).get("openai_api_key_env", "")
        )
        if self.openai_api_key:
            print("✅ OpenAI API key loaded successfully.")
        else:
            print("⚠️ WARNING: OpenAI API key not set or key name misconfigured.")

        # --- Instance variables for runtime state ---
        self.cold_models: List[str] = []
        self.error_models: List[str] = []
        self.timeout_models: List[str] = []

    def print_supported_models(self) -> None:
        """Prints a list of all supported model friendly names and their API identifiers."""
        print("List of supported MLLMs:")
        for mllm, url in self.supported_models.items():
            print(f"{mllm}: {url}")

    def get_all_models(self) -> List[str]:
        """Returns a list of all supported model friendly names."""
        return list(self.supported_models.keys())

    def get_model_url(self, model_id: str) -> str:
        """
        Retrieves the API model identifier for a given friendly name (case-insensitive).
        """
        for key, value in self.supported_models.items():
            if key.upper() == model_id.upper():
                return value
        return ""

    def get_model_ids_startswith(self, prefix: str) -> List[str]:
        """Returns a list of model friendly names that start with a given prefix."""
        return [key for key in self.supported_models if key.startswith(prefix)]

    @staticmethod
    def encode_image(pil_image: Image.Image) -> Optional[str]:
        """Encodes a PIL image to a base64 string."""
        try:
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding PIL image: {e}")
            return None

    def ollama_execute_prompt(
        self,
        model_id: str,
        prompt: str,
        pil_images: Optional[List[Image.Image]],
        format_instructions: str = "",
    ) -> Optional[TraitList]:
        """Executes a multimodal prompt against a local Ollama service."""
        default_model = self.defaults.get("ollama_multimodal_default", "L_Qwen25VL3B")
        model_url = self.get_model_url(model_id) or self.get_model_url(default_model)

        final_prompt = prompt + format_instructions

        image_payload = []
        if pil_images:
            image_payload = [
                b64_img for img in pil_images if (b64_img := self.encode_image(img))
            ]

        messages = [{"role": "user", "content": final_prompt, "images": image_payload}]

        try:
            response = chat(messages=messages, model=model_url, format="json")
            raw_content = response["message"]["content"]
            return TraitList.model_validate_json(raw_content)
        except Exception as exc:
            print(f"Ollama execution error on model {model_id}: {exc}")
            self.error_models.append(model_id)
            return None

    def gpt_execute_prompt(
        self,
        model_id: str,
        prompt: str,
        pil_images: Optional[List[Image.Image]],
        format_instructions: str = "",
        image_detail: str = "",
    ) -> Optional[TraitList]:
        """Executes a multimodal prompt against the OpenAI API."""
        if not self.openai_api_key:
            print("Cannot execute GPT prompt: API key is missing.")
            return None

        # --- Load defaults and parameters from config ---
        default_model = self.defaults.get("openai_multimodal_default", "GPT4oMini")
        api_model_id = self.get_model_url(model_id) or self.get_model_url(default_model)

        openai_params = self.params.get("openai", {})
        max_tokens = openai_params.get("max_tokens", 2048)
        detail_level = image_detail or openai_params.get("image_detail", "auto")

        final_prompt = prompt + format_instructions
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": final_prompt}]

        if pil_images:
            for img in pil_images:
                if base64_image := self.encode_image(img):
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": detail_level,
                            },
                        }
                    )

        payload = {
            "model": api_model_id,
            "messages": [{"role": "user", "content": content_parts}],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }
        api_url = settings.get("services", {}).get("openai", {}).get("base_url")

        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=self.TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            gpt_response_content = response_data["choices"][0]["message"]["content"]

            return TraitList.model_validate_json(gpt_response_content)
        except (requests.RequestException, ValidationError) as exc:
            print(f"GPT execution error on model {model_id}: {exc}")
            self.error_models.append(model_id)
            return None

    def execute_prompt(
        self,
        model_id: str,
        prompt: str,
        pil_images: Optional[List[Image.Image]] = None,
        format_instructions: str = "",
        **kwargs: Any,
    ) -> Optional[TraitList]:
        """Dispatches a prompt to the appropriate multimodal service."""
        if model_id.startswith(("L_", "PCL_")):
            return self.ollama_execute_prompt(
                model_id, prompt, pil_images, format_instructions
            )
        elif model_id.startswith("GPT"):
            return self.gpt_execute_prompt(
                model_id,
                prompt,
                pil_images,
                format_instructions,
                image_detail=kwargs.get("image_detail", ""),
            )
        else:
            print(
                f"Model Skipped: {model_id} (Unknown or unsupported prefix for multimodal service)"
            )
            self.error_models.append(model_id)
            return None
