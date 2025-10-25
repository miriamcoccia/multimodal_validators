import json
import logging
from typing import List, Dict, Any, Optional

from PIL import Image
import aiohttp

from .base import BaseProvider
from ..utils import image_to_data_url, retry_with_backoff
from ..schemas import ValidationSchema

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider using function calling for structured output."""

    def resolve_model_id(self, model_id: str) -> Optional[str]:
        """Resolve OpenAI model ID with fallback to default."""
        if model_id in self.config.model_index:
            return self.config.model_index[model_id]

        fallback = self.config.defaults.get("openai_multimodal_default", "")
        return self.config.model_index.get(fallback)

    # TODO: remove this method because we are using batches instead
    @retry_with_backoff()
    async def _make_request(
        self, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to OpenAI API."""
        if self.config.openai_base_url:
            async with self.session.post(
                self.config.openai_base_url, headers=headers, json=payload
            ) as r:
                if r.status != 200:
                    text = await r.text()
                    logger.warning(f"OpenAI response status {r.status}: {text[:200]}")
                r.raise_for_status()
                return await r.json()

    # TODO: turn this method into a method to create a request line that can be processed through the batch api, do not write it yet, because we need other metadata. The batch jsonl file should be created only ONCE, by the orchestrator
    async def execute(
        self,
        model_id: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        pil_images: Optional[List[Image.Image]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute prompt using OpenAI function calling."""
        if not self.config.openai_api_key:
            logger.error("Cannot execute GPT prompt: API key is missing")
            return None

        model_name = self.resolve_model_id(model_id)
        if not model_name:
            logger.error(f"No OpenAI model found for: {model_id}")
            return None

        if not self.config.openai_base_url:
            logger.error("OpenAI API URL is not set in config")
            return None

        openai_params = self.config.params.get("openai", {})
        max_tokens_cfg = int(openai_params.get("max_tokens", 1024))

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if pil_images:
            for img in pil_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(img)},
                    }
                )
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens_cfg,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "validated_trait",
                        "description": "Return JSON with trait, validity, and reasoning.",
                        "parameters": ValidationSchema.model_json_schema(),
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "validated_trait"},
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openai_api_key}",
        }

        try:
            resp_data = await self._make_request(payload, headers)

            if resp_data is None:
                logger.error(f"OpenAI request failed after retries for {model_id}")
                self.error_models.append(model_id)
                return None

            message = resp_data.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                logger.error(f"No tool calls in OpenAI {model_id} response")
                raise ValueError("No tool calls found in the response.")

            args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
            logger.debug(f"OpenAI {model_id} function args: {args_str}")
            args_dict = json.loads(args_str)

            result = self._safe_validate_response(args_dict, model_id)
            if result:
                return result
            else:
                raise ValueError(f"Validation failed for OpenAI response: {args_dict}")

        except (
            aiohttp.ClientError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            ValueError,
        ) as exc:
            logger.error(f"OpenAI execution error for {model_id}: {exc}")
            self.error_models.append(model_id)
            return None
