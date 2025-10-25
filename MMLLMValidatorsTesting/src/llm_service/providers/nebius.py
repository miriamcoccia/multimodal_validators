"""
Nebius API provider integration
"""

import json
import logging
from typing import List, Dict, Any, Optional

from PIL import Image
import aiohttp

from .base import BaseProvider
from ..utils import image_to_data_url, retry_with_backoff
from ..schemas import ValidationSchema

logger = logging.getLogger(__name__)


class NebiusProvider(BaseProvider):
    """Nebius API provider using function calling for structured output."""

    def resolve_model_id(self, model_id: str) -> Optional[str]:
        """Resolve Nebius model ID with fallback to default."""
        if model_id in self.config.model_index:
            return self.config.model_index[model_id]

        fallback = self.config.defaults.get("nebius_multimodal_default", "")
        return self.config.model_index.get(fallback)

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
        return None

    # TODO: turn this method into a method to create a request line that can be processed through the batch api, do not write it yet, because we need other metadata. The batch jsonl file should be created only ONCE, by the orchestrator
    async def execute(
        self,
        model_id: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        pil_images: Optional[List[Image.Image]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute prompt using Nebius function calling."""
        if not self.config.nebius_api_key:
            logger.error("Cannot execute prompt: API key is missing")
            return None

        model_name = self.resolve_model_id(model_id)
        if not model_name:
            logger.error(f"No Nebius model found for: {model_id}")
            return None

        if not self.config.nebius_base_url:
            logger.error("Nebius API URL is not set in config")
            return None

        nebius_params = self.config.params.get("nebius", {})
        max_tokens_cfg = int(nebius_params.get("max_tokens", 1024))

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
            "Authorization": f"Bearer {self.config.nebius_api_key}",
        }

        try:
            resp_data = await self._make_request(payload, headers)

            if resp_data is None:
                logger.error(f"Nebius request failed after retries for {model_id}")
                self.error_models.append(model_id)
                return None

            message = resp_data.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                logger.error(f"No tool calls in Nebius {model_id} response")
                raise ValueError("No tool calls found in the response.")

            args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
            logger.debug(f"Nebius {model_id} function args: {args_str}")
            args_dict = json.loads(args_str)

            result = self._safe_validate_response(args_dict, model_id)
            if result:
                return result
            else:
                raise ValueError(f"Validation failed for Nebius response: {args_dict}")

        except (
            aiohttp.ClientError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            ValueError,
        ) as exc:
            logger.error(f"Nebius execution error for {model_id}: {exc}")
            self.error_models.append(model_id)
            return None
