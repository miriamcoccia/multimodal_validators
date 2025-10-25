from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

from PIL import Image
import aiohttp

from ..config import ConfigManager
from ..schemas import ValidationSchema
from ..utils import _extract_core_fields

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: ConfigManager, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.error_models: List[str] = []

    @abstractmethod
    async def execute(
        self,
        model_id: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        pil_images: Optional[List[Image.Image]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a prompt and return validated response."""
        pass

    @abstractmethod
    def resolve_model_id(self, model_id: str) -> Optional[str]:
        """Resolve a friendly model ID to the actual model identifier."""
        pass

    def _safe_validate_response(
        self, raw_data: Optional[Dict[str, Any]], model_id: str
    ) -> Optional[Dict]:
        """Safely validate response data against schema."""
        if not raw_data:
            logger.error(f"Empty response data for {model_id}")
            return None

        try:
            validated = ValidationSchema.model_validate(raw_data)
            return validated.model_dump()
        except Exception as strict_error:
            logger.warning(f"Strict validation failed for {model_id}: {strict_error}")

            try:
                core_data = _extract_core_fields(raw_data)
                if core_data:
                    validated = ValidationSchema.model_validate(core_data)
                    logger.info(f"Successfully validated core fields for {model_id}")
                    return validated.model_dump()
            except Exception as fallback_error:
                logger.error(
                    f"Fallback validation also failed for {model_id}: {fallback_error}"
                )
                logger.error(f"Raw data was: {raw_data}")
        return None
