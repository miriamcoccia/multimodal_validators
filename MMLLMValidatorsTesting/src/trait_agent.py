from typing import Optional, Dict, Any, List
from PIL import Image
import logging
import time

from src.llm_service.utils import image_to_data_url
from src.config import settings
from src.img_traits_def import ImgTraitDefinition
from src.prompt_template import system_prompt_template
from src.science_qa import ScienceQA, build_question, build_characteristics
from src_orig.orig_traits_def import OriginalTraitDefinition
from src.llm_service.config import ConfigManager
from src.llm_service.schemas import ValidationSchema


logger = logging.getLogger(__name__)


class TraitAgent:
    """
    A base agent specialized in evaluating a single trait of a question-image pair.
    Designed to be subclassed for different trait definition sources.
    """

    def __init__(
        self,
        trait_name: str,
    ):
        # Stricter input validation
        self.config = ConfigManager()
        if not trait_name or not trait_name.strip():
            raise ValueError("Trait name cannot be empty or contain only whitespace.")

        self.trait_name = trait_name.strip()
        self.trait_definitions = ImgTraitDefinition()
        self.system_prompt = self._build_system_prompt(system_prompt_template)

        logger.info(f"TraitAgent for '{self.trait_name}' initialized.")

    def _build_system_prompt(self, template: str) -> str:
        """Helper to fetch components and format the system prompt."""
        definition = self.trait_definitions.retrieve_definition(self.trait_name)
        note = self.trait_definitions.retrieve_note(self.trait_name)
        evaluation_questions = self.trait_definitions.retrieve_evaluation_questions(
            self.trait_name
        )
        return template.format(
            trait_name=self.trait_name,
            definition=definition,
            note=note,
            evaluation_questions=evaluation_questions,
        )

    def _create_user_prompt(self, question_data: ScienceQA) -> str:
        """
        Combines question, context, options, and characteristics into one user prompt.
        """
        question_str = build_question(question_data, format_str="QCMLE-A")
        characteristics_str = build_characteristics(
            question_data, format_str="GSTCSkTa"
        )
        return f"{characteristics_str}\n\n{question_str}"

    def inputs_for(
        self,
        question_data: ScienceQA,
        provider: str,
        pil_images: Optional[List[Image.Image]] = None,
        image_file_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Builds input messages.
        For OpenAI: returns list for Responses API "input" field
        For Nebius: returns list that will be added to messages array
        """
        user_text = self._create_user_prompt(question_data)

        user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]

        if provider == "openai":
            if image_file_ids:
                for fid in image_file_ids:
                    user_content.append({"type": "input_image", "file_id": fid})

            return [{"role": "user", "content": user_content}]

        elif provider == "nebius":
            # For Nebius, use Chat Completions format
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]

            if pil_images:
                for img in pil_images:
                    data_url = image_to_data_url(img)
                    content.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )

            return [{"role": "user", "content": content}]

        return []

    def prepare_single_request(
        self,
        question_data: ScienceQA,
        provider: str,
        model_id: str, 
        pil_images: Optional[List[Image.Image]] = None,
        image_file_ids: Optional[List[str]] = None,
    ):
        try:
            qid = question_data.id
            if provider == "openai":
                model = settings["models"]["openai"][model_id]
                request_url = "/v1/responses" 
            elif provider == "nebius":
                model = settings["models"]["nebius"][model_id]
                request_url = "/v1/chat/completions"
            else:
                raise ValueError(f"Unknown provider: {provider}")

            trait = "_".join(self.trait_name.split(" ")).lower()
            request_id = f"request-{model_id}-{trait}-{qid}-{int(time.time() * 1000)}"

            openai_params = self.config.params.get("openai", {})
            max_out = int(openai_params.get("max_tokens", 512))

            inputs = self.inputs_for(
                question_data=question_data,
                provider=provider,
                pil_images=pil_images,
                image_file_ids=image_file_ids,
            )

            schema = ValidationSchema.model_json_schema()
            if "properties" in schema:
                schema["required"] = list(schema["properties"].keys())

            # Different payload based on provider
            if provider == "openai":
                # OpenAI uses Responses API format (as requested)
                payload = {
                    "model": model,
                    "instructions": self.system_prompt,
                    "max_output_tokens": max_out,
                    "input": inputs,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "validated_trait",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                }
                
                # GPT5 does not take temperature or top_p parameters in the payload
                if not model_id.startswith("GPT5"):
                    payload["temperature"] = 0
                    payload["top_p"] = 1

            else:  # nebius
                messages = [{"role": "system", "content": self.system_prompt}]

                for msg in inputs:
                    if msg.get("role") == "user":
                        messages.append(msg)

                payload = {
                    "model": model,
                    "messages": messages,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "validated_trait",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    "max_tokens": max_out,
                }


                if not model_id.startswith("GPT5"):
                    payload["temperature"] = 0
                    payload["top_p"] = 1

            request_text = {
                "custom_id": request_id,
                "method": "POST",
                "url": request_url,
                "body": payload,
            }

            return request_text

        except Exception as e:
            logger.error(
                f"Failed to prepare request for trait '{self.trait_name}': {e}"
            )
            return None