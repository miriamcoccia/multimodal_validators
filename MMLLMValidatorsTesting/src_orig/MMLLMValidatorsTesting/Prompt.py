from enum import Enum
from typing import Dict, List, Any, Optional

from pandas import DataFrame

from config import settings
from .TraitList import traits_list, supported_traits
from .ScienceQA import build_question, build_characteristics


class PromptID(Enum):
    """Enumeration of available prompt template identifiers."""

    General_V1 = 0
    General_V2 = 1
    General_V1_Img = 2
    Optimized_V1 = 3

    @classmethod
    def all(cls) -> List["PromptID"]:
        """Returns a list of all prompt IDs."""
        return list(cls)

    @classmethod
    def print_supported_prompts(cls) -> None:
        """Prints the names of all supported prompts."""
        print("List of supported Prompts:")
        for p in cls.all():
            print(f"- {p.name}")


class Prompt:
    """
    A class that constructs a final, formatted prompt string from a template
    and ground truth data, based on settings from the config file.
    """

    id: PromptID
    template: str
    prompt: str
    question_format: str
    characteristics_format: str
    format_instructions: str
    gt_question: Optional[DataFrame]
    selected_traits: List[str]

    def __init__(
        self,
        id: PromptID = PromptID.General_V1,
        question: Optional[DataFrame] = None,
        traits: Optional[List[str]] = None,
    ):
        self.id = id
        self.gt_question = question
        # Fallback to all supported traits if none are provided
        self.selected_traits = (
            traits if traits is not None else list(supported_traits["traits"].values)
        )
        # Initialize all attributes to be safe
        self.template = ""
        self.prompt = ""
        self.question_format = ""
        self.characteristics_format = ""
        self.format_instructions = ""

        self.instantiate_prompt_template()

    def instantiate_prompt_template(self) -> None:
        """
        This method dynamically builds the prompt by looking up
        the templates and formats in the central configuration file.
        """
        prompt_name = self.id.name
        prompt_config = settings.get("prompts", {}).get(prompt_name, {})

        if not prompt_config:
            print(
                f"⚠️ Warning: Prompt '{prompt_name}' not found in config.toml. Using empty prompt."
            )
            return

        defaults = settings.get("defaults", {})
        default_q_format = defaults.get("default_question_format", "QCM-A")
        default_c_format = defaults.get("default_characteristics_format", "GSTCSk")

        self.template = prompt_config.get("template", "")
        self.format_instructions = prompt_config.get("format_instructions", "")
        self.question_format = prompt_config.get("question_format", default_q_format)
        self.characteristics_format = prompt_config.get(
            "characteristics_format", default_c_format
        )

        list_of_conditions = {
            t: d for t, d in traits_list.items() if t in self.selected_traits
        }

        question_text = build_question(self.gt_question, self.question_format)
        characteristics_text = build_characteristics(
            self.gt_question, self.characteristics_format
        )

        format_data = {
            "prompt": characteristics_text,
            "characteristics": characteristics_text,
            "questionnaire": question_text,
            "question": question_text,
            "conditions": str(list_of_conditions),
            "format_instructions": self.format_instructions,
        }

        self.prompt = self.template.format_map(format_data)
