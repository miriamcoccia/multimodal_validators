import json
from typing import List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from config import settings

# Load traits directly from the config file at the module level
traits_list: Dict[str, str] = settings.get("traits", {})

# Create a DataFrame of supported traits for easy access in other modules
supported_traits: pd.DataFrame = pd.DataFrame(
    traits_list.items(), columns=["traits", "condition"]
)


class Trait(BaseModel):
    """Represents the validation result for a single trait."""

    trait: str = Field(..., description="The name of the trait being evaluated.")
    valid: bool = Field(
        ..., description="True if the condition is met, False otherwise."
    )
    reasoning: str = Field(
        ..., description="An explanation for why 'valid' is False, or empty if True."
    )

    @model_validator(mode="after")
    def check_dependencies(self) -> Self:
        """
        Validates the trait against business logic and the supported traits
        list loaded from the configuration file.
        """
        is_invalid = False
        error_message = ""

        # The validation logic correctly checks against the config-loaded traits_list
        if not self.trait:
            is_invalid = True
            error_message = "Invalid trait: name is empty."
        elif self.trait not in traits_list:
            is_invalid = True
            error_message = f"Invalid trait: name '{self.trait}' is not supported."
        elif self.valid and self.reasoning:
            is_invalid = True
            error_message = "Invalid reasoning: must be empty when 'valid' is True."
        elif not self.valid and not self.reasoning:
            is_invalid = True
            error_message = (
                "Invalid reasoning: must not be empty when 'valid' is False."
            )

        if is_invalid:
            # Standardize the error format for downstream processing
            self.trait = "ERROR"
            self.valid = False
            self.reasoning = error_message

        return self


class TraitList(BaseModel):
    """A container for a list of Trait objects, representing the full LLM output."""

    traits_output: List[Trait] = Field(description="A list of Trait objects.")

    def toJSONstr(self) -> str:
        """Serializes the list of traits to a JSON string."""
        return json.dumps([t.model_dump() for t in self.traits_output])

    def toJSON(self) -> List[Dict[str, Any]]:
        """Deserializes the JSON string back into a list of dictionaries."""
        return json.loads(self.toJSONstr())
