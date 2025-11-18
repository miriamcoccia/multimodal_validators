from typing import List, Literal
from pydantic import BaseModel, Field, model_validator

class ValidTrait(BaseModel):
    """Schema that all valid traits should follow"""
    model_config = {"extra": "forbid"}

    trait: str = Field(..., description="The name of the trait being evaluated.")
    validity: Literal[True]
    reasoning: Literal[""]


class InvalidTrait(BaseModel):
    """Schema that all invalid traits should follow"""
    model_config = {"extra": "forbid"}

    trait: str = Field(..., description="The name of the trait being evaluated.")
    validity: Literal[False]
    reasoning: str = Field(..., description="One sentence explaining why the trait is invalid.", min_length=1, max_length=200)

TraitEvaluation = ValidTrait | InvalidTrait

class ValidationSchema(BaseModel):
    """Canonical output schema for a single trait evaluation."""

    model_config = {"extra": "forbid"}

    trait: str = Field(..., description="The name of the trait being evaluated.")
    validity: bool = Field(..., description="True if the trait is valid.")
    reasoning: str = Field(
        ..., 
        description="Explanation for invalid traits. Empty string when validity is True."
    )

    @classmethod
    def model_json_schema(cls):
        """
        Generates a JSON schema where ALL fields are required,
        and 'reasoning' is a non-nullable string.
        """
        schema = super().model_json_schema()

        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())

            reasoning_field = schema["properties"].get("reasoning", {})
            reasoning_field.pop("nullable", None)
            reasoning_field["type"] = "string"

        return schema

    @model_validator(mode="after")
    def _enforce_reasoning_rules(self):
        """
        Internal correctness:
        - validity=True  -> reasoning=""
        - validity=False -> reasoning must be a non-empty string
        """
        if self.validity is True:
            self.reasoning = ""
        else:
            if not self.reasoning or not self.reasoning.strip():
                raise ValueError(
                    "When validity is False, 'reasoning' must be a non-empty string."
                )
        return self


class ValidationListSchema(BaseModel):
    """Output schema for combined evaluation of all traits."""

    model_config = {"extra": "forbid"}

    traits_output: List[TraitEvaluation] = Field(
        ..., description="A list of validation results for all traits."
    )

    @classmethod
    def model_json_schema(cls):
        schema = super().model_json_schema()

        if "properties" in schema:
            # All fields required
            schema["required"] = list(schema["properties"].keys())

            # Ensure nested ValidationSchema.reasoning is non-nullable
            items_schema = schema["properties"]["traits_output"].get("items", {})
            reasoning = items_schema.get("properties", {}).get("reasoning", {})
            reasoning.pop("nullable", None)
            reasoning["type"] = "string"

        return schema


