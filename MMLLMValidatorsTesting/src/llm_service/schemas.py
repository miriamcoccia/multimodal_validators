from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ValidationSchema(BaseModel):
    """Canonical output schema for a single trait evaluation."""

    model_config = {"extra": "forbid"}

    trait: str = Field(..., description="The name of the trait being evaluated.")
    validity: bool = Field(..., description="True if the trait is valid.")
    reasoning: Optional[str] = None

    @classmethod
    def model_json_schema(cls):
        schema = super().model_json_schema()
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
        return schema

    @model_validator(mode="after")
    def _enforce_reasoning_rules(self):
        if self.validity is True:
            self.reasoning = None
        else:
            if self.reasoning is None or not str(self.reasoning).strip():
                raise ValueError(
                    "When validity is False, 'reasoning' must be a non-empty string."
                )
        return self
