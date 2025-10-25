import os
from typing import Dict, Any, Optional
from pathlib import Path

from src.config import settings


class ConfigManager:
    """Centralized configuration management for the LLM service."""

    def __init__(self):
        self._settings = settings

    @property
    def models_openai(self) -> Dict[str, str]:
        return self._settings.get("models", {}).get("openai", {}) or {}

    @property
    def models_nebius(self) -> Dict[str, str]:
        return self._settings.get("models", {}).get("nebius", {}) or {}

    @property
    def model_index(self) -> Dict[str, str]:
        """Combined index of all models across providers."""
        index = {}
        index.update(self.models_openai)
        index.update(self.models_nebius)
        return index

    @property
    def defaults(self) -> Dict[str, str]:
        return self._settings.get("defaults", {})

    @property
    def params(self) -> Dict[str, Any]:
        return self._settings.get("parameters", {})

    @property
    def timeout(self) -> int:
        return self._settings.get("settings", {}).get("default_timeout", 600)

    @property
    def openai_api_key(self) -> Optional[str]:
        env_var = self._settings.get("api", {}).get(
            "openai_api_key_env", "OPENAI_API_KEY"
        )
        return os.environ.get(env_var)

    @property
    def nebius_api_key(self) -> Optional[str]:
        env_var = self._settings.get("api", {}).get(
            "nebius_api_key_env", "NEBIUS_API_KEY"
        )
        return os.environ.get(env_var)

    @property
    def openai_base_url(self) -> Optional[str]:
        return self._settings.get("services", {}).get("openai", {}).get("base_url")

    @property
    def nebius_base_url(self) -> Optional[str]:
        return self._settings.get("services", {}).get("nebius", {}).get("base_url")


# Add this to the end of src/llm_service/config.py for testing
if __name__ == "__main__":
    import os

    # Set a fake key for the test to run
    os.environ["NEBIUS_API_KEY"] = "fake-nebius-key-for-testing"

    manager = ConfigManager()

    print("Checking Nebius models...")
    assert "L_Gemma327B" in manager.models_nebius, "Nebius models not loaded!"
    print("✅ Nebius models loaded.")

    print("Checking Nebius API key...")
    assert (
        manager.nebius_api_key == "fake-nebius-key-for-testing"
    ), "Nebius API key not found!"
    print("✅ Nebius API key loaded.")

    print("Checking Nebius base URL...")
    assert (
        manager.nebius_base_url and "nebius.com" in manager.nebius_base_url
    ), "Nebius base URL not found!"
    print("✅ Nebius base URL loaded.")

    print("Checking combined model index...")
    assert "L_Qwen25VL72B" in manager.model_index, "Nebius model missing from index!"
    assert "GPT4oMini" in manager.model_index, "OpenAI model missing from index!"
    print("✅ Model index is correct.")

    print("\n🎉 All ConfigManager checks passed!")
