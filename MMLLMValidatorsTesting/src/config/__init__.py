import sys
from pathlib import Path
import os
from dotenv import load_dotenv

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        print(
            "ERROR: 'tomli' package not found. Please install it with 'pip install tomli'"
        )
        sys.exit(1)

# --- PATH DEFINITION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent

CONFIG_FILE_PATH = (
    Path(__file__).parent / "config.toml"
)  # path relative to the current file __file__ refers to this file we are on rn
ENV_PATH = PROJECT_ROOT / "secrets.env"

print(f"Attempting to load environment variables from: {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)  # Load the .env file into the environment
print("✅ Environment variables loaded.")


def load_config() -> dict:
    """
    Loads and validates the configuration from the config.toml file.
    Resolves API keys from environment variables into an 'api' section.
    """
    print(f"Attempting to load configuration from: {CONFIG_FILE_PATH}")
    if not CONFIG_FILE_PATH.is_file():
        print(f"FATAL ERROR: Configuration file not found at '{CONFIG_FILE_PATH}'.")
        sys.exit(1)

    try:
        with open(CONFIG_FILE_PATH, "rb") as f:
            config_data = tomllib.load(f)

        api_keys_config = config_data.get("api_keys", {})
        resolved_secrets = {}

        for key_name_in_toml, env_var_name in api_keys_config.items():
            if key_name_in_toml.endswith("_env"):
                secret_value = os.getenv(env_var_name)
                new_key = key_name_in_toml.removesuffix("_env")

                if secret_value:
                    resolved_secrets[new_key] = secret_value
                    print(f"✅ Secret '{new_key}' loaded successfully.")
                else:
                    resolved_secrets[new_key] = None
                    print(
                        f"⚠️ WARNING: Environment variable '{env_var_name}' for '{new_key}' not found."
                    )

        # CRITICAL FIX: Store resolved secrets in a new 'api' section
        if resolved_secrets:
            if "api" not in config_data:
                config_data["api"] = {}
            config_data["api"].update(resolved_secrets)
            print(
                f"✅ Stored {len(resolved_secrets)} resolved API keys in config['api']"
            )

        # Build all_models index
        all_models = {}
        if "models" in config_data:
            for model_group in config_data.get("models", {}).values():
                if isinstance(model_group, dict):
                    all_models.update(model_group)
        config_data["all_models"] = all_models

        print("✅ Configuration loaded and processed successfully.")
        return config_data

    except tomllib.TOMLDecodeError as e:
        print(f"FATAL ERROR: Could not parse '{CONFIG_FILE_PATH}': {e}")
        sys.exit(1)


# --- GLOBAL SETTINGS VARIABLE ---
# This line executes when the module is imported, loading the settings once.
# Any other module can now access settings by using `from src.config import settings`.
settings = load_config()
