from pathlib import Path
import sys

# Use the standard library tomllib if available (Python 3.11+)
# and fall back to 'tomli' for older versions.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# This assumes __init__.py is located in .../PROJECT_ROOT/config/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# CORRECTED PATH: Look for config.toml in the project root, not inside the config folder.
CONFIG_FILE_PATH = PROJECT_ROOT / "src" / "config" / "config.toml"


def load_config() -> dict:
    """Loads the configuration from the config.toml file."""
    try:
        with open(CONFIG_FILE_PATH, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{CONFIG_FILE_PATH}'")
        return {}
    except tomllib.TOMLDecodeError as e:
        print(f"ERROR: Could not parse '{CONFIG_FILE_PATH}': {e}")
        return {}


# Load all settings from the file.
settings = load_config()

# Pre-process the settings to create a combined dictionary of all models.
# This simplifies the code in the service classes.
all_models = {}
if "models" in settings:
    for model_group in settings.get("models", {}).values():
        if isinstance(model_group, dict):
            all_models.update(model_group)

settings["all_models"] = all_models
