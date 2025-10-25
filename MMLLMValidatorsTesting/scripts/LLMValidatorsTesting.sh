#!/bin/bash
# This script sets up the environment and then runs the main Python script.

# --- Path Setup ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/venv/bin/activate"
SECRETS_FILE="$PROJECT_ROOT/secrets.env"

# --- API Key Loading ---
if [ -f "$SECRETS_FILE" ]; then
    source "$SECRETS_FILE"
else
    echo "Error: 'secrets.env' file not found in project root."
    echo "Please create it and add your API keys before running."
    exit 1
fi

# --- Ollama Environment Setup ---
export OLLAMA_DIR=/mnt/beegfs/gpfs/filesets/project_2024_OAT/libs/ollama_v09/
export OLLAMA_LOAD_TIMEOUT=600
export PATH="$PATH:$OLLAMA_DIR/bin"

# --- Virtual Environment Activation ---
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at '$VENV_PATH'."
    exit 1
fi
source "$VENV_PATH"

# --- Execute the Python Script ---
# Change to the 'src' directory, which contains all our Python packages
SRC_DIR="$PROJECT_ROOT/src"
cd "$SRC_DIR"

# Run the 'Main.py' script
python3 -u -m MMLLMValidatorsTesting.Main "$@"