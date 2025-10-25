#!/bin/bash

# This script orchestrates running experiments by looping through model families.
# It leverages the project's config.toml for most settings.

# Ensure the script runs from its own directory
cd "$(dirname "$0")" || exit

# --- Configuration ---

declare -A EXPERIMENTS
EXPERIMENTS["L_Qwen"]="General_V1_Img"
EXPERIMENTS["PCL_Qwen"]="Optimized_V1"
EXPERIMENTS["L_Gemma"]="General_V1_Img"
EXPERIMENTS["PCL_Gemma"]="Optimized_V1"
EXPERIMENTS["L_Llama"]="General_V1_Img"
EXPERIMENTS["PCL_Llama"]="Optimized_V1"
EXPERIMENTS["L_Mistral"]="General_V1_Img"
EXPERIMENTS["PCL_Mistral"]="Optimized_V1"

# Experiment parameters

MAX_QUESTIONS=100

# --- Main Loop ---
echo "--- Starting Experiment Run ---"

for PREFIX in "${!EXPERIMENTS[@]}"; do
    EVAL_PROMPT=${EXPERIMENTS[$PREFIX]}
    
    # Output directory is named after the model prefix (the "family").

    OUT_DIR="/home/ldap/coccia@private.list.lu/oat_2024/MultimodalLLMs/MMLLMValidatorsTesting/data/results/$PREFIX"
    TIME_LOG_FILE="$OUT_DIR/timing_total.log"
    
    FINAL_SUMMARY_FILE="$OUT_DIR/summary.csv"

    mkdir -p "$OUT_DIR"

    # --- Completion Check ---

    if [ -f "$FINAL_SUMMARY_FILE" ]; then
        echo "SKIPPING: Experiment for '$PREFIX' appears complete (found summary.csv)."
        echo ""
        continue
    fi
    
    # --- Launch experiment ---
    echo "Launching experiment for models starting with '$PREFIX'..."
    

    nohup time -o "$TIME_LOG_FILE" ./LLMValidatorsTesting.sh \
        -N "$MAX_QUESTIONS" \
        -p "$EVAL_PROMPT" \
        -o "$OUT_DIR" \
        -sw "$PREFIX" \
        --multimodal \
        > "$OUT_DIR/log.out" \
        2> "$OUT_DIR/err.out" &

    echo "-> Experiment for '$PREFIX' launched in background."
    echo "-> Logs and results will be in: $OUT_DIR"
    echo ""
done

echo "--- All experiments initiated. ---"