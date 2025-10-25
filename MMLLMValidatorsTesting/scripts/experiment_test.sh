#!/bin/bash

declare -A MODEL_FAMILIES
MODEL_FAMILIES["qwen_mm_local"]="L_Qwen25VL3B,L_Qwen25L7B,L_Qwen25VL32B"
MODEL_FAMILIES["gemma_mm_local"]="L_Gemma34B,L_Gemma312B,L_Gemma327B"
MODEL_FAMILIES["llama_mm_local"]="L_Llama32Vision11B,L_Llama32Vision90B,L_Llama4Scout"
MODEL_FAMILIES["mistral_mm_local"]="L_MistralSmall3124B"
MODEL_FAMILIES["openai"]="GPT4o,GPT4oMini,GPT4Turbo"


INPUT="../data/raw/ScienceQA_test_mc_images_mod.csv"
PROMPT="General_V1_Img"
MAX_QUESTIONS=100


for RUN in {1..3}; do
    echo "--- Starting Run: $RUN ---"
    for FAMILY in "${!MODEL_FAMILIES[@]}"; do

        MODELS=${MODEL_FAMILIES[$FAMILY]}
        OUT_DIR="../data/results/test_results/$RUN/$FAMILY"
        OUT_FILE="$OUT_DIR/questions_traits_evaluation_mod.csv"

        mkdir -p "$OUT_DIR"

        NUM_MODELS=$(echo "$MODELS" | tr ',' ' ' | wc -w)
        EXPECTED_ROWS=$((NUM_MODELS * MAX_QUESTIONS))

        if [ -f "$OUT_FILE" ]; then
            # Count actual data rows, skipping the header
            ACTUAL_ROWS=$(tail -n +2 "$OUT_FILE" | wc -l)
            if [ "$ACTUAL_ROWS" -ge "$EXPECTED_ROWS" ]; then
                echo "SKIPPING: $FAMILY (Run $RUN) is complete with $ACTUAL_ROWS rows."
                continue
            else
                echo "INCOMPLETE: $FAMILY (Run $RUN) has $ACTUAL_ROWS of $EXPECTED_ROWS expected rows."
                
                mv "$OUT_FILE" "${OUT_FILE}.incomplete"
                echo "Preserved partial data as ${OUT_FILE}.incomplete"
            fi
        else
            echo "RUNNING: $FAMILY (Run $RUN) - No output file found."
        fi
        
        # --- Launch experiment for the family ---
        echo "Launching full experiment for $FAMILY..."
        nohup ./LLMValidatorsTesting.sh \
            -i "$INPUT" \
            -N "$MAX_QUESTIONS" \
            -p "$PROMPT" \
            -o "$OUT_DIR" \
            -m "$MODELS" \
            1> "$OUT_DIR/log.out" \
            2> "$OUT_DIR/err.out" &

        echo "Experiment launched. Logs -> $OUT_DIR/log.out/.err"
        echo ""
    done
    echo "--- All families checked for Run: $RUN ---"
    echo ""
done
