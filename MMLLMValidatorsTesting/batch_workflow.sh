#!/bin/bash


set -e  # the script will stop if anything returns errors, "fail-quickly" approach

MODELS="GPT4oMini,GPT4o"
NUM_QUESTIONS=10
BATCH_NAME="trait_evaluation_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting batch workflow..."

# 1. Generate and submit batch
echo "📝 Generating batch requests..."
python src/main.py \
    --batch \
    --submit-batch \
    --models "$MODELS" \
    --num-questions $NUM_QUESTIONS \
    --batch-name "$BATCH_NAME"

# Extract batch ID from output (I'll need to implement this)
BATCH_ID=$(python -c "import json; print(json.load(open('data/batch_requests.batch_info.json'))['batch_id'])")

echo "📊 Batch submitted: $BATCH_ID"

# 2. Monitor progress
echo "⏳ Monitoring batch progress..."
while true; do
    STATUS=$(python src/main.py --check-batch "$BATCH_ID" --quiet)
    echo "Status: $STATUS"
    
    if [[ "$STATUS" == "completed" ]]; then
        echo "✅ Batch completed!"
        break
    elif [[ "$STATUS" == "failed" ]]; then
        echo "❌ Batch failed!"
        exit 1
    fi
    
    sleep 300  # Check every 5 minutes
done

# 3. Download results
echo "📥 Downloading results..."
python src/main.py --download-batch "$BATCH_ID"

echo "🎉 Workflow complete!"