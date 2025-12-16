import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load secrets
load_dotenv("secrets.env")

# 2. Initialize Client
try:
    client = OpenAI()
except Exception as e:
    print(f"❌ Error initializing client: {e}")
    exit(1)

# 3. The 5 Failed Batch IDs
failed_batches = [
"batch_6930304b707c81908ea5b2485a2681b3", # GPT5Mini (Single)
    "batch_6930308b8e5081908ab2503196974723", # GPT5Nano (Single)
    "batch_693030a38f5c8190bd4eb92b703af9f0", # GPT5Nano (Combined)
    "batch_693030e7e3a48190a9a014d54ca37402", # GPT4oMini (Single)
    "batch_6930310299988190b243c21e2ab67c7c"
]

print(f"🔍 Checking {len(failed_batches)} failed batches...\n")

for batch_id in failed_batches:
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"🆔 ID: {batch_id}")
        
        if batch.errors:
            print("❌ ERRORS FOUND:")
            # PRINT RAW DATA to avoid attribute errors
            print(f"   {batch.errors}")
        else:
            print(f"❓ Status: {batch.status}")
            
        print("-" * 40)

    except Exception as e:
        print(f"❌ Critical failure checking {batch_id}: {e}")