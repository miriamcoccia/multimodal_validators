import os
import glob
import asyncio
import time
from pathlib import Path
from math import ceil
from datetime import datetime
from src.config import settings, PROJECT_ROOT
from src.llm_service.service import OpenAIBatchService, NebiusBatchService

# --- CONFIG ---
OPENAI_CHUNK_SIZE = 400

def log_submission(provider, model, filename, batch_id):
    """Saves IDs to a text file so you can find them tomorrow."""
    with open("submitted_batches_log.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {provider} | {model} | {filename} | {batch_id}\n")

async def main():
    print(f"🚀 STARTING FIRE-AND-FORGET SUBMISSION")
    
    # 1. FIND THE FILES (Recursively search all 'data' folders)
    # This finds files in data/, data/batch_results/, etc.
    search_pattern = str(PROJECT_ROOT / "data" / "**" / "batch_request_file*.jsonl")
    all_files = glob.glob(search_pattern, recursive=True)
    
    # Filter:
    # 1. Ignore files that are already split parts ("_part")
    # 2. Only take files created in the last 24 hours (86400 seconds)
    current_time = time.time()
    source_files = [
        f for f in all_files 
        if "_part" not in f 
        and (current_time - os.path.getmtime(f)) < 86400
    ]
    
    if not source_files:
        print(f"❌ No recent batch files found in {PROJECT_ROOT}/data")
        return

    print(f"📂 Found {len(source_files)} recent batch files.")

    # Initialize Services
    nebius_svc = NebiusBatchService()
    openai_svc = OpenAIBatchService()

    for filepath in source_files:
        filename = os.path.basename(filepath)
        
        # --- STRATEGY: NEBIUS (Submit Whole) ---
        if "nebius" in filename.lower():
            print(f"\n🔵 NEBIUS: {filename}")
            try:
                # Add timestamp to unique ID to avoid "duplicate" errors
                batch_tag = f"Nebius_{filename[:15]}_{int(time.time())}"
                batch_id = await asyncio.to_thread(nebius_svc.submit_batch, filepath, batch_tag)
                print(f"   ✅ Success! ID: {batch_id}")
                log_submission("NEBIUS", "Unknown", filename, batch_id)
            except Exception as e:
                print(f"   ❌ Failed: {e}")

        # --- STRATEGY: OPENAI (Split & Submit) ---
        elif "openai" in filename.lower():
            print(f"\n🟢 OPENAI: {filename} (Splitting...)")
            
            # Read file
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if not lines: continue
            num_chunks = ceil(len(lines) / OPENAI_CHUNK_SIZE)
            
            for i in range(num_chunks):
                part_num = i + 1
                # Create Part File next to the original
                original_path = Path(filepath)
                part_name = f"{original_path.stem}_part{part_num}{original_path.suffix}"
                part_path = original_path.parent / part_name
                
                # Write part
                with open(part_path, 'w') as out_f:
                    out_f.writelines(lines[i*OPENAI_CHUNK_SIZE : (i+1)*OPENAI_CHUNK_SIZE])
                
                # Submit Part Immediately
                print(f"   🚀 Submitting Part {part_num}/{num_chunks}...", end="\r")
                try:
                    batch_tag = f"OA_P{part_num}_{filename[:10]}_{int(time.time())}"
                    batch_id = await asyncio.to_thread(
                        openai_svc.submit_batch, str(part_path), batch_tag
                    )
                    print(f"   ✅ Part {part_num} Launched: {batch_id}          ")
                    log_submission("OPENAI", f"Part_{part_num}", part_name, batch_id)
                except Exception as e:
                    print(f"   ❌ Part {part_num} Failed: {e}")
        
        else:
            print(f"⚠️ Skipping unknown provider file: {filename}")

    print("\n" + "="*50)
    print("✨ ALL DONE. Batches are in the cloud.")
    print("💤 You can safely turn off your laptop now.")
    print("📝 Check 'submitted_batches_log.txt' tomorrow for IDs.")

if __name__ == "__main__":
    asyncio.run(main())