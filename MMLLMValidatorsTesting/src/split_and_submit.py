#!/usr/bin/env python3
import asyncio
import logging
import glob
import os
import time
from pathlib import Path
from math import ceil
from src.config import settings, PROJECT_ROOT
from src.llm_service.service import OpenAIBatchService

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sequential_submit")

# --- Constants ---
CHUNK_SIZE = 400

async def process_one_batch(service, filepath, model, strategy, part_num):
    """Submits one batch and waits until it is completely finished."""
    
    filename = os.path.basename(filepath)
    batch_tag = f"Seq_{model}_{strategy}_p{part_num}"
    
    logger.info(f"\n🚀 STARTING PART {part_num}: {filename}")

    # 1. Submit
    try:
        batch_id = await asyncio.to_thread(
            service.submit_batch, str(filepath), batch_tag
        )
        if not batch_id:
            logger.error("   ❌ Submission failed (no ID returned).")
            return False
            
        logger.info(f"   ✅ Submitted -> ID: {batch_id}")
        logger.info("   ⏳ Waiting for completion (this keeps the queue clear)...")

    except Exception as e:
        logger.error(f"   ❌ Submission error: {e}")
        return False

    # 2. Monitor until done (Blocking)
    while True:
        try:
            status = service.get_batch_status(batch_id)
            state = status.get("status", "unknown")
            
            if state == "completed":
                logger.info(f"   🎉 Part {part_num} COMPLETED!")
                
                # Download
                output_file_id = status.get("output_file_id")
                if output_file_id:
                    safe_model = "".join(c if c.isalnum() else "_" for c in model)
                    output_dir = (
                        PROJECT_ROOT / "data" / "batch_results" / "openai" / safe_model / strategy
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"results_part{part_num}_{batch_id}.jsonl"
                    
                    await asyncio.to_thread(
                        service.download_batch_results, batch_id, str(output_path)
                    )
                    logger.info(f"   📥 Downloaded results to: {output_path.name}")
                return True

            elif state == "failed":
                errs = status.get("errors", "Unknown error")
                logger.error(f"   ❌ Part {part_num} FAILED. Reason: {errs}")
                return False

            elif state in ["validating", "in_progress", "finalizing"]:
                # Wait 30 seconds before checking again
                await asyncio.sleep(30)
            
            else:
                await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"   ⚠️ Monitor error: {e}")
            await asyncio.sleep(30)

async def main():
    service = OpenAIBatchService()
    batch_dir = Path(settings["paths"]["batch_request_file"]).parent
    
    # 1. Define Targets
    targets = [
        {"model": "GPT5Mini", "strategy": "single"},
        {"model": "GPT5Mini", "strategy": "combined"},
        {"model": "GPT5Nano", "strategy": "single"},
        {"model": "GPT5Nano", "strategy": "combined"},
        {"model": "GPT4oMini", "strategy": "single"},
        {"model": "GPT4oMini", "strategy": "combined"},
    ]

    # 2. Loop through targets
    for target in targets:
        model = target["model"]
        strategy = target["strategy"]
        
        # Find the original large file
        pattern = str(batch_dir / f"*openai_{model}_{strategy}*.jsonl")
        matching_files = glob.glob(pattern)
        
        # Filter out files that already include "_part" to avoid re-splitting splits
        original_files = [f for f in matching_files if "_part" not in f]

        if not original_files:
            continue

        latest_file = max(original_files, key=os.path.getmtime)
        logger.info(f"\n📦 Processing Group: {model} ({strategy})")

        # 3. Read and Split
        with open(latest_file, 'r') as f:
            lines = f.readlines()
        
        if not lines: continue
        
        num_chunks = ceil(len(lines) / CHUNK_SIZE)
        
        # 4. Process Chunks Sequentially
        for i in range(num_chunks):
            part_num = i + 1
            
            # Create/Check Part File
            original_path = Path(latest_file)
            part_filename = f"{original_path.stem}_part{part_num}{original_path.suffix}"
            part_path = original_path.parent / part_filename
            
            # Write if missing
            if not part_path.exists():
                with open(part_path, 'w') as out_f:
                    out_f.writelines(lines[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE])
            
            # CHECK IF DONE: Look for result file to skip
            safe_model = "".join(c if c.isalnum() else "_" for c in model)
            result_dir = PROJECT_ROOT / "data" / "batch_results" / "openai" / safe_model / strategy
            # Simple check if any result file exists for this part
            existing_results = glob.glob(str(result_dir / f"results_part{part_num}_*.jsonl"))
            
            if existing_results:
                logger.info(f"   ⏩ Part {part_num} already done. Skipping.")
                continue

            # RUN BATCH
            success = await process_one_batch(service, part_path, model, strategy, part_num)
            
            if not success:
                logger.error("   🛑 Stopping sequence due to failure.")
                break # Stop this model if a part fails
            
            # Small cooldown
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user.")
    # Locate the folder where your request files live
    
   