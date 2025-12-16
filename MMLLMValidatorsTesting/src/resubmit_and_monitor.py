#!/usr/bin/env python3
import asyncio
import logging
import glob
import os
from pathlib import Path
from src.config import settings, PROJECT_ROOT
from src.llm_service.service import OpenAIBatchService
from src.llm_service.providers.basic_batch import BaseBatchService

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resubmit_tool")

# --- Reusing Your Monitoring Logic ---
async def monitor_batch_progress(
    batch_id: str,
    batch_service: BaseBatchService,
    check_interval: int,
    model_id: str,
    provider: str,
    strategy: str,
):
    """
    Monitors status and downloads results to the standard project path.
    """
    logger.info(f"👀 Monitoring batch {batch_id} for {model_id} ({strategy})...")

    while True:
        try:
            status = batch_service.get_batch_status(batch_id)
            batch_status = status.get("status", "unknown")

            if batch_status == "completed":
                logger.info(f"🎉 Batch {batch_id} completed!")

                output_file_id = status.get("output_file_id")
                if not output_file_id:
                    logger.error(f"❌ Batch {batch_id} has no output_file_id.")
                else:
                    # Construct standard path
                    safe_model = "".join(c if c.isalnum() else "_" for c in model_id)
                    output_dir = (
                        PROJECT_ROOT
                        / "data"
                        / "batch_results"
                        / provider
                        / safe_model
                        / strategy
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_filename = f"results_{batch_id}_{output_file_id}.jsonl"
                    output_path = output_dir / output_filename

                    logger.info(f"📥 Downloading to {output_path}...")
                    
                    # Run the blocking download in a thread to keep async loop alive
                    success = await asyncio.to_thread(
                        batch_service.download_batch_results, batch_id, str(output_path)
                    )
                    
                    if success:
                        logger.info(f"✅ Saved: {output_path.name}")
                    else:
                        logger.warning(f"⚠️ Download failed for {batch_id}.")

                return True

            elif batch_status == "failed":
                logger.error(f"❌ Batch {batch_id} failed!")
                # Try to print error reason
                if status.get("errors"):
                     logger.error(f"Reason: {status.get('errors')}")
                return False

            elif batch_status == "in_progress":
                # Only log progress occasionally to reduce noise
                counts = status.get("request_counts", {})
                logger.debug(f"⏳ {model_id}: {counts}")

            await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Error checking batch {batch_id}: {e}")
            await asyncio.sleep(check_interval)

# --- Main Resubmission Logic ---
async def main():
    service = OpenAIBatchService()
    batch_dir = Path(settings["paths"]["batch_request_file"]).parent
    
    # Map the failed targets to their specific details
    # (Provider is assumed OpenAI since that's what failed)
    failed_targets = [
        {"model": "GPT5Mini", "strategy": "single"},
        {"model": "GPT5Nano", "strategy": "single"},
        {"model": "GPT5Nano", "strategy": "combined"},
        {"model": "GPT4oMini", "strategy": "single"},
        {"model": "GPT4oMini", "strategy": "combined"},
    ]

    tasks = []

    print(f"📂 Scanning {batch_dir} for failed files...")

    for target in failed_targets:
        model = target["model"]
        strategy = target["strategy"]
        
        # specific pattern to match how files were named in run_all_models
        # e.g. *openai_GPT5Nano_single*
        pattern = str(batch_dir / f"*openai_{model}_{strategy}*.jsonl")
        matching_files = glob.glob(pattern)

        if not matching_files:
            logger.warning(f"⚠️  No file found for {model} {strategy}")
            continue

        # Get latest file
        latest_file = max(matching_files, key=os.path.getmtime)
        filename = os.path.basename(latest_file)
        
        logger.info(f"🚀 Resubmitting: {filename}")

        try:
            # Run blocking submit in thread
            batch_id = await asyncio.to_thread(
                service.submit_batch, latest_file, f"Resubmit_{model}_{strategy}"
            )

            if batch_id:
                logger.info(f"   ✅ Submitted {model} ({strategy}) -> ID: {batch_id}")
                
                # Create monitoring task
                task = asyncio.create_task(
                    monitor_batch_progress(
                        batch_id=batch_id,
                        batch_service=service,
                        check_interval=60, # Check every 1 minute
                        model_id=model,
                        provider="openai",
                        strategy=strategy
                    )
                )
                tasks.append(task)
            else:
                logger.error(f"   ❌ Failed to get Batch ID for {filename}")

        except Exception as e:
            logger.error(f"   ❌ Exception submitting {filename}: {e}")

    if tasks:
        logger.info(f"\n👀 Monitoring {len(tasks)} batches in parallel. Do not close this terminal.\n")
        await asyncio.gather(*tasks)
    else:
        logger.info("No batches were submitted.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Process stopped by user.")