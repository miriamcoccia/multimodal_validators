"""
Batch processor that finds all .jsonl results in data/batch_results,
processes them using ResultsHandler, and saves them to data/processed_results.
"""
import logging
import sys
from pathlib import Path
import pandas as pd

# Ensure src is in path if running as script
sys.path.append(str(Path(__file__).parents[1]))

from src.results_handler import ResultsHandler
from src.config import PROJECT_ROOT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("process_all")


def process_all_batches():
    # 1. Define Directories
    input_root = PROJECT_ROOT / "data" / "batch_results"
    output_root = PROJECT_ROOT / "data" / "processed_results"

    if not input_root.exists():
        logger.error(f"❌ Input directory not found: {input_root}")
        return

    # 2. Find all JSONL files recursively
    jsonl_files = list(input_root.rglob("*.jsonl"))
    
    if not jsonl_files:
        logger.warning(f"⚠️ No .jsonl files found in {input_root}")
        return

    logger.info(f"🚀 Found {len(jsonl_files)} batch result files to process.")

    success_count = 0
    error_count = 0

    # 3. Iterate and Process
    for i, jsonl_path in enumerate(jsonl_files, 1):
        try:
            # Calculate relative path structure (e.g., nebius/L_Gemma327B/combined)
            relative_path = jsonl_path.relative_to(input_root).parent
            filename_stem = jsonl_path.stem  # Filename without .jsonl

            # Create output path: processed_results/{provider}/{model}/{mode}/{batch_filename}/
            # We add the filename_stem to avoid overwriting if multiple batches exist for one mode
            target_dir = output_root / relative_path / filename_stem
            
            logger.info(f"[{i}/{len(jsonl_files)}] Processing: {jsonl_path.name}")
            logger.debug(f"   📍 Target: {target_dir}")

            # Initialize Handler
            rh = ResultsHandler(target_dir)

            # Load
            raw_df = rh.load_batch_results_jsonl(jsonl_path)
            if raw_df.empty:
                logger.warning(f"   ⚠️ Empty dataframe for {jsonl_path.name}. Skipping.")
                error_count += 1
                continue

            # Save Raw
            rh.save_raw_results(raw_df)

            # Clean & Save
            clean_df = rh.clean_results(raw_df)
            if clean_df.empty:
                logger.warning(f"   ⚠️ Cleaning resulted in empty data for {jsonl_path.name}.")
            else:
                rh.save_clean_results(clean_df)
                logger.info(f"   ✅ Saved clean CSV with {len(clean_df)} rows.")
                success_count += 1

        except Exception as e:
            logger.error(f"   ❌ Failed to process {jsonl_path.name}: {e}")
            error_count += 1

    # 4. Summary
    logger.info("=" * 40)
    logger.info(f"🎉 Processing Complete.")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed/Empty: {error_count}")
    logger.info(f"📂 Results stored in: {output_root}")


if __name__ == "__main__":
    process_all_batches()