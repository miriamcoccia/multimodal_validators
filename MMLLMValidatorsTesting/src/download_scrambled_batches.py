import logging
from pathlib import Path
from src.config import PROJECT_ROOT
from src.llm_service.service import OpenAIBatchService, NebiusBatchService

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("downloader")

# Paths
LOG_FILE_PATH = PROJECT_ROOT / "data" / "submitted_batches_log.txt"
OUTPUT_BASE_DIR = PROJECT_ROOT / "data" / "batch_results_scrambled_FINAL"

def parse_metadata(filename: str):
    """Extracts model and strategy from filename for folder organization."""
    name = filename.lower()
    
    # Strategy Detection
    strategy = "single" if "single" in name or "_p" in name else "combined" if "combined" in name else "unknown"
    if "combined" in name:
        strategy = "combined"
    
    # Model Detection
    model = "unknown"
    if "gemma" in name: model = "L_Gemma327B"
    elif "qwen" in name: model = "L_Qwen25VL72B"
    elif "gpt5mini" in name: model = "GPT5Mini"
    elif "gpt5nano" in name: model = "GPT5Nano"
    elif "gpt4omini" in name: model = "GPT4oMini"
    
    return model, strategy

def main():
    if not LOG_FILE_PATH.exists():
        logger.error(f"❌ Log file not found at {LOG_FILE_PATH}")
        return

    # Initialize Services
    openai_service = OpenAIBatchService()
    nebius_service = NebiusBatchService()

    with open(LOG_FILE_PATH, "r") as f:
        lines = f.readlines()

    logger.info(f"🚀 Found {len(lines)} batch entries. Starting download...")

    for line in lines:
        if "|" not in line: continue
        
        # Parse log line: [Timestamp] PROVIDER | Part | Filename | BatchID
        parts = [p.strip() for p in line.split("|")]
        provider_label = parts[0].split("]")[-1].strip().lower() 
        filename = parts[2]
        batch_id = parts[3]
        
        model, strategy = parse_metadata(filename)
        provider = "openai" if "openai" in provider_label else "nebius"
        
        # Setup Output Path (Provider/Model/Strategy)
        target_dir = OUTPUT_BASE_DIR / provider / model / strategy
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Output filename: original_name_results.jsonl
        result_name = filename.replace(".jsonl", "_results.jsonl")
        output_path = target_dir / result_name

        if output_path.exists():
            continue

        service = openai_service if provider == "openai" else nebius_service
        
        logger.info(f"📥 Downloading {batch_id} -> {model}/{strategy}/{result_name}")
        
        try:
            success = service.download_batch_results(batch_id, str(output_path))
            if not success:
                logger.warning(f"   ⚠️ Batch {batch_id} is not ready or failed.")
        except Exception as e:
            logger.error(f"   ❌ Error with {batch_id}: {e}")

    logger.info(f"\n✨ Done. Results saved in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()