#!/usr/bin/env python3
import asyncio
import logging # Import logging at the top
from pathlib import Path
from sys import prefix
import pandas as pd
from datetime import datetime
import json # Import json for potential debugging if needed

from src.config import settings
from src.orchestrator import Orchestrator
from src.img_traits_def import ImgTraitDefinition
from src.science_qa import ScienceQA

# ---------- Setup logging at the MODULE LEVEL ----------
# Configure logging basic settings first
logging.basicConfig(
    level=logging.INFO, # Default level, can be overridden later if needed
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
# Create the named logger for this module
logger = logging.getLogger("run_all_models")


# ---------- PROVIDER DETECTION ----------
def get_provider_from_model_id(model_id: str) -> str:
    """Determines provider based on model ID pattern."""
    mid = model_id.upper()
    if mid.startswith("GPT"):     # e.g., GPT4o, GPT4oMini
        return "openai"
    elif mid.startswith("L_"):    # e.g., L_Gemma34B, L_Qwen25VL72B
        return "nebius"
    # Add more prefixes if needed
    # --- FIXED: logger is now accessible here ---
    logger.warning(f"⚠️ Could not determine provider for model '{model_id}', assuming 'unknown'.")
    return "unknown"

# ---------- CLEANING UTIL ----------
def clean_requests(requests: list) -> list:
    """Remove internal metadata before writing to JSONL."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in requests]


# ---------- MAIN RUNNER ----------
async def run_all_models(num_questions: int = 100):
    """
    Prepare **one batch JSONL file per model** defined in config (OpenAI + Nebius).
    """
    # Logging is already configured at module level
    logger.info("🚀 Starting batch preparation - **one file per model**...")

    # --- Load models from config ---
    all_models_dict = settings.get("all_models", {})
    if not all_models_dict:
        # Use logger defined at module level
        logger.error("❌ No models found in settings['all_models']!")
        raise RuntimeError("No models found in settings['all_models']!")

    model_ids_to_process = list(all_models_dict.keys())
    logger.info(f"🧠 Found {len(model_ids_to_process)} total models to process:")
    for m_alias in model_ids_to_process:
        try:
            provider = get_provider_from_model_id(m_alias)
            actual_name = all_models_dict[m_alias]
            logger.info(f"    - {m_alias} (Provider: {provider}, API Name: {actual_name})")
        except ValueError as e:
             logger.error(f"    - Error determining provider for {m_alias}: {e}")
        except KeyError:
             logger.error(f"    - Error: Model alias '{m_alias}' found in list but not in 'all_models' dict.")


    # --- Load dataset ---
    csv_path = Path(settings["paths"]["input_data_csv"])
    if not csv_path.exists():
         logger.error(f"❌ Input data CSV not found at: {csv_path.resolve()}")
         raise FileNotFoundError(f"Input data CSV not found at: {csv_path.resolve()}")
    try:
        df = pd.read_csv(csv_path)
        if num_questions > 0 and num_questions < len(df):
            df = df.head(num_questions)
            logger.info(f"📚 Loaded {len(df)} questions (limited by num_questions={num_questions}) from: {csv_path.resolve()}")
        else:
             logger.info(f"📚 Loaded all {len(df)} questions from: {csv_path.resolve()}")
    except Exception as e:
        logger.error(f"❌ Failed to load or process CSV '{csv_path}': {e}")
        return


    # --- Initialize orchestrator and traits ---
    try:
        trait_def_path = Path(settings["paths"]["trait_definitions_json"])
        if not trait_def_path.exists():
            logger.error(f"Trait definitions JSON not found: {trait_def_path.resolve()}")
            raise FileNotFoundError(f"Trait definitions JSON not found: {trait_def_path.resolve()}")
        trait_def = ImgTraitDefinition(trait_def_path)
        if not trait_def.traits:
             logger.error("Trait definitions loaded but resulted in an empty traits dictionary.")
             raise ValueError("Trait definitions loaded but resulted in an empty traits dictionary.")
        trait_names = list(trait_def.traits.keys())
        logger.info(f"🧬 Loaded {len(trait_names)} traits: {', '.join(trait_names)}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize trait definitions: {e}")
        return

    try:
        checkpoint_file = settings["paths"].get("checkpoint_json")
        orchestrator = Orchestrator(
            trait_names=trait_names,
            checkpoint_file=checkpoint_file,
        )
        logger.info("✅ Orchestrator initialized.")
    except Exception as e:
         logger.error(f"❌ Failed to initialize Orchestrator: {e}")
         return


    # --- Start processing PER MODEL ---
    total_models = len(model_ids_to_process)
    total_questions = len(df)
    files_created = []

    start_time_all = datetime.now()
    logger.info(f"🕓 Beginning request generation at {start_time_all.strftime('%Y-%m-%d %H:%M:%S')}...\n")

    for model_idx, model_id_alias in enumerate(model_ids_to_process, 1):
        model_start_time = datetime.now()
        logger.info(f"--- Processing Model {model_idx}/{total_models}: {model_id_alias} ---")

        requests_for_this_model = []
        try:
            provider = get_provider_from_model_id(model_id_alias)
            if provider == "openai" and not hasattr(orchestrator, 'openai_batch_service'):
                 raise AttributeError("Orchestrator missing 'openai_batch_service'.")
            if provider == "nebius" and not hasattr(orchestrator, 'nebius_batch_service'):
                 raise AttributeError("Orchestrator missing 'nebius_batch_service'.")
            if provider == "unknown":
                logger.error(f"Skipping model {model_id_alias} due to unknown provider.")
                continue

        except (ValueError, AttributeError) as e:
            logger.error(f"❌ Error setting up for model {model_id_alias}: {e}. Skipping model.")
            continue

        processed_questions_for_model = 0
        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                question = ScienceQA.from_df_row(row)
                requests, image_file_ids_or_none = await orchestrator.prepare_batch_requests(
                    question, provider, model_id_alias
                )
                requests_for_this_model.extend(requests)
                processed_questions_for_model += 1

                if i % 20 == 0 or i == total_questions:
                    elapsed_model = (datetime.now() - model_start_time).total_seconds()
                    logger.info(f"    Processed {i}/{total_questions} questions for {model_id_alias} ({len(requests_for_this_model)} requests total) [{elapsed_model:.1f}s]")

            except Exception as e:
                logger.error(f"    ❌ Error processing QID {row.get('question_id', 'N/A')} for model {model_id_alias}: {e}")

        model_end_time = datetime.now()
        model_duration = (model_end_time - model_start_time).total_seconds()
        logger.info(f"✅ Finished model {model_id_alias}. Generated {len(requests_for_this_model)} requests in {model_duration:.2f} seconds.")

        if requests_for_this_model:
            cleaned_requests = clean_requests(requests_for_this_model)
            base_path = Path(settings["paths"]["batch_request_file"])
            timestamp = model_end_time.strftime("%Y%m%d_%H%M%S")
            safe_model_alias = "".join(c if c.isalnum() else "_" for c in model_id_alias)
            output_filename = f"{base_path.stem}_{safe_model_alias}_{timestamp}{base_path.suffix}"
            # Ensure the output directory exists
            output_dir = base_path.parent
            output_dir.mkdir(parents=True, exist_ok=True) # Create dir if it doesn't exist
            output_path = output_dir / output_filename

            try:
                batch_service = None
                if provider == "openai":
                    batch_service = orchestrator.openai_batch_service
                elif provider == "nebius":
                    batch_service = orchestrator.nebius_batch_service

                if batch_service:
                    batch_service.write_jsonl_file(cleaned_requests, str(output_path))
                    logger.info(f"💾 Batch file for {model_id_alias} saved -> {output_path.resolve()}")
                    files_created.append(output_path.resolve())
                else:
                     logger.error(f"Could not find batch service for provider '{provider}' to write file.")

            except Exception as e:
                logger.error(f"❌ Failed to write batch file for model {model_id_alias} to {output_path}: {e}")
        else:
             logger.warning(f"⚠️ No requests generated for model {model_id_alias}, skipping file write.")

        logger.info("-" * (len(f"--- Processing Model {model_idx}/{total_models}: {model_id_alias} ---")) + "\n")


    # --- Summary after all models ---
    end_time_all = datetime.now()
    total_duration_minutes = (end_time_all - start_time_all).total_seconds() / 60
    logger.info("="*50)
    logger.info(f"🎉 All models processed in {total_duration_minutes:.2f} minutes.")
    logger.info(f"📦 Generated {len(files_created)} batch file(s):")
    if files_created:
        for f_path in files_created:
            logger.info(f"   -> {f_path.name}")
    else:
        logger.info("   (No files were generated)")

    logger.info("\n📤 To submit batches (example for one file):")
    if files_created:
         example_path = files_created[0]
         # Infer provider from filename part (assuming format like ..._ModelAlias_timestamp.suffix)
         try:
            stem = example_path.stem
            prefix = "batch_request_file_"
            suffix_start_index = -1
            parts = stem.split("_")
            # looking backwards for time and date
            if len(parts) >= 3 and len(parts[-1]) == 6 and parts[-1].isdigit() and \
                len(parts[-2]) == 8 and parts[-2].isdigit():
                model_alias_in_filename = "_".join(parts[3:-2])
            else:
                model_alias_in_filename = "unknown_alias"
                logger.warning(f"Could not reliably parse filename pattern '{example_path.name}' for example command.")

            if model_alias_in_filename != "unknown_alias":
                example_provider = get_provider_from_model_id(model_alias_in_filename)
                logger.info(f"   python src/main.py --provider {example_provider} --batch-file \"{example_path.resolve()}\" --submit-batch --monitor")
            else:
                # If alias extraction failed, show generic command
                 logger.info("   python src/main.py --provider [openai|nebius] --batch-file \"<YOUR_FILE_PATH>\" --submit-batch --monitor")

         except Exception as e: # Catch any potential errors during parsing
             logger.error(f"Error parsing filename '{example_path.name}' to generate example command: {e}")
             logger.info("uv run python -m src.main --provider [openai|nebius] --batch-file \"<YOUR_FILE_PATH>\" --submit-batch --monitor")
    else: # If no files created, still show generic command
        logger.info("uv run python -m src.main --provider [openai|nebius] --batch-file \"<YOUR_FILE_PATH>\" --submit-batch --monitor")

    logger.info("\n✨ Done!")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate batch request files, one per model.")
    parser.add_argument(
        "-n", "--num-questions",
        type=int,
        default=100,
        help="Number of questions to process (0 for all). Default: 100"
    )
    # Add log level argument
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    args = parser.parse_args()

    # --- Set logging level based on argument ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level_numeric) # Set level for our named logger
    # Optionally set level for root logger if other modules log too
    logging.getLogger().setLevel(log_level_numeric)
    logger.info(f"Logging level set to {args.log_level.upper()}")


    try:
        asyncio.run(run_all_models(num_questions=args.num_questions))
    except (RuntimeError, FileNotFoundError, ValueError, Exception) as e:
        # --- FIXED: logger is now accessible here ---
        logger.error(f"💥 An error occurred during execution: {e}", exc_info=True) # Add traceback
        import sys
        sys.exit(1) # Exit with error code