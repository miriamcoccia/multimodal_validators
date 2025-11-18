#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import argparse
from typing import Optional, List, Dict, Any

from src.config import settings, PROJECT_ROOT  # ✅ Added PROJECT_ROOT
from src.orchestrator import Orchestrator
from src.img_traits_def import ImgTraitDefinition
from src.science_qa import ScienceQA
from src.llm_service.providers.basic_batch import BaseBatchService
from src.llm_service.service import (
    OpenAIBatchService,
    NebiusBatchService,
)

# ---------- Setup logging at the MODULE LEVEL ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all_models")


# ---------- PROVIDER DETECTION (from main.py) ----------
def get_provider_from_model_id(model_id: str) -> str:
    """Determines the provider based on the model ID prefix"""
    if model_id.upper().startswith("GPT"):
        return "openai"
    elif model_id.upper().startswith("L_"):
        return "nebius"
    raise ValueError(f"Could not determine provider for model {model_id}")


# ---------- BATCH HELPERS (from main.py) ----------
async def test_models_batch(
    model_ids: List[str],
    data: pd.DataFrame,
    orchestrator: Orchestrator,
    mode: str,  # "single", "combined", or "both"
) -> List[Dict[str, Any]]:
    """
    Collect all batch requests across models and questions based on the mode.
    """
    all_requests = []
    total_questions = len(data)

    for model_id in model_ids:
        try:
            provider = get_provider_from_model_id(model_id=model_id)
        except ValueError as e:
            logger.error(f"❌ {e}. Skipping model {model_id}.")
            continue

        logger.info(
            f"\n📋 Collecting requests for {model_id} (Provider: {provider}, Mode: {mode})..."
        )

        for i, (_, row) in enumerate(data.iterrows(), 1):
            question = ScienceQA.from_df_row(row)

            try:
                if mode in ("single", "both"):
                    # Assumes prepare_batch_requests adds internal keys:
                    # _provider, _model_id, _strategy="single"
                    single_reqs, _ = await orchestrator.prepare_batch_requests(
                        question, provider, model_id
                    )
                    all_requests.extend(single_reqs)

                if mode in ("combined", "both"):
                    # Assumes prepare_combined_batch_requests adds internal keys:
                    # _provider, _model_id, _strategy="combined"
                    combined_reqs, _ = (
                        await orchestrator.prepare_combined_batch_requests(
                            question, provider, model_id
                        )
                    )
                    all_requests.extend(combined_reqs)

            except Exception as e:
                qid = row.get("question_id", "Unknown")
                logger.error(
                    f"    Failed to prepare request for QID {qid}, Model {model_id}: {e}"
                )

            if (i + 1) % 10 == 0 or (i + 1) == total_questions:
                logger.info(f"    Processed {i + 1}/{total_questions} questions")

    return all_requests


async def monitor_batch_progress(
    batch_id: str,
    batch_service: BaseBatchService,
    check_interval: int,
    # ✅ Added metadata for download path
    model_id: str,
    provider: str,
    strategy: str,
):
    """Monitor batch progress and download results on completion."""
    _log = logging.getLogger(__name__)
    _log.info(f"👀 Monitoring batch {batch_id}...")

    while True:
        try:
            status = batch_service.get_batch_status(batch_id)
            batch_status = status.get("status", "unknown")

            if batch_status == "completed":
                _log.info(f"🎉 Batch {batch_id} completed!")

                # --- ✅ NEW DOWNLOAD LOGIC ---
                output_file_id = status.get("output_file_id")
                if not output_file_id:
                    _log.error(
                        f"❌ Batch {batch_id} completed but has no output_file_id. Cannot download."
                    )
                else:
                    # Construct the path: data/batch_results/<provider>/<model_id>/<strategy>/
                    safe_model = "".join(
                        c if c.isalnum() else "_" for c in model_id
                    )
                    output_dir = (
                        PROJECT_ROOT
                        / "data"
                        / "batch_results"
                        / provider
                        / safe_model
                        / strategy
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Use a unique filename
                    output_filename = f"results_{batch_id}_{output_file_id}.jsonl"
                    output_path = output_dir / output_filename

                    _log.info(
                        f"📥 Downloading results for {batch_id} to {output_path}..."
                    )
                    try:
                        success = batch_service.download_batch_results(
                            batch_id, str(output_path)
                        )
                        if success:
                            _log.info(f"✅ Download complete: {output_path.name}")
                        else:
                            _log.warning(f"⚠️ Download command failed for {batch_id}.")
                    except Exception as e:
                        _log.error(f"❌ Download failed for {batch_id}: {e}")
                # --- END NEW DOWNLOAD LOGIC ---

                error_file_id = status.get("error_file_id")
                if error_file_id:
                    _log.warning("⚠️ Batch has errors, checking error file...")
                    try:
                        error_content = batch_service.client.files.content(
                            error_file_id
                        )
                        text_preview = getattr(error_content, "text", None)
                        if text_preview:
                            _log.error(f"First error: {text_preview[:1000]}...")
                    except Exception as e:
                        _log.error(f"Could not retrieve error file: {e}")
                return True

            elif batch_status == "failed":
                _log.error(f"❌ Batch {batch_id} failed!")
                return False

            elif batch_status == "in_progress":
                counts = status.get("request_counts", {})
                _log.info(f"⏳ Progress: {counts}")

            await asyncio.sleep(check_interval)

        except Exception as e:
            _log.error(f"Error checking batch: {e}")
            await asyncio.sleep(check_interval)


# ---------- MAIN RUNNER ----------
async def run_all_models(args):
    """
    Prepare and optionally submit batch JSONL files for ALL models
    defined in config, respecting the --trait-mode.
    """
    logger.info(
        f"🚀 Starting batch preparation for ALL models (Mode: {args.trait_mode})..."
    )

    # --- Load ALL models from config ---
    all_models_dict = settings.get("all_models", {})
    if not all_models_dict:
        logger.error("❌ No models found in settings['all_models']!")
        raise RuntimeError("No models found in settings['all_models']!")

    model_ids_to_test = list(all_models_dict.keys())
    logger.info(f"🧠 Found {len(model_ids_to_test)} total models to process:")
    for m_alias in model_ids_to_test:
        try:
            provider = get_provider_from_model_id(m_alias)
            actual_name = all_models_dict[m_alias]
            logger.info(
                f"    - {m_alias} (Provider: {provider}, API Name: {actual_name})"
            )
        except Exception as e:
            logger.warning(f"    - Could not determine provider for {m_alias}: {e}")

    # --- Load dataset ---
    csv_path = Path(settings["paths"]["input_data_csv"])
    if not csv_path.exists():
        logger.error(f"❌ Input data CSV not found at: {csv_path.resolve()}")
        raise FileNotFoundError(f"Input data CSV not found at: {csv_path.resolve()}")
    try:
        df = pd.read_csv(csv_path)
        if args.num_questions > 0 and args.num_questions < len(df):
            df = df.head(args.num_questions)
            logger.info(
                f"📚 Loaded {len(df)} questions (limited by --num-questions={args.num_questions}) from: {csv_path.resolve()}"
            )
        else:
            logger.info(
                f"📚 Loaded all {len(df)} questions from: {csv_path.resolve()}"
            )
    except Exception as e:
        logger.error(f"❌ Failed to load or process CSV '{csv_path}': {e}")
        return

    # --- Initialize orchestrator and traits ---
    try:
        trait_def_path = Path(settings["paths"]["trait_definitions_json"])
        trait_def = ImgTraitDefinition(trait_def_path)
        trait_list = list(trait_def.traits.keys())
        if not trait_list:
            raise ValueError("Trait definitions loaded but resulted in an empty list.")
        logger.info(f"🧬 Loaded {len(trait_list)} traits.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize trait definitions: {e}")
        return

    # Initialize Orchestrator (following main.py, no checkpoint file)
    orchestrator = Orchestrator(
        trait_names=trait_list,
    )
    logger.info("✅ Orchestrator initialized.")

    # --- Start request generation ---
    start_time_all = datetime.now()
    logger.info(
        f"🕓 Beginning request generation at {start_time_all.strftime('%Y-%m-%d %H:%M:%S')}..."
    )

    all_requests = await test_models_batch(
        model_ids_to_test, df, orchestrator, mode=args.trait_mode
    )

    elapsed_gen = (datetime.now() - start_time_all).total_seconds()
    logger.info(
        f"📦 Total collected {len(all_requests)} requests in {elapsed_gen:.2f}s."
    )

    if not all_requests:
        logger.warning("No requests were generated. Exiting.")
        return

    # --- Group requests by (provider, strategy, model) ---
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in all_requests:
        provider = r.get("_provider")
        strategy = r.get("_strategy")  # e.g., "single" or "combined"
        model_id = r.get("_model_id")  # e.g., "GPT5Nano"

        if not all([provider, strategy, model_id]):
            logger.warning(
                f"Skipping request with missing metadata: {r.get('custom_id')}"
            )
            continue
        key = (provider, strategy, model_id)
        groups.setdefault(key, []).append(r)

    logger.info(f"📊 Found {len(groups)} unique (provider, strategy, model) groups.")

    # --- Write files and optionally submit ---

    # Helper to drop internal metadata keys
    def _clean(reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in reqs]

    base_path = Path(settings["paths"]["batch_request_file"])
    base_path.parent.mkdir(parents=True, exist_ok=True)
    files_created = []

    # Create a list of tasks to run concurrently (e.g., monitoring)
    monitor_tasks = []

    for (provider, strategy, model_id), reqs in groups.items():
        cleaned_reqs = _clean(reqs)
        safe_model = "".join(c if c.isalnum() else "_" for c in model_id)
        suffix_provider = "openai" if provider == "openai" else "nebius"

        # Generate a timestamp for this specific file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = base_path.with_name(
            f"{base_path.stem}_{suffix_provider}_{safe_model}_{strategy}_{timestamp}{base_path.suffix}"
        )

        batch_service = None
        try:
            if provider == "openai":
                batch_service = orchestrator.openai_batch_service
            elif provider == "nebius":
                batch_service = orchestrator.nebius_batch_service
            else:
                logger.error(f"Unknown provider '{provider}' for group. Skipping.")
                continue

            batch_service.write_jsonl_file(cleaned_reqs, str(output_file))
            logger.info(
                f"💾 Saved {len(cleaned_reqs)} {provider.upper()} {strategy} requests for {model_id} to: {output_file.name}"
            )
            files_created.append(output_file)

        except Exception as e:
            logger.error(f"❌ Failed to write file {output_file.name}: {e}")
            continue

        # --- Submission Logic (from main.py) ---
        if not args.submit_batch:
            continue

        if not (args.trait_mode == strategy or args.trait_mode == "both"):
            logger.info(
                f"    (Skipping submission for {strategy} mode as --trait-mode={args.trait_mode})"
            )
            continue

        logger.info(f"📤 Submitting {output_file.name} for {model_id}...")
        try:
            batch_id = batch_service.submit_batch(str(output_file), args.batch_name)
            if batch_id and args.monitor:
                logger.info(f"    Submitted batch ID: {batch_id}. Monitoring...")
                # ✅ Schedule monitoring as a concurrent task
                task = asyncio.create_task(
                    monitor_batch_progress(
                        batch_id,
                        batch_service,
                        args.check_interval,
                        model_id=model_id,
                        provider=provider,
                        strategy=strategy,
                    )
                )
                monitor_tasks.append(task)
            elif batch_id:
                logger.info(f"    Submitted batch ID: {batch_id}. (Not monitoring)")
            else:
                logger.error(f"    Batch submission failed for {output_file.name}.")

        except Exception as e:
            logger.error(f"    ❌ Submission failed for {output_file.name}: {e}")

    # --- Wait for all monitoring tasks to complete ---
    if monitor_tasks:
        logger.info(f"Waiting for {len(monitor_tasks)} monitoring task(s) to complete...")
        await asyncio.gather(*monitor_tasks)
        logger.info("All monitoring tasks finished.")

    # --- Summary after all models ---
    end_time_all = datetime.now()
    total_duration_minutes = (end_time_all - start_time_all).total_seconds() / 60
    logger.info("=" * 50)
    logger.info(f"🎉 All models processed in {total_duration_minutes:.2f} minutes.")
    logger.info(f"📦 Generated {len(files_created)} batch file(s):")
    if files_created:
        for f_path in files_created:
            logger.info(f"   -> {f_path.name}")
    else:
        logger.info("   (No files were generated)")
    logger.info("\n✨ Done!")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and submit batch request files for ALL configured models."
    )

    # --- Args from run_all_models.py ---
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=1,  # Reduced default for safety
        help="Number of questions to process (0 for all). Default: 10",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )

    # --- Args from main.py ---
    parser.add_argument(
        "-tm",
        "--trait-mode",
        choices=["single", "combined", "both"],
        default="both",
        help="How to build batch requests: per-trait, combined, or both. Default: both.",
    )
    parser.add_argument(
        "--batch-name", type=str, help="Optional custom name for batch identification"
    )
    parser.add_argument(
        "--submit-batch",
        action="store_true",
        help="Actually submit the generated batch files to the API.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Automatically monitor batch after submission (requires --submit-batch).",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Batch monitoring interval in seconds (default: 300).",
    )

    args = parser.parse_args()

    # --- Set logging level based on argument ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level_numeric)
    logging.getLogger().setLevel(log_level_numeric)  # Set root logger level
    logger.info(f"Logging level set to {args.log_level.upper()}")

    if args.monitor and not args.submit_batch:
        logger.warning("--monitor flag ignored as --submit-batch was not specified.")

    try:
        asyncio.run(run_all_models(args))
    except (RuntimeError, FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"💥 An unhandled error occurred: {e}", exc_info=True)
        import sys

        sys.exit(1)