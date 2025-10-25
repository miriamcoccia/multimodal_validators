#!/usr/bin/env python3
import asyncio
import pandas as pd
from pathlib import Path
import argparse
import logging
import time
import json
from typing import Optional, List, Dict, Any

from src.science_qa import ScienceQA
from src.orchestrator import Orchestrator
from src.llm_service.providers.basic_batch import BaseBatchService
from src.llm_service.service import (
    OpenAIBatchService,
    NebiusBatchService,
)
from src.config import settings
from src.img_traits_def import ImgTraitDefinition
from src_orig.orig_traits_def import OriginalTraitDefinition

# ---------- module-level logger to avoid NameError in helpers ----------
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(getattr(logging, level.upper()))


def get_provider_from_model_id(model_id: str) -> str:
    """Determines the provider based on the model ID prefix"""
    if model_id.upper().startswith("GPT"):
        return "openai"
    elif model_id.upper().startswith("L"):
        return "nebius"
    raise ValueError(f"Could not determine provider for model {model_id}")


async def test_models_batch(
    model_ids: List[str],
    data: pd.DataFrame,
    orchestrator: Orchestrator,
) -> List[Dict[str, Any]]:
    """
    Collect all batch requests across models and questions.
    """
    all_requests = []
    total_questions = len(data)

    for model_id in model_ids:
        provider = get_provider_from_model_id(model_id=model_id)
        print(f"\n📋 Collecting requests for {model_id} (Provider: {provider})...")

        for i, (_, row) in enumerate(data.iterrows()):
            question = ScienceQA.from_df_row(row)

            requests, _ = await orchestrator.prepare_batch_requests(
                question, provider, model_id
            )

            for request in requests:
                request["_provider"] = provider

            all_requests.extend(requests)

            if (i + 1) % 5 == 0:
                print(f"  📝 Processed {i + 1}/{total_questions} questions")

    return all_requests


async def monitor_batch_progress(
    batch_id: str, batch_service: BaseBatchService, check_interval: int = 300
):
    """Monitor batch progress until completion"""
    _log = logging.getLogger(__name__)
    _log.info(f"👀 Monitoring batch {batch_id}...")

    while True:
        try:
            status = batch_service.get_batch_status(batch_id)
            batch_status = status.get("status", "unknown")

            if batch_status == "completed":
                _log.info(f"🎉 Batch {batch_id} completed!")

                # Check for errr file
                error_file_id = status.get("error_file_id")
                if error_file_id:
                    _log.warning("⚠️ Batch has errors, checking error file...")
                    try:
                        # TODO: solve this "client" problem
                        error_content = batch_service.client.files.content(
                            error_file_id
                        )
                        text_preview = getattr(error_content, "text", None)
                        if text_preview:
                            _log.error(f"First error: {text_preview[:500]}...")
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


async def main(args):
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # --- Ad-hoc Batch Commands ---
    # These commands operate on a single provider, specified by the --provider flag.
    if (
        args.list_batches
        or args.cancel_batch
        or args.delete_batch
        or args.check_batch
        or args.download_batch
    ):
        batch_service: Optional[BaseBatchService] = None
        if args.provider == "openai":
            batch_service = OpenAIBatchService()
        elif args.provider == "nebius":
            # You might need to add a check here if Nebius doesn't support these commands
            batch_service = NebiusBatchService()

        if not batch_service:
            logger.error(
                f"Provider '{args.provider}' is not supported for ad-hoc commands."
            )
            return

        print(f"--- Running command for provider: {args.provider} ---")

        if args.list_batches:
            logger.info("Listing batches...")
            try:
                batches = batch_service.client.batches.list(limit=20)

                if not batches.data:
                    print("\n📋 No batches found.")
                else:
                    print(f"\n📋 Found {len(batches.data)} batches:\n")

                    for batch in batches.data:
                        print(f"ID: {batch.id}")
                        print(f"  Status: {batch.status}")
                        print(f"  Created: {batch.created_at}")

                        # Show progress if available
                        counts = batch.request_counts
                        print(
                            f"  Requests: {counts.total} total, {counts.completed} completed, {counts.failed} failed"
                        )
                        print("---")

            except Exception as e:
                logger.error(f"Failed to list batches: {e}")
                print(f"\n❌ Error: {e}")

            return

        elif args.cancel_batch:
            success = batch_service.cancel_batch(args.cancel_batch)
            print(f"Cancellation status: {'Success' if success else 'Failed'}")

        elif args.delete_batch:
            success = batch_service.delete_batch(args.delete_batch)
            print(f"Deletion status: {'Success' if success else 'Failed'}")

        elif args.check_batch:
            status_info = batch_service.get_batch_status(args.check_batch)

            print(f"\n📊 Batch Status:")
            print(f"  ID: {status_info.get('id')}")
            print(f"  Status: {status_info.get('status')}")
            print(f"  Created: {status_info.get('created_at')}")
            print(f"  Completed: {status_info.get('completed_at')}")
            print(f"  Failed: {status_info.get('failed_at')}")

            counts = status_info.get("request_counts")
            if counts:
                print(f"\n  Requests:")
                print(f"    Total: {counts.total}")
                print(f"    Completed: {counts.completed}")
                print(f"    Failed: {counts.failed}")

            print(f"\n  Output File: {status_info.get('output_file_id')}")
            print(f"  Error File: {status_info.get('error_file_id')}")

            if status_info.get("error_details"):
                print(f"\n  ⚠️ Error Details:")
                print(f"    {status_info['error_details'][:500]}...")

        elif args.download_batch:
            output_path = Path(f"./{args.download_batch}_results.jsonl")
            success = batch_service.download_batch_results(
                args.download_batch, str(output_path)
            )
            if success:
                print(f"📥 Results downloaded to: {output_path}")

        return  # Exit after running the ad-hoc command

    # --- Main Evaluation Logic ---

    # Load Configuration & Data (no changes here)
    trait_def = ImgTraitDefinition(Path(settings["paths"]["trait_definitions_json"]))
    trait_list = list(trait_def.traits.keys())
    orig_trait_def = OriginalTraitDefinition(
        Path(settings["paths"]["orig_trait_definitions_json"])
    )
    orig_trait_list = list(orig_trait_def.traits.keys())
    test_df = pd.read_csv(Path(settings["paths"]["input_data_csv"])).head(
        args.num_questions
    )
    model_ids_to_test = args.models.split(",") if args.models else ["GPT4o"]

    # Create the Orchestrator once. It now manages all its own internal services.
    orchestrator = Orchestrator(
        trait_names=trait_list,
        checkpoint_file=settings["paths"]["checkpoint_json"],
    )

    if args.batch:
        print("\n🚀 BATCH MODE: Collecting requests...")

        # The test_models_batch function is now much simpler
        all_requests = await test_models_batch(model_ids_to_test, test_df, orchestrator)

        openai_requests = []
        nebius_requests = []

        for request in all_requests:
            provider = request.pop("_provider", "openai")
            if provider == "openai":
                openai_requests.append(request)
            elif provider == "nebius":
                nebius_requests.append(request)

        if openai_requests:
            openai_file = Path(settings["paths"]["batch_request_file"])
            orchestrator.openai_batch_service.write_jsonl_file(
                openai_requests, str(openai_file)
            )
            print(f"💾 Saved {len(openai_requests)} OpenAI requests to: {openai_file}")

            if args.submit_batch:
                batch_id = orchestrator.openai_batch_service.submit_batch(
                    str(openai_file), args.batch_name
                )
                if batch_id and args.monitor:
                    await monitor_batch_progress(
                        batch_id, orchestrator.openai_batch_service, args.check_interval
                    )
        if nebius_requests:
            # Creating Nebius filename by adding _nebius suffix
            base_path = Path(settings["paths"]["batch_request_file"])
            nebius_file = base_path.with_name(
                f"{base_path.stem}_nebius{base_path.suffix}"
            )

            orchestrator.nebius_batch_service.write_jsonl_file(
                nebius_requests, str(nebius_file)
            )
            print(f"💾 Saved {len(nebius_requests)} Nebius requests to: {nebius_file}")

            if args.submit_batch:
                batch_id = orchestrator.nebius_batch_service.submit_batch(
                    str(nebius_file), args.batch_name
                )
                if batch_id and args.monitor:
                    await monitor_batch_progress(
                        batch_id, orchestrator.nebius_batch_service, args.check_interval
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agentic Framework Evaluation.")

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO).",
    )

    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="nebius",
        choices=["nebius", "openai"],
        help="Choose the provider: either 'nebius' or 'openai' (default: 'nebius'). This will influence the models available.",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        help='Comma-separated list of model IDs to test (e.g., "GPT4o,L_Gemma34B").',
    )

    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=2,
        help="Number of questions to process from the test set.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Path to the output directory for results (overrides config.toml).",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use OpenAI batch processing (collect requests, submit batch, exit)",
    )

    parser.add_argument(
        "--batch-name", type=str, help="Optional custom name for batch identification"
    )

    parser.add_argument(
        "--submit-batch",
        action="store_true",
        help="Actually submit the batch to OpenAI (default: just generate JSONL)",
    )

    parser.add_argument("--check-batch", type=str, help="Check status of batch by ID")

    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output for scripting"
    )

    parser.add_argument(
        "--download-batch", type=str, help="Download results for completed batch by ID"
    )

    parser.add_argument(
        "--list-batches", action="store_true", help="List all batches and their status"
    )

    parser.add_argument("--cancel-batch", type=str, help="Cancel a batch by ID")

    parser.add_argument(
        "--delete-batch",
        type=str,
        help="Delete a batch by ID (only works for completed/failed batches)",
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Automatically monitor batch after submission",
    )

    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Batch monitoring interval in seconds (default: 300)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
