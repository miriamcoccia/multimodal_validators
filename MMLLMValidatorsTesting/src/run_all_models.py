#!/usr/bin/env python3
"""
Prepare and (optionally) submit batch requests for multiple models.

- Loads model IDs from settings.
- Loads dataset CSV (config or CLI override).
- Prepares single/combined trait requests via Orchestrator.
- Splits into provider/model/strategy groups and writes JSONL files.
- Optionally submits batches and logs their IDs to a text file.

Behavior preserved; adds clearer logging and small safety checks.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.img_traits_def import ImgTraitDefinition
from src.orchestrator import Orchestrator

# ---------- LOGGING SETUP ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all_models")

# ---------- CONSTANTS ----------
# OpenAI limit is 50K per file; we stick with a conservative split.
MAX_REQUESTS_PER_BATCH = 400
BATCH_LOG_FILE = PROJECT_ROOT / "data" / "submitted_batches_log.txt"


def log_batch_to_file(provider: str, part_info: str, filename: str, batch_id: str) -> None:
    """Append batch submission details to a persistent text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {provider.upper()} | {part_info} | {filename} | {batch_id}\n"

    try:
        BATCH_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with BATCH_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as exc:
        logger.error("Failed to write to log file: %s", exc)


def get_provider_from_model_id(model_id: str) -> str:
    """
    Infer provider from model id string.
    - "GPT..." -> "openai"
    - "L_..."  -> "nebius"
    """
    up = model_id.upper()
    if up.startswith("GPT"):
        return "openai"
    if up.startswith("L_"):
        return "nebius"
    raise ValueError(f"Could not determine provider for model {model_id}")


async def test_models_batch(
    model_ids: List[str],
    data: pd.DataFrame,
    orchestrator: Orchestrator,
    mode: str,
) -> List[Dict[str, Any]]:
    """
    Generate batchable request payloads for the given models across the dataset.
    Returns a flat list of request dicts (possibly many per question).
    """
    all_requests: List[Dict[str, Any]] = []
    total_questions = len(data)

    for model_id in model_ids:
        try:
            provider = get_provider_from_model_id(model_id)
        except ValueError as e:
            logger.error("❌ %s. Skipping model %s.", e, model_id)
            continue

        logger.info("📋 Collecting requests for %s (%s, Mode: %s)...", model_id, provider, mode)

        for i, (_, row) in enumerate(data.iterrows(), start=1):
            # ScienceQA is expected to provide a static constructor from a DataFrame row
            from src.science_qa import ScienceQA  # kept close to use site

            question = ScienceQA.from_df_row(row)
            try:
                if mode in ("single", "both"):
                    single_reqs, _ = await orchestrator.prepare_batch_requests(question, provider, model_id)
                    all_requests.extend(single_reqs)

                if mode in ("combined", "both"):
                    combined_reqs, _ = await orchestrator.prepare_combined_batch_requests(
                        question, provider, model_id
                    )
                    all_requests.extend(combined_reqs)

            except Exception as exc:
                qid = row.get("question_id", "Unknown")
                logger.error("    Failed QID %s: %s", qid, exc)

            if i % 10 == 0 or i == total_questions:
                print(f"    Processed {i}/{total_questions}", end="\r")

    print("")  # newline after progress
    return all_requests


def _group_requests(requests: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """
    Group requests by (provider, strategy, model_id).
    Only groups requests that have all three keys present.
    """
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in requests:
        key = (r.get("_provider"), r.get("_strategy"), r.get("_model_id"))
        if None in key:
            continue
        groups.setdefault(key, []).append(r)
    return groups


async def run_all_models(args: argparse.Namespace) -> None:
    logger.info("🚀 Starting batch run (Mode: %s)", args.trait_mode)

    # 1) Load models
    all_models_dict = settings.get("all_models", {})
    if not all_models_dict:
        raise RuntimeError("No models found in settings['all_models']!")

    # Optional filter from CLI
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        model_ids = [m for m in requested if m in all_models_dict]
        missing = set(requested) - set(model_ids)
        if missing:
            logger.warning("⚠️ Some requested models not found in config: %s", ", ".join(sorted(missing)))
    else:
        model_ids = list(all_models_dict.keys())

    logger.info("🧠 Models: %s", ", ".join(model_ids))

    # 2) Load dataset
    csv_path = Path(args.dataset).resolve() if args.dataset else PROJECT_ROOT / settings["paths"]["input_data_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    logger.info("📚 Loading dataset: %s", csv_path.name)
    df = pd.read_csv(csv_path)
    if args.num_questions > 0:
        df = df.head(args.num_questions)
        logger.info("   Using first %d row(s).", len(df))

    # 3) Initialize orchestrator with trait definitions
    trait_def_path = PROJECT_ROOT / settings["paths"]["trait_definitions_json"]
    trait_def = ImgTraitDefinition(trait_def_path)
    trait_list = list(trait_def.traits.keys())
    orchestrator = Orchestrator(trait_names=trait_list)
    logger.info("✅ Orchestrator ready.")

    # 4) Generate requests
    start_time = datetime.now()
    all_requests = await test_models_batch(model_ids, df, orchestrator, args.trait_mode)
    if not all_requests:
        logger.warning("No requests generated. Exiting.")
        return

    # 5) Group by (provider, strategy, model)
    groups = _group_requests(all_requests)

    # 6) Clean internal keys, split, write, (optionally) submit
    base_path = PROJECT_ROOT / settings["paths"]["batch_request_file"]
    base_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in reqs]

    for (provider, strategy, model_id), reqs in groups.items():
        cleaned_reqs = _clean(reqs)
        total = len(cleaned_reqs)

        chunks = [cleaned_reqs[i : i + MAX_REQUESTS_PER_BATCH] for i in range(0, total, MAX_REQUESTS_PER_BATCH)]
        logger.info("📦 Group %s (%s): %d requests -> %d split(s)", model_id, strategy, total, len(chunks))

        for i, chunk in enumerate(chunks, start=1):
            part_suffix = f"_part{i}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = "".join(c if c.isalnum() else "_" for c in model_id)

            filename = f"{base_path.stem}_{provider}_{safe_model}_{strategy}_{timestamp}{part_suffix}{base_path.suffix}"
            output_file = base_path.with_name(filename)

            # Identify service by provider (kept consistent with original orchestrator)
            service = orchestrator.openai_batch_service if provider == "openai" else orchestrator.nebius_batch_service

            # Write file
            try:
                service.write_jsonl_file(chunk, str(output_file))
                logger.info("   💾 Saved: %s (%d reqs)", output_file.name, len(chunk))
            except Exception as exc:
                logger.error("   ❌ Write failed for %s: %s", output_file.name, exc)
                continue

            # Optionally submit
            if args.submit_batch:
                # Respect the trait_mode filter (submit only matching strategy unless 'both')
                if args.trait_mode != "both" and args.trait_mode != strategy:
                    continue

                logger.info("   📤 Submitting %s...", output_file.name)
                try:
                    batch_id = service.submit_batch(str(output_file), args.batch_name)
                    if batch_id:
                        part_info = f"Part_{i}"
                        log_batch_to_file(provider, part_info, output_file.name, batch_id)
                        logger.info("      ✅ Batch ID: %s (logged)", batch_id)
                    else:
                        logger.error("      ❌ API returned no Batch ID.")
                except Exception as exc:
                    logger.error("      ❌ Submission failed: %s", exc)

    elapsed_min = (datetime.now() - start_time).total_seconds() / 60.0
    logger.info("✨ Done! Total time: %.2f min", elapsed_min)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and optionally submit batch requests for multiple models.")
    # Dataset & Limit
    parser.add_argument("-n", "--num-questions", type=int, default=0, help="0 for all")
    parser.add_argument("--dataset", type=str, help="Override input CSV path")
    # Modes
    parser.add_argument(
        "-tm",
        "--trait-mode",
        choices=["single", "combined", "both"],
        default="both",
        help="Generate requests for single traits, combined traits, or both.",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        help="Comma-separated list of models to run (e.g. GPT4oMini,L_Gemma327B)",
    )
    # Batch Actions
    parser.add_argument("--submit-batch", action="store_true", help="Submit generated files")
    parser.add_argument("--batch-name", type=str, help="Custom batch description")
    # Logging
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)")
    # Compatibility placeholders with original runner (not implemented here)
    parser.add_argument("--check-batch", type=str, help="Check status of batch by ID")
    parser.add_argument("--download-batch", type=str, help="Download results")
    parser.add_argument("--monitor", action="store_true", help="Monitor (not implemented in this simplified runner)")
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    logger.setLevel(args.log_level.upper())
    asyncio.run(run_all_models(args))
