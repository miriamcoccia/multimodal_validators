#!/usr/bin/env python3
"""
Resubmission tool for failed batches.

- Scans the batch request folder for the latest JSONL matching model/strategy patterns.
- Re-submits batches via OpenAI's batch service.
- Monitors submitted batches and downloads results to the standard project structure.

Behavior preserved from the original script; logging and structure improved.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from src.config import PROJECT_ROOT, settings
from src.llm_service.providers.basic_batch import BaseBatchService
from src.llm_service.service import OpenAIBatchService

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resubmit_tool")

# ---------- Config ----------
CHECK_INTERVAL_SECONDS = 60  # default monitor interval

# Targets are fixed to match original behavior (OpenAI provider).
@dataclass(frozen=True)
class FailedTarget:
    model: str
    strategy: str  # "single" or "combined"

FAILED_TARGETS: List[FailedTarget] = [
    FailedTarget("GPT5Mini", "single"),
    FailedTarget("GPT5Nano", "single"),
    FailedTarget("GPT5Nano", "combined"),
    FailedTarget("GPT4oMini", "single"),
    FailedTarget("GPT4oMini", "combined"),
]


# ---------- Monitoring ----------
async def monitor_batch_progress(
    batch_id: str,
    batch_service: BaseBatchService,
    check_interval: int,
    model_id: str,
    provider: str,
    strategy: str,
) -> bool:
    """
    Poll batch status until completion or failure. On success, download results.

    Returns:
        True if batch completed and (attempted) download was initiated; False if failed.
    """
    logger.info("👀 Monitoring batch %s for %s (%s)...", batch_id, model_id, strategy)

    while True:
        try:
            status = batch_service.get_batch_status(batch_id)
            batch_status = status.get("status", "unknown")

            if batch_status == "completed":
                logger.info("🎉 Batch %s completed!", batch_id)

                output_file_id = status.get("output_file_id")
                if not output_file_id:
                    logger.error("❌ Batch %s has no output_file_id.", batch_id)
                    return False

                # Construct standard output path: data/batch_results/<provider>/<model>/<strategy>
                safe_model = "".join(c if c.isalnum() else "_" for c in model_id)
                output_dir = PROJECT_ROOT / "data" / "batch_results" / provider / safe_model / strategy
                output_dir.mkdir(parents=True, exist_ok=True)

                output_filename = f"results_{batch_id}_{output_file_id}.jsonl"
                output_path = output_dir / output_filename

                logger.info("📥 Downloading to %s...", output_path)

                # Run blocking download off the event loop
                success = await asyncio.to_thread(batch_service.download_batch_results, batch_id, str(output_path))
                if success:
                    logger.info("✅ Saved: %s", output_path.name)
                else:
                    logger.warning("⚠️ Download failed for %s.", batch_id)
                return True

            if batch_status == "failed":
                logger.error("❌ Batch %s failed!", batch_id)
                if status.get("errors"):
                    logger.error("Reason: %s", status.get("errors"))
                return False

            if batch_status == "in_progress":
                counts = status.get("request_counts", {})
                logger.debug("⏳ %s: %s", model_id, counts)

            await asyncio.sleep(check_interval)

        except Exception as exc:
            logger.error("Error checking batch %s: %s", batch_id, exc)
            await asyncio.sleep(check_interval)


# ---------- Helpers ----------
def _find_latest_file(paths: Iterable[Path]) -> Optional[Path]:
    """Return the most recently modified file from an iterable of paths (or None)."""
    latest: Optional[Path] = None
    latest_mtime: float = -1.0
    for p in paths:
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest = p
    return latest


def _scan_for_matching_files(batch_dir: Path, model: str, strategy: str) -> List[Path]:
    """
    Find files matching the naming convention used by run_all_models:
      *openai_{model}_{strategy}*.jsonl
    """
    pattern = f"*openai_{model}_{strategy}*.jsonl"
    return list(batch_dir.glob(pattern))


# ---------- Main Resubmission Logic ----------
async def main() -> None:
    service = OpenAIBatchService()
    batch_dir = Path(settings["paths"]["batch_request_file"]).parent

    logger.info("📂 Scanning %s for failed files...", batch_dir)

    monitor_tasks: List[asyncio.Task] = []

    for target in FAILED_TARGETS:
        model = target.model
        strategy = target.strategy

        matches = _scan_for_matching_files(batch_dir, model, strategy)
        if not matches:
            logger.warning("⚠️  No file found for %s %s", model, strategy)
            continue

        latest_file = _find_latest_file(matches)
        if latest_file is None:
            logger.warning("⚠️  No valid file found for %s %s", model, strategy)
            continue

        logger.info("🚀 Resubmitting: %s", latest_file.name)

        try:
            # Run blocking submit in a thread
            batch_id = await asyncio.to_thread(service.submit_batch, str(latest_file), f"Resubmit_{model}_{strategy}")

            if batch_id:
                logger.info("   ✅ Submitted %s (%s) -> ID: %s", model, strategy, batch_id)
                task = asyncio.create_task(
                    monitor_batch_progress(
                        batch_id=batch_id,
                        batch_service=service,
                        check_interval=CHECK_INTERVAL_SECONDS,  # 1 minute
                        model_id=model,
                        provider="openai",
                        strategy=strategy,
                    )
                )
                monitor_tasks.append(task)
            else:
                logger.error("   ❌ Failed to get Batch ID for %s", latest_file.name)

        except Exception as exc:
            logger.error("   ❌ Exception submitting %s: %s", latest_file.name, exc)

    if monitor_tasks:
        logger.info("\n👀 Monitoring %d batch(es) in parallel. Do not close this terminal.\n", len(monitor_tasks))
        await asyncio.gather(*monitor_tasks)
    else:
        logger.info("No batches were submitted.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Process stopped by user.")
