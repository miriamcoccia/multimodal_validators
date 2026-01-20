#!/usr/bin/env python3
"""
Walks a directory tree of JSONL batch results, normalizes them, and writes
raw/clean CSVs beside each input file (in a mirrored structure).

Keeps the original behavior and folder conventions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import PROJECT_ROOT
from src.results_handler import ResultsHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("processor")


def process_all() -> None:
    # Paths retained to match original behavior
    input_root = PROJECT_ROOT / "data" / "batch_results_scrambled_FINAL"
    output_root = PROJECT_ROOT / "data" / "scrambled_processed_FINAL"

    if not input_root.exists():
        logger.error("❌ Folder not found: %s", input_root)
        return

    jsonl_files = list(input_root.rglob("*.jsonl"))
    logger.info("🚀 Found %d parts to process.", len(jsonl_files))

    for i, jsonl_path in enumerate(jsonl_files, start=1):
        try:
            # Mirror folder structure and create a leaf folder per file
            relative_path = jsonl_path.relative_to(input_root).parent
            target_dir = output_root / relative_path / jsonl_path.stem

            logger.info("[%d/%d] Processing %s...", i, len(jsonl_files), jsonl_path.name)
            rh = ResultsHandler(target_dir)

            raw_df = rh.load_batch_results_jsonl(jsonl_path)
            if raw_df.empty:
                continue

            rh.save_raw_results(raw_df)
            clean_df = rh.clean_results(raw_df)
            if not clean_df.empty:
                rh.save_clean_results(clean_df)

        except Exception as exc:
            logger.error("❌ Failed %s: %s", jsonl_path.name, exc)


if __name__ == "__main__":
    process_all()
