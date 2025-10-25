#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import pandas as pd
from src.results_handler import ResultsHandler
from src.config import settings


def main():
    ap = argparse.ArgumentParser(description="Normalize batch JSONL to CSVs")
    ap.add_argument(
        "--jsonl",
        nargs="?",
        default=None,
        help="Path to batch results .jsonl (if omitted, use config + .results.jsonl)",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="Output directory for CSVs (default: settings['paths']['results_dir'])",
    )
    args = ap.parse_args()

    # Resolve JSONL input path
    if args.jsonl:
        jsonl_path = Path(args.jsonl).expanduser().resolve()
    else:
        # Derive from config batch_request_file
        base = Path(settings["paths"]["batch_request_file"]).expanduser().resolve()
        jsonl_path = base.with_suffix(".results.jsonl")

    # Resolve output dir
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else Path(settings["paths"]["results_dir"]).expanduser().resolve()
    )

    print(f"📥 Input JSONL : {jsonl_path}")
    print(f"📤 Output dir  : {outdir}")

    if not jsonl_path.exists():
        print(f"❌ Results JSONL not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    rh = ResultsHandler(outdir)
    raw_df = rh.load_batch_results_jsonl(jsonl_path)
    print(f"🧾 Parsed rows : {len(raw_df)}")

    # Save raw now so you always get a file even if cleaning fails
    rh.save_raw_results(raw_df)

    # Try cleaning; if it fails, keep raw and print helpful info
    try:
        clean_df = rh.clean_results(raw_df)
    except Exception as e:
        print(f"⚠️ Cleaning failed: {e}", file=sys.stderr)
        # Show a small sample of columns to debug
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("🔎 Raw sample:\n", raw_df.head(5))
        sys.exit(2)

    rh.save_clean_results(clean_df)

    print(f"✅ Clean CSV: {rh.CLEAN_RESULTS_FILENAME.resolve()}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(clean_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
