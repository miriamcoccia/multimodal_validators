"""
Script to normalize batch JSONL results to raw and clean CSV files.
Uses ResultsHandler to parse one or more JSONL formats.
"""
from pathlib import Path
import argparse
import sys
import pandas as pd
from src.results_handler import ResultsHandler
from src.config import settings


def main():
    """Main execution function."""
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
    jsonl_path: Path
    if args.jsonl:
        jsonl_path = Path(args.jsonl).expanduser().resolve()
    else:
        # Derive from config batch_request_file
        base = Path(settings["paths"]["batch_request_file"]).expanduser().resolve()
        jsonl_path = base.with_suffix(".results.jsonl")

    # Resolve output dir
    outdir: Path = (
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

    if raw_df.empty:
        print("⚠️ No rows were parsed. Exiting.")
        sys.exit(0)

    rh.save_raw_results(raw_df)

    try:
        clean_df = rh.clean_results(raw_df)
    except Exception as e:
        print(f"⚠️ Cleaning failed: {e}", file=sys.stderr)
        # Show a small sample of columns to debug
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("🔎 Raw sample:\n", raw_df.head(5))
        sys.exit(2)

    if clean_df.empty:
        print("⚠️ Cleaning resulted in an empty DataFrame. Check raw results.")
        sys.exit(0)
        
    rh.save_clean_results(clean_df)

    print(f"✅ Clean CSV: {rh.CLEAN_RESULTS_FILENAME.resolve()}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n--- Clean Results Sample ---")
        print(clean_df.head(10).to_string(index=False))
        print("----------------------------")


if __name__ == "__main__":
    main()