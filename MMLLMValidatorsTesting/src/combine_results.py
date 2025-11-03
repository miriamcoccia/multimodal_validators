#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Combine multiple clean_results CSVs.")
    parser.add_argument(
        "-i",
        "--input-dirs",
        nargs="+",  # Accept one or more directories
        required=True,
        help="List of input directories (e.g., data/results/gemma327)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path for the final combined CSV file.",
    )
    parser.add_argument(
        "-f",
        "--filename",
        default="clean_evaluation_results.csv",
        help="The name of the CSV file to find in each directory.",
    )
    args = parser.parse_args()

    all_dfs = []
    input_dirs = [Path(d).expanduser().resolve() for d in args.input_dirs]
    output_file = Path(args.output_file).expanduser().resolve()

    print(f"Combining CSVs into {output_file}...")

    for directory in input_dirs:
        csv_path = directory / args.filename
        if csv_path.exists():
            print(f"  -> Loading {csv_path.relative_to(Path.cwd())}")
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
        else:
            print(f"⚠️ Warning: File not found, skipping: {csv_path}")

    if not all_dfs:
        print("❌ Error: No CSV files were found to combine.", file=sys.stderr)
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    combined_df.to_csv(output_file, index=False)
    print(f"\n✅ Success! Combined {len(all_dfs)} files ({len(combined_df)} rows) into:")
    print(f"   {output_file}")


if __name__ == "__main__":
    main()