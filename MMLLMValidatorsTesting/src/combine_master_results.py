import logging
import pandas as pd
from pathlib import Path
from src.config import PROJECT_ROOT

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("combiner")

def combine_results():
    # 1. Path Setup
    processed_root = PROJECT_ROOT / "data" / "normal_processed_FINAL"
    master_root = PROJECT_ROOT / "data" / "normal_master_results_FINAL"
    
    if not processed_root.exists():
        logger.error(f"❌ Folder not found: {processed_root}")
        return

    # 2. Find all processed parts
    all_clean_csvs = list(processed_root.rglob("clean_evaluation_results.csv"))
    
    if not all_clean_csvs:
        logger.warning("⚠️ No processed CSVs found. Run process_all_results first.")
        return

    # 3. Group by Model and Strategy
    # Structure: data/normal_processed/{provider}/{model}/{strategy}/{part_folder}/file
    groups = {}
    for csv_path in all_clean_csvs:
        # Get components relative to the root
        rel = csv_path.relative_to(processed_root)
        parts = rel.parts
        
        if len(parts) < 4:
            continue
            
        provider, model, strategy = parts[0], parts[1], parts[2]
        key = (provider, model, strategy)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(csv_path)

    logger.info(f"🚀 Found {len(groups)} Model/Strategy combinations.")

    # 4. Merge and Save
    for (provider, model, strategy), paths in groups.items():
        logger.info(f"📦 Merging {len(paths)} parts for {model} ({strategy})...")
        
        try:
            # Combine all parts into one table
            dfs = [pd.read_csv(p) for p in paths]
            master_df = pd.concat(dfs, ignore_index=True)
            
            # Save to Master Results folder
            master_root.mkdir(parents=True, exist_ok=True)
            filename = f"{provider}_{model}_{strategy}_MASTER.csv"
            output_path = master_root / filename
            
            master_df.to_csv(output_path, index=False)
            logger.info(f"   ✅ Saved {len(master_df)} rows to: {output_path.name}")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to combine {model} {strategy}: {e}")

    logger.info(f"\n✨ All master files generated in: {master_root}")

if __name__ == "__main__":
    combine_results()