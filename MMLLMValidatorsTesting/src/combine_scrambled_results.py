import logging
import pandas as pd
from pathlib import Path
from src.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("scrambled_combiner")

def combine_results():
    # LOCKED PATHS FOR SCRAMBLED DATA
    processed_root = PROJECT_ROOT / "data" / "scrambled_processed_FINAL"
    master_root = PROJECT_ROOT / "data" / "scrambled_master_results_FINAL"
    
    if not processed_root.exists():
        logger.error(f"❌ Folder not found: {processed_root}")
        return

    all_clean_csvs = list(processed_root.rglob("clean_evaluation_results.csv"))
    if not all_clean_csvs:
        logger.warning("⚠️ No processed CSVs found.")
        return

    groups = {}
    for csv_path in all_clean_csvs:
        rel = csv_path.relative_to(processed_root)
        parts = rel.parts
        if len(parts) < 4: continue
        provider, model, strategy = parts[0], parts[1], parts[2]
        key = (provider, model, strategy)
        if key not in groups: groups[key] = []
        groups[key].append(csv_path)

    for (provider, model, strategy), paths in groups.items():
        try:
            dfs = [pd.read_csv(p) for p in paths]
            master_df = pd.concat(dfs, ignore_index=True)
            master_root.mkdir(parents=True, exist_ok=True)
            filename = f"{provider}_{model}_{strategy}_MASTER.csv"
            master_df.to_csv(master_root / filename, index=False)
            logger.info(f"✅ Saved Scrambled Master: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed: {e}")

if __name__ == "__main__":
    combine_results()