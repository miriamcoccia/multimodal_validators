#!/usr/bin/env python3
"""
Statistics & Plots for Batch Evaluation Results

Revisions:
- Supports merging multiple CSVs (Single vs Combined) for side-by-side comparison.
- Auto-tags 'Strategy' based on file path keywords ('single', 'combined').
- Generates aggregate summary and comparison boxplots.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Handle imports if run as script vs module
try:
    from src.config import settings
except ImportError:
    # Fallback if run directly from src/
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import settings


class Statistics:
    """Handles statistical analysis, plots, and timing for the results."""

    def __init__(
        self,
        results_df: pd.DataFrame,
        results_dir: Path,
        ground_truth_col: Optional[str] = None,
    ):
        self.df = results_df.copy()
        self.RESULTS_DIR = results_dir
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Filenames
        self.SUMMARY_FILENAME = results_dir / "evaluation_summary.csv"
        self.PLOT_FILENAME = results_dir / "accuracy_comparison_boxplot.png"
        self.TIMEINFO_FILENAME = results_dir / "time_range.txt"

        # --- Validation ---
        if "validity" not in self.df.columns:
            raise ValueError("Input DataFrame must contain a 'validity' column.")
        self.df["validity"] = self.df["validity"].astype(bool)

        # --- Model ID Standardization ---
        # We extract the base model name, then append Strategy if available
        if "custom_id" in self.df.columns:
            print("Standardizing model_id from custom_id...")
            # Extract base alias (e.g. "L_Gemma327B")
            self.df["base_model"] = self.df["custom_id"].str.extract(
                r"^request-([^-]+)-", expand=False
            )
            
            # Standardize friendly names
            replacements = {
                "L_Gemma327B": "Gemma-3-27B",
                "L_Qwen25VL72B": "Qwen2.5-VL-72B",
                "GPT4oMini": "GPT-4o-Mini",
                "GPT4o": "GPT-4o"
            }
            self.df["base_model"] = self.df["base_model"].replace(replacements)

            # Create Final Model ID: "Model Name (Strategy)"
            if "strategy" in self.df.columns:
                self.df["model_id"] = self.df["base_model"] + " (" + self.df["strategy"] + ")"
            else:
                self.df["model_id"] = self.df["base_model"]
                
            print("Unique Models identified:", self.df["model_id"].unique())
        else:
            print("Warning: 'custom_id' missing, using existing 'model_id'.")

        # --- Ground Truth Logic ---
        self.ground_truth_col = ground_truth_col or next(
            (c for c in ["ground_truth", "label", "expected_validity", "gold_validity"] 
             if c in self.df.columns), None
        )

        if self.ground_truth_col is None:
            self.ground_truth_col = "_assumed_ground_truth_true"
            self.df[self.ground_truth_col] = True
        else:
            self.df[self.ground_truth_col] = self.df[self.ground_truth_col].astype(bool)

        self.traits = sorted(list(set(self.df["trait"].dropna().unique())))
        self._ensure_datetime_column()

    # ---------------- Helpers ----------------
    def _ensure_datetime_column(self) -> None:
        if "timestamp_token" in self.df.columns:
            self.df["timestamp_token_dt"] = pd.to_datetime(
                self.df["timestamp_token"], unit="ms", errors="coerce"
            )

    def _compute_metrics(self, y_pred: pd.Series, y_true: pd.Series) -> Dict[str, float]:
        y_p = y_pred.astype(bool).values
        y_t = y_true.astype(bool).values
        if len(y_p) == 0:
            return {k: np.nan for k in ["accuracy", "precision", "recall", "f1", "mcc"]}

        return {
            "accuracy": accuracy_score(y_t, y_p),
            "precision": precision_score(y_t, y_p, zero_division=0),
            "recall": recall_score(y_t, y_p, zero_division=0),
            "f1": f1_score(y_t, y_p, zero_division=0),
            "mcc": matthews_corrcoef(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
        }

    # ---------------- Outputs ----------------
    def generate_summary(self):
        if self.df.empty: return

        summary_rows = []
        # Group by our new composite Model ID (includes strategy)
        for model_id in sorted(self.df["model_id"].dropna().unique()):
            model_data = self.df[self.df["model_id"] == model_id]
            
            # 1. Overall metrics
            overall = self._compute_metrics(model_data["validity"], model_data[self.ground_truth_col])
            
            row = {
                "model_id": model_id,
                "total_count": len(model_data),
                "accuracy": round(overall["accuracy"], 3),
                "f1": round(overall["f1"], 3)
            }

            # 2. Per-trait metrics
            for trait in self.traits:
                t_data = model_data[model_data["trait"] == trait]
                tm = self._compute_metrics(t_data["validity"], t_data[self.ground_truth_col])
                
                row[f"{trait}_acc"] = round(tm["accuracy"], 3)
                row[f"{trait}_count"] = len(t_data)

            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(self.SUMMARY_FILENAME, index=False)
        print(f"✅ Summary saved: {self.SUMMARY_FILENAME}")

        # Timing
        if "timestamp_token_dt" in self.df.columns:
            dts = self.df["timestamp_token_dt"].dropna()
            if not dts.empty:
                with open(self.TIMEINFO_FILENAME, "w") as f:
                    f.write(f"Start: {dts.min()}\nEnd: {dts.max()}\nDuration: {dts.max() - dts.min()}\n")

    def generate_plots(self):
        if self.df.empty: return
        
        # Prep data: Calculate accuracy per Question
        df_plot = self.df.copy()
        df_plot["correct"] = df_plot["validity"] == df_plot[self.ground_truth_col]
        
        # Group by Model (which includes Strategy) and Question
        per_q = df_plot.groupby(["model_id", "question_id"])["correct"].mean().reset_index()
        per_q.rename(columns={"correct": "accuracy"}, inplace=True)

        try:
            import seaborn as sns
            from matplotlib import pyplot as plt

            plt.figure(figsize=(14, 8))
            
            # Create Boxplot
            # Sort models so 'Combined' and 'Single' for same model appear near each other
            order = sorted(per_q["model_id"].unique())
            
            ax = sns.boxplot(
                data=per_q, x="model_id", y="accuracy",
                order=order, palette="viridis", showmeans=True
            )
            
            plt.xticks(rotation=45, ha="right")
            plt.title("Accuracy Comparison: Single vs Combined Traits")
            plt.ylabel("Accuracy per Question")
            plt.tight_layout()
            plt.savefig(self.PLOT_FILENAME)
            plt.close()
            print(f"✅ Plot saved: {self.PLOT_FILENAME}")
        except ImportError:
            print("⚠️ Seaborn/Matplotlib not found, skipping plots.")


# ---------------- Main Loader ----------------
def load_data_recursively(paths: List[str]) -> pd.DataFrame:
    """Loads CSVs from paths/dirs and tags them with Strategy based on folder name."""
    all_dfs = []
    
    # Resolve all files
    files = []
    for p_str in paths:
        p = Path(p_str).expanduser().resolve()
        if p.is_dir():
            # Find all clean result CSVs recursively
            files.extend(list(p.rglob("clean_evaluation_results.csv")))
        elif p.is_file():
            files.append(p)

    print(f"🔍 Found {len(files)} result files to process.")

    for f in files:
        try:
            df = pd.read_csv(f)
            
            # --- Strategy Auto-Tagging ---
            # Look at parent folders for keywords
            path_str = str(f).lower()
            if "combined" in path_str:
                df["strategy"] = "Combined"
            elif "single" in path_str:
                df["strategy"] = "Single"
            else:
                df["strategy"] = "Unknown"
                
            all_dfs.append(df)
            print(f"  -> Loaded {len(df)} rows from {f.parent.name} (Strategy: {df['strategy'].iloc[0]})")
        except Exception as e:
            print(f"  ❌ Failed to load {f}: {e}")

    if not all_dfs:
        return pd.DataFrame()
        
    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", 
        nargs="+", 
        required=True, 
        help="List of files or folders to scan for 'clean_evaluation_results.csv'"
    )
    ap.add_argument("--outdir", default="./results_comparison", help="Output directory")
    args = ap.parse_args()

    # 1. Load and Merge
    merged_df = load_data_recursively(args.inputs)
    
    if merged_df.empty:
        print("❌ No data found.")
        sys.exit(1)

    # 2. Run Stats
    out_path = Path(args.outdir)
    stats = Statistics(merged_df, out_path)
    stats.generate_summary()
    stats.generate_plots()