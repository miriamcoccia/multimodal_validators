#!/usr/bin/env python3
"""
Statistics & Plots for Batch Evaluation Results

Features
--------
- Recursively loads one or more evaluation CSVs and merges them.
- Auto-tags a 'strategy' column based on file path keywords ('single', 'combined').
- Normalizes model identifiers using `custom_id` (e.g., "L_Gemma327B" -> "Gemma-3-27B").
- Computes aggregate metrics (accuracy, precision, recall, F1, MCC, FPR).
- Writes a summary CSV and, if available, a timing range file.
- Generates a comparison boxplot (accuracy per question) via seaborn (optional).

Usage
-----
python stats_plots.py --inputs results/exp1 results/exp2 --outdir ./results_comparison
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# When executed from within src/, ensure we can resolve imports similar to the original
# (the original tried importing `src.config.settings` mainly to force PYTHONPATH behavior).
try:
    # Only for parity with the original script's behavior; not otherwise used.
    from src.config import settings  # type: ignore # noqa: F401
except Exception:
    # Fallback: add repository root (parent of `src/`) to sys.path
    current = Path(__file__).resolve()
    src_dir = current.parent
    repo_root = src_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


LOG_FORMAT = "%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("stats_plots")


class Statistics:
    """
    Handles statistical analysis, plot generation, and timing export for evaluation results.
    """

    def __init__(
        self,
        results_df: pd.DataFrame,
        results_dir: Path,
        ground_truth_col: Optional[str] = None,
    ) -> None:
        self.df = results_df.copy()
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        self.summary_path = self.results_dir / "evaluation_summary.csv"
        self.plot_path = self.results_dir / "accuracy_comparison_boxplot.png"
        self.timeinfo_path = self.results_dir / "time_range.txt"

        self._validate_and_prepare(ground_truth_col)

    # ---------------- Initialization helpers ----------------

    def _validate_and_prepare(self, ground_truth_col: Optional[str]) -> None:
        if "validity" not in self.df.columns:
            raise ValueError("Input DataFrame must contain a 'validity' boolean column.")
        self.df["validity"] = self.df["validity"].astype(bool)

        # Standardize model identifier
        self._standardize_model_id()

        # Determine ground-truth column
        self._set_ground_truth_column(ground_truth_col)

        # Traits are optional; handle absent column gracefully
        self.traits: List[str] = []
        if "trait" in self.df.columns:
            self.traits = sorted(t for t in pd.Series(self.df["trait"]).dropna().unique())

        # Parse timestamp column if present
        self._ensure_datetime_column()

    def _standardize_model_id(self) -> None:
        """
        Build a friendly 'model_id'. If 'custom_id' exists, extract base name, map
        to friendly names, and append '(Strategy)' if strategy column exists.
        Fallback to existing 'model_id' if 'custom_id' is missing.
        """
        if "custom_id" not in self.df.columns:
            logger.warning("'custom_id' not found; using existing 'model_id' as-is.")
            if "model_id" not in self.df.columns:
                raise ValueError("Neither 'custom_id' nor 'model_id' is present in the data.")
            return

        logger.info("Standardizing model_id from custom_id...")
        self.df["base_model"] = self.df["custom_id"].astype(str).str.extract(
            r"^request-([^-]+)-", expand=False
        )

        replacements = {
            "L_Gemma327B": "Gemma-3-27B",
            "L_Qwen25VL72B": "Qwen2.5-VL-72B",
            "GPT4oMini": "GPT-4o-Mini",
            "GPT4o": "GPT-4o",
        }
        self.df["base_model"] = self.df["base_model"].replace(replacements)

        if "strategy" in self.df.columns:
            self.df["model_id"] = self.df["base_model"] + " (" + self.df["strategy"].astype(str) + ")"
        else:
            self.df["model_id"] = self.df["base_model"]

        unique_models = sorted(pd.Series(self.df["model_id"]).dropna().unique())
        logger.info("Identified %d unique model(s).", len(unique_models))

    def _set_ground_truth_column(self, ground_truth_col: Optional[str]) -> None:
        if ground_truth_col and ground_truth_col in self.df.columns:
            gt = ground_truth_col
        else:
            candidates = ["ground_truth", "label", "expected_validity", "gold_validity"]
            gt = next((c for c in candidates if c in self.df.columns), None)

        if gt is None:
            # Fall back to assuming True for all rows (parity with original behavior)
            gt = "_assumed_ground_truth_true"
            self.df[gt] = True
            logger.warning(
                "No ground truth column found. Assuming all ground truths are True "
                "(column: %s).",
                gt,
            )
        else:
            self.df[gt] = self.df[gt].astype(bool)

        self.ground_truth_col = gt

    def _ensure_datetime_column(self) -> None:
        if "timestamp_token" in self.df.columns:
            self.df["timestamp_token_dt"] = pd.to_datetime(
                self.df["timestamp_token"], unit="ms", errors="coerce"
            )

    # ---------------- Metric computation ----------------

    @staticmethod
    def _compute_metrics(y_pred: pd.Series, y_true: pd.Series) -> Dict[str, float]:
        """
        Compute evaluation metrics with safe defaults.
        Includes accuracy, precision, recall, F1, MCC, and FPR.
        """
        y_p = y_pred.astype(bool).to_numpy()
        y_t = y_true.astype(bool).to_numpy()

        if y_p.size == 0:
            return {k: float("nan") for k in ("accuracy", "precision", "recall", "f1", "mcc", "fpr")}

        # Confusion matrix (labels order ensures TN, FP, FN, TP)
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[False, True]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # If ground truth has only one class, MCC is undefined
        mcc = matthews_corrcoef(y_t, y_p) if np.unique(y_t).size > 1 else float("nan")

        return {
            "accuracy": accuracy_score(y_t, y_p),
            "precision": precision_score(y_t, y_p, zero_division=0),
            "recall": recall_score(y_t, y_p, zero_division=0),
            "f1": f1_score(y_t, y_p, zero_division=0),
            "mcc": mcc,
            "fpr": fpr,
        }

    # ---------------- Outputs ----------------

    def generate_summary(self) -> None:
        if self.df.empty:
            logger.info("No rows to summarize; skipping summary generation.")
            return

        if "model_id" not in self.df.columns:
            raise ValueError("Expected a 'model_id' column after standardization.")

        rows: List[Dict[str, Any]] = []

        for model_id in sorted(pd.Series(self.df["model_id"]).dropna().unique()):
            model_df = self.df[self.df["model_id"] == model_id]

            # Overall metrics
            overall = self._compute_metrics(model_df["validity"], model_df[self.ground_truth_col])
            row: Dict[str, Any] = {
                "model_id": model_id,
                "total_count": len(model_df),
                "accuracy": round(overall["accuracy"], 3) if pd.notna(overall["accuracy"]) else np.nan,
                "precision": round(overall["precision"], 3) if pd.notna(overall["precision"]) else np.nan,
                "recall": round(overall["recall"], 3) if pd.notna(overall["recall"]) else np.nan,
                "f1": round(overall["f1"], 3) if pd.notna(overall["f1"]) else np.nan,
                "fpr": round(overall.get("fpr", np.nan), 3),
                "mcc": round(overall["mcc"], 3) if pd.notna(overall["mcc"]) else np.nan,
            }

            # Per-trait metrics (only if trait column exists)
            for trait in self.traits:
                t_df = model_df[model_df["trait"] == trait]
                tm = self._compute_metrics(t_df["validity"], t_df[self.ground_truth_col])
                row[f"{trait}_acc"] = round(tm["accuracy"], 3) if pd.notna(tm["accuracy"]) else np.nan
                row[f"{trait}_fpr"] = round(tm["fpr"], 3) if pd.notna(tm["fpr"]) else np.nan
                row[f"{trait}_count"] = int(len(t_df))

            rows.append(row)

        summary = pd.DataFrame(rows)
        summary.to_csv(self.summary_path, index=False)
        logger.info("Summary written to %s", self.summary_path)

        # Write timing info if we have a parsed timestamp column
        if "timestamp_token_dt" in self.df.columns:
            dts = pd.Series(self.df["timestamp_token_dt"]).dropna()
            if not dts.empty:
                start, end = dts.min(), dts.max()
                duration = end - start
                self.timeinfo_path.write_text(f"Start: {start}\nEnd: {end}\nDuration: {duration}\n")
                logger.info("Timing info written to %s", self.timeinfo_path)

    def generate_plots(self) -> None:
        if self.df.empty:
            logger.info("No rows to plot; skipping plot generation.")
            return

        if "question_id" not in self.df.columns:
            logger.warning("'question_id' column missing; skipping plot generation.")
            return

        # Accuracy per question (per model)
        df_plot = self.df.copy()
        df_plot["correct"] = df_plot["validity"] == df_plot[self.ground_truth_col]
        per_q = (
            df_plot.groupby(["model_id", "question_id"])["correct"]
            .mean()
            .reset_index()
            .rename(columns={"correct": "accuracy"})
        )

        try:
            import seaborn as sns
            from matplotlib import pyplot as plt

            plt.figure(figsize=(14, 8))
            order = sorted(per_q["model_id"].unique())

            ax = sns.boxplot(
                data=per_q,
                x="model_id",
                y="accuracy",
                order=order,
                palette="viridis",
                showmeans=True,
            )
            ax.set_title("Accuracy Comparison: Single vs Combined Traits")
            ax.set_xlabel("Model (Strategy)")
            ax.set_ylabel("Accuracy per Question")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()
            logger.info("Plot written to %s", self.plot_path)
        except ImportError:
            logger.warning("seaborn/matplotlib not installed; skipping plot generation.")

    # ---------------- End Statistics ----------------


def _discover_files(paths: Iterable[str]) -> List[Path]:
    """
    Resolve CSV files to load.
    - If a path is a directory, search recursively for 'clean_evaluation_results.csv'.
    - If it's a file, include it directly.
    """
    files: List[Path] = []
    for p_str in paths:
        p = Path(p_str).expanduser().resolve()
        if p.is_dir():
            files.extend(p.rglob("clean_evaluation_results.csv"))
        elif p.is_file():
            files.append(p)
        else:
            logger.warning("Path not found or unsupported: %s", p)
    return files


def load_data_recursively(paths: List[str]) -> pd.DataFrame:
    """
    Load CSVs from the provided paths and add a 'strategy' column based on path.
    Strategy detection:
      - contains 'combined' (case-insensitive) -> 'Combined'
      - contains 'single'   (case-insensitive) -> 'Single'
      - otherwise -> 'Unknown'
    """
    files = _discover_files(paths)
    logger.info("Found %d file(s) to process.", len(files))

    dataframes: List[pd.DataFrame] = []

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            path_str = str(fpath).lower()
            if "combined" in path_str:
                strategy = "Combined"
            elif "single" in path_str:
                strategy = "Single"
            else:
                strategy = "Unknown"

            df = df.copy()
            df["strategy"] = strategy
            dataframes.append(df)
            logger.info("Loaded %d row(s) from %s (strategy=%s)", len(df), fpath.parent.name, strategy)
        except Exception as exc:
            logger.error("Failed to load %s: %s", fpath, exc)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistics and plots for batch evaluation results.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Files or folders to scan for 'clean_evaluation_results.csv'.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./results_comparison"),
        help="Output directory for summary/plots.",
    )
    parser.add_argument(
        "--ground-truth-col",
        type=str,
        default=None,
        help="Optional explicit ground-truth column name (boolean). If absent, will auto-detect.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    merged = load_data_recursively(args.inputs)
    if merged.empty:
        logger.error("No data found. Exiting.")
        return 1

    stats = Statistics(merged, args.outdir, ground_truth_col=args.ground_truth_col)
    stats.generate_summary()
    stats.generate_plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
