#!/usr/bin/env python3
"""
Statistics & Plots for Batch Evaluation Results

Revisions (2025-08-27):
- If no ground-truth column is provided/found, we now **assume the base dataset is correct**
  and synthesize a ground-truth column equal to True for all rows. This ensures
  precision/recall/F1 are computed (note: MCC remains undefined without both classes).
- Per-trait metrics are always emitted: accuracy, precision, recall, f1, mcc, count.
- Added robust timestamp handling and a helper to report total elapsed time based on
  `timestamp_token` (ms) or `timestamp_token_dt` (datetime).
"""
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
        self.SUMMARY_FILENAME = results_dir / "evaluation_summary.csv"
        self.PLOT_FILENAME = results_dir / "accuracy_boxplot.png"
        self.TIMEINFO_FILENAME = results_dir / "time_range.txt"

        # --- Required columns ---
        if "validity" not in self.df.columns:
            raise ValueError("Input DataFrame must contain a 'validity' column.")
        self.df["validity"] = self.df["validity"].astype(bool)

        # --- Ground truth handling ---
        # Either use provided ground truth, auto-detect, or synthesize all-True labels.
        self.ground_truth_col = ground_truth_col or next(
            (
                c
                for c in [
                    "ground_truth",
                    "label",
                    "target",
                    "expected_validity",
                    "gold_validity",
                    "gt_validity",
                    "y_true",
                ]
                if c in self.df.columns
            ),
            None,
        )

        if self.ground_truth_col is None:
            # Assume base dataset is always good -> treat ground truth as True.
            self.ground_truth_col = "_assumed_ground_truth_true"
            self.df[self.ground_truth_col] = True
        else:
            self.df[self.ground_truth_col] = self.df[self.ground_truth_col].astype(bool)

        # Extract unique traits for per-trait analysis
        self.traits = self._extract_traits_from_data()

        # Parse timestamps if present (for elapsed time reporting)
        self._ensure_datetime_column()

    # ---------------- internal helpers ----------------
    def _extract_traits_from_data(self) -> List[str]:
        traits = set()
        if "trait" in self.df.columns:
            traits.update(self.df["trait"].dropna().unique())
        return sorted(list(traits))

    @staticmethod
    def _has_two_classes(arr: np.ndarray) -> bool:
        try:
            return len(np.unique(arr)) >= 2
        except Exception:
            return False

    def _compute_metrics(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> Dict[str, float]:
        """
        Compute accuracy, precision, recall, f1, mcc against provided ground truth.
        Note: If y_true has only one class (e.g., all True), MCC is NaN and precision can
        be degenerate (sklearn handles with zero_division=0).
        """
        y_pred_b = y_pred.astype(bool).values
        y_true_b = y_true.astype(bool).values

        if len(y_pred_b) == 0:
            return {
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "mcc": np.nan,
            }

        acc = accuracy_score(y_true_b, y_pred_b)
        prec = precision_score(y_true_b, y_pred_b, zero_division=0)
        rec = recall_score(y_true_b, y_pred_b, zero_division=0)
        f1 = f1_score(y_true_b, y_pred_b, zero_division=0)

        if self._has_two_classes(y_true_b) and self._has_two_classes(y_pred_b):
            mcc = matthews_corrcoef(y_true_b, y_pred_b)
        else:
            mcc = float("nan")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc}

    def _ensure_datetime_column(self) -> None:
        """Create a datetime column from timestamp_token if available."""
        if "timestamp_token_dt" in self.df.columns and np.issubdtype(
            self.df["timestamp_token_dt"].dtype, np.datetime64
        ):
            return
        if "timestamp_token" in self.df.columns:
            # Handle ms-since-epoch integers to datetime
            try:
                self.df["timestamp_token_dt"] = pd.to_datetime(
                    self.df["timestamp_token"], unit="ms", errors="coerce"
                )
            except Exception:
                # Fallback: try parse as generic datetime strings
                self.df["timestamp_token_dt"] = pd.to_datetime(
                    self.df["timestamp_token"], errors="coerce"
                )

    # ---------------- timing ----------------
    def compute_time_range(
        self,
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]:
        """Return (start, end, duration) using timestamp_token_dt if present."""
        if "timestamp_token_dt" not in self.df.columns:
            return None
        dts = self.df["timestamp_token_dt"].dropna()
        if dts.empty:
            return None
        start = dts.min()
        end = dts.max()
        return start, end, end - start

    # ---------------- public API ----------------
    def generate_summary(self):
        """Generates a summary CSV with aggregated metrics per model, including per-trait analysis."""
        if self.df.empty:
            print("⚠️ Cannot generate summary: DataFrame is empty.")
            return

        for col in ["model_id", "trait"]:
            if col not in self.df.columns:
                raise ValueError(f"Input DataFrame must contain '{col}' column.")

        summary_rows: List[Dict[str, Any]] = []

        for model_id in sorted(self.df["model_id"].dropna().unique()):
            model_data = self.df[self.df["model_id"] == model_id]

            # Overall metrics vs ground-truth (assumed True if none provided)
            overall = self._compute_metrics(
                model_data["validity"], model_data[self.ground_truth_col]
            )

            row: Dict[str, Any] = {
                "model_id": model_id,
                "count": int(len(model_data)),
                "mean_accuracy": (
                    round(overall["accuracy"], 3)
                    if not np.isnan(overall["accuracy"])
                    else None
                ),
                "precision": (
                    round(overall["precision"], 3)
                    if not np.isnan(overall["precision"])
                    else None
                ),
                "recall": (
                    round(overall["recall"], 3)
                    if not np.isnan(overall["recall"])
                    else None
                ),
                "f1": round(overall["f1"], 3) if not np.isnan(overall["f1"]) else None,
                "mcc": (
                    round(overall["mcc"], 3) if not np.isnan(overall["mcc"]) else None
                ),
            }

            # Per-trait metrics
            for trait in self.traits:
                md = model_data[model_data["trait"] == trait]
                tm = (
                    self._compute_metrics(md["validity"], md[self.ground_truth_col])
                    if not md.empty
                    else {
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "mcc": np.nan,
                    }
                )
                row[f"{trait}_accuracy"] = (
                    round(tm["accuracy"], 3) if not np.isnan(tm["accuracy"]) else None
                )
                row[f"{trait}_precision"] = (
                    round(tm["precision"], 3) if not np.isnan(tm["precision"]) else None
                )
                row[f"{trait}_recall"] = (
                    round(tm["recall"], 3) if not np.isnan(tm["recall"]) else None
                )
                row[f"{trait}_f1"] = (
                    round(tm["f1"], 3) if not np.isnan(tm["f1"]) else None
                )
                row[f"{trait}_mcc"] = (
                    round(tm["mcc"], 3) if not np.isnan(tm["mcc"]) else None
                )
                row[f"{trait}_count"] = int(len(md))

            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(self.SUMMARY_FILENAME, index=False)
        print(f"✅ Summary saved to {self.SUMMARY_FILENAME}")

        # Also write timing info if available
        tr = self.compute_time_range()
        if tr is not None:
            start, end, duration = tr
            with open(self.TIMEINFO_FILENAME, "w", encoding="utf-8") as f:
                f.write(f"start={start}\nend={end}\nduration={duration}\n")
            print(f"⏱️ Time range saved to {self.TIMEINFO_FILENAME}")

    def generate_plots(self):
        """Generates a boxplot for (per-question) accuracy per model.

        If ground truth is available or synthesized, 'accuracy' means correctness vs labels.
        """
        if self.df.empty:
            print("⚠️ Cannot generate plots: DataFrame is empty.")
            return

        if "model_id" not in self.df.columns or "question_id" not in self.df.columns:
            print("⚠️ Cannot generate plots: missing 'model_id' or 'question_id'.")
            return

        import sys

        script_dir = str(Path(__file__).resolve().parent)
        removed = False
        if script_dir in sys.path:
            try:
                sys.path.remove(script_dir)
                removed = True
            except Exception:
                pass

        try:
            import seaborn as sns  # lazy import
            from matplotlib import pyplot as plt
        finally:
            if removed:
                sys.path.insert(0, script_dir)

        # Build per-question accuracy vs ground truth
        df_plot = self.df.copy()
        df_plot["correct"] = df_plot["validity"].astype(bool) == df_plot[
            self.ground_truth_col
        ].astype(bool)
        per_q = (
            df_plot.groupby(["model_id", "question_id"])["correct"]
            .mean()
            .reset_index()
            .rename(columns={"correct": "accuracy"})
        )

        if per_q.empty:
            print("⚠️ No per-question rows to plot.")
            return

        # --- Plotting Defaults ---
        figsize = (12, 8)
        palette = "viridis"
        rotation = 45

        model_order = sorted(per_q["model_id"].unique())

        plt.figure(figsize=figsize)
        ax = sns.boxplot(
            x="model_id",
            y="accuracy",
            data=per_q,
            hue="model_id",
            dodge=False,
            order=model_order,
            palette=palette,
            showmeans=True,
        )

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        plt.xticks(rotation=rotation, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Per-Question Accuracy vs Ground Truth")
        plt.title("Accuracy Distribution Across Questions")
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(self.PLOT_FILENAME)
        plt.close()
        print(f"✅ Plot saved to {self.PLOT_FILENAME}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute summary & plots from results CSV")
    ap.add_argument(
        "--csv",
        default=str(
            Path(settings["paths"]["results_dir"]) / "clean_evaluation_results.csv"
        ),
        help="Path to clean_evaluation_results.csv",
    )
    ap.add_argument(
        "--outdir",
        default=str(Path(settings["paths"]["results_dir"])),
        help="Where to write summary/plots",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help="Optional CSV with ground-truth labels to merge (must include columns used by --merge-on and a boolean label column)",
    )
    ap.add_argument(
        "--gt-col",
        default=None,
        help=(
            "Name of ground-truth column in the input CSV (or in --labels if provided). "
            "If omitted, we auto-detect; if still missing, we ASSUME all-True ground truth."
        ),
    )
    ap.add_argument(
        "--merge-on",
        default="question_id,trait",
        help="Comma-separated join keys when merging --labels (default: question_id,trait)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.csv).expanduser().resolve())

    # Optionally merge labels
    gt_col = args.gt_col
    if args.labels:
        labels_path = Path(args.labels).expanduser().resolve()
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        labels_df = pd.read_csv(labels_path)

        merge_keys = [k.strip() for k in args.merge_on.split(",") if k.strip()]
        if not all(k in df.columns for k in merge_keys):
            missing = [k for k in merge_keys if k not in df.columns]
            raise ValueError(f"Input CSV missing merge keys: {missing}")
        if not all(k in labels_df.columns for k in merge_keys):
            missing = [k for k in merge_keys if k not in labels_df.columns]
            raise ValueError(f"Labels CSV missing merge keys: {missing}")

        # If ground-truth column not specified, try detect
        if gt_col is None:
            for cand in [
                "ground_truth",
                "label",
                "target",
                "expected_validity",
                "gold_validity",
                "gt_validity",
                "y_true",
            ]:
                if cand in labels_df.columns:
                    gt_col = cand
                    break

        cols = merge_keys + ([gt_col] if gt_col else [])
        cols = [c for c in cols if c]
        df = df.merge(labels_df[cols], on=merge_keys, how="left")

    s = Statistics(df, outdir, ground_truth_col=gt_col)
    s.generate_summary()
    s.generate_plots()

    # Always report time range if possible
    tr = s.compute_time_range()
    if tr is not None:
        start, end, duration = tr
        print(f"⏱️ Start: {start} | End: {end} | Duration: {duration}")

    print(f"✅ Wrote: {s.SUMMARY_FILENAME} and {s.PLOT_FILENAME}")
