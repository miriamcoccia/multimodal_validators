import csv
import json
from pathlib import Path
from bisect import bisect_left
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.metrics import accuracy_score

from config import settings
from .Metrics import Metrics


class Statistics:
    """
    Handles statistical analysis, data cleaning, and plot generation for the results.
    """

    metrics: Metrics
    selected_traits: List[str]
    RESULTS_DIR: Path
    RESULTS_FILENAME: Path
    CLEAN_RESULTS_FILENAME: Path
    CLASSIFICATION_FILENAME: Path
    RESULTS_STATISTICS: Path
    RESULTS_SUMMARY: Path

    def __init__(
        self,
        input_filename: str,
        results_dir: str,
        metrics: Metrics,
        selected_traits: List[str],
    ):
        self.metrics = metrics
        self.selected_traits = selected_traits

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR = results_path

        filenames_config = settings.get("filenames", {})
        self.RESULTS_FILENAME = results_path / filenames_config.get(
            "results_evaluation", "questions_traits_evaluation.csv"
        )
        self.CLEAN_RESULTS_FILENAME = results_path / filenames_config.get(
            "clean_results_evaluation", "clean_questions_traits_evaluation.csv"
        )
        self.CLASSIFICATION_FILENAME = results_path / filenames_config.get(
            "classification", "classification.csv"
        )
        self.RESULTS_STATISTICS = results_path / filenames_config.get(
            "statistics", "statistics.csv"
        )
        self.RESULTS_SUMMARY = results_path / filenames_config.get(
            "summary", "summary.csv"
        )

    def VD_A(self, treatment: List[float], control: List[float]) -> Tuple[float, str]:
        """
        Computes Vargha and Delaney A index.
        """
        m, n = len(treatment), len(control)
        if m != n:
            raise ValueError("Data d and f must have the same length")

        r = ss.rankdata(treatment + control)
        r1 = sum(r[:m])
        A = (2 * r1 - m * (m + 1)) / (2 * n * m)

        levels = [0.147, 0.33, 0.474]
        magnitudes = ["negligible", "small", "medium", "large"]
        magnitude = magnitudes[bisect_left(levels, abs((A - 0.5) * 2))]

        return A, magnitude

    def clean_generated_questions(self) -> None:
        """Filters out invalid or error-filled rows from the raw results file."""
        df = pd.read_csv(self.RESULTS_FILENAME, keep_default_na=False)

        with open(
            self.CLEAN_RESULTS_FILENAME, "w", newline="", encoding="utf-8"
        ) as data_file:
            csv_writer = csv.writer(data_file, quoting=csv.QUOTE_NONNUMERIC)
            csv_writer.writerow(df.columns)

            for _, row in df.iterrows():
                traits_output = row.get("traits_output", "")
                if not traits_output or traits_output == "{}":
                    continue
                try:
                    traits_list = json.loads(traits_output)
                    if isinstance(traits_list, list) and all(
                        t.get("trait") == "ERROR" for t in traits_list
                    ):
                        continue
                except (json.JSONDecodeError, AttributeError):
                    continue
                csv_writer.writerow(row)

    def compute_statistics(self) -> None:
        """Computes and saves statistical comparisons between different prompts."""
        df_all = pd.read_csv(self.CLEAN_RESULTS_FILENAME, keep_default_na=False)
        with open(
            self.RESULTS_STATISTICS, "w", newline="", encoding="utf-8"
        ) as statistics_file:
            statistics_writer = csv.writer(statistics_file)
            statistics_writer.writerow(
                [
                    "model_id",
                    "metric",
                    "treatment",
                    "control",
                    "median_treatment",
                    "mean_treatment",
                    "median_control",
                    "mean_control",
                    "wilcoxon",
                    "kendall",
                    "A12",
                ]
            )

            for mid in df_all["model_id"].unique():
                df = df_all[df_all["model_id"] == mid]
                if df["traits_output"].astype(str).str.len().sum() == 0:
                    continue

                for metric in ["accuracy"]:
                    for treatment in df["prompt_id"].unique():
                        for control in df["prompt_id"].unique():
                            if treatment == control:
                                continue

                            treatment_values, control_values = [], []
                            for qid in df["question_id"].unique():
                                t_val = df[
                                    (df["question_id"] == qid)
                                    & (df["prompt_id"] == treatment)
                                ][metric].values.astype(float)
                                c_val = df[
                                    (df["question_id"] == qid)
                                    & (df["prompt_id"] == control)
                                ][metric].values.astype(float)
                                if t_val.size == 1 and c_val.size == 1:
                                    treatment_values.append(t_val[0])
                                    control_values.append(c_val[0])

                            wil_p, ken_p, A12 = 0.0, 0.0, 0.0
                            if treatment_values and not np.allclose(
                                treatment_values, control_values
                            ):
                                _, wil_p = ss.wilcoxon(treatment_values, control_values)
                                _, ken_p = ss.kendalltau(
                                    treatment_values, control_values
                                )
                                A12, _ = self.VD_A(treatment_values, control_values)

                            statistics_writer.writerow(
                                [
                                    mid,
                                    metric,
                                    treatment,
                                    control,
                                    f"{np.median(treatment_values):.2f}",
                                    f"{np.mean(treatment_values):.2f}",
                                    f"{np.median(control_values):.2f}",
                                    f"{np.mean(control_values):.2f}",
                                    f"{wil_p:.2f}",
                                    f"{ken_p:.2f}",
                                    f"{A12:.2f}",
                                ]
                            )

    def generate_summary(self) -> None:
        """Generates a summary CSV file with aggregated metrics per model and prompt."""
        df = pd.read_csv(self.CLEAN_RESULTS_FILENAME, keep_default_na=False)
        df_gen = pd.read_csv(self.RESULTS_FILENAME, keep_default_na=False)
        total_questions = df_gen["question_id"].nunique()
        total_questions = 1 if total_questions == 0 else total_questions

        with open(
            self.RESULTS_SUMMARY, "w", newline="", encoding="utf-8"
        ) as summary_file:
            summary_writer = csv.writer(summary_file)
            header = (
                ["prompt_id", "model_id", "accuracy"]
                + self.selected_traits
                + ["num_questions", "success_ratio"]
            )
            summary_writer.writerow(header)

            for pid in df["prompt_id"].unique():
                for mid in df["model_id"].unique():
                    subset = df[(df["prompt_id"] == pid) & (df["model_id"] == mid)]
                    if subset.empty:
                        continue

                    num_gen_questions = len(subset)
                    accuracy = subset["accuracy"].astype(float).mean()
                    stats_row = [pid, mid, f"{accuracy:.2f}"]

                    for trait in self.selected_traits:
                        trait_obtained = np.zeros(num_gen_questions, dtype=bool)
                        for i, traits_json in enumerate(subset["traits_output"]):
                            try:
                                for t in json.loads(traits_json):
                                    if t.get("trait") == trait and t.get("valid"):
                                        trait_obtained[i] = True
                                        break
                            except (json.JSONDecodeError, TypeError):
                                continue

                        trait_accuracy = accuracy_score(
                            np.ones(num_gen_questions, dtype=bool), trait_obtained
                        )
                        stats_row.append(f"{trait_accuracy:.2f}")

                    stats_row.append(f"{num_gen_questions}")
                    stats_row.append(f"{num_gen_questions / total_questions:.2f}")
                    summary_writer.writerow(stats_row)

    def generate_plots(self) -> None:
        """Generates boxplots for metrics, configured via config.toml."""
        df = pd.read_csv(self.CLEAN_RESULTS_FILENAME, keep_default_na=False)

        plotting_config = settings.get("plotting", {})
        figsize = (
            plotting_config.get("figsize_width", 12),
            plotting_config.get("figsize_height", 7),
        )
        palette = plotting_config.get("palette", "Set2")
        y_bottom = plotting_config.get("y_limit_bottom", -0.5)
        y_top = plotting_config.get("y_limit_top", 1.5)
        rotation = plotting_config.get("x_label_rotation", 90)
        legend_loc = plotting_config.get("legend_location", "lower center")

        filenames_config = settings.get("filenames", {})
        plot_template = filenames_config.get("plot_template", "{metric}.png")

        for metric in ["accuracy"]:
            plt.figure(figsize=figsize)
            ax = sns.boxplot(
                x="model_id",
                y=metric,
                hue="prompt_id",
                data=df.astype({metric: float}),
                showmeans=True,
                palette=palette,
            )
            ax.legend(
                title="Prompt",
                loc=legend_loc,
                bbox_to_anchor=(0.5, 1.05),
                ncol=3,
                frameon=False,
            )
            plt.xticks(rotation=rotation)
            plt.xlabel("Large Language Model")
            plt.ylabel(metric.capitalize())
            plt.ylim(y_bottom, y_top)
            plt.tight_layout()

            plot_path = self.RESULTS_DIR / plot_template.format(metric=metric)
            plt.savefig(plot_path)
            plt.close()
            print(f"✅ Plot saved to {plot_path}")

    def classify_generated_questions(self) -> None:
        """Classifies questions based on how many LLMs failed a given trait."""
        df = pd.read_csv(self.CLEAN_RESULTS_FILENAME, keep_default_na=False)
        class_labels = settings.get("classification", {}).get(
            "labels", ["passing", "doubtful", "failing"]
        )

        with open(
            self.CLASSIFICATION_FILENAME, "w", newline="", encoding="utf-8"
        ) as data_file:
            csv_writer = csv.writer(data_file)
            csv_writer.writerow(
                ["question_id", "prompt_id", "trait", "trait_class", "failing_llms"]
            )

            for name, group in df.groupby(["question_id", "prompt_id"]):
                qid, pid = name
                for trait in self.selected_traits:
                    failing_llms = []
                    for _, row in group.iterrows():
                        try:
                            for t in json.loads(row["traits_output"]):
                                if t.get("trait") == trait and not t.get("valid"):
                                    failing_llms.append(row["model_id"])
                                    break
                        except (json.JSONDecodeError, TypeError):
                            continue

                    num_fails = len(failing_llms)
                    class_label = class_labels[min(num_fails, len(class_labels) - 1)]
                    csv_writer.writerow([qid, pid, trait, class_label, failing_llms])
