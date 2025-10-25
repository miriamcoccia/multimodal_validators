import argparse
import os
import pandas as pd
from pathlib import Path
from .LLMEnsembleEvaluation import LLMEnsembleEvaluation, MultiJudgesType

RESULTS_DIR = os.path.join(os.getcwd(), "results/")

GROUND_TRUTH_POSSIBILITIES = ["gtincorrect"]  # ["gtcorrect", "gtincorrect"]
CLEAN_RESULTS_FILENAME = RESULTS_DIR + "clean_questions_traits_evaluation.csv"
ENSEMBLE_RESULTS_FILENAME = RESULTS_DIR + "ensemble_evaluation.csv"
ENSEMBLE_COMPARISON_FILENAME = RESULTS_DIR + "ensemble_comparison.csv"


def load_evaluation_data(input_csv: str) -> pd.DataFrame:
    return pd.read_csv(input_csv)


def save_ensemble_results(results_df: pd.DataFrame, output_csv: str):
    results_df.to_csv(output_csv, index=False, quoting=1)
    print(f"Saved ensemble majority voting results to: {output_csv}")


# Not needed as we already have a functionality to plot and compare accordingly
# def run_ensemble_comparison(input_csv: str, ensemble_df: pd.DataFrame, output_csv: str):
#     LLMEnsembleEvaluation.compare_ensemble_to_individuals(input_csv, ensemble_df, output_csv)


def main():
    for gt in GROUND_TRUTH_POSSIBILITIES:
        input_csv = CLEAN_RESULTS_FILENAME.replace(".csv", "_" + gt + ".csv")
        output_csv = ENSEMBLE_RESULTS_FILENAME.replace(".csv", "_" + gt + ".csv")
        df = load_evaluation_data(input_csv)
        evaluator = LLMEnsembleEvaluation(df)
        traits_evaluation_results = pd.DataFrame(
            columns=[
                "question_id",
                "prompt_id",
                "model_id",
                "traits_output",
                "accuracy",
            ]
        )
        for judge in [MultiJudgesType.MajorityVoting, MultiJudgesType.AllTrue]:

            # Will only be needed in case we have to save evaluation numbers per strategy in a separate csv
            # path = Path(RESULTS_DIR + "/"+judge.name)
            # path.mkdir(parents=True, exist_ok=True)

            ensemble_results = evaluator.run_multiple_judges_strategies(judge)
            traits_evaluation_results = traits_evaluation_results._append(
                ensemble_results, ignore_index=True
            )

        traits_evaluation_results = traits_evaluation_results._append(df)
        save_ensemble_results(traits_evaluation_results, output_csv)

        # Not needed as we already have a functionality to plot and compare accordingly
        # output_comparison_csv = RESULTS_DIR + "/"+ judge.name+ "/ensemble_comparison_" + judge.name + ".csv"
        # run_ensemble_comparison(input_csv, ensemble_results, output_comparison_csv)


# To run it via command-line: python LLMEnsembleEvaluator.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="LLMEnsembleEvaluator",
        description="Evaluate ensemble performance of multiple LLMs acting as judges over question traits.",
    )

    main()
