from enum import Enum
import pandas as pd
import json
from collections import Counter
from typing import List, Dict, Tuple


class MultiJudgesType(Enum):
    MajorityVoting = 0
    AllTrue = 1
    AtLeastOne = 2

    @classmethod
    def all(self):
        return list(map(lambda c: c, self))


class LLMEnsembleEvaluation:
    def __init__(self, records_df: pd.DataFrame):
        """
        :param records_df: DataFrame with columns ['question_id', 'prompt_id', 'model_id', 'traits_output', 'accuracy']
        """
        self.records_df = records_df.copy()
        self.records_df["traits_output"] = self.records_df["traits_output"].apply(
            self._parse_traits_output
        )

    @staticmethod
    def _parse_traits_output(raw_output: str) -> List[Dict]:
        try:
            return json.loads(raw_output)
        except Exception:
            return []

    def _group_key(self, row) -> Tuple:
        return (row["question_id"], row["prompt_id"])

    def _vote_trait_majority(self, trait_outputs: List[Dict]) -> List[Dict]:
        """
        Given a list of trait outputs from multiple models, apply majority voting on each trait.
        """
        trait_votes = {}
        for output in trait_outputs:
            for item in output:
                trait = item["trait"]
                vote = item["valid"]
                trait_votes.setdefault(trait, []).append(vote)

        voted_traits = []
        for trait, votes in trait_votes.items():
            vote_count = Counter(votes)
            majority = (
                vote_count[True] > vote_count[False]
            )  # vote_count.most_common(1)[0][0]
            voted_traits.append(
                {
                    "trait": trait,
                    "valid": majority,
                    "reasoning": (
                        ""
                        if majority
                        else f"Majority voted invalid ({vote_count[False]}/{len(votes)})"
                    ),
                }
            )
        return voted_traits

    def _all_true(self, trait_outputs: List[Dict]) -> List[Dict]:
        """
        Given a list of trait outputs from multiple models, apply majority voting on each trait.
        """
        trait_votes = {}
        for output in trait_outputs:
            for item in output:
                trait = item["trait"]
                vote = item["valid"]
                trait_votes.setdefault(trait, []).append(vote)

        voted_traits = []
        for trait, votes in trait_votes.items():
            vote_count = Counter(votes)
            majority = vote_count[True] == len(votes)
            voted_traits.append(
                {
                    "trait": trait,
                    "valid": majority,
                    "reasoning": (
                        ""
                        if majority
                        else f"AllTrue invalid ({vote_count[False]}/{len(votes)})"
                    ),
                }
            )
        return voted_traits

    def _at_least_one(self, trait_outputs: List[Dict]) -> List[Dict]:
        """
        Given a list of trait outputs from multiple models, apply majority voting on each trait.
        """
        trait_votes = {}
        for output in trait_outputs:
            for item in output:
                trait = item["trait"]
                vote = item["valid"]
                trait_votes.setdefault(trait, []).append(vote)

        voted_traits = []
        for trait, votes in trait_votes.items():
            vote_count = Counter(votes)
            majority = vote_count[True] > 0
            voted_traits.append(
                {
                    "trait": trait,
                    "valid": majority,
                    "reasoning": (
                        ""
                        if majority
                        else f"AllTrue invalid ({vote_count[False]}/{len(votes)})"
                    ),
                }
            )
        return voted_traits

    def run_multiple_judges_strategies(
        self, judge=MultiJudgesType.MajorityVoting
    ) -> pd.DataFrame:
        """
        Performs majority voting for each (question_id, prompt_id) pair.
        Returns a new DataFrame with ensemble accuracy per question.
        """
        grouped = self.records_df.groupby(["question_id", "prompt_id"])
        results = []
        model_combinations = [
            # Same Family: Same version, different sizes
            # ("L_Llama38Instruct", "L_Llama370Instruct", ""),
            # ("L_Llama318Instruct", "L_Llama3170Instruct_Q4", ""),
            # ("L_DeepSeekR1Llama8", "L_DeepSeekR1Llama70_Q4", ""),
            # ("L_DeepSeekR1Qwen7", "DeepSeekR1Qwen32", ""),
            # ("Mixtral8x7B01Instruct", "L_Mixtral8x22B01Instruct_Q4", ""),
            # ("GPT4oMini", "GPT4o", ""),
            # ("GPT4oMini", "GPT4Turbo", ""),
            #
            # # Same Family: Different versions, same size
            # ("L_Llama38Instruct", "L_Llama318Instruct", ""),  # Llama 8B (v3, v3.1)
            # ("L_Llama370Instruct", "L_Llama3170Instruct_Q4", ""),  # Llama 70B (v3, v3.1)
            # ("L_Llama370Instruct", "Llama3370Instruct", ""),  # Llama 70B (v3, v3.3)
            # ("L_Llama3170Instruct_Q4", "Llama3370Instruct", ""),  # Llama 70B (v3.1, v3.3)
            # ("L_Llama370Instruct", "L_Llama3170Instruct_Q4", "Llama3370Instruct"),  # Llama 70B all versions
            # ("Mistral7B03Instruct", "L_Mistral7B02Instruct", ""),  # Mistral 7B (v0.2, v0.3)
            # ("Mistral7B03Instruct", "Mixtral8x7B01Instruct", ""),  # Mistral 7B (v0.1, v0.3)
            # ("L_Mistral7B02Instruct", "Mixtral8x7B01Instruct", ""),  # Mistral 7B (v0.2, v0.1)
            # ("Mistral7B03Instruct", "L_Mistral7B02Instruct", "Mixtral8x7B01Instruct"),  # Mistral 7B (v0.2, v0.3)
            # ("GPT35Turbo1106", "GPT4Turbo", ""),
            # ("GPT35Turbo1106", "GPT4o", ""),
            # Different Families: Same size, same/different versions
            ("L_DeepSeekR1Qwen7", "Mistral7B03Instruct", ""),  # DeepSeek and Mistral
            ("L_DeepSeekR1Qwen7", "L_Mistral7B02Instruct", ""),  # DeepSeek and Mistral
            ("L_DeepSeekR1Qwen7", "Mixtral8x7B01Instruct", ""),  # DeepSeek and Mistral
            ("L_Llama38Instruct", "L_DeepSeekR1Llama8", ""),  # Llama and DeepSeek
            ("L_Llama318Instruct", "L_DeepSeekR1Llama8", ""),  # Llama and DeepSeek
            ("L_Llama370Instruct", "L_DeepSeekR1Llama70_Q4", ""),  # Llama and DeepSeek
            (
                "L_Llama3170Instruct_Q4",
                "L_DeepSeekR1Llama70_Q4",
                "",
            ),  # Llama and DeepSeek
            ("Llama3370Instruct", "L_DeepSeekR1Llama70_Q4", ""),  # Llama and DeepSeek
            ("GPT4o", "L_DeepSeekR1Qwen7", ""),  # GPT and DeepSeek
            ("GPT4o", "L_Llama3170Instruct_Q4", ""),  # GPT and Llama
            ("GPT4o", "Mixtral8x7B01Instruct", ""),  # GPT and Mistral
            # Different Families: Best and Second performing models
            # Best 3 models individually
            # ("GPT4o","L_Llama3170Instruct_Q4", "L_DeepSeekR1Qwen7"),
            # ("GPT4o", "L_Mixtral8x22B01Instruct_Q4", "L_DeepSeekR1Qwen7"),
            # ("GPT4o", "L_Mixtral8x22B01Instruct_Q4", "L_Llama3170Instruct_Q4"),
            # ("L_DeepSeekR1Qwen7", "L_Mixtral8x22B01Instruct_Q4", "L_Llama3170Instruct_Q4"),
            #
            # # 1 Second, and 2 Best models
            # ("GPT4Turbo", "L_Llama3170Instruct_Q4", "L_DeepSeekR1Qwen7"),  # second GPT
            # ("GPT4o", "L_Llama318Instruct", "L_DeepSeekR1Qwen7"),   # second Llama
            # ("GPT4o", "L_Llama3170Instruct_Q4", "DeepSeekR1Qwen32"), #second DeepSeek
        ]

        for M1, M2, M3 in model_combinations:
            ensemble_name = judge.name + "__" + M1 + "__" + M2
            if M3 != "":
                ensemble_name = ensemble_name + "__" + M3
            for (qid, pid), group in grouped:
                model_trait_outputs = group["traits_output"][
                    (group["model_id"] == M1)
                    | (group["model_id"] == M2)
                    | (group["model_id"] == M3)
                ].tolist()
                if not model_trait_outputs:
                    continue

                voted_traits = None
                if judge == MultiJudgesType.MajorityVoting:
                    voted_traits = self._vote_trait_majority(model_trait_outputs)
                elif judge == MultiJudgesType.AllTrue:
                    voted_traits = self._all_true(model_trait_outputs)
                elif judge == MultiJudgesType.AtLeastOne:
                    voted_traits = self._at_least_one(model_trait_outputs)
                accuracy = sum(t["valid"] for t in voted_traits) / len(voted_traits)

                results.append(
                    {
                        "question_id": qid,
                        "prompt_id": pid,
                        "model_id": ensemble_name,
                        "traits_output": json.dumps(voted_traits),
                        "accuracy": accuracy,
                    }
                )

        return pd.DataFrame(results)

    # Not needed as we already have a functionality to plot and compare accordingly
    # def compare_ensemble_to_individuals(input_csv:str, ensemble_df: pd.DataFrame, output_csv:str, label: str = "ensemble") -> pd.DataFrame:
    #     """
    #     Compares accuracy of ensemble model vs individual models.
    #     Returns a DataFrame with average accuracy per model and ensemble.
    #     """
    #     df_clean = pd.read_csv(input_csv, keep_default_na=False)
    #     df_ensemble = ensemble_df.copy()
    #     df_ensemble["dataset"] = label

    #     # Normalize schema if needed
    #     df_clean["dataset"] = "individual"
    #     combined = pd.concat([df_clean[["model_id", "dataset", "accuracy"]],
    #                         df_ensemble[["model_id", "dataset", "accuracy"]]], ignore_index=True)

    #     summary = combined.groupby(["dataset", "model_id"]).agg(
    #         count=("accuracy", "count"),
    #         mean_accuracy=("accuracy", "mean"),
    #         std_accuracy=("accuracy", "std")
    #     ).reset_index()

    #     summary.to_csv(output_csv, index=False)
    #     print(f"Saved ensemble vs individual comparison to: {output_csv}")
    #     return summary
