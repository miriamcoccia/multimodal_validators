from enum import Enum
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Tuple, Union
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class MultiJudgesType(Enum):
    """Enumeration of available ensemble strategies."""

    MajorityVoting = 0
    AllTrue = 1
    AtLeastOne = 2


class EnsembleEvaluation:
    """
    Performs ensemble evaluation on trait validation results from multiple models.
    """

    def __init__(self, records_df: pd.DataFrame):
        self.records_df = records_df.copy()

    def _apply_voting_logic(
        self, trait_outputs: List[bool], judge: MultiJudgesType
    ) -> bool:
        """Applies the selected voting logic to a list of boolean votes."""
        if judge == MultiJudgesType.MajorityVoting:
            vote_count = Counter(trait_outputs)
            return vote_count.get(True, 0) > vote_count.get(False, 0)
        elif judge == MultiJudgesType.AllTrue:
            return all(trait_outputs)
        elif judge == MultiJudgesType.AtLeastOne:
            return any(trait_outputs)
        return False

    def _get_model_combinations(self) -> List[Tuple[str, ...]]:
        """
        Get model combinations from config or generate defaults.
        Returns list of tuples containing model IDs.
        """
        # Try to get from config
        ensemble_config = settings.get("ensemble_models", {})
        combinations = ensemble_config.get("ensemble_combinations", [])

        # Validate and normalize combinations
        valid_combinations = []
        for combo in combinations:
            if isinstance(combo, (list, tuple)):
                # Filter out empty strings and None values
                valid_models = [str(m).strip() for m in combo if m and str(m).strip()]
                if len(valid_models) >= 2:  # Need at least 2 models for ensemble
                    valid_combinations.append(tuple(valid_models))
            elif isinstance(combo, str) and combo.strip():
                # Single model string - skip as we need pairs
                logger.warning(f"Skipping single model combination: {combo}")

        # If no valid combinations from config, create defaults from available models
        if not valid_combinations:
            unique_models = self.records_df["model_id"].unique().tolist()
            logger.info(
                f"No valid ensemble combinations in config. Available models: {unique_models}"
            )

            # Create default pairs if we have at least 2 models
            if len(unique_models) >= 2:
                # Just use first two models as a default combination
                valid_combinations = [(unique_models[0], unique_models[1])]
                logger.info(f"Using default combination: {valid_combinations[0]}")

        return valid_combinations

    def run_ensemble_strategy(self, judge: MultiJudgesType) -> pd.DataFrame:
        """
        Performs the chosen ensemble strategy for each (question_id, trait) pair.
        Returns a new DataFrame with the ensemble result.
        """
        if self.records_df.empty:
            logger.warning("Empty records DataFrame provided to ensemble evaluation")
            return pd.DataFrame()

        # Get model combinations
        model_combinations = self._get_model_combinations()

        if not model_combinations:
            logger.warning("No valid model combinations for ensemble evaluation")
            return pd.DataFrame()

        ensemble_results = []
        grouped = self.records_df.groupby(["question_id", "trait"])

        for combo in model_combinations:
            # Build ensemble name from judge type and model IDs
            ensemble_name = f"{judge.name}_{'_'.join(combo)}"
            logger.debug(f"Processing ensemble: {ensemble_name}")

            for (qid, trait), group in grouped:
                # Get votes from models in this combination
                votes = []
                for model_id in combo:
                    model_votes = group[group["model_id"] == model_id]["validity"]
                    if not model_votes.empty:
                        # Convert to boolean and get the value (handle Series properly)
                        vote_value = (
                            model_votes.iloc[0]
                            if len(model_votes) == 1
                            else model_votes.mode().iloc[0]
                        )
                        votes.append(bool(vote_value))

                if not votes:
                    logger.debug(
                        f"No votes found for {qid}/{trait} with models {combo}"
                    )
                    continue

                final_validity = self._apply_voting_logic(votes, judge)

                ensemble_results.append(
                    {
                        "question_id": qid,
                        "trait": trait,
                        "model_id": ensemble_name,
                        "validity": final_validity,
                        "contributing_models": len(votes),
                    }
                )

        result_df = pd.DataFrame(ensemble_results)
        logger.info(
            f"Ensemble evaluation complete. Generated {len(result_df)} results."
        )
        return result_df
