import unittest
from .LLMValidatorsTesting import LLMValidatorsTesting
from .Prompt import Prompt, PromptID
from pathlib import Path

print("Starting LLMValidatorsTestingTest...")  # Debugging line to indicate test start


class MyTestCase(unittest.TestCase):
    """
    Test suite for the LLMValidatorsTesting class.

    This class contains various test methods to evaluate the functionality
    of LLMValidatorsTesting under different configurations, including
    different models, prompts, datasets, and multimodal capabilities.
    Each method typically sets up a specific scenario by defining
    the dataset, number of questions (MAX), selection strategy (random_choice),
    output directory, prompt types, and target LLM models.
    """

    def test_multimodal(self):
        """
        Tests the LLMValidatorsTesting pipeline with a multimodal model.

        This test processes a single randomly chosen question (MAX=1, random_choice=True)
        from a ScienceQA dataset that includes image references (`ScienceQA_test_mc_images.csv`).
        It uses the General_V1 prompt and targets the "GPT4oMini" multimodal model.
        The `multimodal=True` flag is passed to `LLMValidatorsTesting` to initialize
        the appropriate multimodal service. Paths are resolved using `pathlib.Path`.
        """
        print("Starting multimodal test...")
        MAX = 5
        random_choice = True
        SciQdataset = str(Path(r"../data/raw/ScienceQA_test_mc_images.csv").resolve())
        print("SciQdataset:", SciQdataset)
        results_dir = str(Path(r"../data/results/test_results" + str(MAX)).resolve())
        LLMrunner = LLMValidatorsTesting(
            SciQdataset, results_dir, MAX, random_choice, multimodal=True
        )
        prompt_IDs = [PromptID.General_V1_Img]
        models = ["GPT4oMini", "GPT4o", "GPT4Turbo"]
        # "L_Qwen25VL3B", "L_Qwen25L7B", "L_Qwen25VL32B", "L_Qwen25VL72B",
        #   "L_Gemma34B", "L_Gemma312B", "L_Gemma327B",
        #     "L_Llama32Vision11B", "L_Llama32Vision90B",
        #     "L_Llama4Maverick", "L_Llama4Scout",
        #     "L_MistralSmall3124B"
        LLMrunner.run_per_qid(prompt_IDs, models)


if __name__ == "__main__":
    unittest.main()
