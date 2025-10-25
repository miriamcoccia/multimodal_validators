import csv
import random
import time
from typing import Union, List, Optional
import numpy as np
import pandas as pd

from config import settings
from .Prompt import PromptID, Prompt
from .LLM_Service import LLM_Service
from .MultimodalLLM_Service import MultimodalLLM_Service
from .ScienceQA import get_image_files
from .TraitList import supported_traits

from .Metrics import Metrics
from .Statistics import Statistics

from sklearn.metrics import accuracy_score


class LLMValidatorsTesting:
    """
    A class to test and evaluate LLM validators based on various prompts and models.
    """

    metrics: Metrics
    statistics: Statistics
    llm_service: Union[LLM_Service, MultimodalLLM_Service]
    ground_truth_questions: pd.DataFrame
    selected_questions: List[int]
    selected_traits: List[str]
    prompts: pd.DataFrame
    traits_evaluation_results: pd.DataFrame

    def __init__(
        self,
        input_filename: Optional[str] = None,
        results_dir: Optional[str] = None,
        MAX: int = 0,
        random_choice: bool = False,
        multimodal: bool = False,
    ):
        """
        Initializes the LLMValidatorsTesting instance.
        """
        self.metrics = Metrics()
        self.selected_traits = list(supported_traits["traits"].values)

        input_path = input_filename or settings.get("paths", {}).get("input_data_csv")
        results_path = results_dir or settings.get("paths", {}).get("results_dir")

        self.statistics = Statistics(
            input_path, results_path, self.metrics, self.selected_traits
        )

        if multimodal:
            print("Initializing in MULTIMODAL mode.")
            self.llm_service = MultimodalLLM_Service()
        else:
            print("Initializing in standard text-only mode.")
            self.llm_service = LLM_Service()

        self.prompts = pd.DataFrame(columns=["question_id", "prompt"])
        self.traits_evaluation_results = pd.DataFrame(
            columns=[
                "question_id",
                "prompt_id",
                "model_id",
                "traits_output",
                "accuracy",
                "time_taken_ms",
            ]
        )
        self.load_testing_data(input_path, MAX, random_choice)

    def generate_prompts(self, qid: int, pid: PromptID):
        """
        Generates a prompt and appends it to the prompts DataFrame using pd.concat.
        """
        qid_indexes = self.ground_truth_questions["question_id"] == qid
        question_df = self.ground_truth_questions[qid_indexes]

        new_prompt_row = pd.DataFrame(
            [{"question_id": qid, "prompt": Prompt(pid, question_df)}]
        )
        self.prompts = pd.concat([self.prompts, new_prompt_row], ignore_index=True)

    def execute(self, qid: int, pid: PromptID, mid: str):
        """
        Executes a prompt and records the evaluation, including timing.
        """
        prompt_series = self.prompts.loc[self.prompts["question_id"] == qid, "prompt"]
        for prompt_obj in prompt_series:
            if prompt_obj.id is not pid:
                continue

            print(f"Running QID: {qid}, PID: {prompt_obj.id.name}, MID: {mid}.")

            start_time = time.perf_counter()

            question_row = self.ground_truth_questions.loc[
                self.ground_truth_questions["question_id"] == qid
            ]
            pil_images = (
                get_image_files(question_row)
                if isinstance(self.llm_service, MultimodalLLM_Service)
                else None
            )

            exec_args = {
                "model_id": mid,
                "prompt": str(prompt_obj.prompt),
                "format_instructions": str(prompt_obj.format_instructions),
            }
            if pil_images:
                exec_args["pil_images"] = pil_images

            response = self.llm_service.execute_prompt(**exec_args)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            answer_result = self._process_response(qid, pid, mid, response, duration_ms)

            new_result_row = pd.DataFrame([answer_result])

            if not new_result_row.empty:
                self.traits_evaluation_results = pd.concat(
                    [self.traits_evaluation_results, new_result_row], ignore_index=True
                )
            break

    def _process_response(
        self, qid: int, pid: PromptID, mid: str, response: any, duration_ms: float
    ) -> dict:
        """Helper method to process the model's response and return a result dictionary."""
        rate_limit_msg = settings.get("constants", {}).get("rate_limit_error")

        if response is None or response == rate_limit_msg:
            return {
                "question_id": qid,
                "prompt_id": pid.name,
                "model_id": mid,
                "traits_output": rate_limit_msg if response == rate_limit_msg else "{}",
                "accuracy": "0.0",
                "time_taken_ms": round(duration_ms, 2),
            }

        expected = np.ones(len(self.selected_traits), dtype=bool)
        obtained = np.zeros(len(self.selected_traits), dtype=bool)

        if hasattr(response, "traits_output") and response.traits_output:
            for trait_res in response.traits_output:
                if trait_res.trait in self.selected_traits and trait_res.valid:
                    try:
                        index = self.selected_traits.index(trait_res.trait)
                        obtained[index] = True
                    except ValueError:
                        continue

            accuracy = accuracy_score(expected, obtained)
            traits_json = (
                response.model_dump_json()
                if hasattr(response, "model_dump_json")
                else "{}"
            )
        else:
            accuracy, traits_json = 0.0, "{}"

        return {
            "question_id": qid,
            "prompt_id": pid.name,
            "model_id": mid,
            "traits_output": traits_json,
            "accuracy": f"{accuracy}",
            "time_taken_ms": round(duration_ms, 2),
        }

    def load_testing_data(
        self, inputname: str, MAX: int = 0, random_choice: bool = False
    ):
        """Loads and selects question data from the input CSV."""
        self.ground_truth_questions = pd.read_csv(inputname, keep_default_na=False)
        all_qids = sorted(self.ground_truth_questions["question_id"].unique())

        if 0 < MAX < len(all_qids):
            self.selected_questions = (
                random.sample(all_qids, MAX) if random_choice else all_qids[:MAX]
            )
        else:
            self.selected_questions = all_qids

    def is_valid_question(self, qid: int) -> bool:
        """Checks if a question has sufficient context (lecture or solution) to be valid."""
        question_data = self.ground_truth_questions[
            self.ground_truth_questions["question_id"] == qid
        ]
        if question_data.empty:
            return False

        lecture = str(question_data["lecture"].values[0] or "")
        solution = str(question_data["solution"].values[0] or "")

        has_lecture = lecture and lecture.lower() != "nan"
        has_solution = solution and solution.lower() != "nan"

        return has_lecture or has_solution

    def report(self, qid: int, pid: PromptID, mid: str, first_time: bool = False):
        """Writes the evaluation result for a single run to the main CSV file."""
        out_filename = self.statistics.RESULTS_FILENAME
        file_mode = "w" if first_time else "a"

        try:
            with open(
                out_filename, file_mode, newline="", encoding="utf-8"
            ) as data_file:
                csv_writer = csv.writer(data_file, quoting=csv.QUOTE_NONNUMERIC)
                if first_time:
                    csv_writer.writerow(self.traits_evaluation_results.columns)

                row_to_write = self.traits_evaluation_results.loc[
                    (self.traits_evaluation_results["question_id"] == qid)
                    & (self.traits_evaluation_results["prompt_id"] == pid.name)
                    & (self.traits_evaluation_results["model_id"] == mid)
                ]

                if not row_to_write.empty:
                    csv_writer.writerow(row_to_write.values[0])
        except IOError as e:
            print(f"Error writing to report file {out_filename}: {e}")

    def run_per_qid(
        self,
        prompt_ids: Optional[List[PromptID]] = None,
        model_ids: Optional[List[str]] = None,
        new_file: bool = True,
    ):
        """
        Runs the complete testing and evaluation pipeline for all specified models and prompts.
        """
        prompt_ids = prompt_ids or PromptID.all()
        model_ids = model_ids or []

        for mid_idx, mid in enumerate(model_ids):
            print(f"\n--- Processing Model: {mid} ---")
            for qid_idx, qid in enumerate(self.selected_questions):
                if not self.is_valid_question(qid):
                    continue
                for pid_idx, pid in enumerate(prompt_ids):
                    is_first_report = (
                        new_file and mid_idx == 0 and qid_idx == 0 and pid_idx == 0
                    )

                    self.generate_prompts(qid, pid)
                    self.execute(qid, pid, mid)
                    self.report(qid, pid, mid, first_time=is_first_report)

        print("\n--- Finalizing Statistics and Reports ---")
        self.statistics.clean_generated_questions()
        self.statistics.generate_summary()
        self.statistics.compute_statistics()
        self.statistics.generate_plots()
        self.statistics.classify_generated_questions()

        print("\n" + "=" * 30)
        print("Processing Complete!")
        print(f"Cold Models Encountered: {self.llm_service.cold_models}")
        print(f"Unsupported/Error Models: {self.llm_service.error_models}")
        print("=" * 30 + "\n")
