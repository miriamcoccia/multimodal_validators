import json
from typing import Optional
from pathlib import Path
from src.config import PROJECT_ROOT


class ImgTraitDefinition:
    """
    Loads and provides trait definitions and example prompts for multimodal evaluation.
    """

    def __init__(self, path: Optional[Path] = None):
        prompt_file = path or PROJECT_ROOT / "data" / "prompt_fillers.json"
        try:
            with open(prompt_file, "r", encoding="utf-8") as file:
                self.traits = json.load(file)
        except FileNotFoundError:
            print(f"❌ Could not find prompt file at {prompt_file}")
            self.traits = {}

    def retrieve_definition(self, trait_name: str) -> str:
        trait = self.traits.get(trait_name.title())
        if trait:
            return trait.get("definition", "Definition not found.")
        return "Trait not found."

    def retrieve_note(self, trait_name: str) -> str:
        note = self.traits.get(trait_name.title(), {}).get("note", "")
        if note:
            return note
        return "Note not found."

    def retrieve_evaluation_questions(self, trait_name: str) -> str:
        eval_questions_list = self.traits.get(trait_name.title(), {}).get(
            "evaluation_questions", []
        )
        if eval_questions_list:
            return "\n".join([f"- {q}" for q in eval_questions_list])
        return "Evaluation Questions not found."

    def get_examples_for_prompt(self, trait_name: str) -> str:
        examples = self.traits.get(trait_name.title(), {}).get("examples", [])
        if not examples:
            return "No examples available."

        formatted = []
        for i, ex in enumerate(examples, 1):
            input_data = ex["input"]
            output_data = ex["output"]

            output = {
                "trait": output_data["trait"],
                "validity": output_data["validity"],
            }
            if "reasoning" in output_data:
                output["reasoning"] = output_data["reasoning"]

            formatted.append(
                f"Example {i}:\n"
                f"Input:\n"
                f"  Trait: {input_data['trait']}\n"
                f'  Question: "{input_data["question"]}"\n'
                f"  Image: {input_data['image']}\n"
                f"Correct Output:\n{json.dumps(output, indent=2)}\n"
            )
        return "\n".join(formatted)
