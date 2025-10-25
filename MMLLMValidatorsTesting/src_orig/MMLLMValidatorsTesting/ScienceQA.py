import pandas as pd
from PIL import Image
from pathlib import Path
import ast
from typing import List, Union
from config import settings, PROJECT_ROOT


class ScienceQA:
    """
    Represents a single problem from the ScienceQA dataset.
    """

    id: str
    question: str
    choices: List[str]
    answer: int
    hint: str
    image: str
    task: str
    grade: str
    subject: str
    topic: str
    category: str
    skill: str
    lecture: str
    solution: str
    split: str

    def __init__(self, problem_data: pd.DataFrame):
        """
        Initializes a ScienceQA problem instance from a DataFrame row.
        """
        if not problem_data.empty:
            row = problem_data.iloc[0]
            self.id = str(row.get("question_id", ""))
            self.question = str(row.get("question", ""))
            # Safely evaluate the 'choices' string into a list
            try:
                choices_val = row.get("choices", "[]")
                self.choices = (
                    ast.literal_eval(choices_val)
                    if isinstance(choices_val, str)
                    else choices_val or []
                )
            except (ValueError, SyntaxError):
                self.choices = []
            self.answer = int(row.get("answer", -1))
            self.hint = str(row.get("hint", ""))
            self.image = str(row.get("image", ""))
            self.task = str(row.get("task", ""))
            self.grade = str(row.get("grade", ""))
            self.subject = str(row.get("subject", ""))
            self.topic = str(row.get("topic", ""))
            self.category = str(row.get("category", ""))
            self.skill = str(row.get("skill", ""))
            self.lecture = str(row.get("lecture", ""))
            self.solution = str(row.get("solution", ""))
            self.split = str(row.get("set", ""))
        else:
            # Initialize with empty values if no data is provided
            self.id, self.question, self.hint, self.image, self.task = (
                "",
                "",
                "",
                "",
                "",
            )
            self.grade, self.subject, self.topic, self.category = "", "", "", ""
            self.skill, self.lecture, self.solution, self.split = "", "", "", ""
            self.choices, self.answer = [], -1


### --- Generate QUESTION / RESPONSE part of the Validators ---


def get_question_text(problem: pd.DataFrame) -> str:
    """Extracts the question text from the problem data."""
    return str(problem["question"].iloc[0])


def get_context_text(problem: pd.DataFrame, use_caption: bool) -> str:
    """Generates a context string from the problem's hint and optionally its image caption."""
    txt_context = str(problem["hint"].iloc[0] or "")
    img_context = ""
    if use_caption and "caption" in problem.columns:
        img_context = str(problem["caption"].iloc[0] or "")

    context = " ".join([txt_context, img_context]).strip()

    na_string = settings.get("constants", {}).get("not_applicable_string", "N/A")
    return context or na_string


def get_image_files(problem: pd.DataFrame) -> List[Image.Image]:
    """
    Loads and returns all .png images for a problem from the directory specified in the config.
    """
    images: List[Image.Image] = []
    image_base_dir: Path = PROJECT_ROOT / settings["paths"]["image_base_dir"]

    question_id = str(problem["question_id"].iloc[0]).strip()
    image_folder_name = str(problem["image"].iloc[0]).strip()

    placeholders = settings.get("settings", {}).get(
        "image_folder_placeholders", ["", "none", "image.png"]
    )
    if image_folder_name.lower() in placeholders:
        image_folder_name = question_id

    image_dir = image_base_dir / image_folder_name
    if not image_dir.is_dir():
        return []

    for img_path in sorted(image_dir.glob("*.png")):
        try:
            with Image.open(img_path) as img:
                images.append(img.convert("RGB"))
        except Exception as e:
            print(f"⚠️ Could not load image {img_path}: {e}")

    return images


def get_choice_text(problem: pd.DataFrame, options: List[str]) -> str:
    """Formats the multiple-choice options into a single string."""
    choices_str: Union[str, list] = problem["choices"].iloc[0]
    choices_list: List[str] = []
    try:
        choices_list = (
            ast.literal_eval(choices_str)
            if isinstance(choices_str, str)
            else choices_str
        )
    except (ValueError, SyntaxError):
        print(f"Warning: Could not parse choices: '{choices_str}'.")
        choices_list = []

    return " ".join(
        [f"({options[i]}) {c}" for i, c in enumerate(choices_list) if i < len(options)]
    )


def get_answer(problem: pd.DataFrame, options: List[str]) -> str:
    """Retrieves the correct answer option letter for the problem."""
    answer_index: int = problem["answer"].iloc[0]
    return options[answer_index] if 0 <= answer_index < len(options) else ""


def get_lecture_text(problem: pd.DataFrame) -> str:
    """Extracts and formats the lecture text from the problem data."""
    return str(problem["lecture"].iloc[0] or "").replace("\n", " ")


def get_solution_text(problem: pd.DataFrame) -> str:
    """Extracts and formats the solution text from the problem data."""
    return str(problem["solution"].iloc[0] or "").replace("\n", " ")


def create_one_question(
    format_str: str,
    question: str,
    context: str,
    choice: str,
    answer: str,
    lecture: str,
    solution: str,
) -> str:
    """Constructs a formatted question string based on the specified format and components."""
    input_format, output_format = format_str.split("-")

    input_map = {
        "Q": f"Question: {question}\n",
        "C": f"Context: {context}\n",
        "M": f"Options: {choice}\n",
        "L": f"BECAUSE: {lecture}\n",
        "E": f"BECAUSE: {solution}\n",
    }
    input_str = "".join(input_map.get(char, "") for char in input_format)

    output_map = {"A": f"The answer is {answer}.", "L": lecture, "E": solution}
    output_parts = [output_map.get(char) for char in output_format]
    output_str = "Answer: " + " ".join(filter(None, output_parts))

    text = (input_str + output_str).replace("  ", " ").strip()
    return text.replace("BECAUSE:", "").strip() if text.endswith("BECAUSE:") else text


def build_question(problem: pd.DataFrame, format_str: str) -> str:
    """
    Builds a complete question string from problem data using choices from the config.
    """
    options = settings.get("settings", {}).get("choice_options", [])

    return create_one_question(
        format_str,
        get_question_text(problem),
        get_context_text(problem, use_caption=False),
        get_choice_text(problem, options),
        get_answer(problem, options),
        get_lecture_text(problem),
        get_solution_text(problem),
    )


### --- Generate CHARACTERISTICS / PROMPT part of the Validators ---


def get_skill(problem: pd.DataFrame) -> str:
    return str(problem["skill"].iloc[0])


def get_topic(problem: pd.DataFrame) -> str:
    return str(problem["topic"].iloc[0])


def get_subject(problem: pd.DataFrame) -> str:
    return str(problem["subject"].iloc[0])


def get_task(problem: pd.DataFrame) -> str:
    return str(problem["task"].iloc[0])


def get_grade(problem: pd.DataFrame) -> str:
    return str(problem["grade"].iloc[0])


def get_category(problem: pd.DataFrame) -> str:
    return str(problem["category"].iloc[0])


def create_one_characteristic(
    input_format: str,
    grade: str,
    subject: str,
    topic: str,
    category: str,
    skill: str,
) -> str:
    """Constructs a formatted string of educational characteristics based on the input format."""
    parts = {
        "G": f"Student Grade: {grade}\n",
        "S": f"Subject: {subject}\n",
        "T": f"Topic: {topic}\n",
        "C": f"Category: {category}\n",
        "Sk": f"Skill: {skill}\n",
    }

    if input_format == "GSTCSk":
        return "".join(parts.values())
    elif input_format == "GSTC":
        return parts["G"] + parts["S"] + parts["T"] + parts["C"]
    elif input_format == "GST":
        return parts["G"] + parts["S"] + parts["T"]
    elif input_format == "GS":
        return parts["G"] + parts["S"]
    elif input_format == "G":
        return parts["G"]

    print(f"Warning: Unknown characteristic format '{input_format}'.")
    return ""


def build_characteristics(problem: pd.DataFrame, format_str: str) -> str:
    """Builds a formatted string of educational characteristics from problem data."""
    grade_val = get_grade(problem)
    grade_cleaned = grade_val.lower().replace("grade", "").strip()

    return create_one_characteristic(
        format_str,
        grade_cleaned,
        get_subject(problem),
        get_topic(problem),
        get_category(problem),
        get_skill(problem),
    )
