import re
import ast
import asyncio
import string
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from PIL import Image
from pydantic import BaseModel

from src.config import settings, PROJECT_ROOT


class ScienceQA(BaseModel):
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
    _raw_row: Optional[pd.Series] = None

    @classmethod
    def from_df_row(cls, row: pd.Series) -> "ScienceQA":
        try:
            parsed_choices = (
                ast.literal_eval(row["choices"])
                if isinstance(row["choices"], str)
                else row["choices"]
            )
        except Exception:
            parsed_choices = []

        instance = cls(
            id=str(row.get("question_id", "")),
            question=str(row.get("question", "")),
            choices=parsed_choices or [],
            answer=int(row.get("answer", -1)),
            hint=str(row.get("hint", "")),
            image=str(row.get("image", "")),
            task=str(row.get("task", "")),
            grade=str(row.get("grade", "")),
            subject=str(row.get("subject", "")),
            topic=str(row.get("topic", "")),
            category=str(row.get("category", "")),
            skill=str(row.get("skill", "")),
            lecture=str(row.get("lecture", "")),
            solution=str(row.get("solution", "")),
            split=str(row.get("set", "")),
        )

        # Optional: attach the original row if needed later
        instance._raw_row = row
        return instance

    async def load_images(self) -> List[Image.Image]:
        """
        Asynchronously loads and returns .png images associated with this question.
        """
        images: List[Image.Image] = []
        image_base_dir: Path = PROJECT_ROOT / settings["paths"]["image_base_dir"]

        folder = self.image.strip() or self.id
        placeholders = settings.get("settings", {}).get(
            "image_folder_placeholders", ["", "none", "image.png"]
        )

        if folder.lower() in placeholders:
            folder = self.id

        image_dir = image_base_dir / folder
        if not image_dir.is_dir():
            return []

        loop = asyncio.get_running_loop()

        async def load(path: Path):
            try:
                return await loop.run_in_executor(
                    None, lambda: Image.open(path).convert("RGB")
                )
            except Exception as e:
                print(f"⚠️ Could not load image {path}: {e}")
                return None

        tasks = [load(p) for p in sorted(image_dir.glob("*.png"))]
        results = await asyncio.gather(*tasks)
        return [img for img in results if img]

    async def upload_images_to_openai(self, openai_client) -> List[str]:
        """
        Asynchronously uploads .png images to OpenAI and returns file IDs.
        """
        file_ids: List[str] = []
        image_base_dir: Path = PROJECT_ROOT / settings["paths"]["image_base_dir"]

        folder = self.image.strip() or self.id
        placeholders = settings.get("settings", {}).get(
            "image_folder_placeholders", ["", "none", "image.png"]
        )

        if folder.lower() in placeholders:
            folder = self.id

        image_dir = image_base_dir / folder
        if not image_dir.is_dir():
            return []

        loop = asyncio.get_running_loop()

        async def upload_openai(path: Path):
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: openai_client.files.create(
                        file=open(str(path), "rb"), purpose="vision"
                    ).id,
                )
            except Exception as e:
                print(f"⚠️ Could not upload image {path}: {e}")
                return None

        tasks = [upload_openai(p) for p in sorted(image_dir.glob("*.png"))]
        results = await asyncio.gather(*tasks)
        return [file_id for file_id in results if file_id]


# === Utility: Question formatting ===
def build_question(problem: ScienceQA, format_str: str) -> str:
    options = settings.get("settings", {}).get("choice_options", [])

    def choice_text():
        return " ".join(
            [
                f"({options[i]}) {c}"
                for i, c in enumerate(problem.choices)
                if i < len(options)
            ]
        )

    def answer_text():
        idx = problem.answer
        return options[idx] if 0 <= idx < len(options) else ""

    def clean(text: str) -> str:
        return str(text or "").replace("\n", " ").strip()

    def make_question():
        input_map = {
            "Q": f"Question: {clean(problem.question)}\n",
            "C": f"Context: {clean(problem.hint)}\n",
            "M": f"Options: {choice_text()}\n",
            "L": f"BECAUSE: {clean(problem.lecture)}\n",
            "E": f"BECAUSE: {clean(problem.solution)}\n",
        }
        input_fmt, output_fmt = format_str.split("-")
        input_str = "".join([input_map.get(c, "") for c in input_fmt])

        output_map = {
            "A": f"The answer is {answer_text()}.",
            "L": clean(problem.lecture),
            "E": clean(problem.solution),
        }
        output_str = "Answer: " + " ".join(
            [output_map[c] for c in output_fmt if c in output_map and output_map[c]]
        )

        return (input_str + output_str).strip()

    return make_question()


# === Utility: Characteristics formatting ===
def build_characteristics(problem: ScienceQA, format_str: str) -> str:
    """Builds a formatted string of characteristics based on the format string."""
    parts = {
        "G": f"Student Grade: {problem.grade.lower().replace('grade', '').strip()}\n",
        "S": f"Subject: {problem.subject}\n",
        "T": f"Topic: {problem.topic}\n",
        "C": f"Category: {problem.category}\n",
        "Sk": f"Skill: {problem.skill}\n",
        "Ta": f"Task Type: {problem.task}\n",
    }

    keys_in_format = re.findall(f"({'|'.join(parts.keys())})", format_str)

    if not keys_in_format:
        print(f"⚠️ Unknown or empty format '{format_str}' for characteristics.")
        return ""

    return "".join(parts[key] for key in keys_in_format)


# === Utility: ACTIVITY block for OriginalTraitAgent ===
def build_activity_components(problem: ScienceQA) -> str:
    """
    STRICT, labeled block that matches OriginalTraitAgent's system prompt.
    Ensures stable A–Z labels even if settings choice_options is missing/short.
    Keeps 'solution' minimal (letter only) to reduce bias.
    """

    def clean(x: str) -> str:
        return str(x or "").replace("\n", " ").strip()

    # Letters fallback
    configured = settings.get("settings", {}).get("choice_options", [])
    letters = list(configured) if configured else list(string.ascii_uppercase)

    # Choices list
    choices = problem.choices or []
    choice_lines = []
    for i, c in enumerate(choices):
        label = letters[i] if i < len(letters) else f"({i})"
        choice_lines.append(f"- ({label}) {clean(str(c))}")
    choices_block = "\n".join(choice_lines) if choice_lines else "(none)"

    # Solution label only
    ans_idx = problem.answer if isinstance(problem.answer, int) else -1
    ans_label = letters[ans_idx] if 0 <= ans_idx < len(letters) else "(not provided)"

    # Optional extras (kept separate so they don’t pollute evaluation)
    lecture = clean(problem.lecture)
    solution_expl = clean(problem.solution)
    extras = []
    if lecture:
        extras.append(f"- lecture: {lecture}")
    if solution_expl:
        extras.append(f"- solution_explanation: {solution_expl}")
    extra_block = "\n".join(extras)

    return (
        f"""ACTIVITY:
- skill: {clean(problem.skill)}
- question: {clean(problem.question)}
- passage: {clean(problem.hint)}
- choices:
{choices_block}
- solution: {ans_label}
{extra_block if extra_block else ""}"""
    ).strip()
