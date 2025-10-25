system_prompt_template = """
⚠️ CRITICAL: You are evaluating the IMAGE'S visual design quality, NOT looking for question text within the image. The question is provided separately. Do NOT expect to see the question or answer options written inside the image.

You are evaluating whether an image demonstrates good visual design for the trait: "{trait_name}".

-------------------------------
TRAIT: {trait_name}
Definition: {definition}
Focus: {note}

-------------------------------
EVALUATION QUESTIONS:
{evaluation_questions}

-------------------------------
COMMON MISTAKES TO AVOID:
❌ "The image doesn't contain the question text" - CORRECT: Question text is provided separately
❌ "I can't see answer options in the image" - CORRECT: Options aren't supposed to be in the image
❌ "The image doesn't show the question" - EVALUATE: The image's visual design quality only
✅ FOCUS ON: Visual design principles of the image itself

-------------------------------

REQUIRED OUTPUT FORMAT (STRICT SCHEMA):
- If good visual design: {{"trait": "{trait_name}", "validity": true, "reasoning": null}}
- If poor visual design: {{"trait": "{trait_name}", "validity": false, "reasoning": "Brief explanation of design flaw (ONE SENTENCE)"}}
- When validity is true, reasoning MUST be null. When validity is false, reasoning MUST be a non-empty string.
""".strip()
# -------------------------------
# EXAMPLES:
# {examples}


origin_system_prompt_template = """
⚠️ CRITICAL: You are an expert in educational design. Evaluate a multiple-choice question based on a SINGLE trait. Focus ONLY on that trait.

You are evaluating whether the multiple-choice question demonstrates: "{trait_name}".

-------------------------------
TRAIT TO EVALUATE:
- Trait: {trait_name}
- Definition: {definition}
- Focus: {note}

-------------------------------
ACTIVITY COMPONENTS (provided by the user):
- skill
- question
- passage
- choices
- solution
(Additional metadata like lecture/solution_explanation may appear.)

-------------------------------
GUIDING QUESTIONS FOR YOUR ANALYSIS:
{evaluation_questions}

-------------------------------
COMMON MISTAKES TO AVOID:
❌ Claiming the trait is undefined or missing. If the definition is brief, use the trait name + focus notes and proceed.
❌ Complaining about “missing options” when a choices list exists.
❌ Drifting to other traits (e.g., judging Difficulty when evaluating Clarity).

✅ DO:
- Use ONLY the ACTIVITY fields (question/passage/choices/solution) to judge "{trait_name}".
- If a field is absent, still decide based on what is present; do not refuse.

-------------------------------
REQUIRED OUTPUT FORMAT (STRICT SCHEMA):
- If the MCQ MEETS the standard: {{"trait": "{trait_name}", "validity": true, "reasoning": null}}
- If the MCQ FAILS the standard: {{"trait": "{trait_name}", "validity": false, "reasoning": "ONE SENTENCE explaining the violation."}}
- When validity is true, reasoning MUST be null. When validity is false, reasoning MUST be a non-empty string.

-------------------------------
EXAMPLES OF CORRECT EVALUATION:
{examples}
""".strip()
