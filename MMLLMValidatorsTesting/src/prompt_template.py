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
You must choose strictly between two output formats:

1. [VALID] If the image MEETS the design standard:
   {{"trait": "{trait_name}", "validity": true, "reasoning": ""}}

2. [INVALID] If the image FAILS the design standard:
   {{"trait": "{trait_name}", "validity": false, "reasoning": "Brief explanation of design flaw (ONE SENTENCE)"}}

⚠️ CRITICAL RULES:
- When validity is true, reasoning MUST be an empty string (""). It cannot be null or omitted.
- When validity is false, reasoning MUST be a non-empty string.
""".strip()
# -------------------------------
# EXAMPLES:
# {examples}

combi_system_prompt_template = """
⚠️ CRITICAL: You are evaluating the IMAGE'S visual design quality, NOT looking for question text within the image. The question is provided separately. Do NOT expect to see the question or answer options written inside the image.

You are evaluating whether an image demonstrates good visual design for the following traits: "{traits_list}".
-------------------------------
TRAITS TO EVALUATE:
{traits_info}

-------------------------------
COMMON MISTAKES TO AVOID:
❌ "The image doesn't contain the question text" - CORRECT: Question text is provided separately
❌ "I can't see answer options in the image" - CORRECT: Options aren't supposed to be in the image
❌ "The image doesn't show the question" - EVALUATE: The image's visual design quality only
✅ FOCUS ON: Visual design principles of the image itself

-------------------------------

REQUIRED OUTPUT FORMAT (STRICT SCHEMA):
You MUST return a JSON object with a single key "traits_output".
"traits_output" MUST be a LIST of JSON objects.

For EACH trait in the list, you must strictly adhere to one of these two formats:

1. [VALID]
   {{"trait": "TraitName", "validity": true,  "reasoning": ""}}

2. [INVALID]
   {{"trait": "TraitName", "validity": false, "reasoning": "Brief explanation of design flaw (ONE SENTENCE)"}}

⚠️ CRITICAL RULES:
- If validity is true, reasoning MUST be an empty string ("").
- If validity is false, reasoning MUST be a non-empty string.

The final output structure MUST look like:
{{
  "traits_output": [
    {{"trait": "TraitName1", "validity": true,  "reasoning": ""}},
    {{"trait": "TraitName2", "validity": false, "reasoning": "Brief explanation of design flaw (ONE SENTENCE)"}},
    {{"trait": "TraitName3", "validity": true,  "reasoning": ""}}
  ]
}}
""".strip()

