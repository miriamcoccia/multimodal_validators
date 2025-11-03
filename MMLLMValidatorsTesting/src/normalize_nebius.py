#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def transform_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Transforms a Nebius JSONL line into the specific
    format expected by the user's ResultsHandler.
    (i.e., response.body.output[...].text)
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        print(f"Skipping bad JSON line: {line[:50]}...", file=sys.stderr)
        return None

    # Check if it's already in the target format (has body.output)
    if (
        data.get("response")
        and isinstance(data["response"], dict)
        and data["response"].get("body")
        and isinstance(data["response"]["body"], dict)
        and "output" in data["response"]["body"]
    ):
        return data  # It's already converted or is an OpenAI file

    original_response = data.get("response")
    original_error = data.get("error")
    new_response_wrapper: Dict[str, Any] = {}

    # If there's no response, just format the error
    if not original_response or not isinstance(original_response, dict):
        new_response_wrapper = {
            "status_code": 500,
            "request_id": None,
            "body": {"error": original_error or "Missing response object"},
        }
    else:
        # This is the Nebius "response" object:
        # {"id": "chatcmpl-...", "choices": [...], "created": ..., "model": ...}

        # Extract the content string
        text_content = None
        try:
            text_content = original_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print(
                f"Warning: Could not find text content for custom_id {data.get('custom_id')}",
                file=sys.stderr,
            )

        # --- THIS IS THE CRITICAL CHANGE ---
        # Build the nested 'body' structure that ResultsHandler expects
        new_body = {
            "model": original_response.get("model"),
            "created_at": original_response.get("created"),
            "status": "completed" if text_content else "failed",
            "error": None,
            "output": [  # The 'output' list
                {
                    "content": [  # The 'content' list
                        {
                            "type": "output_text",
                            "text": text_content,  # Put the JSON string here
                        }
                    ]
                }
            ],
        }
        # --- END OF CHANGE ---

        new_response_wrapper = {
            "status_code": 200,
            "request_id": original_response.get("id"),
            "body": new_body,  # Use our newly constructed body
        }

    # Reconstruct the top-level object
    new_data = {
        "id": data.get("id"),
        "custom_id": data.get("custom_id"),
        "response": new_response_wrapper,
        "error": original_error,  # This is the top-level Nebius error, if any
    }

    return new_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nebius batch JSONL to the format expected by ResultsHandler."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the original Nebius results JSONL file.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to write the new custom-formatted JSONL file.",
    )
    args = parser.parse_args()

    args.input_file = args.input_file.expanduser().resolve()
    args.output_file = args.output_file.expanduser().resolve()

    if not args.input_file.exists():
        print(f"❌ Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {args.input_file.name} -> {args.output_file.name}")

    lines_processed = 0
    lines_written = 0

    with open(args.input_file, "r") as infile, open(args.output_file, "w") as outfile:
        for line in infile:
            lines_processed += 1
            transformed_data = transform_line(line)

            if transformed_data:
                outfile.write(json.dumps(transformed_data) + "\n")
                lines_written += 1

    print(f"✅ Done. Processed {lines_processed} lines, wrote {lines_written} lines.")


if __name__ == "__main__":
    main()