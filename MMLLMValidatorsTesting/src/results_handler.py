import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class ResultsHandler:
    """Handles saving, cleaning, and managing evaluation result files."""

    # Pattern based on your custom_id convention:
    # request-<ModelAlias>-<trait_name>-<question_id>-<timestamp>
    CID_RX = re.compile(
        r"^request-(?P<model>[^-]+)-(?P<trait>[^-]+)-(?P<qid>\d+)-(?P<ts>\d+)$"
    )

    def __init__(self, results_dir: Path):
        self.RESULTS_DIR = results_dir
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.RAW_RESULTS_FILENAME = self.RESULTS_DIR / "raw_evaluation_results.csv"
        self.CLEAN_RESULTS_FILENAME = self.RESULTS_DIR / "clean_evaluation_results.csv"

    def load_batch_results_jsonl(self, jsonl_path: Path) -> pd.DataFrame:
        """
        Load Batch API JSONL output and normalize into a DataFrame.
        Handles flattening of 'combined' trait results (1 request -> N rows).
        """
        rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as e:
                    rows.append(
                        {
                            "_line": ln,
                            "_parse_error": str(e),
                            "custom_id": None,
                            "model_id": "PARSE_ERROR",
                            "trait": None,
                            "question_id": None,
                            "validity": None,
                            "reasoning": None,
                        }
                    )
                    continue

                # Parse the object. This returns a LIST of rows.
                # We use .extend() to flatten them into the main list.
                parsed_items = self._parse_batch_object(obj, ln)
                rows.extend(parsed_items)

        df = pd.DataFrame(rows)

        # Define column order (standard fields first, extras last)
        preferred = [
            "custom_id",
            "question_id",
            "model_id",
            "model_reported",
            "trait",
            "validity",
            "reasoning",
            "http_status",
            "request_status",
            "created_at",
            "error",
            "_raw_text",
            "_line",
            "_parse_error",
            "timestamp_token",
        ]
        cols = [c for c in preferred if c in df.columns] + [
            c for c in df.columns if c not in preferred
        ]
        return df[cols]

    @staticmethod
    def _extract_output_text(response_obj: Dict[str, Any]) -> Optional[str]:
        """
        Extract the inner JSON string emitted by the model.
        Handles both OpenAI 'Responses' API (body -> output)
        and Standard/Nebius 'Chat' API (choices -> message -> content).
        """
        # 1. Normalize: Get the inner 'body' or use response_obj itself if no body wrapper
        body = response_obj.get("body")
        if not body:
            # Nebius/Standard format often puts 'choices' directly in 'response'
            body = response_obj

        # 2. Try OpenAI "Responses" API format (output_text)
        out = body.get("output", [])
        if isinstance(out, list):
            for msg in out:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "output_text":
                            return part.get("text", "")

        # 3. Try Standard Chat Completions format (Nebius/OpenAI Chat)
        choices = body.get("choices", [])
        if isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return str(content)

        return None

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strips markdown code fences (```json ... ```) to ensure json.loads works."""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        if t.startswith("```"):
            # Remove first line (e.g. ```json)
            parts = t.split("\n", 1)
            if len(parts) > 1:
                t = parts[1]
            # Remove last line (```)
            if t.endswith("```"):
                t = t.rsplit("\n", 1)[0]
            else:
                t = t.strip("`")
        return t.strip()

    @staticmethod
    def _coerce_bool(val: Any) -> Optional[bool]:
        """Convert diverse truthy/falsey representations to real booleans."""
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            if val in (0, 1):
                return bool(val)
            return None
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "t", "yes", "y", "1"}:
                return True
            if v in {"false", "f", "no", "n", "0"}:
                return False
        return None

    def _parse_batch_object(self, obj: Dict[str, Any], ln: int) -> List[Dict[str, Any]]:
        """
        Parse a single JSONL object from the batch output.
        Returns a LIST of normalized rows to handle combined traits (One-to-Many).
        """
        custom_id = obj.get("custom_id", "")
        response = obj.get("response", {}) or {}
        
        status_code = response.get("status_code")
        # Nebius might not have a status code in the response dict, usually 200 if present
        if status_code is None and "choices" in response:
            status_code = 200

        # Metadata extraction
        # Check both 'body' and 'response' levels for metadata
        body = response.get("body") or response
        
        model_reported = body.get("model")
        created_at = body.get("created_at")
        req_status = body.get("status") 
        error = obj.get("error") # Check top level error too
        request_id = response.get("request_id") or body.get("id")

        # Derive fields from custom_id
        trait_hint = qid = ts = None
        model_alias = None
        m = self.CID_RX.search(custom_id) if isinstance(custom_id, str) else None
        if m:
            model_alias = m.group("model")
            trait_hint = m.group("trait") 
            qid = m.group("qid")
            ts = m.group("ts")

        # --- Base row template (common to all traits in this request) ---
        base_row = {
            "_line": ln,
            "custom_id": custom_id,
            "request_id": request_id,
            "http_status": status_code,
            "request_status": req_status,
            "created_at": created_at,
            "error": error,
            "model_reported": model_reported,
            "model_id": model_alias or model_reported,
            "question_id": qid,
            "timestamp_token": ts,
            "trait": trait_hint, # Fallback trait
            "validity": None,
            "reasoning": None,
            "_raw_text": None,
        }

        # Extract Text Payload
        raw_text = self._extract_output_text(response)
        if not raw_text:
            # If no text (e.g. error), return one row with error details
            return [base_row]

        # Clean and Parse JSON
        cleaned_text = self._clean_json_text(raw_text)
        try:
            inner_json = json.loads(cleaned_text)
        except json.JSONDecodeError:
            base_row["_raw_text"] = raw_text # preserve raw for debugging
            return [base_row]

        # --- LOGIC: Determine if output is Singular or Combined ---
        
        items_to_process = []

        # Case A: It's a list directly (e.g., [ {trait...}, {trait...} ])
        if isinstance(inner_json, list):
            items_to_process = inner_json
        
        # Case B: It's a dict wrapper (e.g. { "traits_output": [...] } )
        elif isinstance(inner_json, dict):
            if "traits_output" in inner_json and isinstance(inner_json["traits_output"], list):
                items_to_process = inner_json["traits_output"]
            elif "traits" in inner_json and isinstance(inner_json["traits"], list):
                items_to_process = inner_json["traits"]
            else:
                # Case C: It's a single trait object (e.g. { "trait": ..., "validity": ... })
                items_to_process = [inner_json]
        
        if not items_to_process:
            # Empty list or dict? Return base row
            return [base_row]

        # --- Generate a row for each trait found ---
        generated_rows = []
        for item in items_to_process:
            if not isinstance(item, dict): 
                continue

            new_row = base_row.copy()
            
            # 1. Get specific trait name from JSON, fall back to custom_id hint
            json_trait = item.get("trait")
            new_row["trait"] = json_trait if json_trait else trait_hint
            
            # 2. Get validity/reasoning
            new_row["validity"] = self._coerce_bool(item.get("validity"))
            new_row["reasoning"] = item.get("reasoning")
            
            generated_rows.append(new_row)

        return generated_rows

    def save_raw_results(self, df: pd.DataFrame):
        """Saves the raw DataFrame of results to a CSV file."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.RAW_RESULTS_FILENAME, index=False, quoting=csv.QUOTE_ALL)
        ok = self.RAW_RESULTS_FILENAME.exists()
        size = self.RAW_RESULTS_FILENAME.stat().st_size if ok else 0
        print(
            f"💾 Raw results saved to {self.RAW_RESULTS_FILENAME} ({'OK' if ok else 'MISSING'}, {size} bytes)"
        )

    def save_clean_results(self, df: pd.DataFrame):
        """Saves the cleaned DataFrame to a CSV file."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.CLEAN_RESULTS_FILENAME, index=False, quoting=csv.QUOTE_ALL)
        ok = self.CLEAN_RESULTS_FILENAME.exists()
        size = self.CLEAN_RESULTS_FILENAME.stat().st_size if ok else 0
        print(
            f"✅ Clean results saved to {self.CLEAN_RESULTS_FILENAME} ({'OK' if ok else 'MISSING'}, {size} bytes)"
        )

    def clean_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out rows with errors, normalizes columns, and attempts data recovery.
        """
        clean_df = df.copy()
        print(f"Starting cleaning with {len(clean_df)} rows")

        # 1. Question ID fallback logic (Regex extraction from custom_id)
        if "question_id" in clean_df.columns:
            null_qid_mask = clean_df["question_id"].isna()
            if null_qid_mask.sum() > 0:
                print(f"DIAGNOSTIC: {null_qid_mask.sum()} rows have null question_id. Attempting recovery.")
                
                def extract_qid_fallback(custom_id):
                    if pd.isna(custom_id) or custom_id == "": 
                        return None
                    import re
                    # Patterns to find ID in strings like "request-model-trait-123-ts"
                    patterns = [r"question[_-](\d+)", r"qid[_-](\d+)", r"id[_-](\d+)", r"-(\d+)-"]
                    for pattern in patterns: 
                        match = re.search(pattern, str(custom_id), re.IGNORECASE)
                        if match: return match.group(1)
                    # Last resort: find numbers
                    numbers = re.findall(r"\d+", str(custom_id))
                    return max(numbers, key=len) if numbers else None

                clean_df.loc[null_qid_mask, "question_id"] = clean_df.loc[null_qid_mask, "custom_id"].apply(extract_qid_fallback)

            # Standardize type
            clean_df["question_id"] = clean_df["question_id"].astype(str)

        # 2. Validity Coercion & Recovery
        if "validity" in clean_df.columns:
            clean_df["validity"] = clean_df["validity"].apply(self._coerce_bool)
            
            # Try to recover from raw text if validity is missing but we have text
            missing_validity = clean_df["validity"].isna() & clean_df["_raw_text"].notna()
            if missing_validity.sum() > 0:
                print(f"RECOVERY: Attempting to recover {missing_validity.sum()} validity values from raw text")
                def recover_validity(text):
                    t = str(text).lower()
                    if "true" in t and "false" not in t: return True
                    if "false" in t and "true" not in t: return False
                    return None
                
                recovered = clean_df.loc[missing_validity, "_raw_text"].apply(recover_validity)
                clean_df.loc[missing_validity, "validity"] = recovered

        # 3. Trait Recovery from custom_id if missing in JSON
        if "trait" in clean_df.columns:
             missing_trait = clean_df["trait"].isna() & clean_df["custom_id"].notna()
             if missing_trait.sum() > 0:
                 print(f"RECOVERY: Attempting to recover {missing_trait.sum()} traits from custom_id")
                 # Map your filename slugs to nice names
                 traits_map = {
                     "functional_relevance": "Functional Relevance",
                     "visual_clarity": "Visual Clarity",
                     "technical_quality": "Technical Quality", 
                     "standard_presentation": "Standard Presentation",
                     "text-image_coherence": "Text-Image Coherence",
                     "fair_representation": "Fair Representation"
                 }
                 def recover_trait(cid):
                     c_lower = str(cid).lower().replace("-", "_")
                     for key, val in traits_map.items():
                         if key in c_lower: return val
                     return None
                 
                 clean_df.loc[missing_trait, "trait"] = clean_df.loc[missing_trait, "custom_id"].apply(recover_trait)

        # 4. Model ID Standardization
        if "model_id" in clean_df.columns:
             # Fill from reported if missing
             clean_df["model_id"] = clean_df["model_id"].fillna(clean_df.get("model_reported"))
             
             # Standardize common names
             mask_mini = clean_df["model_id"].str.startswith("gpt-4o-mini", na=False)
             clean_df.loc[mask_mini, "model_id"] = "GPT4oMini"
             
             mask_4o = clean_df["model_id"].str.startswith("gpt-4o", na=False) & ~mask_mini
             clean_df.loc[mask_4o, "model_id"] = "GPT4o"

        # 5. Final Filtering
        required = ["question_id", "trait", "validity"]
        for col in required:
            if col in clean_df.columns:
                missing = clean_df[col].isna().sum()
                if missing > 0:
                    print(f"WARNING: {missing} rows missing '{col}'")

        # Only keep rows that actually have a validity result
        mask = clean_df["validity"].notna()
        clean_df = clean_df[mask]

        print(f"Final clean count: {len(clean_df)}")
        return clean_df