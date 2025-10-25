import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    # ========= New: Batch JSONL ingestion =========
    def load_batch_results_jsonl(self, jsonl_path: Path) -> pd.DataFrame:
        """
        Load OpenAI Batch Responses API JSONL output and normalize into a DataFrame
        with canonical columns: question_id, model_id, trait, validity, reasoning, etc.
        Does NOT modify the JSONL file; purely reads and structures it.
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
                            "model_id": None,
                            "trait": None,
                            "question_id": None,
                            "validity": None,
                            "reasoning": None,
                        }
                    )
                    continue

                rows.append(self._parse_batch_object(obj, ln))

        df = pd.DataFrame(rows)

        # Reorder columns (keep extras at the end)
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

    def _parse_batch_object(self, obj: Dict[str, Any], ln: int) -> Dict[str, Any]:
        """Parse a single JSONL object from the batch output into normalized fields."""
        custom_id = obj.get("custom_id", "")
        response = obj.get("response", {}) or {}
        status_code = response.get("status_code")
        body = response.get("body", {}) or {}

        # OpenAI response metadata
        model_reported = body.get("model")
        created_at = body.get("created_at")
        req_status = body.get("status")
        error = body.get("error")
        request_id = response.get("request_id")

        # Derive fields from custom_id
        trait_hint = qid = ts = None
        model_alias = None
        m = self.CID_RX.search(custom_id) if isinstance(custom_id, str) else None
        if m:
            model_alias = m.group("model")
            trait_hint = m.group("trait")
            qid = m.group("qid")
            ts = m.group("ts")

        # Extract JSON payload returned by the model (strict schema)
        text_payload = self._extract_output_text(body)
        inner_json: Dict[str, Any] = {}
        raw_text = None
        if text_payload:
            try:
                inner_json = json.loads(text_payload)
            except json.JSONDecodeError:
                raw_text = text_payload  # preserve raw for debugging

        trait = inner_json.get("trait") if inner_json else None
        if trait is None:
            trait = trait_hint  # fallback to custom_id

        validity = self._coerce_bool(inner_json.get("validity") if inner_json else None)
        reasoning = inner_json.get("reasoning") if inner_json else None

        return {
            "_line": ln,
            "custom_id": custom_id,
            "request_id": request_id,
            "http_status": status_code,
            "request_status": req_status,
            "created_at": created_at,
            "error": error,
            "model_reported": model_reported,
            "model_id": model_alias
            or model_reported,  # prefer alias, fallback to reported
            "trait": trait,
            "question_id": qid,
            "timestamp_token": ts,
            "validity": validity,
            "reasoning": reasoning,
            "_raw_text": raw_text,
        }

    @staticmethod
    def _extract_output_text(body: Dict[str, Any]) -> Optional[str]:
        """
        Extract the inner JSON string emitted by the model under Responses API:
        body.output -> list[message]; message.content -> list[parts]; parts[type=="output_text"].text
        """
        out = body.get("output", [])
        if not isinstance(out, list):
            return None
        for msg in out:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        # strip accidental code fences if present
                        t = text.strip()
                        if t.startswith("```"):
                            t = t.strip("`")
                            # after stripping ticks, if it contains json prefix, try to cut it down
                            if t.lower().startswith("json"):
                                t = t[4:].strip()
                        return t
        return None

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

    # ========= Existing CSV helpers (unchanged API) =========
    def save_raw_results(self, df: pd.DataFrame):
        """Saves the raw DataFrame of results to a CSV file."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.RAW_RESULTS_FILENAME, index=False, quoting=csv.QUOTE_ALL)
        ok = self.RAW_RESULTS_FILENAME.exists()
        size = self.RAW_RESULTS_FILENAME.stat().st_size if ok else 0
        print(
            f"💾 Raw results saved to {self.RAW_RESULTS_FILENAME} ({'OK' if ok else 'MISSING'}, {size} bytes)"
        )

    # Add this method to your ResultsHandler class:

    def save_clean_results(self, df: pd.DataFrame):
        """Saves the cleaned DataFrame to a CSV file."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.CLEAN_RESULTS_FILENAME, index=False, quoting=csv.QUOTE_ALL)
        ok = self.CLEAN_RESULTS_FILENAME.exists()
        size = self.CLEAN_RESULTS_FILENAME.stat().st_size if ok else 0
        print(
            f"✅ Clean results saved to {self.CLEAN_RESULTS_FILENAME} ({'OK' if ok else 'MISSING'}, {size} bytes)"
        )

    # Also, fix the _parse_batch_object method to handle question_id parsing better:

    def _parse_batch_object(self, obj: Dict[str, Any], ln: int) -> Dict[str, Any]:
        """Parse a single JSONL object from the batch output into normalized fields."""
        custom_id = obj.get("custom_id", "")
        response = obj.get("response", {}) or {}
        status_code = response.get("status_code")
        body = response.get("body", {}) or {}

        # OpenAI response metadata
        model_reported = body.get("model")
        created_at = body.get("created_at")
        req_status = body.get("status")
        error = body.get("error")
        request_id = response.get("request_id")

        # Derive fields from custom_id
        trait_hint = qid = ts = None
        model_alias = None
        m = self.CID_RX.search(custom_id) if isinstance(custom_id, str) else None
        if m:
            model_alias = m.group("model")
            trait_hint = m.group("trait")
            qid = m.group("qid")
            ts = m.group("ts")
        else:
            # Debug: print problematic custom_ids
            print(f"Warning: Could not parse custom_id at line {ln}: '{custom_id}'")

        # Extract JSON payload returned by the model (strict schema)
        text_payload = self._extract_output_text(body)
        inner_json: Dict[str, Any] = {}
        raw_text = None
        if text_payload:
            try:
                inner_json = json.loads(text_payload)
            except json.JSONDecodeError:
                raw_text = text_payload  # preserve raw for debugging

        trait = inner_json.get("trait") if inner_json else None
        if trait is None:
            trait = trait_hint  # fallback to custom_id

        validity = self._coerce_bool(inner_json.get("validity") if inner_json else None)
        reasoning = inner_json.get("reasoning") if inner_json else None

        return {
            "_line": ln,
            "custom_id": custom_id,
            "request_id": request_id,
            "http_status": status_code,
            "request_status": req_status,
            "created_at": created_at,
            "error": error,
            "model_reported": model_reported,
            "model_id": model_alias
            or model_reported,  # prefer alias, fallback to reported
            "trait": trait,
            "question_id": qid,  # This will be None if custom_id parsing failed
            "timestamp_token": ts,
            "validity": validity,
            "reasoning": reasoning,
            "_raw_text": raw_text,
        }

    # And update the clean_results method to be more informative about the question_id issue:

    def clean_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out rows with errors or invalid data.
        Modified to preserve all rows for comparable results by handling missing data gracefully.
        """
        clean_df = df.copy()

        print(f"Starting cleaning with {len(clean_df)} rows")
        print("GOAL: Preserve all rows for comparable results")

        # Diagnostic: Check what's causing question_id to be null
        if "question_id" in clean_df.columns:
            null_qid_count = clean_df["question_id"].isna().sum()
            if null_qid_count > 0:
                print(f"DIAGNOSTIC: {null_qid_count} rows have null question_id")
                # Show sample of problematic custom_ids to debug regex issue
                null_rows = clean_df[clean_df["question_id"].isna()]
                if "custom_id" in null_rows.columns and not null_rows.empty:
                    sample_custom_ids = null_rows["custom_id"].head(5).tolist()
                    print(f"Sample problematic custom_ids: {sample_custom_ids}")

                    # Try to extract question_id from custom_id manually for these rows
                    def extract_qid_fallback(custom_id):
                        if pd.isna(custom_id) or custom_id == "":
                            return None
                        # Try to find numeric patterns that could be question IDs
                        import re

                        # Look for patterns like "question_123" or just "123" in the string
                        patterns = [
                            r"question[_-](\d+)",
                            r"qid[_-](\d+)",
                            r"id[_-](\d+)",
                            r"-(\d+)-",
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, str(custom_id), re.IGNORECASE)
                            if match:
                                return match.group(1)
                        # Last resort: find any number in the string
                        numbers = re.findall(r"\d+", str(custom_id))
                        if numbers:
                            # Return the longest number found (likely to be question_id)
                            return max(numbers, key=len)
                        return None

                    # Try to recover question_ids using fallback method
                    recovered = null_rows["custom_id"].apply(extract_qid_fallback)
                    recovered_count = recovered.notna().sum()
                    print(
                        f"Fallback method could recover {recovered_count} question_ids"
                    )

                    # Apply the fallback
                    clean_df.loc[clean_df["question_id"].isna(), "question_id"] = (
                        clean_df.loc[clean_df["question_id"].isna(), "custom_id"].apply(
                            extract_qid_fallback
                        )
                    )

        # Handle validity - try multiple approaches to preserve rows
        if "validity" in clean_df.columns:
            before_validity = len(clean_df)
            original_validity = clean_df["validity"].copy()

            # Apply boolean coercion
            clean_df["validity"] = clean_df["validity"].apply(self._coerce_bool)

            # For rows where coercion failed, try alternative approaches
            failed_coercion = clean_df["validity"].isna() & original_validity.notna()
            if failed_coercion.sum() > 0:
                print(
                    f"RECOVERY: Attempting to recover {failed_coercion.sum()} validity values"
                )

                # Look at the raw text to extract validity
                if "_raw_text" in clean_df.columns:

                    def extract_validity_from_text(raw_text):
                        if pd.isna(raw_text):
                            return None
                        text_lower = str(raw_text).lower()
                        if "true" in text_lower and "false" not in text_lower:
                            return True
                        elif "false" in text_lower and "true" not in text_lower:
                            return False
                        return None

                    recovered_validity = clean_df.loc[
                        failed_coercion, "_raw_text"
                    ].apply(extract_validity_from_text)
                    recovery_count = recovered_validity.notna().sum()
                    print(f"Recovered {recovery_count} validity values from raw text")

                    # Apply recovered values
                    clean_df.loc[
                        failed_coercion & recovered_validity.notna(), "validity"
                    ] = recovered_validity.dropna()

            final_invalid = clean_df["validity"].isna().sum()
            if final_invalid > 0:
                print(
                    f"WARNING: Still have {final_invalid} rows with invalid validity - these will be excluded from analysis"
                )

        # Ensure model_id is populated - use model_reported as fallback
        if "model_reported" in clean_df.columns:
            clean_df["model_id"] = clean_df["model_reported"]

            # Apply standardization
            mask_mini = clean_df["model_id"].str.startswith("gpt-4o-mini", na=False)
            clean_df.loc[mask_mini, "model_id"] = "GPT4oMini"

            # Handle regular GPT-4o (including versioned ones like gpt-4o-2024-08-06)
            mask_4o = clean_df["model_id"].str.startswith(
                "gpt-4o", na=False
            ) & ~clean_df["model_id"].str.startswith("gpt-4o-mini", na=False)
            clean_df.loc[mask_4o, "model_id"] = "GPT4o"

        # Fill missing trait values from custom_id if possible
        if "trait" in clean_df.columns:
            null_trait_count = clean_df["trait"].isna().sum()
            if null_trait_count > 0 and "custom_id" in clean_df.columns:
                print(
                    f"RECOVERY: Attempting to recover {null_trait_count} missing trait values"
                )

                def extract_trait_from_custom_id(custom_id):
                    if pd.isna(custom_id):
                        return None
                    # Common trait names to look for
                    traits = [
                        "Functional_Relevance",
                        "Visual_Clarity",
                        "Technical_Quality",
                        "Standard_Presentation",
                        "Text-Image_Coherence",
                        "Fair_Representation",
                    ]
                    custom_id_str = str(custom_id)
                    for trait in traits:
                        if trait.lower().replace("_", "").replace(
                            "-", ""
                        ) in custom_id_str.lower().replace("_", "").replace("-", ""):
                            return trait
                    return None

                recovered_traits = clean_df.loc[
                    clean_df["trait"].isna(), "custom_id"
                ].apply(extract_trait_from_custom_id)
                recovery_count = recovered_traits.notna().sum()
                print(f"Recovered {recovery_count} trait values from custom_id")
                clean_df.loc[
                    clean_df["trait"].isna() & recovered_traits.notna(), "trait"
                ] = recovered_traits.dropna()

        # Report final data quality but DON'T drop rows
        required_cols = ["question_id", "model_id", "trait", "validity"]
        for col in required_cols:
            if col in clean_df.columns:
                null_count = clean_df[col].isna().sum()
                if null_count > 0:
                    print(
                        f"WARNING: {null_count}/{len(clean_df)} rows have missing {col}"
                    )
            else:
                print(f"ERROR: Missing required column: {col}")

        # Only remove rows that are completely unusable (all key fields missing)
        unusable_mask = (
            clean_df["question_id"].isna()
            & clean_df["trait"].isna()
            & clean_df["validity"].isna()
        )
        unusable_count = unusable_mask.sum()

        if unusable_count > 0:
            print(f"Removing {unusable_count} completely unusable rows")
            clean_df = clean_df[~unusable_mask]

        # Basic type normalization for non-null values
        if "question_id" in clean_df.columns:
            clean_df.loc[clean_df["question_id"].notna(), "question_id"] = clean_df.loc[
                clean_df["question_id"].notna(), "question_id"
            ].astype(str)

        if len(clean_df) > 0 and "model_id" in clean_df.columns:
            unique_models = sorted(
                [str(x) for x in clean_df["model_id"].unique() if pd.notna(x)]
            )
            print(f"All model_ids after cleaning: {unique_models}")

        print(
            f"Final result: {len(df)} -> {len(clean_df)} rows ({len(df) - len(clean_df)} removed as completely unusable)"
        )
        print(f"Data completeness:")
        for col in required_cols:
            if col in clean_df.columns:
                complete_pct = (1 - clean_df[col].isna().mean()) * 100
                print(f"  {col}: {complete_pct:.1f}% complete")

        return clean_df
