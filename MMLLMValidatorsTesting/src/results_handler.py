#!/usr/bin/env python3
"""
Utilities for loading, cleaning, and saving batch evaluation results.

- Reads provider JSONL outputs and normalizes them into rows.
- Handles "combined trait" responses by expanding a single request into N rows.
- Attempts recovery for missing fields (question_id, trait, validity).
- Saves both raw and cleaned CSVs.

The outputs are:
- raw_evaluation_results.csv
- clean_evaluation_results.csv
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


class ResultsHandler:
    """Handles saving, cleaning, and managing evaluation result files."""

    # request-<ModelAlias>-<trait_name>-<question_id>-<timestamp>
    CID_RX = re.compile(
        r"^request-(?P<model>[^-]+)-(?P<trait>[^-]+)-(?P<qid>\d+)-(?P<ts>\d+)$"
    )

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.raw_results_path = self.results_dir / "raw_evaluation_results.csv"
        self.clean_results_path = self.results_dir / "clean_evaluation_results.csv"

    # ---------------- Loading & Parsing ----------------

    def load_batch_results_jsonl(self, jsonl_path: Path) -> pd.DataFrame:
        """
        Load Batch API JSONL output and normalize into a DataFrame.
        Handles flattening of 'combined' trait results (1 request -> N rows).
        """
        rows: List[Dict[str, Any]] = []

        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            logger.error("JSONL file not found: %s", jsonl_path)
            return pd.DataFrame()

        with jsonl_path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as e:
                    # Preserve basic debug signal; keep parity with original fields
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

                rows.extend(self._parse_batch_object(obj, ln))

        df = pd.DataFrame(rows)

        # Preferred column order; keep extras at the end
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
        ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        return df[ordered]

    @staticmethod
    def _extract_output_text(response_obj: Dict[str, Any]) -> Optional[str]:
        """
        Extract the inner JSON string emitted by the model.

        Supports:
        - OpenAI "Responses" API: body.output[i].content[j] with type="output_text"
        - Standard/Nebius Chat API: response.choices[0].message.content
        """
        # Normalize: prefer nested 'body', otherwise treat the object as the body
        body = response_obj.get("body") or response_obj

        # OpenAI "Responses" API style
        out = body.get("output", [])
        if isinstance(out, list):
            for msg in out:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if part.get("type") == "output_text":
                        return part.get("text", "")

        # Chat Completions style (OpenAI / Nebius)
        choices = body.get("choices", [])
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return str(content)

        return None

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strip markdown code fences (e.g., ```json ... ```), returning plain JSON text."""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        if t.startswith("```"):
            # Remove opening fence line
            parts = t.split("\n", 1)
            if len(parts) > 1:
                t = parts[1]
            # Remove closing fence or trailing backticks
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
        Returns a LIST of normalized rows to handle combined traits (one-to-many).
        """
        custom_id = obj.get("custom_id", "")
        response = obj.get("response", {}) or {}

        status_code = response.get("status_code")
        # Nebius often omits status code but includes "choices"; treat as 200
        if status_code is None and "choices" in response:
            status_code = 200

        body = response.get("body") or response
        model_reported = body.get("model")
        created_at = body.get("created_at")
        req_status = body.get("status")
        error = obj.get("error")
        request_id = response.get("request_id") or body.get("id")

        # Derive fields from custom_id
        trait_hint = qid = ts = None
        model_alias: Optional[str] = None
        m = self.CID_RX.search(custom_id) if isinstance(custom_id, str) else None
        if m:
            model_alias = m.group("model")
            trait_hint = m.group("trait")
            qid = m.group("qid")
            ts = m.group("ts")

        base_row: Dict[str, Any] = {
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
            "trait": trait_hint,  # Fallback trait
            "validity": None,
            "reasoning": None,
            "_raw_text": None,
        }

        # Extract model output text
        raw_text = self._extract_output_text(response)
        if not raw_text:
            return [base_row]

        # Try parsing inner JSON payload
        cleaned_text = self._clean_json_text(raw_text)
        try:
            inner_json = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Keep the unparsed raw payload to help recovery
            row = base_row.copy()
            row["_raw_text"] = raw_text
            return [row]

        # Determine if this is a single-object or a collection of traits
        items_to_process: List[Dict[str, Any]] = []
        if isinstance(inner_json, list):
            items_to_process = [it for it in inner_json if isinstance(it, dict)]
        elif isinstance(inner_json, dict):
            if isinstance(inner_json.get("traits_output"), list):
                items_to_process = [it for it in inner_json["traits_output"] if isinstance(it, dict)]
            elif isinstance(inner_json.get("traits"), list):
                items_to_process = [it for it in inner_json["traits"] if isinstance(it, dict)]
            else:
                items_to_process = [inner_json]

        if not items_to_process:
            return [base_row]

        rows: List[Dict[str, Any]] = []
        for item in items_to_process:
            new_row = base_row.copy()
            json_trait = item.get("trait")
            new_row["trait"] = json_trait if json_trait else trait_hint
            new_row["validity"] = self._coerce_bool(item.get("validity"))
            new_row["reasoning"] = item.get("reasoning")
            rows.append(new_row)

        return rows

    # ---------------- Saving ----------------

    def save_raw_results(self, df: pd.DataFrame) -> None:
        """Save the raw DataFrame of results to CSV."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.raw_results_path, index=False, quoting=csv.QUOTE_ALL)
        ok = self.raw_results_path.exists()
        size = self.raw_results_path.stat().st_size if ok else 0
        logger.info("Raw results saved to %s (%s, %d bytes)", self.raw_results_path, "OK" if ok else "MISSING", size)

    def save_clean_results(self, df: pd.DataFrame) -> None:
        """Save the cleaned DataFrame to CSV."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.clean_results_path, index=False, quoting=csv.QUOTE_ALL)
        ok = self.clean_results_path.exists()
        size = self.clean_results_path.stat().st_size if ok else 0
        logger.info(
            "Clean results saved to %s (%s, %d bytes)",
            self.clean_results_path,
            "OK" if ok else "MISSING",
            size,
        )

    # ---------------- Cleaning ----------------

    def clean_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out rows with errors, normalizes columns, and attempts data recovery.
        """
        clean_df = df.copy()
        logger.info("Starting cleaning with %d rows", len(clean_df))

        # 1) Question ID recovery (from custom_id) if null
        if "question_id" in clean_df.columns:
            null_qid_mask = clean_df["question_id"].isna()
            if null_qid_mask.any():
                logger.info("Recovering %d missing question_id value(s) from custom_id", int(null_qid_mask.sum()))

                def extract_qid_fallback(custom_id: Any) -> Optional[str]:
                    if pd.isna(custom_id) or custom_id == "":
                        return None
                    patterns = [r"question[_-](\d+)", r"qid[_-](\d+)", r"id[_-](\d+)", r"-(\d+)-"]
                    s = str(custom_id)
                    for pattern in patterns:
                        m = re.search(pattern, s, re.IGNORECASE)
                        if m:
                            return m.group(1)
                    # Last resort: grab the longest number sequence
                    numbers = re.findall(r"\d+", s)
                    return max(numbers, key=len) if numbers else None

                clean_df.loc[null_qid_mask, "question_id"] = clean_df.loc[null_qid_mask, "custom_id"].apply(
                    extract_qid_fallback
                )

            clean_df["question_id"] = clean_df["question_id"].astype(str, errors="ignore")

        # 2) Validity coercion & heuristic recovery from raw text
        if "validity" in clean_df.columns:
            clean_df["validity"] = clean_df["validity"].apply(self._coerce_bool)

            missing_validity = clean_df["validity"].isna() & clean_df.get("_raw_text", pd.Series([None] * len(clean_df))).notna()
            if missing_validity.any():
                logger.info("Recovering %d validity value(s) from raw text", int(missing_validity.sum()))

                def recover_validity(text: Any) -> Optional[bool]:
                    t = str(text).lower()
                    if "true" in t and "false" not in t:
                        return True
                    if "false" in t and "true" not in t:
                        return False
                    return None

                recovered = clean_df.loc[missing_validity, "_raw_text"].apply(recover_validity)
                clean_df.loc[missing_validity, "validity"] = recovered

        # 3) Trait recovery from custom_id
        if "trait" in clean_df.columns:
            missing_trait = clean_df["trait"].isna() & clean_df["custom_id"].notna()
            if missing_trait.any():
                logger.info("Recovering %d trait value(s) from custom_id", int(missing_trait.sum()))
                traits_map = {
                    "functional_relevance": "Functional Relevance",
                    "visual_clarity": "Visual Clarity",
                    "technical_quality": "Technical Quality",
                    "standard_presentation": "Standard Presentation",
                    "text-image_coherence": "Text-Image Coherence",
                    "fair_representation": "Fair Representation",
                }

                def recover_trait(cid: Any) -> Optional[str]:
                    c_lower = str(cid).lower().replace("-", "_")
                    for key, val in traits_map.items():
                        if key in c_lower:
                            return val
                    return None

                clean_df.loc[missing_trait, "trait"] = clean_df.loc[missing_trait, "custom_id"].apply(recover_trait)

        # 4) Model ID standardization
        if "model_id" in clean_df.columns:
            clean_df["model_id"] = clean_df["model_id"].fillna(clean_df.get("model_reported"))
            mask_mini = clean_df["model_id"].astype(str).str.startswith("gpt-4o-mini", na=False)
            clean_df.loc[mask_mini, "model_id"] = "GPT4oMini"

            mask_4o = clean_df["model_id"].astype(str).str.startswith("gpt-4o", na=False) & ~mask_mini
            clean_df.loc[mask_4o, "model_id"] = "GPT4o"

        # 5) Report missing required fields and filter to rows with a validity result
        required = ["question_id", "trait", "validity"]
        for col in required:
            if col in clean_df.columns:
                missing = int(clean_df[col].isna().sum())
                if missing > 0:
                    logger.warning("%d row(s) missing '%s'", missing, col)

        clean_df = clean_df[clean_df["validity"].notna()].copy()
        logger.info("Final clean count: %d", len(clean_df))
        return clean_df
