import os
import json
from typing import Dict, Any, Optional
from filelock import FileLock, Timeout
import logging

logger = logging.getLogger(__name__)


def is_checkpoint(qid: str, trait: str, modelid: str, out_file: str) -> bool:
    """
    Checks if a result for a given question id, trait, and model id is already present in the checkpoint file.
    Args:
        qid: The question ID to check
        trait: the trait to check
        modelid: the model id to check
        out_file: Path to the checkpoint file.
    Returns:
        True if entry exists, False otherwise.
    """
    if not os.path.exists(out_file):
        return False

    lock_path = out_file + ".lock"
    try:
        with FileLock(lock_path, timeout=5):
            with open(out_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if (
                            str(entry.get("question_id")) == str(qid)
                            and entry.get("trait") == trait
                            and entry.get("model_id") == modelid
                        ):
                            return True
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping malformed JSON line in {out_file}: {line.strip()}"
                        )
                        continue
    except Timeout:
        logger.warning(f"Could not acquire lock to read {out_file} in time.")
    except (OSError, IOError) as e:
        logger.warning(f"Could not perform resume check on {out_file}: {e}")

    return False


def checkpoint(output: Dict[str, Any], out_file: str) -> bool:
    """
    Appends an output dictionary to a JSONL file if it's not already present.
    Args:
        output: Dictionary containing question id, trait, model id, and other results.
        out_file: Path to the checkpoint file.
    """
    if (
        not isinstance(output, dict)
        or "qid" not in output
        or "trait" not in output
        or "model_id" not in output
    ):
        logger.error(f"Invalid output format for checkpointing: {output}")
        return False

    qid = str(output["qid"])
    trait = str(output["trait"])
    modelid = str(output["model_id"])
    lock_path = out_file + ".lock"

    try:
        with FileLock(lock_path, timeout=10):
            already_exists = False
            if os.path.exists(out_file):
                with open(out_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            if (
                                str(entry.get("qid")) == qid
                                and entry.get("trait") == trait
                                and entry.get("model_id") == modelid
                            ):
                                already_exists = True
                                break
                        except json.JSONDecodeError:
                            continue
            if already_exists:
                logger.info(
                    f"Skipping existing entry found inside lock: question_id={qid}, trait={trait}, model={modelid}"
                )
                return False

            with open(out_file, "a") as f:
                json.dump(output, f)
                f.write("\n")
            logger.info(
                f"Checkpointed: question_id={qid}, trait={trait}, model={modelid}"
            )
            return True
    except Timeout:
        logger.error(
            f"Could not acquire lock to write checkpoint for question_id={qid}, trait={trait}, model={modelid}"
        )

    except (OSError, IOError, TypeError) as e:
        logger.error(
            f"Failed to write checkpoint for question_id={qid}, trait={trait}, model={modelid}"
        )
    return False


def get_checkpointed_result(
    qid: str, trait: str, modelid: str, out_file: str
) -> Optional[Dict[str, Any]]:
    """
    Finds a result in the checkpoint file and returns it as a dictionary.
    Returns None if no matching result is found.
    """

    if not os.path.exists(out_file):
        return None

    lock_path = out_file + ".lock"
    try:
        with FileLock(lock_path, timeout=5):
            with open(out_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    entry = json.loads(line)
                    if (
                        str(entry.get("qid")) == qid
                        and entry.get("trait") == trait
                        and modelid == entry.get("model_id")
                    ):
                        return entry
    except (Timeout, IOError):
        return None

    return None
