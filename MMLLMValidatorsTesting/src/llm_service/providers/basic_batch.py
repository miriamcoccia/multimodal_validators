"""
Base class from which we can inherit to create batch services that point to specific API providers (OpenAI, or Nebius)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class BaseBatchService(ABC):
    """
    Abstract base class for a batch processing service.
    Defines the common interface for submitting, monitoring, and retrieving batch jobs.
    """

    @abstractmethod
    def submit_batch(self, filepath: str, name: Optional[str] = None) -> str:
        """
        Submits a batch job for processing.

        Args:
            filepath: The path to the file containing batch requests.
            name: An optional name or description for the batch job.

        Returns:
            A unique identifier for the submbitted batch job.
        """
        pass

    @abstractmethod
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Retrieves the status of a specific batch job."""
        pass

    @abstractmethod
    def download_batch_results(self, batch_id: str, output_path: str) -> bool:
        """Downloads the resutls of a completed batch job."""
        pass

    @abstractmethod
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancels an in-progress batch job."""
        pass

    @abstractmethod
    def delete_batch(self, batch_id: str) -> bool:
        """Deletes a batch job and its associated files."""
        pass

    def write_jsonl_file(self, requests: List[Dict[str, Any]], filepath: str) -> None:
        """
        Writes batch requests to a JSONL file that will be used by the different batch services.

        Args:
            requests: List of request dictionaries
            filepath: Path where the JSONL file should be written
        """

        with open(filepath, "w+", encoding="utf-8") as f:
            for request in requests:
                json.dump(request, f)
                f.write("\n")
