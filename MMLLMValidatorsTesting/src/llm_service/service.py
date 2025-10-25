import logging
from typing import List, Dict, Any, Optional
import json
import asyncio

from PIL import Image
import aiohttp

from .config import ConfigManager
from .providers import OpenAIProvider, BaseProvider, NebiusProvider
from .providers.basic_batch import BaseBatchService

from openai import OpenAI


logger = logging.getLogger(__name__)


class OpenAIBatchService(BaseBatchService):
    """
    An implementation of the abstract BaseBatchService that can handle batches for OpenAI
    """

    def __init__(self):
        """
        Initializes the service with the necessary API key.
        """

        logger.info("🔧 Initializing OpenAIBatchService...")
        self.config = ConfigManager()
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=self.config.openai_api_key)
        logger.info("✅ OpenAI client initialized.")

    def submit_batch(self, filepath: str, name: Optional[str] = None) -> str:
        """
        Uploads a file and creates a new batch job on OpenAI
        """
        logger.info(f"Submitting batch file: {filepath}...")
        try:
            with open(filepath, "rb") as f:
                batch_input_file = self.client.files.create(file=f, purpose="batch")
            logger.debug(f"Uploaded file ID {batch_input_file.id}")

            # creating batch job based on the uploaded file
            description = name or f"Batch job for {filepath}"
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/responses",
                completion_window="24h",
                metadata={"description": description},
            )

            logger.info(f"🚀 Submitted batch job with ID: {batch.id}")
            return batch.id
        except Exception as e:
            logger.error(f"❌ Batch submission failed: {e}")
            raise

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieves the status and metadata of a specific batch job from OpenAI
        """
        logger.info(f"Checking status for batch ID: {batch_id}...")
        try:
            batch = self.client.batches.retrieve(batch_id)

            status_info = {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "request_counts": batch.request_counts,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }
            logger.info(f"Status for batch {batch_id}: {batch.status.upper()}")

            return status_info
        except Exception as e:
            logger.error(f"❌ Failed to retrieve status for batch {batch_id}: {e}")
            raise

    def download_batch_results(self, batch_id: str, output_path: str) -> bool:
        """Download complete batch results."""
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.status != "completed":
                logger.warning(f"Batch {batch_id} status: {batch.status}")
                return False

            if not batch.output_file_id:
                logger.error(f"No output file for batch {batch_id}")
                return False

            result_content = self.client.files.content(batch.output_file_id)

            with open(output_path, "wb") as f:
                f.write(result_content.content)

            logger.info(f"📥 Downloaded results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

    def cancel_batch(self, batch_id: str) -> bool:
        try:
            self.client.batches.cancel(batch_id)
            return True

        except Exception as e:
            logger.error(f"❌ Cancel batch failed: {e}")
            return False

    def delete_batch(self, batch_id: str) -> bool:
        """Deletes batch files if the batch is completed/failed/canceled"""
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.status == "in_progress":
                logger.warning(
                    f"⚠️ Cannot delete a batch (batch id {batch.id}) that is being currently processed. Cancel it first."
                )
                return False

            if batch.input_file_id:
                self.client.files.delete(batch.input_file_id)
                logger.info(f"🗑️ Deleted input file: {batch.input_file_id}")

            if batch.output_file_id:
                self.client.files.delete(batch.output_file_id)
                logger.info(f"🗑️ Deleted output file: {batch.output_file_id}")

            if batch.error_file_id:
                self.client.files.delete(batch.error_file_id)
                logger.info(f"🗑️ Deleted error file: {batch.error_file_id}")

            logger.info(f"✅ All files related to batch: {batch_id} have been deleted.")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete batch {batch_id}: {e}")
            return False


class NebiusBatchService(BaseBatchService):
    """
    An implementation of the abstract BaseBatchService that can handle batches for OpenAI
    """

    def __init__(self):
        """
        Initializes the service with the necessary API key.
        """

        logger.info("🔧 Initializing OpenAIBatchService...")
        self.config = ConfigManager()
        if not self.config.nebius_api_key:
            raise ValueError("Nebius API key is required.")
        self.client = OpenAI(
            base_url=self.config.nebius_base_url, api_key=self.config.nebius_api_key
        )
        logger.info("✅ Nebius client initialized.")

    def submit_batch(self, filepath: str, name: Optional[str] = None) -> str:
        """
        Uploads a file and creates a new batch job on Nebius
        """
        logger.info(f"Submitting batch file: {filepath}...")
        try:
            with open(filepath, "rb") as f:
                batch_input_file = self.client.files.create(file=f, purpose="batch")
            logger.debug(f"Uploaded file ID {batch_input_file.id}")

            # creating batch job based on the uploaded file
            description = name or f"Batch job for {filepath}"
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": description},
            )

            logger.info(f"🚀 Submitted batch job with ID: {batch.id}")
            return batch.id
        except Exception as e:
            logger.error(f"❌ Batch submission failed: {e}")
            raise

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieves the status and metadata of a specific batch job from Nebius
        """
        logger.info(f"Checking status for batch ID: {batch_id}...")
        try:
            batch = self.client.batches.retrieve(batch_id)

            status_info = {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "request_counts": batch.request_counts,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }
            logger.info(f"Status for batch {batch_id}: {batch.status.upper()}")

            return status_info
        except Exception as e:
            logger.error(f"❌ Failed to retrieve status for batch {batch_id}: {e}")
            raise

    def download_batch_results(self, batch_id: str, output_path: str) -> bool:
        """Download complete batch results."""
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.status != "completed":
                logger.warning(f"Batch {batch_id} status: {batch.status}")
                return False

            if not batch.output_file_id:
                logger.error(f"No output file for batch {batch_id}")
                return False

            result_content = self.client.files.content(batch.output_file_id)

            with open(output_path, "wb") as f:
                f.write(result_content.content)

            logger.info(f"📥 Downloaded results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

    def cancel_batch(self, batch_id: str) -> bool:
        try:
            self.client.batches.cancel(batch_id)
            return True

        except Exception as e:
            logger.error(f"❌ Cancel batch failed: {e}")
            return False

    def delete_batch(self, batch_id: str) -> bool:
        """Deletes batch files if the batch is completed/failed/canceled"""
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.status == "in_progress":
                logger.warning(
                    f"⚠️ Cannot delete a batch (batch id {batch.id}) that is being currently processed. Cancel it first."
                )
                return False

            if batch.input_file_id:
                self.client.files.delete(batch.input_file_id)
                logger.info(f"🗑️ Deleted input file: {batch.input_file_id}")

            if batch.output_file_id:
                self.client.files.delete(batch.output_file_id)
                logger.info(f"🗑️ Deleted output file: {batch.output_file_id}")

            if batch.error_file_id:
                self.client.files.delete(batch.error_file_id)
                logger.info(f"🗑️ Deleted error file: {batch.error_file_id}")

            logger.info(f"✅ All files related to batch: {batch_id} have been deleted.")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete batch {batch_id}: {e}")
            return False
