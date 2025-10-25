from typing import List, Dict, Any, Optional
import logging

from src.trait_agent import TraitAgent
from src.llm_service.service import OpenAIBatchService, NebiusBatchService
from src.science_qa import ScienceQA

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Manages a team of TraitAgents to evaluate a ScienceQA problem.
    This version runs agents sequentially to ensure stability on systems
    with computation bottlenecks (e.g., CPU-only or driver issues).
    The evaluations can be either performed in real-time or via batches.
    """

    def __init__(
        self,
        trait_names: List[str],
        checkpoint_file: Optional[str] = None,
    ):
        logger.info("🎶 Initializing Sequential Orchestrator...")

        self.openai_batch_service = OpenAIBatchService()
        self.nebius_batch_service = NebiusBatchService()

        self.all_agents = [TraitAgent(name) for name in trait_names]
        self.checkpoint_file = checkpoint_file

        logger.info(f"✅ Orchestrator created with {len(self.all_agents)} agents.")

    async def prepare_batch_requests(
        self, question_data: ScienceQA, provider: str, model_id: str
    ) -> tuple[list[Dict[str, Any]], list[str]]:
        """
        Prepares batch requests for a single question by collecting requests from all agents.
        It handles image formats based on the provider.
        Returns list of request dictionaries ready for JSONL batch file.
        """

        pil_images = []
        image_file_ids = []

        if provider == "openai":
            image_file_ids = await question_data.upload_images_to_openai(
                self.openai_batch_service.client
            )

        else:
            pil_images = await question_data.load_images()

        requests = []
        qid = question_data.id
        total_agents = len(self.all_agents)
        logger.info(
            f"🔧 Orchestrator preparing batch requests for QID {qid} with {total_agents} agents..."
        )
        for _, agent in enumerate(self.all_agents):
            agent_name = agent.trait_name
            logger.debug(f"Preparing request for agent '{agent_name}' (QID {qid})")

            try:
                request = agent.prepare_single_request(
                    question_data=question_data,
                    provider=provider,
                    model_id=model_id,
                    pil_images=pil_images,
                    image_file_ids=image_file_ids,
                )

                if request is not None:
                    request["_provider"] = provider
                    requests.append(request)
                else:
                    logger.warning(
                        f"Agent '{agent_name}' returned None for batch request preparation"
                    )
            except Exception as e:
                logger.error(
                    f"Error preparing batch request for agent '{agent_name}': {e}"
                )
                continue

        logger.info(
            f"📦 Orchestrator prepared {len(requests)} batch requests for QID {qid}"
        )
        return requests, image_file_ids
