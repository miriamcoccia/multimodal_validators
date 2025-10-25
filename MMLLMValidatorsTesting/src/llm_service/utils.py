import io
import json
import base64
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, List, Awaitable
from functools import wraps

from PIL import Image
import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _extract_core_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the core fields needed for ValidationSchema, ignoring extras."""
    if not isinstance(data, dict):
        return {}

    core_keys = {"trait", "validity", "reasoning"}
    return {k: v for k, v in data.items() if k in core_keys}


def image_to_data_url(pil_image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL image into a base64 data URL."""
    buf = io.BytesIO()
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")

    pil_image.save(buf, format=format)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/{format.lower()};base64,{b64_str}"


def write_jsonl_file(requests: List[Dict[str, Any]], filepath: str) -> None:
    with open(filepath, "a+", encoding="utf-8") as f:
        for request in requests:
            json.dump(request, f)
            f.write("\n")
    logger.info(f"📝 Wrote {len(requests)} requests to {filepath}")


def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0
) -> Callable:
    """Decorator for automatic retry with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., Awaitable[Optional[T]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Optional[T]:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    start_time = asyncio.get_event_loop().time()
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                    elapsed = asyncio.get_event_loop().time() - start_time
                    logger.debug(
                        f"Request completed in {elapsed:.2f}s on attempt {attempt + 1}"
                    )
                    return result

                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        retry_after = (
                            dict(e.headers).get("Retry-After", "60")
                            if e.headers
                            else "60"
                        )
                        delay = float(retry_after)
                        logger.warning(
                            f"Rate limited. Waiting {delay}s as requested by server"
                        )
                        await asyncio.sleep(delay)
                        continue
                    last_exception = e

                except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries ({max_retries}) exceeded: {e}")
                        break

                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"Non-retryable error on attempt {attempt + 1}: {e}")
                    last_exception = e
                    break

            if last_exception:
                logger.error(f"All retry attempts failed. Last error: {last_exception}")
            return None

        return wrapper

    return decorator
