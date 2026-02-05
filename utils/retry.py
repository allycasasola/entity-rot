"""Utility functions for retry logic with exponential backoff."""

import random
import time
from typing import Callable, Optional, TypeVar

from utils.cost_estimator import TokenUsage

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    base: float = 0.5,
    jitter: float = 0.25,
    token_usage: Optional[TokenUsage] = None,
) -> T:
    """Simple exponential backoff for transient errors.

    If token_usage is provided, accumulates token counts across all attempts
    (including failed ones that returned a response with usage metadata).

    Args:
        fn: Function to call with retries
        retries: Maximum number of retry attempts
        base: Base delay in seconds for exponential backoff
        jitter: Random jitter to add to delay
        token_usage: Optional TokenUsage object to accumulate token counts

    Returns:
        The result of fn() on success

    Raises:
        The last exception if all retries fail
    """
    for i in range(retries):
        try:
            result = fn()
            # Accumulate tokens from successful attempt
            if (
                token_usage is not None
                and hasattr(result, "usage_metadata")
                and result.usage_metadata
            ):
                token_usage.cached_content_token_count += (
                    result.usage_metadata.cached_content_token_count or 0
                )
                token_usage.prompt_token_count += (
                    result.usage_metadata.prompt_token_count or 0
                )
                token_usage.thoughts_token_count += (
                    result.usage_metadata.thoughts_token_count or 0
                )
                token_usage.candidates_token_count += (
                    result.usage_metadata.candidates_token_count or 0
                )
            return result
        except Exception as e:
            # Try to extract token usage from failed response if available
            if (
                token_usage is not None
                and hasattr(e, "response")
                and hasattr(e.response, "usage_metadata")
            ):
                usage = e.response.usage_metadata
                if usage:
                    token_usage.cached_content_token_count += (
                        usage.cached_content_token_count or 0
                    )
                    token_usage.prompt_token_count += usage.prompt_token_count or 0
                    token_usage.thoughts_token_count += usage.thoughts_token_count or 0
                    token_usage.candidates_token_count += (
                        usage.candidates_token_count or 0
                    )
            if i == retries - 1:
                raise
            sleep = base * (2**i) + random.random() * jitter
            print(f"Transient error: {e} — retrying in {sleep:.2f}s")
            time.sleep(sleep)
