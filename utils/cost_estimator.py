# Utility functions for estimating costs of LLM calls
from pydantic import BaseModel


class TokenUsage(BaseModel):
    cached_content_token_count: int = 0
    prompt_token_count: int = 0
    thoughts_token_count: int = 0
    candidates_token_count: int = 0


class Cost(BaseModel):
    cached_content_cost: float = 0.0
    prompt_cost: float = 0.0
    thoughts_cost: float = 0.0
    candidates_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.cached_content_cost + self.prompt_cost + self.thoughts_cost + self.candidates_cost


# Pricing constants (per 1M tokens)
PRICING = {
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
        "cache_input": 0.01,
        "cache_storage_per_hour": 1.00,
    },
    "gemini-2.5-flash-lite-preview-09-2025": {
        "input": 0.10,
        "output": 0.40,
        "cache_input": 0.01,
        "cache_storage_per_hour": 1.00,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 0.30,
        "cache_input": 0.01875,
        "cache_storage_per_hour": 1.00,
    },
}

DEFAULT_PRICING = PRICING["gemini-2.5-flash-lite"]


def get_pricing(model_name: str) -> dict:
    """Get pricing for a model, falling back to default if not found."""
    return PRICING.get(model_name, DEFAULT_PRICING)


def estimated_token_usage_for_extraction(
    model_name: str, n_city_words: int, n_sections: int, instructions: str
) -> TokenUsage:
    """Estimate total token usage for extraction task."""
    # System instruction tokens (counted once due to Gemini's automatic caching)
    instruction_tokens = int(len(instructions.split()) * 1.4)

    # Assume 1.4 tokens per word for content input
    content_tokens = int(n_city_words * 1.4)

    # Total prompt tokens = instruction tokens (once) + content tokens
    prompt_tokens = instruction_tokens + content_tokens

    # Assume each section generates about 150 output tokens (10 entities)
    output_tokens = n_sections * 150

    return TokenUsage(
        cached_content_token_count=0,  # Not using caching
        prompt_token_count=prompt_tokens,
        thoughts_token_count=0,
        candidates_token_count=output_tokens,
    )


def cost_from_token_usage(model_name: str, token_usage: TokenUsage) -> Cost:
    """Calculate cost from token usage.

    Note: cached_content_cost only uses the cache_input rate (cost per cached token read).
    Storage costs are charged per hour separately, not per token.
    """
    pricing = get_pricing(model_name)

    return Cost(
        cached_content_cost=token_usage.cached_content_token_count * pricing["cache_input"] / 1_000_000,
        prompt_cost=token_usage.prompt_token_count * pricing["input"] / 1_000_000,
        thoughts_cost=token_usage.thoughts_token_count * pricing["output"] / 1_000_000,
        candidates_cost=token_usage.candidates_token_count * pricing["output"] / 1_000_000,
    )
