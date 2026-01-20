# Utility functions for estimating costs of LLM calls

def calculate_input_price_for_extraction(model_name: str, n_city_words: int) -> float:
    """Calculate the price of input tokens for a given model for the extract_entities task."""
    if model_name in ["gemini-2.5-flash-lite-preview-09-2025", "gemini-2.5-flash-lite"]:
        # Costs $0.10 per 1M tokens
        # Assume 1.3 tokens per word
        return n_city_words * 1.3 * (0.10 / 1_000_000)
    elif model_name == "gemini-2.5-flash":
        # Costs $0.075 per 1M tokens (for prompts <= 128k)
        # Assume 1.3 tokens per word
        return n_city_words * 1.3 * (0.075 / 1_000_000)
    else:
        # Default fallback estimate
        return n_city_words * 1.3 * (0.10 / 1_000_000)

def calculate_output_price_for_extraction(model_name: str, num_sections: int) -> float:
    """Calculate the price of output tokens for a given model for the extract_entities task.
    Provides a lofty upper bound on the cost, as it assumes each section generates at most 10 entities."""
    if model_name in ["gemini-2.5-flash-lite-preview-09-2025", "gemini-2.5-flash-lite"]:
        # Suppose each section generates at most 10 entities (which is about 150 tokens)
        # Costs $0.40 per 1M tokens
        return num_sections * 150 * (0.40 / 1_000_000)
    elif model_name == "gemini-2.5-flash":
        # Costs $0.30 per 1M tokens (for prompts <= 128k)
        return num_sections * 150 * (0.30 / 1_000_000)
    else:
        # Default fallback estimate
        return num_sections * 150 * (0.40 / 1_000_000)

def calculate_context_caching_price_for_extraction(model_name: str, instructions: str) -> float:
    """Calculate the price of context caching for a given model for the extract_entities task."""
    if model_name in ["gemini-2.5-flash-lite-preview-09-2025", "gemini-2.5-flash-lite"]:
        # Costs $0.01 per 1M tokens + $1.00 per 1M tokens per hour for storage
        # Assume 1.3 tokens per word
        # Assume it takes 1 hour to run the extraction task? (TO-DO: Not really sure about this estimate)
        num_tokens = len(instructions.split()) * 1.3
        return num_tokens * (0.01 / 1_000_000) + num_tokens * (1.00 / 1_000_000)
    elif model_name == "gemini-2.5-flash":
        # Costs $0.01875 per 1M tokens + $1.00 per 1M tokens per hour for storage
        num_tokens = len(instructions.split()) * 1.3
        return num_tokens * (0.01875 / 1_000_000) + num_tokens * (1.00 / 1_000_000)
    else:
        # Default fallback estimate
        num_tokens = len(instructions.split()) * 1.3
        return num_tokens * (0.01 / 1_000_000) + num_tokens * (1.00 / 1_000_000)

