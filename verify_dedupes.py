"""
Verify that fuzzy-matched entity groups are correctly deduplicated using Gemini LLM.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm

from utils.cost_estimator import TokenUsage, Cost, cost_from_token_usage
from utils.retry import with_retries

load_dotenv()

SYSTEM_PROMPT = """You are an expert at verifying that entities have been deduplicated correctly. You will be given a list of entities that were deduplicated and grouped together by fuzzy matching. As you know, however, fuzzy matching is not perfect. Therefore, your task is to verify that the entity names belonging to this list indeed refer to the same entity.

Your response must be a list of lists, where each list contains the proper group of entity names that refer to the same entity. If all entity names in the input list refer to the same entity, return a list with a single list containing all the entity names. If the entity names do not refer to the same entity, return a list with multiple lists, where each list contains the entity names that refer to the same entity. Make sure that each entity name is only included in one list and that all entity names are included in the response.

Example 1:
Input:
["County of San Bernardino", "San Bernardino County", "San Bernardino County Code", "Code of San Bernardino County"]

Output:
[["County of San Bernardino", "San Bernardino County", "San Bernardino County Code", "Code of San Bernardino County"]]

Example 2:
Input:
["Ord. No. NS-1102.2", "Ord. No. NS-209.3", "Ord. No. NS-248.2", "Ord. No. NS-248.3"]

Output:
[["Ord. No. NS-1102.2"], ["Ord. No. NS-209.3"], ["Ord. No. NS-248.2"], ["Ord. No. NS-248.3"]]
"""

DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_ROOT_DIR = Path("/Users/allisoncasasola/reglab/entity-rot")


class VerificationMetadata(BaseModel):
    """Metadata for a verification run."""
    city_slug: str
    state_code: str
    model_name: str
    start_time: str
    end_time: str
    total_groups: int
    groups_processed: int
    groups_skipped: int
    groups_split: int
    estimated_token_usage: dict
    estimated_cost: dict
    actual_token_usage: dict
    actual_cost: dict


class OutputEntityGroup(BaseModel):
    representative_name: str
    alternate_names: List[str]
    type: str
    node_ids: List[str]
    occurrence_count: int


def load_fuzzy_matched(file_path: Path) -> List[Dict[str, Any]]:
    """Load fuzzy matched groups from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load existing results from JSONL file."""
    results = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


def save_result_jsonl(result: Dict[str, Any], output_path: Path) -> None:
    """Append a single result to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def save_metadata(metadata: VerificationMetadata, output_path: Path) -> None:
    """Save metadata to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)


def _clean_and_parse_response(response_text: str) -> List[List[str]]:
    """Clean and parse LLM response."""
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:].strip()
    elif response_text.startswith("```"):
        response_text = response_text[3:].strip()
    if response_text.endswith("```"):
        response_text = response_text[:-3].strip()
    return json.loads(response_text)


def estimate_token_usage(groups: List[Dict[str, Any]]) -> TokenUsage:
    """Estimate token usage for verification task."""
    # System instruction tokens (counted once due to Gemini's automatic caching)
    instruction_tokens = int(len(SYSTEM_PROMPT.split()) * 1.4)

    # Estimate input tokens for all groups
    # Each group with >1 entity makes an API call with the system prompt
    total_input_tokens = 0
    groups_needing_verification = 0
    for group in groups:
        entities = group.get("entities", [])
        if len(entities) <= 1:
            continue  # No API call needed for single-entity groups

        groups_needing_verification += 1
        entity_names = [e.get("name", "") for e in entities]
        # Estimate tokens: ~4 chars per token for JSON content
        json_str = json.dumps(entity_names, ensure_ascii=False)
        content_tokens = len(json_str) // 4
        total_input_tokens += content_tokens

    # Total prompt tokens = instruction tokens per API call + content tokens
    prompt_tokens = (instruction_tokens * groups_needing_verification) + total_input_tokens

    # Estimate output tokens (each group returns a list of lists)
    # Assume output is roughly same size as input for entity names
    output_tokens = total_input_tokens

    return TokenUsage(
        cached_content_token_count=0,
        prompt_token_count=prompt_tokens,
        thoughts_token_count=0,
        candidates_token_count=output_tokens,
    )


def verify_group(
    client: genai.Client,
    model_name: str,
    group: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], bool, TokenUsage]:
    """Verify a single fuzzy-matched group using Gemini.

    Returns:
        - List of verified group dicts (each with 'entities' and 'occurrence_count')
        - Boolean indicating if the group was split
        - TokenUsage for this call
    """
    entities = group.get("entities", [])
    entity_names = [e.get("name", "") for e in entities]

    # If only one entity, no verification needed - return as-is
    if len(entity_names) <= 1:
        return ([group], False, TokenUsage())

    # Build prompt content
    content = json.dumps(entity_names, ensure_ascii=False)

    token_usage = TokenUsage()

    try:
        resp = with_retries(
            lambda: client.models.generate_content(
                model=model_name,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            ),
            token_usage=token_usage,
        )

        parsed_response = _clean_and_parse_response(resp.text)

        # Validate response structure
        if not isinstance(parsed_response, list):
            raise ValueError(f"Expected list, got {type(parsed_response)}")

        # Build output groups in the same format as input
        output_groups = []
        for subgroup_names in parsed_response:
            if not isinstance(subgroup_names, list):
                raise ValueError(f"Expected list of lists, got list containing {type(subgroup_names)}")

            # Match entity names back to full entity dicts
            subgroup_entities = []
            for entity_name in subgroup_names:
                for entity in entities:
                    if entity.get("name") == entity_name:
                        subgroup_entities.append(entity)
                        break

            # Calculate occurrence count for this subgroup
            occurrence_count = sum(len(e.get("node_ids", [])) for e in subgroup_entities)

            output_groups.append({
                "entities": subgroup_entities,
                "occurrence_count": occurrence_count,
            })

        was_split = len(output_groups) > 1
        return (output_groups, was_split, token_usage)

    except Exception as e:
        print(f"\nError verifying group: {e}")
        # Return original group as-is on error
        return ([group], False, token_usage)


def main():
    parser = argparse.ArgumentParser(
        description="Verify fuzzy-matched entity deduplication using Gemini LLM"
    )
    parser.add_argument(
        "city",
        help="City to process in format 'city_slug:state_code' (e.g., 'santa_clara_county:ca')",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=f"Root directory for inputs/outputs (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Override input file path (default: outputs/deduplication/fuzzy-matches/{city}/fuzzy_matched.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        default=False,
        help="Run non-interactively (auto-confirm prompts)",
    )

    args = parser.parse_args()

    # Parse city specification
    parts = args.city.split(":")
    if len(parts) != 2:
        print(f"Error: Invalid city specification '{args.city}'")
        print("Expected format: 'city_slug:state_code'")
        return

    city_slug, state_code = parts
    city_name = f"{city_slug}_{state_code}"

    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = (
            args.root_dir
            / "outputs"
            / "deduplication"
            / "fuzzy-matches"
            / city_name
            / "fuzzy_matched.json"
        )

    # Validate input file
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    # Set output paths
    output_dir = args.root_dir / "outputs" / "deduplication" / "verification" / city_name
    output_jsonl = output_dir / "verified_groups.jsonl"
    metadata_file = output_dir / "inference_metadata.json"

    print(f"Processing: {city_slug} ({state_code})")
    print("=" * 80)

    # Load fuzzy matched groups
    print(f"Loading fuzzy matched groups from {input_file}...")
    groups = load_fuzzy_matched(input_file)
    print(f"Loaded {len(groups)} groups")

    # Estimate token usage and cost
    estimated_token_usage = estimate_token_usage(groups)
    estimated_cost = cost_from_token_usage(args.model, estimated_token_usage)

    print(f"\nEstimated token usage:")
    print(f"  - Prompt: {estimated_token_usage.prompt_token_count:,}")
    print(f"  - Output: {estimated_token_usage.candidates_token_count:,}")
    print(f"\nEstimated cost: ${estimated_cost.total_cost:.4f}")

    if not args.auto_confirm:
        resp = input("\nContinue with verification? (y/n): ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    # Check for existing results
    processed_indices = set()
    if output_jsonl.exists():
        existing_results = load_existing_results_jsonl(output_jsonl)
        print(f"Found existing results for {len(existing_results)} groups.")

        resp = input("Choose an action: 1) Overwrite, 2) Resume, 3) Skip (1/2/3): ").strip()
        if resp == "1":
            print("Overwriting existing results.")
            output_jsonl.unlink()
        elif resp == "2":
            print("Resuming from existing results.")
            processed_indices = set(range(len(existing_results)))
        elif resp == "3":
            print("Skipping.")
            return
        else:
            print("Invalid response. Skipping.")
            return

    # Set up Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Track metrics
    start_time = datetime.now()
    actual_token_usage = TokenUsage()
    groups_processed = 0
    groups_split = 0

    to_process = len(groups) - len(processed_indices)
    print(f"\nVerifying {to_process} groups...")

    try:
        with tqdm(total=to_process, desc="Verifying groups") as pbar:

            all_output_groups = []

            for i, group in enumerate(groups):
                if i in processed_indices:
                    continue

                output_groups, was_split, token_usage = verify_group(
                    client=client,
                    model_name=args.model,
                    group=group,
                )

                all_output_groups.extend(output_groups)

                # Accumulate token usage
                actual_token_usage.prompt_token_count += token_usage.prompt_token_count
                actual_token_usage.candidates_token_count += token_usage.candidates_token_count
                actual_token_usage.cached_content_token_count += token_usage.cached_content_token_count
                actual_token_usage.thoughts_token_count += token_usage.thoughts_token_count

                groups_processed += 1
                if was_split:
                    groups_split += 1

                pbar.update(1)

        # Calculate actual cost
        actual_cost = cost_from_token_usage(args.model, actual_token_usage)

        # Save metadata
        metadata = VerificationMetadata(
            city_slug=city_slug,
            state_code=state_code,
            model_name=args.model,
            start_time=start_time.isoformat(),
            end_time=datetime.now().isoformat(),
            total_groups=len(groups),
            groups_processed=groups_processed,
            groups_skipped=len(processed_indices),
            groups_split=groups_split,
            estimated_token_usage=estimated_token_usage.model_dump(),
            estimated_cost=estimated_cost.model_dump(),
            actual_token_usage=actual_token_usage.model_dump(),
            actual_cost=actual_cost.model_dump(),
        )
        save_metadata(metadata, metadata_file)

        # Format all_output_groups so that each line in output JSONL is an OutputEntityGroup
        for group in all_output_groups:
            entities = group["entities"]

            # Get the representative name as the entity with the most node_ids
            representative_entity = max(entities, key=lambda x: len(x.get("node_ids", [])))
            representative_name = representative_entity.get("name", "")
            alternate_names = [e.get("name", "") for e in entities if e.get("name") != representative_name]

            # For each type, count how many entities have that type
            type_counts = {
                "organization": 0,
                "location": 0,
                "role": 0,
                "event": 0,
                "legal document": 0,
                "other": 0,
            }
            for entity in entities:
                entity_type = entity.get("type", "other")
                if entity_type in type_counts:
                    type_counts[entity_type] += 1
                else:
                    type_counts["other"] += 1

            # Flatten all node_ids from all entities
            all_node_ids = []
            for entity in entities:
                all_node_ids.extend(entity.get("node_ids", []))

            output_group = OutputEntityGroup(
                representative_name=representative_name,
                alternate_names=alternate_names,
                type=max(type_counts, key=type_counts.get),
                node_ids=all_node_ids,
                occurrence_count=group["occurrence_count"],
            )
            save_result_jsonl(output_group.model_dump(), output_jsonl)


        print(f"\nResults saved to: {output_jsonl}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"\nVerified {groups_processed} groups")
        print(f"Groups split: {groups_split}")
        print(f"Total output groups: {len(all_output_groups)}")
        print(f"\nActual token usage:")
        print(f"  - Cached content: {actual_token_usage.cached_content_token_count:,}")
        print(f"  - Prompt: {actual_token_usage.prompt_token_count:,}")
        print(f"  - Output: {actual_token_usage.candidates_token_count:,}")
        print(f"\nActual cost: ${actual_cost.total_cost:.4f}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise e

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
