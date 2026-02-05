"""
Analyze entities using LLM with web search to assess existence.
- Uses Google Search grounding via the Google GenAI SDK.
- Takes verified deduplicated entities and analyzes them in batches.
- Looks up section content from the original parquet file using node_ids.
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Literal

from google import genai
from google.genai import types

from utils.cost_estimator import TokenUsage, Cost, cost_from_token_usage
from utils.retry import with_retries

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EntityType = Literal[
    "organization", "location", "role", "event", "legal document", "other"
]
NonexistenceStatus = Literal[
    "abolished", "merged", "replaced", "renamed", "dormant", "expired", "defunct"
]
ProcessStatus = Literal["skipped", "processed"]

DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_ROOT_DIR = Path("/Users/allisoncasasola/reglab/entity-rot")
DEFAULT_PATH_TO_PARQUET_FILE = "/Users/allisoncasasola/reglab/entity-rot/data/city_ordinances_token_filtered.parquet"


class AssessmentMetadata(BaseModel):
    """Metadata for the assessment inference run."""

    city_slug: str
    state_code: str
    model_name: str
    start_time: str
    end_time: Optional[str] = None
    total_entities: int
    entities_processed: int = 0
    entities_skipped: int = 0
    actual_token_usage: dict
    actual_cost: dict


class Citation(BaseModel):
    url: str
    quote: str


# TODO: Fix the municipality that is being passed in to the prompt
# TODO: Fix all of the new lines I think in the content? See how extract_entities.py handles it

# TODO: Fix these errors: "Failed batch: object of type 'NoneType has no len()"
class AnalyzedEntity(BaseModel):
    """Result of analyzing a single entity."""

    entity_name: str  # Same as representative_name
    alternate_names: List[str]
    exists: Optional[Any] = None  # Can be bool, str like "irrelevant", "inconclusive"
    nonexistence_status: Optional[NonexistenceStatus] = None
    reasoning: Optional[str] = None
    reasoning_type: Optional[str] = None
    citations: List[Citation]
    node_ids: List[str]
    occurrence_count: int
    sampled_node_ids: List[str]  # The node_ids that were sampled for analysis


ANALYSIS_PROMPT_TEMPLATE = """You are an expert at assessing whether entities referenced in municipal code text still exist. This is part of a larger project to evaluate the outdatedness of municipal code text.

You will be given multiple entities to analyze. For EACH entity, you will see:
- The entity name and multiple name variations found in the code that were fuzzy matched together
- The entity type(s)
- Up to 3 sample sections from the municipal code where this entity is mentioned


Follow these instructions strictly for each entity: 
1. Determine the relevance of the entity. An entity is relevant if it is real-world, specific, and formally named. Its existence must also have legal or factual relevance to the interpretation and enforcement of the municipal code. If the entity is relevant, continue with the next step. If it is not relevant, label the field "exists" as "irrelevant," provide a 1-5 sentence reasoning on your determination in the "reasoning" field, and label all other fields as null. Then, terminate your analysis for this entity and move on to the next entity.
2. Now, determine if the entity still exists using Google Search grounding. Use reliable sources (official .gov sites preferred). Use up to 3 sources to justify your determination.
    - If you find sufficient information to determine that the entity still exists, label the "exists" field as "true" and the "nonexistence_status" field as null.
    - If you find sufficient information to determine that the entity no longer exists, label the "exists" field as "false."
    - If you cannot find sufficient information to determine whether the entity still exists or not (the information is inconsistent or unclear), label the "exists" field as "inconclusive," provide a 1-5 sentence reasoning on your determination in the "reasoning" field, and label the "nonexistence_status" field as null. Then, terminate your analysis for this entity and move on to the next entity.
3. If the entity is nonexistent, label the "nonexistence_status" field as one of the following values:
    - "abolished": The entity was formally dissolved or eliminated, and no direct successor entity continues its functions.
    - "merged": The entity was combined with another existing entity, and the merged entity continues its functions.
    - "replaced": The entity was replaced by a new entity, and the new entity continues its functions.
    - "renamed": The entity continues to exist but under a new official name.
    - "dormant": The entity has not been formally abolished, but there is evidence of inactivity or lack of official records.
    - "expired": The entity existed for a fixed term and ended automatically.
    - "defunct": The entity is widely known to have ceased operations, though no evidence of an official abolishment is found.
4. If you were able to determine the entity's existence or nonexistence (exists = true/false), provide a 1-5 sentence explanation in the "reasoning" field to indicate how you reached your conclusion, and label the "reasoning_type" field as one of the following values to indicate how you reached your conclusion:
    - "explicit": You were able to find direct evidence.
    - "inferred": You had to make certain inferences to determine the entity's existence or nonexistence.
---

**Output Format:**
Return ONLY a JSON array with this exact structure:
[
  {{
    "entity_name": "exact name from input (not the name variations)",
    "exists": true/false/conflicting_entities/irrelevant/inconclusive,
    "nonexistence_status": "abolished"/"merged"/"replaced"/"renamed"/"dormant"/"expired"/"defunct"/null,
    "reasoning": "1-5 sentence explanation"/null,
    "reasoning_type": "explicit"/"inferred"/null
  }},
  ...
]

The jurisdiction that this municipal code belongs to is {city}, {state}.

**Entities to analyze:**

{entities_text}
"""

load_dotenv()


def setup_gemini_api(model_name: str):
    """
    Initialize the Google GenAI client and a GenerateContentConfig
    with Google Search grounding enabled.

    Per docs, enable grounding by adding the google_search Tool to config.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.2,
    )

    return client, config, model_name


def add_citations(response):
    text = response.candidates[0].content.parts[0].text
    chunks = response.candidates[0].grounding_metadata.grounding_chunks
    supports = response.candidates[0].grounding_metadata.grounding_supports

    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i+1}]({uri})")
            citation_string = ", ".join(citation_links)
            if citation_string:
                text = text[:end_index] + citation_string + text[end_index:]
    return text


def load_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load entities from JSONL file (output of verify_dedupes.py)."""
    entities = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entities.append(json.loads(line))
    return entities


def load_sections_for_node_ids(
    parquet_path: Path, node_ids: List[str]
) -> List[Dict[str, Any]]:
    """Load section content from parquet file for given node_ids."""
    if not node_ids:
        return []

    conn = duckdb.connect()
    try:
        # Build placeholders for IN clause
        placeholders = ", ".join(["?" for _ in node_ids])
        query = f"""
            SELECT node_id, section_heading, content
            FROM '{parquet_path}'
            WHERE node_id IN ({placeholders})
        """
        df = conn.execute(query, node_ids).df()

        sections = []
        for rec in df.to_dict("records"):
            sections.append(
                {
                    "node_id": rec["node_id"],
                    "section_heading": rec.get("section_heading", ""),
                    "content": rec.get("content", ""),
                }
            )
        return sections
    finally:
        conn.close()


def save_result_jsonl(result: AnalyzedEntity, output_path: Path) -> None:
    """Append a single result to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")


def load_existing_results_jsonl(path: Path) -> List[AnalyzedEntity]:
    """Load existing results from JSONL file."""
    results = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(AnalyzedEntity(**json.loads(line)))
    return results


def save_results_csv(
    results: List[AnalyzedEntity], output_path: Path, parquet_path: Path
) -> None:
    """Save analyzed entities to CSV file with full sampled section content."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all sampled node_ids to load sections in one query
    all_sampled_node_ids = []
    for result in results:
        all_sampled_node_ids.extend(result.sampled_node_ids)

    # Load all sampled sections from parquet
    sections = load_sections_for_node_ids(parquet_path, all_sampled_node_ids)
    sections_by_node_id = {s.get("node_id"): s for s in sections}

    rows = []
    for result in results:
        # Build sampled section dicts for CSV
        sampled_secs = []
        for nid in result.sampled_node_ids[:3]:
            sec = sections_by_node_id.get(nid)
            if sec:
                sampled_secs.append({
                    "node_id": sec.get("node_id", ""),
                    "content": sec.get("content", ""),
                })
            else:
                sampled_secs.append(None)

        # Pad to 3 elements
        while len(sampled_secs) < 3:
            sampled_secs.append(None)

        row = {
            "entity_name": result.entity_name,
            "alternate_names": json.dumps(result.alternate_names),
            "exists": result.exists,
            "nonexistence_status": result.nonexistence_status,
            "reasoning": result.reasoning,
            "reasoning_type": result.reasoning_type,
            "citations": json.dumps([c.model_dump() for c in result.citations]),
            "node_ids": json.dumps(result.node_ids),
            "occurrence_count": result.occurrence_count,
            "sampled_sec_1": json.dumps(sampled_secs[0]) if sampled_secs[0] else None,
            "sampled_sec_2": json.dumps(sampled_secs[1]) if sampled_secs[1] else None,
            "sampled_sec_3": json.dumps(sampled_secs[2]) if sampled_secs[2] else None,
        }
        rows.append(row)

    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def save_metadata(metadata: AssessmentMetadata, output_path: Path) -> None:
    """Save metadata to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)


def analyze_entity_batch(
    client: genai.Client,
    config: types.GenerateContentConfig,
    model_name: str,
    batch: List[Dict[str, Any]],
    sections_by_entity: Dict[str, List[Dict[str, Any]]],
    city: str,
    state: str,
    max_sections_per_entity: int = 3,
    debug: bool = False,
) -> Tuple[List[AnalyzedEntity], TokenUsage]:
    """Analyze a batch of entities in one grounded request.

    Args:
        batch: List of entity dicts from verify_dedupes.py output
        sections_by_entity: Dict mapping representative_name to list of section dicts
    """

    # Track which sections were actually used for each entity
    used_sections_by_entity: Dict[str, List[Dict[str, Any]]] = {}

    entities_text = ""
    for i, entity_data in enumerate(batch, 1):
        name = entity_data.get("representative_name", "")
        alternate_names = entity_data.get("alternate_names", [])
        entity_type = entity_data.get("type", "other")
        sections = sections_by_entity.get(name, [])

        # Validate sections is not None (should never happen)
        if sections is None:
            raise ValueError(f"sections is None for entity '{name}' - this should not happen")

        # Sections are already sampled in process_batch, just use them
        sampled_sections = sections[:max_sections_per_entity]
        used_sections_by_entity[name] = sampled_sections

        entities_text += f"\n\nEntity {i}: {name}\n"
        entities_text += f"Name variations: {', '.join(alternate_names)}\n"
        entities_text += f"Type: {entity_type}\n"
        entities_text += f"\nSample sections (showing {len(sampled_sections)} of {len(sections)} total):\n"

        for j, section in enumerate(sampled_sections, 1):
            content = section.get("content", "")

            entities_text += f"\n--- Section {j} ---\n"
            entities_text += f"Content:\n{content}\n"

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        city=city, state=state, entities_text=entities_text
    )

    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("=" * 80)
        logger.debug("PROMPT BEING SENT TO MODEL:")
        logger.debug("=" * 80)
        logger.debug(prompt)
        logger.debug("=" * 80)

    # Track token usage across retries
    token_usage = TokenUsage()

    def call_api_and_parse():
        """Make API call and parse response. Raises on any failure to trigger retry."""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        # Validate response structure
        if not response.candidates:
            raise ValueError("Empty candidates in response")
        if not response.candidates[0].content:
            raise ValueError("Empty content in response")
        if not response.candidates[0].content.parts:
            raise ValueError("Empty parts in response")

        text = response.candidates[0].content.parts[0].text
        if not text or not text.strip():
            raise ValueError("Empty text in response")

        # Clean up text
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON - raises on invalid JSON
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list response, got {type(parsed)}")

        return response, parsed

    try:
        response, parsed = with_retries(
            call_api_and_parse,
            retries=3,
            base=1.0,
            jitter=1.0,
            token_usage=token_usage,
        )

        if debug:
            logger.debug("RAW RESPONSE:")
            logger.debug(response.candidates[0].content.parts[0].text)
            logger.debug("=" * 80)

        text_with_citations = add_citations(response)

        results: List[AnalyzedEntity] = []
        for entity_result, original in zip(parsed, batch):
            citations = []
            if (
                hasattr(response.candidates[0], "grounding_metadata")
                and response.candidates[0].grounding_metadata
                and response.candidates[0].grounding_metadata.grounding_chunks
            ):
                chunks = response.candidates[0].grounding_metadata.grounding_chunks
                for chunk in chunks[:3]:
                    if hasattr(chunk, "web") and chunk.web:
                        citations.append(
                            Citation(url=chunk.web.uri, quote=chunk.web.title or "")
                        )

            entity_name = original.get("representative_name", "")
            used_sections = used_sections_by_entity.get(entity_name, [])
            sampled_node_ids = [s.get("node_id", "") for s in used_sections]

            results.append(
                AnalyzedEntity(
                    entity_name=entity_name,
                    alternate_names=original.get("alternate_names", []),
                    exists=entity_result.get("exists"),
                    nonexistence_status=entity_result.get("nonexistence_status"),
                    reasoning=entity_result.get("reasoning"),
                    reasoning_type=entity_result.get("reasoning_type"),
                    citations=citations,
                    node_ids=original.get("node_ids", []),
                    occurrence_count=original.get("occurrence_count", 0),
                    sampled_node_ids=sampled_node_ids,
                )
            )
        return (results, token_usage)

    except Exception as err:
        print(f"Failed batch: {err}")
        fallback_results: List[AnalyzedEntity] = []
        for entity_data in batch:
            entity_name = entity_data.get("representative_name", "")
            used_sections = used_sections_by_entity.get(entity_name, [])
            sampled_node_ids = [s.get("node_id", "") for s in used_sections]

            fallback_results.append(
                AnalyzedEntity(
                    entity_name=entity_name,
                    alternate_names=entity_data.get("alternate_names", []),
                    exists=None,
                    nonexistence_status=None,
                    reasoning=f"Error: {str(err)}",
                    reasoning_type=None,
                    citations=[],
                    node_ids=entity_data.get("node_ids", []),
                    occurrence_count=entity_data.get("occurrence_count", 0),
                    sampled_node_ids=sampled_node_ids,
                )
            )
        return (fallback_results, token_usage)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze verified deduplicated entities using Google Search grounding"
    )
    parser.add_argument(
        "city",
        help="City to process in format 'city_slug:state_code' (e.g., 'santa_clara_county:ca')",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=f"Root directory for outputs (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path(DEFAULT_PATH_TO_PARQUET_FILE),
        help=f"Path to parquet file for section content (default: {DEFAULT_PATH_TO_PARQUET_FILE})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to verified entities JSONL file (overrides default location)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--num-entities-in-prompt",
        type=int,
        default=3,
        help="Number of entities to analyze per API call (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of entities to process (for testing)",
    )
    parser.add_argument(
        "--max-sections-per-entity",
        type=int,
        default=3,
        help="Maximum number of sections to sample per entity (default: 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode to print prompts sent to the model",
    )
    parser.add_argument(
        "--output-csv",
        action="store_true",
        default=False,
        help="Also output a CSV file alongside the JSONL",
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

    # Build paths based on new structure
    assessment_dir = args.root_dir / "outputs" / "assessment" / city_name
    output_jsonl = assessment_dir / "assessed_entities.jsonl"
    metadata_file = assessment_dir / "inference_metadata.json"

    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        # Default: look for verified groups file in deduplication output
        input_file = (
            args.root_dir
            / "outputs"
            / "deduplication"
            / "verification"
            / city_name
            / "verified_groups.jsonl"
        )

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Loading input data from {input_file}...")
    data = load_data(input_file)
    print(f"Loaded {len(data)} entities")

    if args.limit:
        data = data[: args.limit]
        print(f"Limited to {len(data)} entities for testing")

    # Check for existing results to resume
    processed_names = set()
    if output_jsonl.exists():
        existing_results = load_existing_results_jsonl(output_jsonl)
        processed_names = {r.entity_name for r in existing_results}
        print(f"Found {len(existing_results)} existing results, will skip these.")

    # Filter out already processed entities
    data_to_process = [
        d for d in data if d.get("representative_name") not in processed_names
    ]

    if not data_to_process:
        print("All entities already processed!")
        return

    print(
        f"\nWill analyze {len(data_to_process)} entities (skipping {len(processed_names)} already done)"
    )

    client, config, model_name = setup_gemini_api(args.model)

    # Get city and state names for the prompt (use slug as fallback)
    city_display = city_slug.replace("_", " ").title()
    state_display = state_code.upper()

    batch: List[dict[str, Any]] = []

    # Track token usage and timing
    start_time = datetime.now()
    actual_token_usage = TokenUsage()
    entities_processed = 0

    print(f"\nAnalyzing entities in batches of {args.num_entities_in_prompt}...")

    def process_batch(batch: List[Dict[str, Any]]) -> Tuple[List[AnalyzedEntity], TokenUsage]:
        """Sample node_ids, load sections, and process a batch of entities."""
        # Sample up to max_sections_per_entity node_ids per entity first
        sampled_node_ids_by_entity: Dict[str, List[str]] = {}
        all_sampled_node_ids = []

        for entity_data in batch:
            entity_name = entity_data.get("representative_name", "")
            node_ids = entity_data.get("node_ids", [])

            if len(node_ids) > args.max_sections_per_entity:
                sampled = random.sample(node_ids, args.max_sections_per_entity)
            else:
                sampled = node_ids

            sampled_node_ids_by_entity[entity_name] = sampled
            all_sampled_node_ids.extend(sampled)

        # Load only the sampled sections from parquet
        sections = load_sections_for_node_ids(args.parquet_path, all_sampled_node_ids)

        # Build sections_by_entity mapping from loaded sections
        sections_by_node_id = {s.get("node_id"): s for s in sections}
        sections_by_entity: Dict[str, List[Dict[str, Any]]] = {}

        for entity_name, sampled_ids in sampled_node_ids_by_entity.items():
            sections_by_entity[entity_name] = [
                sections_by_node_id[nid] for nid in sampled_ids if nid in sections_by_node_id
            ]

        return analyze_entity_batch(
            client=client,
            config=config,
            model_name=model_name,
            batch=batch,
            sections_by_entity=sections_by_entity,
            city=city_display,
            state=state_display,
            max_sections_per_entity=args.max_sections_per_entity,
            debug=args.debug,
        )

    try:
        with tqdm(total=len(data_to_process), desc="Analyzing entities") as pbar:
            for entity_data in data_to_process:
                batch.append(entity_data)

                if len(batch) >= args.num_entities_in_prompt:
                    batch_results, batch_token_usage = process_batch(batch)

                    # Accumulate token usage
                    actual_token_usage.prompt_token_count += (
                        batch_token_usage.prompt_token_count
                    )
                    actual_token_usage.candidates_token_count += (
                        batch_token_usage.candidates_token_count
                    )
                    actual_token_usage.cached_content_token_count += (
                        batch_token_usage.cached_content_token_count
                    )
                    actual_token_usage.thoughts_token_count += (
                        batch_token_usage.thoughts_token_count
                    )

                    # Save each result to JSONL immediately
                    for result in batch_results:
                        save_result_jsonl(result, output_jsonl)

                    entities_processed += len(batch)
                    pbar.update(len(batch))
                    batch.clear()

            # Process leftover batch
            if batch:
                batch_results, batch_token_usage = process_batch(batch)

                # Accumulate token usage
                actual_token_usage.prompt_token_count += (
                    batch_token_usage.prompt_token_count
                )
                actual_token_usage.candidates_token_count += (
                    batch_token_usage.candidates_token_count
                )
                actual_token_usage.cached_content_token_count += (
                    batch_token_usage.cached_content_token_count
                )
                actual_token_usage.thoughts_token_count += (
                    batch_token_usage.thoughts_token_count
                )

                for result in batch_results:
                    save_result_jsonl(result, output_jsonl)

                entities_processed += len(batch)
                pbar.update(len(batch))

        # Calculate actual cost
        actual_cost = cost_from_token_usage(model_name, actual_token_usage)

        # Save metadata
        metadata = AssessmentMetadata(
            city_slug=city_slug,
            state_code=state_code,
            model_name=model_name,
            start_time=start_time.isoformat(),
            end_time=datetime.now().isoformat(),
            total_entities=len(data),
            entities_processed=entities_processed,
            entities_skipped=len(processed_names),
            actual_token_usage=actual_token_usage.model_dump(),
            actual_cost=actual_cost.model_dump(),
        )
        save_metadata(metadata, metadata_file)

        print(f"\n✓ Results saved to: {output_jsonl}")

        # Optionally save CSV
        if args.output_csv:
            all_results = load_existing_results_jsonl(output_jsonl)
            output_csv = assessment_dir / "assessed_entities.csv"
            save_results_csv(all_results, output_csv, args.parquet_path)
            print(f"✓ CSV saved to: {output_csv}")
        print(f"✓ Metadata saved to: {metadata_file}")
        print(f"✓ Analyzed {entities_processed} entities")
        print(f"\nActual token usage:")
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
