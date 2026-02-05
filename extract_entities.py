import argparse
import csv
import json
import os
import random
import re
import time

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import duckdb
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Literal

from utils.cost_estimator import (
    TokenUsage,
    Cost,
    estimated_token_usage_for_extraction,
    cost_from_token_usage,
)
from utils.retry import with_retries

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# TODO: Handle input text that may be greater than model's max input length
# TODO: Check that it forces the outputs to match the schema

# TODO: Check how off the estimates are

EntityType = Literal["organization", "location", "role", "event", "legal document", "other"]


class Entity(BaseModel):
    name: str
    type: EntityType


class ExtractedEntities(BaseModel):
    """Schema for extracted entities from a section."""
    entities: List[Entity]
    node_id: str


class ExtractionMetadata(BaseModel):
    """Metadata for an extraction run."""
    city_slug: str
    state_code: str
    model_name: str
    timestamp: str
    n_sections: int
    n_words: int
    estimated_token_usage: TokenUsage
    estimated_cost: Cost
    actual_token_usage: TokenUsage
    actual_cost: Cost


@dataclass
class SectionRow:
    source: str
    source_format: Optional[str]
    city_slug: str
    jurisdiction_name: str
    state: str
    state_code: str
    source_file: Optional[str]
    node_id: str
    section_heading: Optional[str]
    hierarchy_path: Optional[str]
    content: str
    tiktoken: Optional[int]
    word_tokens_approx: Optional[int]



DEFAULT_PATH_TO_PARQUET_FILE = (
    "/Users/allisoncasasola/reglab/entity-rot/data/city_ordinances_token_filtered.parquet"
)
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_ROOT_DIR = Path("/Users/allisoncasasola/reglab/entity-rot")

ENTITY_EXTRACTION_PROMPT = """
You are working on a project to identify municipal code sections that may be outdated. One indicator that a section might be outdated is if it refers to  real-world entities whose existence, name, or authority may have changed or no longer exists.

Your task is to extract relevant entities from the text. A relevant entity is one that meets the following criteria:
- Is specific
- Is formally named
- Is real-world
- Is a referent whose continued existence or name is necessary for the legal text to remain accurate.

### Instructions

Extract entities and classify each into one of the following types:

- **organization**: Include only formally-named organizations, governing bodies, institutions, companies, or other groups.
 organizations or formally named bodies (e.g., "City of Aurora Planning and Zoning Commission", "Federal Emergency Management Agency").
  Exclude generic terms like "city," "board," or "department," unless they appear as part of a *full proper name*.
- **location**: Include only proper-named geographic or jurisdictional locations.
  Exclude generic or descriptive locations like "city," "school," "public property," "building," or "hospital area."
- **role**: Include only *uniquely titled* offices or positions (not occupations, job classes, or generic titles).
- **event**: Only include *proper-named events* (e.g., "Planning Commission Meeting," "Special Election," "Memorial Day"). These may be recurring or one-time events.
  Exclude vague or generic events, historical events, and dates.
- **legal document**: Include only references to legal documents, statutes, acts, regulations, etc. that are NOT internal to the municipal code.
- **other**: Proper-named entities with a stable real-world referent relevant to the municipal code that do not fit the above categories. This category should be used sparingly.

Do NOT extract any term that is:
    - a generic category, concept, or classification
    - an object, species, disease, or material
    - an occupation, job class, or generic title.


### Output Format
Return **only** a JSON object with this exact structure:

{
  "entities": [
    {"name": "entity text here", "type": "type here"},
    {"name": "another entity", "type": "type here"}
  ]
}

### Few-shot examples
**Example 1:**
Text: "§ 92.24 RECOVERY OF COST.\\n\\n(A)\\nPersonal liability.\\nThe owner of premises on which a nuisance has been abated by the city shall be personally liable for the cost to the city of the abatement, including administrative costs. As soon as the work has been completed and the cost determined, the City Clerk/Treasurer or other official shall prepare a bill for the cost and mail it to the owner. Thereupon the amount shall be immediately due and payable at the office of the City Clerk/Treasurer.\\n\\n(B)\\nAssessment.\\nAfter notice and hearing as provided in M.S. § 429.061, as it may be amended from time to time, if the nuisance is a public health or safety hazard on private property, the accumulation of snow and ice on public sidewalks, the growth of weeds on private property or outside the traveled portion of streets, or unsound or insect-infected trees, the City Clerk/Treasurer shall, on or before September 1 next following abatement of the nuisance, list the total unpaid charges along with all other charges as well as other charges for current services to be assessed under M.S. § 429.101 against each separate lot or parcel to which the charges are attributable. The City Council may then spread the charges against the property under that statute and other pertinent statutes for certification to the County Auditor and collection along with current taxes the following year or in annual installments, not exceeding 10, as the City Council may determine in each case.	"

Output:
{
  "entities": [
    {"name": "City Clerk/Treasurer", "type": "role"},
    {"name": "City Council", "type": "organization"},
  ]
}

Explanation: The City Clerk/Treasurer and City Council are all specific proper-named entities. M.S. § 429.061 is excluded as it is a legal reference, and September 1 is excluded as it is a date.

**Example 2:**
Text: "5.68.060 Signs and Solicitation.\\n\\nA. Notwithstanding any County code provisions to the contrary, signs advertising a non-profit car wash shall be displayed only on the premises on which the car wash is conducted; and no sign shall exceed nine (9) square feet. No signs may be placed on any sidewalks or within any public right-of-way.\\n\\nB. No solicitation may be done from any sidewalk or within any public right-of-way in a manner which impedes or otherwise endangers or interferes with the public's use thereof.\\n\\n[Ord. 462, Section 3, 5/9/07.]\\n\\nChapter 5.72 ENTERTAINMENT IN BUSINESS ESTABLISHMENTS\\n\\nSections:	"

Output:
{
  "entities": []
}

Explanation: There are no specific nor formally named entities in this text.

**Example 3:**
Text: "§ 91.20 POLICY AND PURPOSE.\\n\\nThe city has determined that the health of oak and elm trees is threatened by fatal diseases known as oak wilt and Dutch Elm Disease. It has further determined that the loss of oak and elm trees located on public and private property would substantially depreciate the value of property and impair the safety, good order, general welfare and convenience of the public. It is declared to be the intention of the Council to control and prevent the spread of these diseases, and provide for the removal of dead or diseased trees, as nuisances.\\n\\n(Prior Code, § 10.20)"

Output:
{
  "entities": [
    {"name": "Council", "type": "organization"},
    {"name": "Tree Inspector", "type": "role"},
    {"name": "Public Works Director", "type": "role"},
  ]
}

Explanation: The Council, Tree Inspector, Public Works Director, and State Commissioner of Agriculture are all specific proper-named entities. M.S. §§ 89.001 et seq. is excluded as it is a legal reference. Oak wilt and Dutch Elm Disease are excluded because they refer to
"""

load_dotenv()


def _connect_duckdb() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def count_city_sections(parquet_path: Path, city_slug: str, state_code: str) -> int:
    conn = _connect_duckdb()
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM '{parquet_path}' WHERE city_slug = ? AND state_code = ?",
            [city_slug, state_code],
        ).fetchone()[0]
    finally:
        conn.close()


def count_city_words(parquet_path: Path, city_slug: str, state_code: str) -> int:
    conn = _connect_duckdb()
    try:
        df = conn.execute(
            f"SELECT content FROM '{parquet_path}' WHERE city_slug = ? AND state_code = ?",
            [city_slug, state_code],
        ).df()
        return sum(len(str(c).split()) for c in df["content"].dropna())
    finally:
        conn.close()


def iter_city_rows(
    parquet_path: Path,
    city_slug: str,
    state_code: str,
    batch_size: int = 1000,
) -> Iterable[SectionRow]:
    """Stream rows for a city in batches to avoid memory spikes."""
    conn = _connect_duckdb()
    try:
        total = conn.execute(
            f"SELECT COUNT(*) FROM '{parquet_path}' WHERE city_slug = ? AND state_code = ?",
            [city_slug, state_code],
        ).fetchone()[0]

        offset = 0
        while offset < total:
            df = conn.execute(
                f"""
                SELECT source, source_format, city_slug, jurisdiction_name, state,
                       state_code, source_file, node_id, section_heading,
                       hierarchy_path, content, tiktoken, word_tokens_approx
                FROM '{parquet_path}'
                WHERE city_slug = ? AND state_code = ?
                LIMIT ? OFFSET ?
                """,
                [city_slug, state_code, batch_size, offset],
            ).df()

            for rec in df.to_dict("records"):
                yield SectionRow(**rec)

            offset += batch_size
    finally:
        conn.close()


_WS = re.compile(r"\s+")


def _normalize_entity(e: str) -> str:
    e = e.strip()
    e = _WS.sub(" ", e)
    return e


def _clean_and_parse_response(response_text: str) -> dict:
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:].strip()
    elif response_text.startswith("```"):
        response_text = response_text[3:].strip()
    if response_text.endswith("```"):
        response_text = response_text[:-3].strip()
    return json.loads(response_text)


def extract_entities_from_content(
    client: genai.Client,
    model_name: str,
    system_instruction: str,
    content: str,
    node_id: str,
    section_heading: Optional[str] = None,
    hierarchy_path: Optional[str] = None,
    debug: bool = False,
) -> Tuple[ExtractedEntities, TokenUsage]:
    """Extract entities for one section using Gemini in JSON-mode with retries."""

    if debug:
        logger.debug(f"PROCESSING SECTION: {node_id}")
        logger.debug(f"Section Heading: {section_heading}")
        logger.debug(f"Hierarchy Path: {hierarchy_path}")
        logger.debug("CONTENT SENT TO LLM:")
        logger.debug(content)

    token_usage = TokenUsage(
        cached_content_token_count=0,
        prompt_token_count=0,
        thoughts_token_count=0,
        candidates_token_count=0,
    )

    try:
        resp = with_retries(
            lambda: client.models.generate_content(
                model=model_name,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
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

        if debug:
            print(resp.usage_metadata)

        parsed_response = _clean_and_parse_response(resp.text)
        entities_raw = parsed_response.get("entities", [])
        entity_objects = []
        seen = set()

        for entity in entities_raw:
            name = _normalize_entity(entity.get("name", ""))
            entity_type = entity.get("type", "other")

            if entity_type not in [
                "organization",
                "location",
                "role",
                "event",
                "legal document",
                "other",
            ]:
                entity_type = "other"

            key = name.lower()
            if key and key not in seen:
                seen.add(key)
                entity_objects.append(Entity(name=name, type=entity_type))

        return (ExtractedEntities(entities=entity_objects, node_id=node_id), token_usage)
    except Exception as e:
        print(f"\nError processing section {node_id}: {e}")
        if "resp" in locals():
            print(f"Raw response: {resp.text}")
        return (ExtractedEntities(entities=[], node_id=node_id), token_usage)


def load_existing_results_jsonl(path: Path) -> List[ExtractedEntities]:
    """Load existing results from JSONL file."""
    results = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(ExtractedEntities(**json.loads(line)))
    return results


def save_result_jsonl(result: ExtractedEntities, output_path: Path) -> None:
    """Append a single result to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")


def save_metadata(metadata: ExtractionMetadata, output_path: Path) -> None:
    """Save metadata to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)


def process_city(
    city_slug: str,
    state_code: str,
    client: genai.Client,
    model_name: str,
    root_dir: Path,
    parquet_path: Path,
    auto_confirm: bool,
    output_csv: bool,
    batch_size: int = 1000,
    autosave_every: int = 25,
    debug: bool = False,
) -> None:
    print(f"Processing: {city_slug} ({state_code})")
    print("=" * 80)

    # Build output paths based on new structure
    city_name = f"{city_slug}_{state_code}"
    extraction_dir = root_dir / "outputs" / "extraction" / city_name

    output_jsonl = extraction_dir / "entities.jsonl"
    output_csv_file = extraction_dir / "entities.csv"
    metadata_file = extraction_dir / "inference_metadata.json"

    # Count sections and words
    n_sections = count_city_sections(parquet_path, city_slug, state_code)
    n_words = count_city_words(parquet_path, city_slug, state_code)

    if n_sections == 0:
        print(f"No data found for: {city_slug} ({state_code})")
        return

    print(f"Found {n_sections} sections with {n_words} words")

    # Estimate token usage and cost
    estimated_token_usage = estimated_token_usage_for_extraction(
        model_name, n_words, n_sections, ENTITY_EXTRACTION_PROMPT
    )
    estimated_cost = cost_from_token_usage(model_name, estimated_token_usage)

    print(f"\nEstimated token usage:")
    print(f"  - Cached content: {estimated_token_usage.cached_content_token_count:,}")
    print(f"  - Prompt: {estimated_token_usage.prompt_token_count:,}")
    print(f"  - Output: {estimated_token_usage.candidates_token_count:,}")
    print(f"\nEstimated cost: ${estimated_cost.total_cost:.4f}")

    if not auto_confirm:
        resp = input("\nContinue with entity extraction? (y/n): ").strip().lower()
        if resp not in ("y", "yes"):
            print("Skipping this city.")
            return

    # Initialize results tracking
    processed_node_ids = set()

    # Check for existing results to resume
    if output_jsonl.exists():
        existing_results = load_existing_results_jsonl(output_jsonl)
        print(f"Found existing results for {len(existing_results)} out of {n_sections} sections.")

        if len(existing_results) == n_sections:
            print("All sections already processed!")
            return

        resp = input("Choose an action: 1) Overwrite, 2) Resume, 3) Skip (1/2/3): ").strip()
        if resp == "1":
            print("Overwriting existing results.")
            output_jsonl.unlink()  # Delete existing file
        elif resp == "2":
            print("Resuming from existing results.")
            processed_node_ids = {r.node_id for r in existing_results}
        elif resp == "3":
            print("Skipping this city.")
            return
        else:
            print("Invalid response. Skipping this city.")
            return

    to_process = n_sections - len(processed_node_ids)
    if to_process <= 0:
        print("All sections already processed!")
        return

    print(f"\nProcessing {to_process} sections (skipping {len(processed_node_ids)} already done)...")

    # Track actual token usage
    actual_token_usage = TokenUsage(
        cached_content_token_count=0,
        prompt_token_count=0,
        thoughts_token_count=0,
        candidates_token_count=0,
    )

    rows_iter = iter_city_rows(parquet_path, city_slug, state_code, batch_size=batch_size)
    results_for_csv = []

    with tqdm(total=to_process, desc=f"Extracting entities from {city_slug}, {state_code}") as pbar:
        for row in rows_iter:
            if row.node_id in processed_node_ids:
                continue

            extracted, token_usage = extract_entities_from_content(
                client=client,
                model_name=model_name,
                system_instruction=ENTITY_EXTRACTION_PROMPT,
                content=row.content,
                node_id=row.node_id,
                section_heading=row.section_heading,
                hierarchy_path=row.hierarchy_path,
                debug=debug,
            )

            # Save result immediately to JSONL
            save_result_jsonl(extracted, output_jsonl)

            if output_csv:
                results_for_csv.append((extracted, row.content))

            # Accumulate actual token usage
            actual_token_usage.cached_content_token_count += token_usage.cached_content_token_count
            actual_token_usage.prompt_token_count += token_usage.prompt_token_count
            actual_token_usage.thoughts_token_count += token_usage.thoughts_token_count
            actual_token_usage.candidates_token_count += token_usage.candidates_token_count

            processed_node_ids.add(row.node_id)
            pbar.update(1)

    # Calculate actual cost
    actual_cost = cost_from_token_usage(model_name, actual_token_usage)

    # Save metadata
    metadata = ExtractionMetadata(
        city_slug=city_slug,
        state_code=state_code,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        n_sections=n_sections,
        n_words=n_words,
        estimated_token_usage=estimated_token_usage,
        estimated_cost=estimated_cost,
        actual_token_usage=actual_token_usage,
        actual_cost=actual_cost,
    )
    save_metadata(metadata, metadata_file)

    print(f"\n✓ Results saved to: {output_jsonl}")
    print(f"✓ Metadata saved to: {metadata_file}")
    print(f"✓ Processed {len(processed_node_ids)} total sections for {city_slug}, {state_code}")
    print(f"\nActual token usage:")
    print(f"  - Cached content: {actual_token_usage.cached_content_token_count:,}")
    print(f"  - Prompt: {actual_token_usage.prompt_token_count:,}")
    print(f"  - Output: {actual_token_usage.candidates_token_count:,}")
    print(f"\nActual cost: ${actual_cost.total_cost:.4f}")

    # Output CSV if requested
    if output_csv and results_for_csv:
        with open(output_csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "content", "entities"])
            for extracted, content in results_for_csv:
                entities_json = json.dumps([e.model_dump() for e in extracted.entities])
                writer.writerow([extracted.node_id, content, entities_json])
        print(f"✓ CSV saved to: {output_csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from municipal code sections using Gemini API"
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
        help=f"Path to parquet file (default: {DEFAULT_PATH_TO_PARQUET_FILE})",
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
    parser.add_argument(
        "--output-csv",
        action="store_true",
        default=False,
        help="Also output CSV file with entities",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per DuckDB batch (default: 1000)",
    )
    parser.add_argument(
        "--autosave-every",
        type=int,
        default=25,
        help="Autosave after this many newly processed sections (default: 25)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debugging logging",
    )

    args = parser.parse_args()

    # Parse city specification
    parts = args.city.split(":")
    if len(parts) != 2:
        print(f"Error: Invalid city specification '{args.city}'")
        print("Expected format: 'city_slug:state_code'")
        return

    city_slug, state_code = parts

    # Set up client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        process_city(
            city_slug=city_slug,
            state_code=state_code,
            client=client,
            model_name=args.model,
            root_dir=args.root_dir,
            parquet_path=args.parquet_path,
            auto_confirm=args.auto_confirm,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            autosave_every=args.autosave_every,
            debug=args.debug,
        )
    except Exception as e:
        print(f"\nError processing city {args.city}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
