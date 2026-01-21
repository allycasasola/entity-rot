import argparse
import json
import os
import random
import re
import time

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional
from typing_extensions import Literal
import duckdb
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
import logging

from utils.cost_estimator import (
    calculate_input_price_for_extraction,
    calculate_output_price_for_extraction,
    calculate_context_caching_price_for_extraction,
)

logging.basicConfig(
    level=logging.WARNING
)  # Changed from DEBUG to WARNING for cleaner output
logger = logging.getLogger(__name__)


EntityType = Literal["organization", "location", "role", "event", "other"]


class Entity(BaseModel):
    name: str
    type: EntityType


class ExtractedEntities(BaseModel):
    """Schema for extracted entities from a section."""

    entities: List[Entity]
    node_id: str  # always present now
    section_heading: Optional[str] = None
    hierarchy_path: Optional[str] = None
    content: Optional[str] = None


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
    tfidf_score: Optional[float]
    tfidf_label: Optional[str]


# TODO: Add more varied examples of entities and non-entities to the prompt. Right now, the prompt only includes sections from Aurora, MN.


DEFAULT_PATH_TO_PARQUET_FILE = "/oak/stanford/groups/deho/allyc/city_ordinances.parquet"
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"

ENTITY_EXTRACTION_PROMPT = """
You are working on a project to extract entities specific to the given jurisdiction and identify whether or not they still exist.
You are specifically tasked with the first step of this project, which is to extract *specific, proper-named entities* (not generic categories, nouns, or references) from municipal code text.

### Instructions

Extract entities and classify each into one of the following types:
- **organization**: Only include *specific* organizations or formally named bodies (e.g., "City of Aurora Planning and Zoning Commission", "Federal Emergency Management Agency").
  Exclude generic terms like “city,” “board,” or “department,” unless they appear as part of a *full proper name*.
- **location**: Only include *specific proper locations* (e.g., "1st Street East", "Aurora", "St. Louis County", "I-1 Industrial Park"). 
  Exclude generic or descriptive locations like “city,” “cemetery,” “school,” “public property,” “building,” or “hospital area.”
- **role**: Only include *titled or uniquely identifying roles* (e.g., “City Administrator/Clerk-Treasurer,” “Planning and Zoning Official,” “State Commissioner of Public Safety”). 
  Exclude generic roles like “owner,” “citizen,” “employee,” “director,” “officer,” or “person in charge.”
- **event**: Only include *specific named events or recurring meetings* (e.g., “Planning Commission Meeting,” “Special Election,” “Memorial Day”). 
  Exclude vague or generic phrases like “emergency,” “fire,” or “hearing.”
- **other**: Only include significant named references that don't fit the above but are *not* citations, section numbers, or internal code references.
  Exclude any references to sections, ordinances, code citations, or statute numbers (e.g., “§ 11.99,” “M.S. §§ 349.11 through 349.23,” “Ord. 85”).

### Additional Rules
- Ignore all numeric or legal references (sections, ordinances, or statute citations).
- Exclude entities that do not have a proper name.
- Do **not** include duplicates or repeated generic placeholders.
- If unsure whether an entity is specific enough, **include it**.

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
Text: "Except for controlled breeding purposes, every female animal in heat shall be kept confined in a building or secure enclosure, or in a veterinary hospital or boarding kennel, in a manner that the female animal cannot come in contact with other animals."

Output:
{
  "entities": []
}

Explanation: There are no specific entities in this text. Female animals, veterinary hospitals, and boarding kennels are all generic nouns, not specific proper-named entities.

**Example 2:**
Text: "(A)   Personal liability. The owner of premises on which a nuisance has been abated by the city shall be personally liable for the cost to the city of the abatement, including administrative costs. As soon as the work has been completed and the cost determined, the Deputy Clerk or other official shall prepare a bill for the cost and mail it to the owner. Thereupon the amount shall be immediately due and payable at the office of the Deputy Clerk.
(B)   Assessment.
    (1)   After notice and hearing as provided in M.S. § 429.061, as it may be amended from time to time, if the nuisance is a public health or safety hazard on private property, the accumulation of snow and ice on public sidewalks, the growth of weeds on private property or outside the traveled portion of streets, or unsound or insect-infected trees, the Deputy Clerk shall, on or before September 1 next following abatement of the nuisance, list the total unpaid charges along with all other charges as well as other charges for current services to be assessed under M.S. § 429.101, as it may be amended from time to time, against each separate lot or parcel to which the charges are attributable.
    (2)   The City Council may then spread the charges against the property under that statute and other pertinent statutes for certification to the County Auditor and collection along with current taxes the following year or in annual installments, not exceeding 10, as the City Council may determine in each case.
Penalty, see § 10.99"

Output:
{
  "entities": [
    {"name": "Deputy Clerk", "type": "role"},
    {"name": "City Council", "type": "organization"},
    {"name": "County Auditor", "type": "role"},
  ]
}

Explanation: The Deputy Clerk, City Council, and County Auditor are all specific proper-named entities. M.S. § 429.061 and § 10.99 are excluded as they are legal references.

**Example 3:**
Text: "It is unlawful for any person to drive or operate a motorized vehicle, except a wheelchair powered by electricity and occupied by a handicapped person, on any public sidewalk or public property designated for use as a pedestrian walkway or bicycle trail, except when crossing the same for ingress and egress through a curb cut to property lying on the other side thereof."

Output:
{
  "entities": []
}  

Explanation: There are no specific entities in this text. Motorized vehicles, wheelchair powered by electricity, and handicapped persons are all generic nouns, not specific proper-named entities. Public sidewalks, public property, pedestrian walkways, and bicycle trails are all generic locations, not specific proper-named entities.

**Example 4:**
Text: "  (A)   Proof. It is prima facie evidence of exhibition driving when a motor vehicle stops, starts, accelerates, decelerates or turns at an unnecessary rate of speed so as to cause tires to squeal, gears to grind, soil to be thrown, engine backfire, fishtailing or skidding or as to two-wheeled motor vehicles, the front wheel to lost contact with the ground or roadway surface.
   (B)   Unlawful act. No person shall do any exhibition driving on any street, parking lot or other public or private property, except when an emergency creates necessity for such operation to prevent injury to persons or damage to property; provided, that this section shall not apply to driving on a racetrack. For the purposes of this section, a RACETRACK means any track or premises whereon motorized vehicles legally compete in a race or timed contest for an audience, the members of which have directly or indirectly paid a consideration for admission.
   (C)   Violation. Any person violating this section shall be guilty of a petty misdemeanor subject to a fine as set forth in § 10.99 and the costs of prosecution.
Penalty, see § 10.99"

Output:
{
  "entities": []
}

Explanation: There are no specific entities in this text. Exhibition driving, motorized vehicless,  and racetracks are all generic nouns, not specific proper-named entities. § 10.99 is excluded as it is a legal reference.

**Example 5:**
Text: "(A)   Policy and purpose. The city has determined that the health of oak and elm trees is threatened by fatal diseases known as oak wilt and Dutch elm disease. It has further determined that the loss of oak and elm trees located on public and private property would substantially depreciate the value of property and impair the safety, good order, general welfare and convenience of the public. It is declared to be the intention of the Council to control and prevent the spread of these diseases, and provide for the removal of dead or diseased trees, as nuisances.
(B)   Definitions.    For the purpose of this section, the following definitions shall apply unless the context clearly indicates or requires a different meaning.
    NUISANCE. 
         (a)   Any living or standing tree infected to any degree with a shade tree disease; or
         (b)   Any logs, branches, stumps or other parts of any dead or dying tree, so infected, unless the parts have been fully burned or treated under the direction of the Tree Inspector.
    SHADE TREE DISEASE. Dutch elm disease or oak wilt disease.
    TREE INSPECTOR. The Public Works Director, or any other employee of the city as the Council may designate and who shall thereafter qualify.
(C)   Scope and adoption by reference. M.S. §§ 89.001 et seq., as it may be amended from time to time, is hereby adopted by reference, together with the rules and regulations of the State Commissioner of Agriculture relating to shade tree diseases; provided, that this section shall supersede those statutes, rules and regulations only to the extent of inconsistencies.
(D)   Stockpiling of elm wood. The stockpiling of bark-bearing elmwood shall be permitted during the period from September 15 through April 1 of the following year if a permit has been issued therefor. Any wood not utilized by April 1 must then be removed and disposed of as provided by this section and the regulations incorporated thereby. Prior to April 1 of each year, the Tree Inspector shall inspect all public and private properties for elmwood logs or stumps that could serve as bark beetle breeding sites, and require by April 1, removal or debarking of all wood, logs or stumps to be retained."

Output:
{
  "entities": [
    {"name": "Council", "type": "organization"},
    {"name": "Tree Inspector", "type": "role"},
    {"name": "Public Works Director", "type": "role"},
    {"name": "State Commissioner of Agriculture", "type": "role"},
  ]
}

Explanation: The Council, Tree Inspector, Public Works Director, and State Commissioner of Agriculture are all specific proper-named entities. M.S. §§ 89.001 et seq. is excluded as it is a legal reference.
"""

load_dotenv()


def _connect_duckdb() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def count_city_sections(
    parquet_path: Path, city_slug: str, jurisdiction_name: str, state_code: str
) -> int:
    conn = _connect_duckdb()
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM '{parquet_path}' WHERE city_slug = ? AND jurisdiction_name = ? AND state_code = ?",
            [city_slug, jurisdiction_name, state_code],
        ).fetchone()[0]
    finally:
        conn.close()


def count_city_words(
    parquet_path: Path, city_slug: str, jurisdiction_name: str, state_code: str
) -> int:
    conn = _connect_duckdb()
    try:
        df = conn.execute(
            f"SELECT content FROM '{parquet_path}' WHERE city_slug = ? AND jurisdiction_name = ? AND state_code = ?",
            [city_slug, jurisdiction_name, state_code],
        ).df()
        return sum(len(str(c).split()) for c in df["content"].dropna())
    finally:
        conn.close()


def iter_city_rows(
    parquet_path: Path,
    city_slug: str,
    jurisdiction_name: str,
    state_code: str,
    batch_size: int = 1000,
) -> Iterable[SectionRow]:
    """Stream rows for a city in batches to avoid memory spikes."""
    conn = _connect_duckdb()
    try:
        total = conn.execute(
            f"SELECT COUNT(*) FROM '{parquet_path}' WHERE city_slug = ? AND jurisdiction_name = ? AND state_code = ?",
            [city_slug, jurisdiction_name, state_code],
        ).fetchone()[0]

        offset = 0
        while offset < total:
            df = conn.execute(
                f"""
                SELECT source, source_format, city_slug, jurisdiction_name, state, 
                       state_code, source_file, node_id, section_heading,
                       hierarchy_path, content, tiktoken, word_tokens_approx,
                       tfidf_score, tfidf_label
                FROM '{parquet_path}'
                WHERE city_slug = ? AND jurisdiction_name = ? AND state_code = ?
                LIMIT ? OFFSET ?
                """,
                [city_slug, jurisdiction_name, state_code, batch_size, offset],
            ).df()

            for rec in df.to_dict("records"):
                yield SectionRow(**rec)

            offset += batch_size
    finally:
        conn.close()


def _with_retries(fn, *, retries=5, base=0.5, jitter=0.25):
    """Simple exponential backoff for transient errors."""
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            sleep = base * (2**i) + random.random() * jitter
            print(f"Transient error: {e} — retrying in {sleep:.2f}s")
            time.sleep(sleep)


_WS = re.compile(r"\s+")


def _normalize_entity(e: str) -> str:
    e = e.strip()
    e = _WS.sub(" ", e)
    return e


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for raw in items:
        if not isinstance(raw, str):
            continue
        n = _normalize_entity(raw)
        key = n.lower()
        if key not in seen and n:
            seen.add(key)
            out.append(n)
    return out


def extract_entities_from_content(
    cache: types.CachedContent,
    client: genai.Client,
    model_name: str,
    content: str,
    node_id: str,
    section_heading: Optional[str] = None,
    hierarchy_path: Optional[str] = None,
    debug: bool = False,
) -> ExtractedEntities:
    """Extract entities using Gemini in JSON-mode with retries."""

    # The prompt is just the content since ENTITY_EXTRACTION_PROMPT is in the cached system instruction
    prompt = content

    if debug:
        logger.debug("=" * 80)
        logger.debug(f"PROCESSING SECTION: {node_id}")
        logger.debug("=" * 80)
        logger.debug(f"Section Heading: {section_heading}")
        logger.debug(f"Hierarchy Path: {hierarchy_path}")
        logger.debug("-" * 80)
        logger.debug("CONTENT SENT TO LLM:")
        logger.debug("-" * 80)
        logger.debug(content)
        logger.debug("=" * 80)

    try:
        resp = _with_retries(
            lambda: client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    cached_content=cache.name,
                    temperature=0.1,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        )

        if debug:
            print(resp.usage_metadata)

        # Clean response text
        response_text = resp.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:].strip()

        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        # Parse JSON
        parsed = json.loads(response_text)
        entities_raw = parsed.get("entities", [])

        # Process entities - handle both string and object formats
        entity_objects = []
        seen = set()

        for entity in entities_raw:
            if isinstance(entity, str):
                # Old format: just a string
                name = _normalize_entity(entity)
                entity_type = "other"
            elif isinstance(entity, dict):
                # New format: object with name and type
                name = _normalize_entity(entity.get("name", ""))
                entity_type = entity.get("type", "other")

                # Validate entity_type
                if entity_type not in [
                    "organization",
                    "location",
                    "role",
                    "event",
                    "other",
                ]:
                    entity_type = "other"
            else:
                continue

            # Deduplicate by normalized name (case-insensitive)
            key = name.lower()
            if key and key not in seen:
                seen.add(key)
                entity_objects.append(Entity(name=name, type=entity_type))

        return ExtractedEntities(
            entities=entity_objects,
            node_id=node_id,
            section_heading=section_heading,
            hierarchy_path=hierarchy_path,
            content=content,
        )
    except Exception as e:
        print(f"\nError processing section {node_id}: {e}")
        if "resp" in locals():
            print(f"Raw response (first 500 chars): {resp.text[:500]}")
        return ExtractedEntities(
            entities=[],
            node_id=node_id,
            section_heading=section_heading,
            hierarchy_path=hierarchy_path,
            content=content,
        )


def load_existing_results(path: Path) -> List[ExtractedEntities]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ExtractedEntities(**item) for item in data]


def save_results(results: List[ExtractedEntities], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)


def process_city(
    city_slug: str,
    jurisdiction_name: str,
    state_code: str,
    client: genai.Client,
    model_name: str,
    output_dir: Path,
    parquet_path: Path,
    yes: bool,
    resume: bool,
    batch_size: int = 1000,
    autosave_every: int = 25,
    debug: bool = False,
) -> None:
    print(f"\n{'='*80}")
    print(f"Processing: {jurisdiction_name} ({city_slug}), {state_code}")
    print("=" * 80)

    # Determine work size
    n_sections = count_city_sections(
        parquet_path, city_slug, jurisdiction_name, state_code
    )
    n_words = count_city_words(parquet_path, city_slug, jurisdiction_name, state_code)
    if n_sections == 0:
        print(f"No data found for: {jurisdiction_name} ({city_slug}), {state_code}")
        return
    print(f"Found {n_sections} sections with {n_words} words")

    # Calculate costs
    estimated_cost = (
        calculate_input_price_for_extraction(model_name, n_words)
        + calculate_output_price_for_extraction(model_name, n_sections)
        + calculate_context_caching_price_for_extraction(
            model_name, ENTITY_EXTRACTION_PROMPT
        )
    )
    print(f"Estimated cost: {estimated_cost}")

    # Confirm if interactive is desired
    if not yes:
        resp = input("\nContinue with entity extraction? (y/n): ").strip().lower()
        if resp not in ("y", "yes"):
            print("Skipping this city.")
            return

    # Output file - use city_slug and state_code for uniqueness
    output_file = (
        output_dir
        / f"{city_slug}_{state_code}_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Resume (optional)
    results: List[ExtractedEntities] = []
    processed_node_ids = set()

    if output_file.exists() and (
        resume
        or (
            not resume
            and input(f"Found {output_file}. Resume from existing results? (y/n): ")
            .strip()
            .lower()
            in ("y", "yes")
        )
    ):
        try:
            results = load_existing_results(output_file)
            processed_node_ids = {r.node_id for r in results}
            print(f"Loaded {len(results)} existing results (will skip these sections).")
        except Exception as e:
            print(f"Failed to load existing results (continuing fresh): {e}")

    # Stream rows, skipping already processed
    processed_since_save = 0
    rows_iter = iter_city_rows(
        parquet_path, city_slug, jurisdiction_name, state_code, batch_size=batch_size
    )
    to_process = n_sections - len(processed_node_ids)
    if to_process <= 0:
        print("All sections already processed!")
        return

    print(
        f"\nProcessing {to_process} sections (skipping {len(processed_node_ids)} already done)..."
    )

    # Set up context caching
    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            display_name="extraction_instruction",
            system_instruction=ENTITY_EXTRACTION_PROMPT,
        ),
    )

    with tqdm(
        total=to_process,
        desc=f"Extracting entities from {jurisdiction_name}, {state_code}",
    ) as pbar:
        for row in rows_iter:
            if row.node_id in processed_node_ids:
                continue

            extracted = extract_entities_from_content(
                cache=cache,
                client=client,
                model_name=model_name,
                content=row.content,
                node_id=row.node_id,
                section_heading=row.section_heading,
                hierarchy_path=row.hierarchy_path,
                debug=debug,
            )
            results.append(extracted)
            processed_node_ids.add(row.node_id)
            processed_since_save += 1
            pbar.update(1)

            if processed_since_save >= autosave_every:
                save_results(results, output_file)
                processed_since_save = 0

    # Final save
    save_results(results, output_file)
    print(f"\n✓ Results saved to: {output_file}")
    print(
        f"✓ Processed {len(results)} total sections for {jurisdiction_name} ({city_slug}), {state_code}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from municipal code sections using Gemini API"
    )
    parser.add_argument(
        "cities",
        nargs="+",
        help="One or more cities to process in format 'city_slug:jurisdiction_name:state_code' (e.g., 'aurora-co:Aurora:CO', 'minneapolis-mn:Minneapolis:MN')",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save output JSON files (default: output/)",
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
        "--yes",
        action="store_true",
        help="Run non-interactively (auto-confirm prompts)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, resume from it automatically",
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

    # Set up client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Process each city
    for city_spec in args.cities:
        try:
            # Parse city specification in format: city_slug:jurisdiction_name:state_code
            parts = city_spec.split(":")
            if len(parts) != 3:
                print(f"\nError: Invalid city specification '{city_spec}'")
                print("Expected format: 'city_slug:jurisdiction_name:state_code'")
                print("Example: 'aurora-co:Aurora:CO'")
                continue

            city_slug, jurisdiction_name, state_code = parts

            process_city(
                city_slug=city_slug,
                jurisdiction_name=jurisdiction_name,
                state_code=state_code,
                client=client,
                model_name=args.model,
                output_dir=args.output_dir,
                parquet_path=args.parquet_path,
                yes=args.yes,
                resume=args.resume,
                batch_size=args.batch_size,
                autosave_every=args.autosave_every,
                debug=args.debug,
            )
        except Exception as e:
            print(f"\nError processing city {city_spec}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("All cities processed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
