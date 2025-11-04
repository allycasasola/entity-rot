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
import google.generativeai as genai
from pydantic import BaseModel
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

EntityType = Literal["organization", "location", "role", "event", "other"]


class Entity(BaseModel):
    name: str
    type: EntityType


class ExtractedEntities(BaseModel):
    """Schema for extracted entities from a section."""

    entities: List[Entity]
    section_id: str  # always present now
    section_heading: Optional[str] = None
    citation: Optional[str] = None
    hierarchy_path: Optional[str] = None
    content: Optional[str] = None


@dataclass
class SectionRow:
    city: str
    source_file: str
    section_id: str
    section_heading: Optional[str]
    citation: Optional[str]
    hierarchy_path: Optional[str]
    content: str
    token_count: Optional[int]
    chunk_ids: Optional[str]


DEFAULT_PATH_TO_PARQUET_FILE = "/oak/stanford/groups/deho/dbateyko/municipal_codes/data/output/municode_sections.parquet"
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"

ENTITY_EXTRACTION_PROMPT = """
You are an expert at extracting *specific named entities* from municipal code text. 
Your goal is to identify only *properly named and distinct entities*, not generic categories or references.

### Instructions

Extract entities and classify each into one of the following types:
- **organization**: Only include *specific* organizations or formally named bodies (e.g., "City of Aurora Planning and Zoning Commission", "Federal Emergency Management Agency").
  Exclude generic terms like “city,” “board,” “department,” or “council” unless they appear as part of a *full proper name*.
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
- Prefer named entities that begin with capital letters or include formal titles.
- Do **not** include duplicates or repeated generic placeholders.
- If unsure whether an entity is specific enough, **include it**.

### Output Format
Return **only** a JSON object with this exact structure:

{
  "entities": [
    {"name": "entity text here", "type": "organization"},
    {"name": "another entity", "type": "location"}
  ]
}

### Text
{content}
"""
load_dotenv()


def setup_gemini_api(model_name: str) -> genai.GenerativeModel:
    """Initialize and configure the Gemini API with JSON-mode."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it with: export GEMINI_API_KEY='your-api-key'"
        )
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        # Force the model to emit valid JSON (no code fences)
        "response_mime_type": "application/json",
    }

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )


def _connect_duckdb() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def count_city_sections(parquet_path: Path, city_name: str) -> int:
    conn = _connect_duckdb()
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM '{parquet_path}' WHERE city = ?", [city_name]
        ).fetchone()[0]
    finally:
        conn.close()


def count_city_words(parquet_path: Path, city_name: str) -> int:
    conn = _connect_duckdb()
    try:
        df = conn.execute(
            f"SELECT content FROM '{parquet_path}' WHERE city = ?",
            [city_name],
        ).df()
        return sum(len(str(c).split()) for c in df["content"].dropna())
    finally:
        conn.close()


def iter_city_rows(
    parquet_path: Path,
    city_name: str,
    batch_size: int = 1000,
) -> Iterable[SectionRow]:
    """Stream rows for a city in batches to avoid memory spikes."""
    conn = _connect_duckdb()
    try:
        total = conn.execute(
            f"SELECT COUNT(*) FROM '{parquet_path}' WHERE city = ?", [city_name]
        ).fetchone()[0]

        offset = 0
        while offset < total:
            df = conn.execute(
                f"""
                SELECT city, source_file, section_id, section_heading, citation,
                       hierarchy_path, content, token_count, chunk_ids
                FROM '{parquet_path}'
                WHERE city = ?
                LIMIT ? OFFSET ?
                """,
                [city_name, batch_size, offset],
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
    model: genai.GenerativeModel,
    content: str,
    section_id: str,
    section_heading: Optional[str] = None,
    citation: Optional[str] = None,
    hierarchy_path: Optional[str] = None,
) -> ExtractedEntities:
    """Extract entities using Gemini in JSON-mode with retries."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(content=content)
    try:
        resp = _with_retries(lambda: model.generate_content(prompt))

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
            section_id=section_id,
            section_heading=section_heading,
            citation=citation,
            hierarchy_path=hierarchy_path,
            content=content,
        )
    except Exception as e:
        print(f"\nError processing section {section_id}: {e}")
        if "resp" in locals():
            print(f"Raw response (first 500 chars): {resp.text[:500]}")
        return ExtractedEntities(
            entities=[],
            section_id=section_id,
            section_heading=section_heading,
            citation=citation,
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
    city_name: str,
    model: genai.GenerativeModel,
    output_dir: Path,
    parquet_path: Path,
    yes: bool,
    resume: bool,
    batch_size: int = 1000,
    autosave_every: int = 25,
) -> None:
    print(f"\n{'='*80}")
    print(f"Processing city: {city_name}")
    print("=" * 80)

    # Determine work size
    n_sections = count_city_sections(parquet_path, city_name)
    n_words = count_city_words(parquet_path, city_name)
    if n_sections == 0:
        print(f"No data found for city: {city_name}")
        return
    print(f"Found {n_sections} sections for {city_name} with {n_words} words")

    # Confirm if interactive is desired
    if not yes:
        resp = input("\nContinue with entity extraction? (y/n): ").strip().lower()
        if resp not in ("y", "yes"):
            print("Skipping this city.")
            return

    # Output file
    output_file = (
        output_dir
        / f"{city_name.lower().replace(' ', '_')}_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Resume (optional)
    results: List[ExtractedEntities] = []
    processed_section_ids = set()

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
            processed_section_ids = {r.section_id for r in results}
            print(f"Loaded {len(results)} existing results (will skip these sections).")
        except Exception as e:
            print(f"Failed to load existing results (continuing fresh): {e}")

    # Stream rows, skipping already processed
    processed_since_save = 0
    rows_iter = iter_city_rows(parquet_path, city_name, batch_size=batch_size)
    to_process = n_sections - len(processed_section_ids)
    if to_process <= 0:
        print("All sections already processed!")
        return

    print(
        f"\nProcessing {to_process} sections (skipping {len(processed_section_ids)} already done)..."
    )

    with tqdm(total=to_process, desc=f"Extracting entities from {city_name}") as pbar:
        for row in rows_iter:
            if row.section_id in processed_section_ids:
                continue

            extracted = extract_entities_from_content(
                model=model,
                content=row.content,
                section_id=row.section_id,
                section_heading=row.section_heading,
                citation=row.citation,
                hierarchy_path=row.hierarchy_path,
            )
            results.append(extracted)
            processed_section_ids.add(row.section_id)
            processed_since_save += 1
            pbar.update(1)

            if processed_since_save >= autosave_every:
                save_results(results, output_file)
                processed_since_save = 0

    # Final save
    save_results(results, output_file)
    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Processed {len(results)} total sections for {city_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from municipal code sections using Gemini API"
    )
    parser.add_argument("cities", nargs="+", help="One or more city names to process")

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

    args = parser.parse_args()

    # Setup Gemini API
    print("Initializing Gemini API...")
    try:
        model = setup_gemini_api(args.model)
        print(f"✓ Using model: {args.model}")
    except Exception as e:
        print(f"Error setting up Gemini API: {e}")
        return

    # Process each city
    for city_name in args.cities:
        try:
            process_city(
                city_name=city_name,
                model=model,
                output_dir=args.output_dir,
                parquet_path=args.parquet_path,
                yes=args.yes,
                resume=args.resume,
                batch_size=args.batch_size,
                autosave_every=args.autosave_every,
            )
        except Exception as e:
            print(f"\nError processing city {city_name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("All cities processed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
