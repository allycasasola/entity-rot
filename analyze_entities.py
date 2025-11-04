"""
Analyze extracted entities using LLM with web search to assess existence.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Literal

# Schema definitions
EntityType = Literal["organization", "location", "role", "event", "other"]
NonexistenceStatus = Literal[
    "abolished", "merged", "replaced", "renamed", "dormant", "expired", "defunct"
]
ProcessStatus = Literal["skipped", "processed"]


class Citation(BaseModel):
    url: str
    quote: str


class Entity(BaseModel):
    name: str
    type: EntityType
    processed: ProcessStatus
    exists: bool | None
    nonexistence_status: NonexistenceStatus | None
    citations: List[Citation]
    reasoning: str


class EvaluatedEntities(BaseModel):
    """Schema for evaluated entities from a section."""

    entities: List[Entity]
    section_id: str
    section_heading: Optional[str] = None
    citation: Optional[str] = None
    hierarchy_path: Optional[str] = None
    content: Optional[str] = None


DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

ANALYSIS_PROMPT = """You are an expert at assessing whether entities within a specific jurisdiction still exist.

**Municipality:** {city}, {state}

**Municipal Code Section:**
- Section ID: {section_id}
- Heading: {section_heading}
- Citation: {citation}
- Hierarchy: {hierarchy_path}
- Content excerpt: {content_excerpt}

**Entities to Analyze:**
{entity_list}

**Instructions:**

For each entity listed above, follow these steps:

1. **Assess Specificity:** Determine if the entity name is specific enough to be a proper noun (i.e., a formally named entity, not a generic reference like "city" or "department"). 
   - If NOT specific enough → Set processed="skipped", exists=null, provide brief reasoning.
   - If specific enough → Continue to step 2.

2. **Check Existence:** Use Google Search to determine if this entity currently exists in {city}, {state}.
   - Search for the entity name combined with the municipality name.
   - Look for official websites, news articles, government documents.

3. **Determine Status:**
   - **If entity EXISTS:** Set exists=true, provide URL(s) and relevant quotes showing it exists.
   - **If entity DOES NOT EXIST:** Set exists=false, categorize the nonexistence_status as one of:
     * "abolished" - formally dissolved or eliminated
     * "merged" - absorbed into another entity
     * "replaced" - a new entity took over its functions
     * "renamed" - same core entity but name/structure changed
     * "dormant" - legally exists but no activity or staff
     * "expired" - ended automatically due to time limit
     * "defunct" - ceased to exist, unclear how/when
   - **If UNCLEAR:** Set exists=null, explain why it's unclear.

4. **Provide Evidence:** Include citations with:
   - url: The web source
   - quote: Specific text from the source supporting your conclusion

5. **Reasoning:** Explain your assessment process and conclusion.

**Output Format:**

Return ONLY a JSON object with this structure:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type",
      "processed": "processed" or "skipped",
      "exists": true/false/null,
      "nonexistence_status": "category" or null,
      "citations": [
        {{"url": "...", "quote": "..."}}
      ],
      "reasoning": "explanation"
    }}
  ]
}}

Be thorough and evidence-based in your assessment."""


load_dotenv()


def setup_gemini_api(model_name: str) -> genai.GenerativeModel:
    """Initialize Gemini API with Google Search grounding."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it with: export GEMINI_API_KEY='your-api-key'"
        )
    genai.configure(api_key=api_key)

    # Configure with Google Search grounding
    tools = [
        genai.protos.Tool(google_search_retrieval=genai.protos.GoogleSearchRetrieval())
    ]

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        tools=tools,
    )


def load_data(file_path: Path) -> List[dict[str, Any]]:
    """Load JSON data file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: List[EvaluatedEntities], output_path: Path) -> None:
    """Save evaluated results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)


def analyze_section_entities(
    model: genai.GenerativeModel,
    section_data: dict[str, Any],
    city: str,
    state: str,
) -> EvaluatedEntities:
    """Analyze all entities in a section."""

    entities = section_data.get("entities", [])
    section_id = section_data.get("section_id", "")
    section_heading = section_data.get("section_heading", "")
    citation = section_data.get("citation", "")
    hierarchy_path = section_data.get("hierarchy_path", "")
    content_excerpt = section_data.get("content_excerpt", "")

    # Truncate content for prompt
    content_excerpt = 

    # Format entity list
    entity_list = "\n".join(
        f"- {i+1}. Name: \"{e.get('name', '')}\", Type: {e.get('type', 'unknown')}"
        for i, e in enumerate(entities)
    )

    if not entity_list:
        entity_list = "(No entities to analyze)"

    # Build prompt
    prompt = ANALYSIS_PROMPT.format(
        city=city,
        state=state,
        section_id=section_id,
        section_heading=section_heading,
        citation=citation,
        hierarchy_path=hierarchy_path,
        content_excerpt=content_excerpt,
        entity_list=entity_list,
    )

    try:
        # Call model with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)

                # Clean response text
                response_text = response.text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:].strip()

                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()

                # Parse JSON
                parsed = json.loads(response_text)
                evaluated_entities = [Entity(**e) for e in parsed.get("entities", [])]

                return EvaluatedEntities(
                    entities=evaluated_entities,
                    section_id=section_id,
                    section_heading=section_heading,
                    citation=citation,
                    hierarchy_path=hierarchy_path,
                    content=content,
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.random()
                    print(
                        f"  Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    raise

    except Exception as e:
        print(f"\nError analyzing section {section_id}: {e}")
        if "response" in locals():
            print(f"Response (first 500 chars): {response.text[:500]}")

        # Return with skipped entities
        skipped_entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "other"),
                processed="skipped",
                exists=None,
                nonexistence_status=None,
                citations=[],
                reasoning=f"Error during analysis: {str(e)}",
            )
            for e in entities
        ]

        return EvaluatedEntities(
            entities=skipped_entities,
            section_id=section_id,
            section_heading=section_heading,
            citation=citation,
            hierarchy_path=hierarchy_path,
            content=content,
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze entities using LLM with Google Search grounding"
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to JSON file containing extracted entities",
    )
    parser.add_argument("city", help="City name")
    parser.add_argument("state", help="State name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analyzed"),
        help="Directory to save analyzed results (default: output/analyzed/)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--autosave-every",
        type=int,
        default=10,
        help="Autosave after this many sections (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of sections to process (for testing)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.file_path.exists():
        print(f"Error: Input file not found: {args.file_path}")
        return

    print(f"\n{'='*80}")
    print(f"Analyzing Entities: {args.city}, {args.state}")
    print("=" * 80)
    print(f"Input file: {args.file_path}")
    print(f"Model: {args.model}")

    # Setup Gemini API
    print("\nInitializing Gemini API with Google Search grounding...")
    try:
        model = setup_gemini_api(args.model)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"Error setting up Gemini API: {e}")
        return

    # Load data
    print(f"\nLoading data from {args.file_path}...")
    data = load_data(args.file_path)

    if args.limit:
        data = data[: args.limit]
        print(f"Limited to first {args.limit} sections for testing")

    print(f"Loaded {len(data)} sections")

    # Prepare output file
    input_stem = args.file_path.stem
    output_file = args.output_dir / f"{input_stem}_analyzed.json"

    # Process each section
    results = []
    processed_count = 0
    skipped_sections = 0

    print(f"\nProcessing sections...")

    with tqdm(total=len(data), desc="Analyzing entities") as pbar:
        for section_data in data:
            try:
                # Skip sections with no entities
                entities = section_data.get("entities", [])
                if not entities:
                    skipped_sections += 1
                    pbar.update(1)
                    continue

                evaluated = analyze_section_entities(
                    model=model,
                    section_data=section_data,
                    city=args.city,
                    state=args.state,
                )
                results.append(evaluated)
                processed_count += 1

                # Autosave
                if processed_count % args.autosave_every == 0:
                    save_results(results, output_file)

                pbar.update(1)

            except Exception as e:
                print(f"\nFailed to process section: {e}")
                pbar.update(1)
                continue

    # Final save
    print(f"\nSaving results to: {output_file}")
    save_results(results, output_file)

    # Summary statistics
    total_entities = sum(len(r.entities) for r in results)
    processed_entities = sum(
        1 for r in results for e in r.entities if e.processed == "processed"
    )
    skipped_entities = total_entities - processed_entities
    exists_count = sum(1 for r in results for e in r.entities if e.exists is True)
    not_exists_count = sum(1 for r in results for e in r.entities if e.exists is False)
    unclear_count = sum(
        1
        for r in results
        for e in r.entities
        if e.exists is None and e.processed == "processed"
    )

    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total sections in file: {len(data)}")
    print(f"Sections skipped (no entities): {skipped_sections}")
    print(f"Sections analyzed: {len(results)}")
    print(f"Total entities: {total_entities}")
    print(f"  Processed: {processed_entities}")
    print(f"  Skipped: {skipped_entities}")
    print(f"\nExistence Status:")
    print(f"  Exists: {exists_count}")
    print(f"  Does not exist: {not_exists_count}")
    print(f"  Unclear: {unclear_count}")
    print("=" * 80)
    print(f"\n✓ Analysis complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
