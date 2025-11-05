"""
Analyze extracted entities using LLM with web search to assess existence (batched for cost efficiency).
- Uses Google Search grounding via the Google GenAI SDK.
- Returns strictly JSON (the model is configured for application/json).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Literal

from google import generativeai as genai
from google.generativeai import types



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
    entities: List[Entity]
    section_id: str
    section_heading: Optional[str] = None
    citation: Optional[str] = None
    hierarchy_path: Optional[str] = None
    content: Optional[str] = None

DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# Note the prompt supports multiple sections per request.
BATCH_ANALYSIS_PROMPT = """You are an expert at assessing whether entities within a specific jurisdiction still exist.

**Municipality:** {city}, {state}

You will be given multiple municipal code sections to analyze. For EACH section, independently assess entities using Google Search grounding as needed.
Use only reliable sources (official .gov sites preferred) and provide short quotes with URLs to justify determinations.

Return a JSON list, where each element corresponds to a single section.

---

### Sections to analyze:
{sections_text}

---

### Output Format

Return ONLY a JSON array of this structure:
[
  {{
    "section_id": "...",
    "entities": [
      {{
        "name": "...",
        "type": "...",
        "processed": "...",
        "exists": true/false/null,
        "nonexistence_status": "... or null",
        "citations": [{{"url": "...", "quote": "..."}}],
        "reasoning": "..."
      }}
    ]
  }}
]
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
        max_output_tokens=8192,
        response_mime_type="application/json",  # ensures JSON-only responses
    )

    return client, config, model_name


def load_data(file_path: Path) -> List[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: List[EvaluatedEntities], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)


def analyze_section_batch(
    client: genai.Client,
    config: types.GenerateContentConfig,
    model_name: str,
    batch: List[dict[str, Any]],
    city: str,
    state: str,
) -> List[EvaluatedEntities]:
    """Analyze a batch of sections in one grounded request."""
    # Build text for all sections in the batch
    sections_text = ""
    for i, section in enumerate(batch):
        entities = section.get("entities", [])
        entity_list = (
            "\n".join(
                f"- {j+1}. Name: \"{e.get('name', '')}\", Type: {e.get('type', 'unknown')}"
                for j, e in enumerate(entities)
            )
            or "(No entities)"
        )
        excerpt = section.get("content", "") or ""
        excerpt = (excerpt[:500] + "...") if len(excerpt) > 500 else excerpt

        sections_text += (
            f"\n\n---\nSection {i+1}\n"
            f"ID: {section.get('section_id')}\n"
            f"Heading: {section.get('section_heading')}\n"
            f"Citation: {section.get('citation')}\n"
            f"Entities:\n{entity_list}\n"
            f"Content excerpt: {excerpt}\n"
        )

    prompt = BATCH_ANALYSIS_PROMPT.format(
        city=city, state=state, sections_text=sections_text
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # ✅ New SDK calling style with config (tools) attached
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            # The model is constrained to JSON via response_mime_type,
            # but we still defensively strip code fences if any slipped in.
            text = (response.text or "").strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]

            parsed = json.loads(text)

            results: List[EvaluatedEntities] = []
            for section_result, original in zip(parsed, batch):
                entities = [Entity(**e) for e in section_result.get("entities", [])]
                results.append(
                    EvaluatedEntities(
                        entities=entities,
                        section_id=original.get("section_id"),
                        section_heading=original.get("section_heading"),
                        citation=original.get("citation"),
                        hierarchy_path=original.get("hierarchy_path"),
                        content=original.get("content"),
                    )
                )
            return results

        except Exception as err:
            if attempt < max_retries - 1:
                wait = 2**attempt + random.random()
                print(f"Retry {attempt+1}: {err} (wait {wait:.1f}s)")
                time.sleep(wait)
            else:
                print(f"Failed batch: {err}")
                # Return "skipped" for each entity in each section
                fallback_results: List[EvaluatedEntities] = []
                for s in batch:
                    fallback_entities: List[Entity] = []
                    for ent in s.get("entities", []):
                        fallback_entities.append(
                            Entity(
                                name=ent.get("name", ""),
                                type=ent.get("type", "other"),
                                processed="skipped",
                                exists=None,
                                nonexistence_status=None,
                                citations=[],
                                reasoning=f"Error: {str(err)}",
                            )
                        )
                    fallback_results.append(
                        EvaluatedEntities(
                            entities=fallback_entities,
                            section_id=s.get("section_id"),
                            section_heading=s.get("section_heading"),
                            citation=s.get("citation"),
                            hierarchy_path=s.get("hierarchy_path"),
                            content=s.get("content"),
                        )
                    )
                return fallback_results


def main():
    parser = argparse.ArgumentParser(description="Analyze entities (batched)")
    parser.add_argument("file_path", type=Path)
    parser.add_argument("city")
    parser.add_argument("state")
    parser.add_argument("--output-dir", type=Path, default=Path("output/analyzed"))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--batch-size", type=int, default=3, help="Number of sections per prompt"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    data = load_data(args.file_path)
    if args.limit:
        data = data[: args.limit]

    client, config, model_name = setup_gemini_api(args.model)
    results: List[EvaluatedEntities] = []
    batch: List[dict[str, Any]] = []

    output_file = args.output_dir / f"{args.file_path.stem}_batched.json"

    with tqdm(total=len(data), desc="Analyzing batched entities") as pbar:
        for section in data:
            batch.append(section)
            if len(batch) >= args.batch_size:
                batch_results = analyze_section_batch(
                    client, config, model_name, batch, args.city, args.state
                )
                results.extend(batch_results)
                batch.clear()
                save_results(results, output_file)
                pbar.update(args.batch_size)

        # leftover batch
        if batch:
            batch_results = analyze_section_batch(
                client, config, model_name, batch, args.city, args.state
            )
            results.extend(batch_results)
            save_results(results, output_file)
            pbar.update(len(batch))

    print(f"\n✓ Saved batched results to {output_file}")


if __name__ == "__main__":
    main()
