"""
Summarize analyzed entities and convert to CSV format.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, List


def load_data(file_path: Path) -> List[dict[str, Any]]:
    """Load the analyzed entities JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_entities(data: List[dict[str, Any]]) -> dict:
    """Generate summary statistics for the entities."""
    total_entities = 0
    exists_true = 0
    exists_false = 0
    exists_null = 0

    nonexistent_entities = []

    for section in data:
        entities = section.get("entities", [])
        for entity in entities:
            total_entities += 1
            exists = entity.get("exists")

            if exists is True:
                exists_true += 1
            elif exists is False:
                exists_false += 1
                nonexistent_entities.append(entity.get("name", ""))
            else:  # None/null
                exists_null += 1

    # Deduplicate nonexistent entities (case-insensitive)
    nonexistent_lower = [name.lower().strip() for name in nonexistent_entities if name]
    deduplicated_nonexistent = set(nonexistent_lower)

    return {
        "total_entities": total_entities,
        "exists_true": exists_true,
        "exists_false": exists_false,
        "exists_null": exists_null,
        "nonexistent_entities": nonexistent_entities,
        "deduplicated_nonexistent": deduplicated_nonexistent,
    }


def sample_entities(data: List[dict[str, Any]], sample_size: int = 10) -> List[dict]:
    """Sample random non-existent entities from the dataset."""
    nonexistent_entities = []

    for section in data:
        entities = section.get("entities", [])
        for entity in entities:
            # Only include entities where exists=False
            if entity.get("exists") is False:
                # Add section context to the entity
                entity_with_context = {
                    **entity,
                    "section_id": section.get("section_id"),
                    "section_heading": section.get("section_heading"),
                }
                nonexistent_entities.append(entity_with_context)

    if len(nonexistent_entities) <= sample_size:
        return nonexistent_entities

    return random.sample(nonexistent_entities, sample_size)


def convert_to_csv(data: List[dict[str, Any]], output_path: Path) -> None:
    """Convert JSON data to CSV format."""
    rows = []

    for section in data:
        section_id = section.get("section_id", "")
        section_heading = section.get("section_heading", "")
        citation = section.get("citation", "")
        hierarchy_path = section.get("hierarchy_path", "")
        content = section.get("content", "")

        entities = section.get("entities", [])

        for entity in entities:
            # Format citations as a single string
            citations = entity.get("citations", [])
            citations_str = " | ".join(
                [f"{c.get('url', '')}: {c.get('quote', '')}" for c in citations]
            )

            row = {
                "section_id": section_id,
                "section_heading": section_heading,
                "citation": citation,
                "hierarchy_path": hierarchy_path,
                "entity_name": entity.get("name", ""),
                "entity_type": entity.get("type", ""),
                "processed": entity.get("processed", ""),
                "exists": entity.get("exists", ""),
                "nonexistence_status": entity.get("nonexistence_status", ""),
                "reasoning": entity.get("reasoning", ""),
                "citations": citations_str,
                "content_excerpt": (
                    (content[:200] + "...") if len(content) > 200 else content
                ),
            }
            rows.append(row)

    # Write to CSV
    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def print_summary(summary: dict, samples: List[dict]) -> None:
    """Print a formatted summary to console."""
    print("=" * 80)
    print("ENTITY ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    print(f"Total entities: {summary['total_entities']:,}")
    print(
        f"  - Exist (exists=true): {summary['exists_true']:,} "
        f"({summary['exists_true']/summary['total_entities']*100:.1f}%)"
    )
    print(
        f"  - Don't exist (exists=false): {summary['exists_false']:,} "
        f"({summary['exists_false']/summary['total_entities']*100:.1f}%)"
    )
    print(
        f"  - Unknown (exists=null): {summary['exists_null']:,} "
        f"({summary['exists_null']/summary['total_entities']*100:.1f}%)"
    )
    print()

    print(f"Nonexistent entities (exists=false):")
    print(f"  - Total occurrences: {len(summary['nonexistent_entities']):,}")
    print(f"  - Deduplicated count: {len(summary['deduplicated_nonexistent']):,}")
    print()

    # Show most common nonexistent entities
    if summary["nonexistent_entities"]:
        counter = Counter(
            [e.lower().strip() for e in summary["nonexistent_entities"] if e]
        )
        print(f"Top 10 most common nonexistent entities:")
        for entity, count in counter.most_common(10):
            print(f"  - {entity}: {count} occurrence(s)")
        print()

    print("=" * 80)
    print(f"SAMPLE OF {len(samples)} NON-EXISTENT ENTITIES (exists=false)")
    print("=" * 80)
    print()

    for i, entity in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(f"  Name: {entity.get('name', 'N/A')}")
        print(f"  Type: {entity.get('entity_type', entity.get('type', 'N/A'))}")
        print(f"  Exists: {entity.get('exists', 'N/A')}")
        print(f"  Nonexistence Status: {entity.get('nonexistence_status', 'N/A')}")
        print(f"  Section: {entity.get('section_id', 'N/A')}")
        print(f"  Heading: {entity.get('section_heading', 'N/A')}")
        print(f"  Reasoning: {entity.get('reasoning', 'N/A')[:200]}...")

        citations = entity.get("citations", [])
        if citations:
            print(f"  Citations:")
            for cite in citations[:2]:  # Show first 2 citations
                print(f"    - {cite.get('url', 'N/A')}")
                print(f"      Quote: {cite.get('quote', 'N/A')[:100]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Summarize analyzed entities and convert to CSV"
    )
    parser.add_argument(
        "input_file", type=Path, help="Input JSON file with analyzed entities"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV file path (default: same name as input with .csv extension)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of non-existent entities to sample for display (default: 10)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV generation",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} sections")
    print()

    # Generate summary
    summary = summarize_entities(data)

    # Sample entities
    samples = sample_entities(data, args.sample_size)

    # Print summary
    print_summary(summary, samples)

    # Generate CSV
    if not args.no_csv:
        if args.output_csv:
            csv_path = args.output_csv
        else:
            csv_path = args.input_file.with_suffix(".csv")

        print("=" * 80)
        print(f"Converting to CSV: {csv_path}")
        convert_to_csv(data, csv_path)
        print(f"✓ CSV file created successfully")
        print(f"  Rows: {summary['total_entities']:,}")
        print("=" * 80)


if __name__ == "__main__":
    main()
